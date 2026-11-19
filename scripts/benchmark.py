#!/usr/bin/env python3
"""Head-to-head summarization benchmark: LiteRT-LM vs llama.cpp on IBM Granite 4.0 1B.

Measures both:
  1. Synthetic prefill/decode microbenchmarks (litert-lm benchmark vs llama-bench)
  2. Real CNN/DailyMail summary-v3 workload (TTFT, prefill tok/s, decode tok/s,
     end-to-end latency, total wall time, and ROUGE-1/2/L parity under greedy decoding)

Takes full advantage of Apple Silicon hardware on macOS:
  - GPU mode: LiteRT-LM Metal GPU vs llama.cpp Metal GPU (-ngl 99 -fa on)
  - CPU mode: LiteRT-LM XNNPACK (P-cores) vs llama.cpp CPU+Accelerate (P-cores, -ngl 0 -fa on)
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from litert_vs_llamacpp.dataset import load_jsonl, sha256_file, write_jsonl
from litert_vs_llamacpp.hardware import HardwareInfo, detect_hardware
from litert_vs_llamacpp.metrics import format_comparison_markdown, summarize_records
from litert_vs_llamacpp.prompts import MODEL_ID, PROMPT_VERSION, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _find_litert_lm_cli() -> str:
    sibling = Path(sys.executable).parent / "litert-lm"
    if sibling.exists():
        return str(sibling)
    on_path = shutil.which("litert-lm")
    if on_path:
        return on_path
    raise FileNotFoundError("Could not find 'litert-lm' CLI in virtualenv or on PATH")


def _parse_csv_choices(raw: str, allowed: set[str], flag: str, parser: argparse.ArgumentParser) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        parser.error(f"{flag} must specify at least one value from {sorted(allowed)}")
    invalid = [v for v in values if v not in allowed]
    if invalid:
        parser.error(f"Invalid {flag} value(s): {invalid}; allowed: {sorted(allowed)}")
    return values


def _find_llama_bin(llama_bin_dir: Path, name: str) -> Path:
    candidate = llama_bin_dir / name
    if candidate.exists():
        return candidate
    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)
    raise FileNotFoundError(
        f"Could not find {name} in {llama_bin_dir} or on PATH. "
        "Run scripts/setup_and_convert_gguf.sh first or pass --llama-bin-dir."
    )


def run_litert_summarization(
    model_path: Path,
    dataset: list[dict[str, Any]],
    *,
    backend_name: str,
    hw: HardwareInfo,
    max_num_tokens: int,
    max_output_tokens: int,
    warmup: int,
    threads: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import litert_lm

    thread_count = threads or hw.perf_cpus
    backend = (
        litert_lm.Backend.GPU()
        if backend_name == "gpu"
        else litert_lm.Backend.CPU(thread_count=thread_count)
    )
    sampler = litert_lm.SamplerConfig(top_k=1, temperature=0.0, seed=42)

    records: list[dict[str, Any]] = []
    # Use a dedicated cache dir per run so GPU shader compilation is warmed once before timed runs
    # and never pollutes the model directory.
    with tempfile.TemporaryDirectory(prefix="litert_bench_cache_") as cache_dir:
        init_t0 = time.perf_counter()
        with litert_lm.Engine(
            str(model_path),
            backend=backend,
            max_num_tokens=max_num_tokens,
            cache_dir=cache_dir,
            enable_benchmark=True,
        ) as engine:
            init_s = time.perf_counter() - init_t0

            # Warmup runs (compiles Metal shaders / XNNPACK packs)
            for idx in range(warmup):
                row = dataset[idx % len(dataset)]
                with engine.create_conversation(
                    system_message=SYSTEM_PROMPT,
                    sampler_config=sampler,
                    max_output_tokens=max_output_tokens,
                ) as conv:
                    _ = conv.send_message(USER_PROMPT_TEMPLATE.format(document=row["document"]))

            for idx, row in enumerate(dataset, start=1):
                prompt = USER_PROMPT_TEMPLATE.format(document=row["document"])
                t0 = time.perf_counter()
                with engine.create_conversation(
                    system_message=SYSTEM_PROMPT,
                    sampler_config=sampler,
                    max_output_tokens=max_output_tokens,
                ) as conv:
                    msg = conv.send_message(prompt)
                    elapsed = time.perf_counter() - t0
                    bench = conv.get_benchmark_info()

                prediction = str(msg).strip()
                out_tokens = int(bench.last_decode_token_count)
                records.append(
                    {
                        "id": row["id"],
                        "reference": row["reference"],
                        "prediction": prediction,
                        "latency_s": round(elapsed, 4),
                        "ttft_s": round(float(bench.time_to_first_token_in_second), 4),
                        "prompt_tokens": int(bench.last_prefill_token_count),
                        "output_tokens": out_tokens,
                        "prefill_tok_s": round(float(bench.last_prefill_tokens_per_second), 2),
                        "decode_tok_s": round(float(bench.last_decode_tokens_per_second), 2),
                        "hit_max_tokens": out_tokens >= max_output_tokens,
                    }
                )
                if idx % 10 == 0 or idx == len(dataset):
                    print(
                        f"  [litert-lm/{backend_name}] {idx}/{len(dataset)} "
                        f"last={elapsed:.2f}s pf={bench.last_prefill_tokens_per_second:.1f}t/s "
                        f"dec={bench.last_decode_tokens_per_second:.1f}t/s",
                        flush=True,
                    )

    try:
        lt_version = importlib.metadata.version("litert-lm")
    except importlib.metadata.PackageNotFoundError:
        lt_version = "unknown"
    meta = {
        "engine_version": lt_version,
        "model_path": str(model_path),
        "model_size_mib": round(model_path.stat().st_size / (1024**2), 1),
        "init_time_s": round(init_s, 3),
        "threads": None if backend_name == "gpu" else thread_count,
        "max_num_tokens": max_num_tokens,
        "max_output_tokens": max_output_tokens,
    }
    return records, meta


def _read_stderr_tail(stderr_file: IO[str] | None, max_chars: int = 2000) -> str:
    if stderr_file is None:
        return ""
    try:
        stderr_file.flush()
        stderr_file.seek(0)
        text = stderr_file.read()
        return text[-max_chars:].strip()
    except OSError:
        return ""


def _wait_for_health(
    base_url: str,
    proc: subprocess.Popen[Any] | None = None,
    stderr_file: IO[str] | None = None,
    timeout_s: float = 120.0,
) -> None:
    deadline = time.perf_counter() + timeout_s
    health_url = f"{base_url}/health"
    while time.perf_counter() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = _read_stderr_tail(stderr_file)
            detail = f"\nllama-server stderr:\n{tail}" if tail else ""
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}{detail}")
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.25)
    tail = _read_stderr_tail(stderr_file)
    detail = f"\nllama-server stderr:\n{tail}" if tail else ""
    raise TimeoutError(
        f"llama-server failed to become healthy at {health_url} within {timeout_s}s{detail}"
    )


def _llama_server_request(
    base_url: str,
    document: str,
    max_output_tokens: int,
    timeout_s: float = 300.0,
) -> tuple[str, dict[str, Any], float]:
    body = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(document=document)},
        ],
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "max_tokens": max_output_tokens,
        "stream": False,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    choice = payload["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    return text, payload, elapsed


def run_llamacpp_summarization(
    model_path: Path,
    llama_bin_dir: Path,
    dataset: list[dict[str, Any]],
    *,
    backend_name: str,
    hw: HardwareInfo,
    max_num_tokens: int,
    max_output_tokens: int,
    warmup: int,
    threads: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    server_bin = _find_llama_bin(llama_bin_dir, "llama-server")
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    thread_count = threads or hw.perf_cpus

    cmd = [
        str(server_bin),
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(max_num_tokens),
        "-np",
        "1",
        "-b",
        "1024",
        "-ub",
        "1024",
        "-fa",
        "on",
        "-t",
        str(thread_count),
        "--cache-ram",
        "0",
        "--no-warmup",
    ]
    if backend_name == "gpu":
        cmd.extend(["-ngl", "99"])
    else:
        cmd.extend(["-ngl", "0"])

    init_t0 = time.perf_counter()
    records: list[dict[str, Any]] = []
    # Redirect stderr to a temp file instead of a PIPE so verbose server logs
    # never fill the OS pipe buffer and deadlock long runs.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stderr_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )
        try:
            _wait_for_health(base_url, proc=proc, stderr_file=stderr_file)
            init_s = time.perf_counter() - init_t0

            for idx in range(warmup):
                row = dataset[idx % len(dataset)]
                _llama_server_request(base_url, row["document"], max_output_tokens)

            for idx, row in enumerate(dataset, start=1):
                prediction, payload, elapsed = _llama_server_request(
                    base_url, row["document"], max_output_tokens
                )
                timings = payload.get("timings") or {}
                usage = payload.get("usage") or {}
                prompt_n = int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0)
                pred_n = int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)
                pf_tps = timings.get("prompt_per_second")
                dec_tps = timings.get("predicted_per_second")
                prompt_ms = timings.get("prompt_ms")
                ttft_s = (float(prompt_ms) / 1000.0) if prompt_ms is not None else None
                finish_reason = payload["choices"][0].get("finish_reason")

                records.append(
                    {
                        "id": row["id"],
                        "reference": row["reference"],
                        "prediction": prediction,
                        "latency_s": round(elapsed, 4),
                        "ttft_s": round(ttft_s, 4) if ttft_s is not None else None,
                        "prompt_tokens": prompt_n,
                        "output_tokens": pred_n,
                        "prefill_tok_s": round(float(pf_tps), 2) if pf_tps is not None else None,
                        "decode_tok_s": round(float(dec_tps), 2) if dec_tps is not None else None,
                        "hit_max_tokens": finish_reason == "length" or pred_n >= max_output_tokens,
                    }
                )
                if idx % 10 == 0 or idx == len(dataset):
                    pf_str = f"{pf_tps:.1f}" if pf_tps is not None else "n/a"
                    dc_str = f"{dec_tps:.1f}" if dec_tps is not None else "n/a"
                    print(
                        f"  [llama.cpp/{backend_name}] {idx}/{len(dataset)} "
                        f"last={elapsed:.2f}s pf={pf_str}t/s dec={dc_str}t/s",
                        flush=True,
                    )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    version_out = subprocess.run(
        [str(server_bin), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    meta = {
        "engine_version": (version_out.stderr or version_out.stdout).strip().splitlines()[0]
        if (version_out.stderr or version_out.stdout).strip()
        else "unknown",
        "model_path": str(model_path),
        "model_size_mib": round(model_path.stat().st_size / (1024**2), 1),
        "init_time_s": round(init_s, 3),
        "threads": thread_count,
        "cmd": cmd,
        "max_num_tokens": max_num_tokens,
        "max_output_tokens": max_output_tokens,
    }
    return records, meta


def run_synthetic_microbench(
    litert_models: dict[str, Path],
    gguf_models: dict[str, Path],
    llama_bin_dir: Path,
    backends: list[str],
    hw: HardwareInfo,
    runs: int,
    threads: int | None,
) -> list[dict[str, Any]]:
    """Run matched synthetic prefill/decode microbenchmarks (litert-lm benchmark vs llama-bench)."""
    results: list[dict[str, Any]] = []
    shapes = [(512, 128), (1024, 256)]
    thread_count = threads or hw.perf_cpus

    litert_cli = _find_litert_lm_cli()
    for quant, model_path in litert_models.items():
        if not model_path.exists():
            continue
        for backend_name in backends:
            for p_len, d_len in shapes:
                cmd = [
                    litert_cli,
                    "benchmark",
                    str(model_path),
                    "--backend",
                    backend_name,
                    "-p",
                    str(p_len),
                    "-d",
                    str(d_len),
                    "--max-num-tokens",
                    "4096",
                    "--runs",
                    str(runs),
                    "--cache",
                    "no",
                ]
                if backend_name == "cpu":
                    cmd.extend(["--cpu-thread-count", str(thread_count)])
                print(f"Running synthetic: {' '.join(cmd)}", flush=True)
                out = subprocess.run(cmd, capture_output=True, text=True, check=False)
                results.append(
                    {
                        "engine": "litert-lm",
                        "backend": backend_name,
                        "quant": quant,
                        "prefill": p_len,
                        "decode": d_len,
                        "returncode": out.returncode,
                        "stdout": out.stdout,
                        "stderr": out.stderr,
                    }
                )

    try:
        bench_bin = _find_llama_bin(llama_bin_dir, "llama-bench")
    except FileNotFoundError:
        return results

    for quant, model_path in gguf_models.items():
        if not model_path.exists():
            continue
        for backend_name in backends:
            ngl = "99" if backend_name == "gpu" else "0"
            cmd = [
                str(bench_bin),
                "-m",
                str(model_path),
                "-ngl",
                ngl,
                "-fa",
                "1",
                "-t",
                str(thread_count),
                "-p",
                "512,1024",
                "-n",
                "128,256",
                "-pg",
                "512,128",
                "-pg",
                "1024,256",
                "-r",
                str(runs),
                "-o",
                "json",
            ]
            print(f"Running synthetic: {' '.join(cmd)}", flush=True)
            out = subprocess.run(cmd, capture_output=True, text=True, check=False)
            parsed = None
            if out.returncode == 0:
                try:
                    parsed = json.loads(out.stdout)
                except json.JSONDecodeError:
                    pass
            results.append(
                {
                    "engine": "llama.cpp",
                    "backend": backend_name,
                    "quant": quant,
                    "returncode": out.returncode,
                    "parsed": parsed,
                    "stdout": out.stdout,
                    "stderr": out.stderr,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/cnn_dailymail_test_pilot.jsonl"),
        help="Prepared CNN/DailyMail JSONL file (default: 50-example pilot).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples (for quick runs).")
    parser.add_argument("--backends", default="gpu,cpu", help="Comma-separated backends: gpu,cpu")
    parser.add_argument("--quants", default="int8,int4", help="Comma-separated quant tiers: int8,int4,int4-cal")
    parser.add_argument("--litert-int8", type=Path, default=Path("models/litertlm/int8/granite-4.0-1b-int8.litertlm"))
    parser.add_argument("--litert-int4", type=Path, default=Path("models/litertlm/int4/granite-4.0-1b-int4.litertlm"))
    parser.add_argument(
        "--litert-int4-cal",
        type=Path,
        default=Path("models/litertlm/int4-cal/granite-4.0-1b-int4-cal.litertlm"),
    )
    parser.add_argument("--gguf-int8", type=Path, default=Path("models/gguf/granite-4.0-1b-Q8_0.gguf"))
    parser.add_argument("--gguf-int4", type=Path, default=Path("models/gguf/granite-4.0-1b-Q4_K_M.gguf"))
    parser.add_argument("--llama-bin-dir", type=Path, default=Path("third_party/llama.cpp/build/bin"))
    parser.add_argument("--max-num-tokens", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=None, help="Override CPU thread count (default: perf cores).")
    parser.add_argument("--run-synthetic", action="store_true", help="Also run litert-lm benchmark vs llama-bench.")
    parser.add_argument("--synthetic-runs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    hw = detect_hardware()
    print("=== Hardware ===")
    print(json.dumps(hw.to_dict(), indent=2))

    dataset = load_jsonl(args.dataset)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        dataset = dataset[: args.limit]
    print(f"Loaded {len(dataset)} examples from {args.dataset}")

    backends = _parse_csv_choices(args.backends, {"gpu", "cpu"}, "--backends", parser)
    quants = _parse_csv_choices(args.quants, {"int8", "int4", "int4-cal"}, "--quants", parser)

    litert_map = {
        "int8": args.litert_int8,
        "int4": args.litert_int4,
        "int4-cal": args.litert_int4_cal,
    }
    gguf_map = {"int8": args.gguf_int8, "int4": args.gguf_int4}
    gguf_labels = {"int8": "Q8_0", "int4": args.gguf_int4.stem.split("granite-4.0-1b-")[-1]}
    litert_labels = {
        "int8": "dynamic_wi8",
        "int4": "int4_block32",
        "int4-cal": "int4_block32_cal",
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, Any]] = []

    for q in quants:
        for backend_name in backends:
            lt_path = litert_map[q]
            if lt_path.exists():
                print(f"\n>>> LiteRT-LM | backend={backend_name} | quant={litert_labels[q]} ({lt_path})")
                lt_records, lt_meta = run_litert_summarization(
                    lt_path,
                    dataset,
                    backend_name=backend_name,
                    hw=hw,
                    max_num_tokens=args.max_num_tokens,
                    max_output_tokens=args.max_output_tokens,
                    warmup=args.warmup,
                    threads=args.threads,
                )
                pred_path = run_dir / f"litert-lm_{backend_name}_{q}.jsonl"
                write_jsonl(pred_path, lt_records)
                run_summaries.append(
                    {
                        "engine": "LiteRT-LM",
                        "backend": backend_name,
                        "quant": litert_labels[q],
                        "meta": lt_meta,
                        "summary": summarize_records(lt_records),
                        "predictions_path": str(pred_path),
                    }
                )
            else:
                print(f"Skipping missing LiteRT-LM bundle: {lt_path}")

            gguf_path = gguf_map.get(q)
            if gguf_path is not None and gguf_path.exists():
                print(f"\n>>> llama.cpp | backend={backend_name} | quant={gguf_labels[q]} ({gguf_path})")
                lc_records, lc_meta = run_llamacpp_summarization(
                    gguf_path,
                    args.llama_bin_dir,
                    dataset,
                    backend_name=backend_name,
                    hw=hw,
                    max_num_tokens=args.max_num_tokens,
                    max_output_tokens=args.max_output_tokens,
                    warmup=args.warmup,
                    threads=args.threads,
                )
                pred_path = run_dir / f"llamacpp_{backend_name}_{q}.jsonl"
                write_jsonl(pred_path, lc_records)
                run_summaries.append(
                    {
                        "engine": "llama.cpp",
                        "backend": backend_name,
                        "quant": gguf_labels[q],
                        "meta": lc_meta,
                        "summary": summarize_records(lc_records),
                        "predictions_path": str(pred_path),
                    }
                )
            elif gguf_path is not None:
                print(f"Skipping missing GGUF model: {gguf_path}")

    synthetic: list[dict[str, Any]] = []
    if args.run_synthetic:
        synthetic = run_synthetic_microbench(
            {q: litert_map[q] for q in quants if q in litert_map},
            {q: gguf_map[q] for q in quants if q in gguf_map},
            args.llama_bin_dir,
            backends,
            hw,
            args.synthetic_runs,
            args.threads,
        )

    if not run_summaries:
        print("No models were found to benchmark. Convert models first (see README.md).", file=sys.stderr)
        sys.exit(1)

    table_md = format_comparison_markdown(run_summaries)
    print("\n=== Head-to-Head Results ===")
    print(table_md)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "example_count": len(dataset),
        "hardware": hw.to_dict(),
        "python": platform.python_version(),
        "runs": run_summaries,
        "synthetic": synthetic,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"# LiteRT-LM vs llama.cpp — Granite 4.0 1B Summarization\n\n"
        f"- **Host:** `{hw.cpu_model}` ({hw.os_name} {hw.arch}, {hw.memory_gb} GiB RAM, {hw.gpu_summary})\n"
        f"- **Dataset:** `{args.dataset}` ({len(dataset)} examples, `{PROMPT_VERSION}`)\n"
        f"- **Decoding:** greedy (`temperature=0`, `top_k=1`, `seed=42`), `max_num_tokens={args.max_num_tokens}`, `max_output_tokens={args.max_output_tokens}`\n\n"
        f"{table_md}\n",
        encoding="utf-8",
    )
    print(f"\nSaved full report to {run_dir / 'report.json'} and {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
