"""Unit tests for dataset sampling, Granite template rendering, hardware detection, and metrics."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import jinja2
import pytest

from litert_vs_llamacpp.dataset import load_jsonl, sample_stream, write_jsonl
from litert_vs_llamacpp.hardware import detect_hardware
from litert_vs_llamacpp.metrics import compute_rouge, format_comparison_markdown, summarize_records
from litert_vs_llamacpp.prompts import GRANITE4_MINIMAL_CHAT_TEMPLATE, SYSTEM_PROMPT


def test_sample_stream_is_deterministic_and_respects_exclusions() -> None:
    stream = [
        {"id": f"doc-{i}", "article": f"Article text {i}", "highlights": f"Summary {i}"}
        for i in range(40)
    ]
    first = sample_stream(stream, count=10, seed=20261028)
    second = sample_stream(stream, count=10, seed=20261028)
    assert first == second
    assert len(first) == 10

    excluded = {row["id"] for row in first}
    disjoint = sample_stream(stream, count=15, seed=20261029, exclude_ids=excluded)
    assert len(disjoint) == 15
    assert excluded.isdisjoint({row["id"] for row in disjoint})


def test_jsonl_roundtrip_and_validation(tmp_path: Path) -> None:
    rows = [{"id": "1", "document": "Some text", "reference": "Short summary"}]
    out = tmp_path / "sample.jsonl"
    write_jsonl(out, rows)
    assert load_jsonl(out) == rows

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "1", "document": "", "reference": "ok"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty string"):
        load_jsonl(bad)


def test_granite4_minimal_template_matches_expected_format() -> None:
    tmpl = jinja2.Template(GRANITE4_MINIMAL_CHAT_TEMPLATE)
    rendered = tmpl.render(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Hello world."},
        ],
        add_generation_prompt=True,
    )
    expected = (
        f"<|start_of_role|>system<|end_of_role|>{SYSTEM_PROMPT}<|end_of_text|>\n"
        "<|start_of_role|>user<|end_of_role|>Hello world.<|end_of_text|>\n"
        "<|start_of_role|>assistant<|end_of_role|>"
    )
    assert rendered == expected


def test_metrics_and_markdown_table() -> None:
    records = [
        {
            "id": "1",
            "reference": "The quick brown fox jumps over the lazy dog.",
            "prediction": "The quick brown fox jumps over the lazy dog.",
            "latency_s": 1.2,
            "ttft_s": 0.2,
            "prompt_tokens": 400,
            "output_tokens": 100,
            "prefill_tok_s": 2000.0,
            "decode_tok_s": 100.0,
            "hit_max_tokens": False,
        },
        {
            "id": "2",
            "reference": "A test article about space exploration.",
            "prediction": "Space exploration test article.",
            "latency_s": 0.8,
            "ttft_s": 0.1,
            "prompt_tokens": 200,
            "output_tokens": 50,
            "prefill_tok_s": 1800.0,
            "decode_tok_s": 120.0,
            "hit_max_tokens": True,
        },
    ]
    rouge = compute_rouge(records)
    assert rouge["rouge1"] > 0.8

    summary = summarize_records(records)
    assert summary["count"] == 2
    assert summary["cap_stops"] == 1
    assert summary["total_output_tokens"] == 150

    md = format_comparison_markdown(
        [{"engine": "LiteRT-LM", "backend": "gpu", "quant": "int4_block32", "summary": summary}]
    )
    assert "LiteRT-LM" in md
    assert "int4_block32" in md
    assert "1/2" in md


def test_detect_hardware_returns_sane_values() -> None:
    hw = detect_hardware()
    assert hw.logical_cpus >= 1
    assert hw.perf_cpus >= 1
    assert hw.os_name
    assert hw.arch


def test_wait_for_health_fails_fast_on_early_exit() -> None:
    bench_path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("bench_script", bench_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as err:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stderr.write('boom metal init'); sys.exit(2)"],
            stdout=subprocess.DEVNULL,
            stderr=err,
        )
        proc.wait(timeout=5)
        with pytest.raises(RuntimeError, match=r"(?s)exited early with code 2.*boom metal init"):
            mod._wait_for_health("http://127.0.0.1:1", proc=proc, stderr_file=err, timeout_s=5.0)

