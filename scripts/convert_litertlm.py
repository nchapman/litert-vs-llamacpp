#!/usr/bin/env python3
"""Convert ibm-granite/granite-4.0-1b to LiteRT-LM (.litertlm) and verify quality.

Designed for macOS Apple Silicon (native litert-converter wheel) or Linux x86_64.
Produces both:
  - int8 (dynamic_wi8_afp32): strong CPU prefill/quality baseline (pairs with GGUF Q8_0)
  - int4 (blockwise-32 OCTAV on linears + int8 embedding): fastest Apple Metal GPU decode
    (pairs with GGUF Q4_K_M / Q4_0)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from litert_vs_llamacpp.prompts import (
    FLOOR_GATE_QUESTIONS,
    GRANITE4_MINIMAL_CHAT_TEMPLATE,
    MODEL_ID,
    SYSTEM_PROMPT,
)

# Full prefill ladder avoids padded-chunk overhead on variable-length news articles.
PREFILL_LENGTHS = "1024,512,256,128,64,32,16,8,4,2,1"
CACHE_LENGTH = 4096

# Per google-ai-edge/litert-samples/models/conversion.md:
# Blockwise-32 OCTAV on linears + channelwise INT8 on embeddings is the quality-safe
# and fastest Apple Metal GPU int4 recipe for 1B-3B dense decoders.
INT4_BLOCK32_RECIPE = [
    {
        "regex": ".*",
        "operation": "FULLY_CONNECTED",
        "algorithm_key": "OCTAV",
        "op_config": {
            "weight_tensor_config": {
                "num_bits": 4,
                "granularity": "BLOCKWISE_32",
                "symmetric": True,
            },
            "compute_precision": "INTEGER",
        },
    },
    {
        "regex": ".*",
        "operation": "EMBEDDING_LOOKUP",
        "algorithm_key": "min_max_uniform_quantize",
        "op_config": {
            "weight_tensor_config": {
                "num_bits": 8,
                "granularity": "CHANNELWISE",
                "symmetric": True,
            },
            "compute_precision": "INTEGER",
        },
    },
]


def export_bundle(
    model_id: str,
    output_dir: Path,
    quant: str,
    *,
    cache_length: int = CACHE_LENGTH,
    prefill_lengths: str = PREFILL_LENGTHS,
) -> Path:
    """Run litert-torch export_hf in a clean subprocess so dialect loading stays isolated."""
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe_arg = "dynamic_wi8_afp32"
    recipe_file: Path | None = None

    if quant == "int4":
        recipe_file = output_dir / "int4_block32_octav.json"
        recipe_file.write_text(json.dumps(INT4_BLOCK32_RECIPE, indent=2) + "\n", encoding="utf-8")
        recipe_arg = str(recipe_file)
    elif quant != "int8":
        raise ValueError(f"Unsupported quant: {quant!r} (expected 'int8' or 'int4')")

    # Subprocess ensures litert_torch is imported before transformers model modules
    # and lets us cleanly swap the vendor Jinja template for the minimal Granite 4.0 template.
    driver = f"""
import sys
from transformers import AutoTokenizer

MINIMAL_TEMPLATE = {GRANITE4_MINIMAL_CHAT_TEMPLATE!r}
_orig = AutoTokenizer.from_pretrained.__func__

def _patched(cls, *args, **kwargs):
    tok = _orig(cls, *args, **kwargs)
    tok.chat_template = MINIMAL_TEMPLATE
    return tok

AutoTokenizer.from_pretrained = classmethod(_patched)

from litert_torch.cli import main

sys.argv = [
    "litert-torch",
    "export_hf",
    "--model", {model_id!r},
    "--output_dir", {str(output_dir)!r},
    "--prefill_lengths", {prefill_lengths!r},
    "--cache_length", {str(cache_length)!r},
    "--use_jinja_template", "False",
    "--bundle_litert_lm", "True",
    "--quantization_recipe", {recipe_arg!r},
]
sys.exit(main())
"""
    print(f"=== Exporting {model_id} ({quant}) -> {output_dir} ===")
    for stale in output_dir.glob("*.litertlm"):
        stale.unlink()
    subprocess.run([sys.executable, "-c", driver], check=True)

    bundles = sorted(output_dir.glob("*.litertlm"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not bundles:
        raise FileNotFoundError(f"No .litertlm file produced in {output_dir}")
    target = output_dir / f"granite-4.0-1b-{quant}.litertlm"
    if bundles[0] != target:
        bundles[0].replace(target)
    print(f"Exported bundle: {target} ({target.stat().st_size / (1024**2):.1f} MiB)")
    return target


def _find_litert_lm_cli() -> str:
    sibling = Path(sys.executable).parent / "litert-lm"
    if sibling.exists():
        return str(sibling)
    return shutil.which("litert-lm") or "litert-lm"


def verify_bundle(bundle_path: Path, model_id: str, backends: list[str]) -> None:
    """Verify metadata, chat template parity against HF tokenizer, and 8-question floor gate."""
    import litert_lm
    from transformers import AutoTokenizer

    print(f"\n=== Describing {bundle_path.name} ===")
    subprocess.run([_find_litert_lm_cli(), "describe", str(bundle_path)], check=False)

    hf_tok = AutoTokenizer.from_pretrained(model_id)
    expected_prompt = hf_tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Sample article text."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    for backend_name in backends:
        backend = litert_lm.Backend.GPU() if backend_name == "gpu" else litert_lm.Backend.CPU()
        print(f"\n=== Verifying {bundle_path.name} on backend={backend_name} ===")
        with tempfile.TemporaryDirectory(prefix="litert_verify_") as tmp_cache:
            with litert_lm.Engine(
                str(bundle_path),
                backend=backend,
                max_num_tokens=1024,
                cache_dir=tmp_cache,
            ) as engine:
                # Check BOS/EOS metadata: granite-4.0-1b has bos == eos == 100257 and add_bos_token=False
                print(
                    f"Engine token IDs: bos_token_id={engine.bos_token_id}, "
                    f"eos_token_ids={engine.eos_token_ids}"
                )

                # Verify chat template rendering matches HF tokenizer
                with engine.create_conversation(system_message=SYSTEM_PROMPT) as conv:
                    rendered = conv.render_message_to_string(
                        litert_lm.Message.user("Sample article text.")
                    )
                    if rendered.strip() != expected_prompt.strip():
                        print("WARNING: Rendered prompt differs from HF tokenizer!")
                        print("  Expected:", repr(expected_prompt))
                        print("  Got:     ", repr(rendered))
                    else:
                        print("Chat template parity: EXACT MATCH with HF AutoTokenizer.")

                # Run 8-question greedy floor gate
                passed = 0
                sampler = litert_lm.SamplerConfig(top_k=1, temperature=0.0, seed=0)
                for question, expected_substrings in FLOOR_GATE_QUESTIONS:
                    with engine.create_conversation(
                        sampler_config=sampler,
                        max_output_tokens=64,
                    ) as conv:
                        reply = str(conv.send_message(question)).strip()
                    low = reply.lower()
                    ok = bool(reply) and any(sub in low for sub in expected_substrings)
                    passed += int(ok)
                    status = "PASS" if ok else "FAIL"
                    print(f"  [{status}] Q: {question} -> {reply!r}")

                print(f"Floor gate ({backend_name}): {passed}/{len(FLOOR_GATE_QUESTIONS)}")
                if passed < 6:
                    raise RuntimeError(
                        f"Floor gate failed on {backend_name} ({passed}/{len(FLOOR_GATE_QUESTIONS)} < 6). "
                        "Inspect BOS/EOS metadata and quantization scales before benchmarking."
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output-root", type=Path, default=Path("models/litertlm"))
    parser.add_argument(
        "--quant",
        choices=["int8", "int4", "all"],
        default="all",
        help="Which quantization(s) to export (default: all = both int8 and int4).",
    )
    parser.add_argument("--cache-length", type=int, default=CACHE_LENGTH)
    parser.add_argument("--prefill-lengths", default=PREFILL_LENGTHS)
    parser.add_argument(
        "--verify-backends",
        default="gpu,cpu",
        help="Comma-separated backends to verify after export (e.g. 'gpu,cpu' or 'cpu').",
    )
    parser.add_argument("--skip-export", action="store_true", help="Only run verification on existing bundles.")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    quants = ["int8", "int4"] if args.quant == "all" else [args.quant]
    backends = [b.strip() for b in args.verify_backends.split(",") if b.strip()]
    invalid = [b for b in backends if b not in {"gpu", "cpu"}]
    if invalid:
        parser.error(f"Invalid --verify-backends value(s): {invalid}; allowed: ['cpu', 'gpu']")

    for q in quants:
        out_dir = args.output_root / q
        bundle_path = out_dir / f"granite-4.0-1b-{q}.litertlm"
        if not args.skip_export:
            bundle_path = export_bundle(
                args.model,
                out_dir,
                q,
                cache_length=args.cache_length,
                prefill_lengths=args.prefill_lengths,
            )
        if not args.skip_verify:
            verify_bundle(bundle_path, args.model, backends)


if __name__ == "__main__":
    main()
