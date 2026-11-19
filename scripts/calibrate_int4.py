#!/usr/bin/env python3
"""Offline calibrated INT4 (blockwise-32) quantization for Granite 4.0 1B.

Runs:
  1. Non-leaking calibration sampling from CNN/DailyMail train split (strictly disjoint
     from data/cnn_dailymail_test_*.jsonl).
  2. Teacher BF16 final hidden-state caching on calibration train/val sequences.
  3. Sequential layer-by-layer Cholesky GPTQ (actorder="static", no column permutation)
     + Gauss-Seidel H-CD + exact DEQUANTIZED_WEIGHT_RECOVERY grid guarantee across all
     240 decoder linear layers (leaving tied embed_tokens/lm_head untouched).
  4. End-to-end FP32-master RMSNorm KL QAT against the BF16 teacher (frozen INT4 weights).
  5. Export of the exact FP32 dequantized HF checkpoint ready for `scripts/convert_litertlm.py --quant int4-cal`.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from litert_vs_llamacpp.calibrate_int4 import (
    GROUP_SIZE,
    build_calibration_sequences,
    calibrate_granite_decoder_linears,
    collect_eval_ids,
    compute_teacher_hidden_states,
    run_rmsnorm_kl_qat,
    save_dequantized_hf_checkpoint,
)


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/hf/granite-4.0-1b"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/hf/granite-4.0-1b-int4-cal"))
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--num-train", type=int, default=128, help="Calibration train sequences (cnn_dailymail train).")
    parser.add_argument("--num-val", type=int, default=16, help="Calibration val sequences (cnn_dailymail train).")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--calib-seed", type=int, default=20260401)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--actorder", choices=["static", "weight", "none"], default="static")
    parser.add_argument("--cd-passes", type=int, default=4, help="Gauss-Seidel H-CD refinement passes per linear.")
    parser.add_argument("--dampening-frac", type=float, default=0.01)
    parser.add_argument("--qat-epochs", type=int, default=2, help="FP32-master RMSNorm KL QAT epochs (0 to skip).")
    parser.add_argument("--qat-batch-size", type=int, default=4)
    parser.add_argument("--qat-lr", type=float, default=1.5e-4)
    parser.add_argument("--qat-anchor-wd", type=float, default=0.02)
    parser.add_argument("--qat-kl-chunk-size", type=int, default=128)
    parser.add_argument(
        "--calib-cache",
        type=Path,
        default=None,
        help="Optional .pt cache of pre-tokenized calibration sequences (avoids importing pyarrow/datasets in the calibration process).",
    )
    parser.add_argument(
        "--save-calib-only",
        action="store_true",
        help="Only build and save --calib-cache, then exit (for isolating pyarrow/datasets from PyTorch).",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="*",
        type=Path,
        default=[
            Path("data/cnn_dailymail_test_pilot.jsonl"),
            Path("data/cnn_dailymail_test_confirmation.jsonl"),
        ],
        help="Eval split JSONL files to assert zero ID overlap against.",
    )
    args = parser.parse_args()
    if args.save_calib_only and args.calib_cache is None:
        parser.error("--save-calib-only requires --calib-cache to be specified")

    t_start = time.time()
    device = _select_device(args.device)
    _log(f"Calibrating {args.model} -> {args.output_dir} on device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.calib_cache is not None and args.calib_cache.exists() and not args.save_calib_only:
        _log(f"Loading pre-tokenized calibration cache from {args.calib_cache} (no datasets/pyarrow import)...")
        cache_obj = torch.load(args.calib_cache, map_location="cpu", weights_only=True)
        if not isinstance(cache_obj, dict) or not {"train_rows", "val_rows", "used_ids"}.issubset(cache_obj):
            raise ValueError(
                f"Invalid calibration cache {args.calib_cache}: expected dict with keys train_rows, val_rows, used_ids"
            )
        raw_train = cache_obj["train_rows"]
        raw_val = cache_obj["val_rows"]
        if not isinstance(raw_train, list) or not raw_train:
            raise ValueError(f"Calibration cache {args.calib_cache} has empty or invalid train_rows")
        if args.qat_epochs > 0 and (not isinstance(raw_val, list) or not raw_val):
            raise ValueError(f"Calibration cache {args.calib_cache} has empty or invalid val_rows")
        train_rows = []
        for idx, t in enumerate(raw_train):
            if not isinstance(t, torch.Tensor) or t.ndim != 1 or t.numel() < 32 or t.is_floating_point():
                raise ValueError(
                    f"Calibration cache {args.calib_cache}: train_rows[{idx}] must be a 1-D integer tensor "
                    f"with >= 32 tokens"
                )
            train_rows.append(t.to(dtype=torch.long))
        val_rows = []
        for idx, t in enumerate(raw_val):
            if not isinstance(t, torch.Tensor) or t.ndim != 1 or t.numel() < 32 or t.is_floating_point():
                raise ValueError(
                    f"Calibration cache {args.calib_cache}: val_rows[{idx}] must be a 1-D integer tensor "
                    f"with >= 32 tokens"
                )
            val_rows.append(t.to(dtype=torch.long))
        used_ids = [str(x) for x in cache_obj["used_ids"]]
        overlap = set(used_ids) & collect_eval_ids(args.eval_splits)
        if overlap:
            raise RuntimeError(
                f"Calibration leakage detected in cache {args.calib_cache}: {len(overlap)} IDs overlap eval splits"
            )
    else:
        _log(
            f"Sampling {args.num_train} train + {args.num_val} val calibration sequences "
            f"(max_seq_len={args.max_seq_len}, seed={args.calib_seed}) from cnn_dailymail train split..."
        )
        train_rows, val_rows, used_ids = build_calibration_sequences(
            tokenizer,
            num_train=args.num_train,
            num_val=args.num_val,
            max_seq_len=args.max_seq_len,
            seed=args.calib_seed,
            eval_split_paths=args.eval_splits,
        )
        if args.calib_cache is not None:
            args.calib_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"train_rows": train_rows, "val_rows": val_rows, "used_ids": used_ids},
                args.calib_cache,
            )
            _log(f"Saved calibration cache to {args.calib_cache}")
        if args.save_calib_only:
            return
    total_train_toks = sum(int(r.numel()) for r in train_rows)
    total_val_toks = sum(int(r.numel()) for r in val_rows)
    _log(f"Calibration tokens ready: train={len(train_rows)} seqs ({total_train_toks:,} toks), val={len(val_rows)} seqs ({total_val_toks:,} toks)")

    _log("Loading BF16 Granite 4.0 1B model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(device=device)
    model.eval()

    # Step 1: Cache BF16 teacher final hidden states BEFORE mutating weights in-place
    train_teacher_states: list[torch.Tensor] = []
    val_teacher_states: list[torch.Tensor] = []
    if args.qat_epochs > 0:
        _log("Caching unquantized BF16 teacher final hidden states for KL QAT...")
        t_tc = time.time()
        train_teacher_states = compute_teacher_hidden_states(model, train_rows, device)
        val_teacher_states = compute_teacher_hidden_states(model, val_rows, device)
        _log(f"Cached teacher hidden states in {time.time() - t_tc:.1f}s")

    # Step 2: Sequential GPTQ (actorder="static") + Gauss-Seidel H-CD on all 40 decoder layers
    _log(
        f"Running sequential GPTQ (actorder={args.actorder}, group_size={args.group_size}) "
        f"+ H-CD ({args.cd_passes} passes)..."
    )
    dequantized_weights, linear_stats = calibrate_granite_decoder_linears(
        model,
        train_rows,
        device=device,
        group_size=args.group_size,
        actorder=args.actorder,
        cd_passes=args.cd_passes,
        dampening_frac=args.dampening_frac,
        progress_cb=_log,
    )

    total_rtn = sum(s.rtn_hessian_loss for s in linear_stats)
    total_gptq = sum(s.gptq_hessian_loss for s in linear_stats)
    total_hcd = sum(s.hcd_hessian_loss for s in linear_stats)
    total_fixed_blocks = sum(s.fixed_blocks for s in linear_stats)
    worst_rec_diff = max(s.recovery_max_diff for s in linear_stats)
    _log(
        f"GPTQ + H-CD complete across {len(linear_stats)} linears: "
        f"total H-loss RTN={total_rtn:.4f} -> GPTQ={total_gptq:.4f} -> H-CD={total_hcd:.4f} "
        f"({100.0 * (1.0 - total_hcd / max(total_rtn, 1e-12)):.1f}% reduction) | "
        f"fixed_blocks={total_fixed_blocks} | max_recovery_diff={worst_rec_diff:.2e}"
    )

    # Step 3: End-to-end FP32-master RMSNorm KL QAT over frozen quantized INT4 decoder weights
    qat_summary: dict = {"skipped": True}
    norm_overrides: dict[str, torch.Tensor] = {}
    if args.qat_epochs > 0:
        if device.type == "mps":
            torch.mps.empty_cache()
        qat_res = run_rmsnorm_kl_qat(
            model,
            train_rows,
            train_teacher_states,
            val_rows,
            val_teacher_states,
            device=device,
            num_epochs=args.qat_epochs,
            batch_size=args.qat_batch_size,
            norm_lr=args.qat_lr,
            anchor_wd=args.qat_anchor_wd,
            kl_chunk_size=args.qat_kl_chunk_size,
            progress_cb=_log,
        )
        norm_overrides = qat_res["norm_overrides"]
        qat_summary = {
            "init_val_kl": qat_res["init_val_kl"],
            "best_val_kl": qat_res["best_val_kl"],
            "history": qat_res["history"],
            "num_norms_tuned": len(norm_overrides),
        }
        _log(
            f"RMSNorm KL QAT complete: val_kl {qat_res['init_val_kl']:.6f} -> {qat_res['best_val_kl']:.6f}"
        )

    del model, train_teacher_states, val_teacher_states
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Step 4: Save exact FP32 dequantized HF checkpoint
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": str(args.model),
        "device": str(device),
        "group_size": args.group_size,
        "actorder": args.actorder,
        "cd_passes": args.cd_passes,
        "dampening_frac": args.dampening_frac,
        "calib_seed": args.calib_seed,
        "num_train_seqs": len(train_rows),
        "num_val_seqs": len(val_rows),
        "total_train_tokens": total_train_toks,
        "total_val_tokens": total_val_toks,
        "calib_id_count": len(used_ids),
        "linear_count": len(linear_stats),
        "total_rtn_hessian_loss": total_rtn,
        "total_gptq_hessian_loss": total_gptq,
        "total_hcd_hessian_loss": total_hcd,
        "total_fixed_blocks": total_fixed_blocks,
        "max_recovery_diff": worst_rec_diff,
        "qat": qat_summary,
        "elapsed_s": round(time.time() - t_start, 2),
        "linears": [dataclasses.asdict(s) for s in linear_stats],
    }
    _log(f"Saving exact FP32 fake-quantized HF checkpoint to {args.output_dir}...")
    save_dequantized_hf_checkpoint(
        args.model,
        args.output_dir,
        dequantized_weights,
        norm_overrides,
        report=report,
    )
    _log(f"Done in {time.time() - t_start:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
