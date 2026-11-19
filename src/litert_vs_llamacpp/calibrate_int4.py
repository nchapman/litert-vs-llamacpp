"""Calibrated symmetric INT4 (blockwise-32) quantization for Granite 4.0 1B in LiteRT-LM.

Pipeline (adapted from warp34-quant-lab for LiteRT's native BLOCKWISE_32 INT4 format):
  1. Sequential layer-by-layer Hessian accumulation (H = X^T X) over calibration sequences
     strictly disjoint from evaluation splits.
  2. Symmetric group_size=32 MSE clipping scale initialization + exact block Cholesky GPTQ
     (actorder="static", no column permutation) + Gauss-Seidel Hessian Coordinate Descent (H-CD)
     over all 240 decoder linear layers (40 layers x {q,k,v,o}_proj, input_linear, output_linear).
  3. Exact DEQUANTIZED_WEIGHT_RECOVERY grid guarantee: block scales are projected onto
     LiteRT's exact bf16->fp16 blockwise scale grid and any rare block lacking a unit step
     gap whose codes are not divisible by min_diff is minimally adjusted so ai-edge-quantizer
     recovers the exact integer codes and scales with zero re-rounding.
  4. End-to-end FP32-master RMSNorm KL QAT against the BF16 teacher (accounting for Granite 4.0's
     logits_scaling) with all INT4 linear weights frozen and tied embed_tokens/lm_head untouched.
"""

from __future__ import annotations

import functools
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

from litert_vs_llamacpp.dataset import load_jsonl
from litert_vs_llamacpp.prompts import (
    GRANITE4_MINIMAL_CHAT_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

INT4_QMIN = -8
INT4_QMAX = 7
GROUP_SIZE = 32
DEFAULT_MIN_SCALE = 1e-6
# Maximum finite value representable cleanly in both bfloat16 and float16
# (float16 max is 65504, which rounds up to 65536 in bfloat16 -> +inf in float16).
LITERT_MAX_BLOCK_SCALE = 65280.0


@dataclass
class LinearQuantStats:
    name: str
    shape: tuple[int, int]
    rtn_hessian_loss: float
    gptq_hessian_loss: float
    hcd_hessian_loss: float
    rel_mse: float
    fixed_blocks: int
    recovery_max_diff: float


def quantize_litert_block_scale(
    scales: torch.Tensor,
    *,
    min_scale: float = DEFAULT_MIN_SCALE,
    max_scale: float = LITERT_MAX_BLOCK_SCALE,
) -> torch.Tensor:
    """Project positive scales onto LiteRT's exact BlockwiseQuantization scale grid (bf16 -> fp16).

    In ai_edge_quantizer/transformations/quantize_tensor.py, blockwise scales are serialized via:
      scale.astype(ml_dtypes.bfloat16).astype(np.float16)
    Keeping scales on this exact grid during GPTQ and H-CD ensures zero scale re-rounding on export.
    """
    s_safe = torch.nan_to_num(scales.float(), nan=min_scale, posinf=max_scale, neginf=min_scale)
    return s_safe.clamp(min=min_scale, max=max_scale).to(torch.bfloat16).to(torch.float16).to(torch.float32)


@torch.no_grad()
def find_mse_scale_int4(
    w_f32: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
    grid_points: int = 81,
    min_scale: float = DEFAULT_MIN_SCALE,
) -> torch.Tensor:
    """Vectorized MSE clipping search for signed INT4 [-8, 7] per group of `group_size` columns."""
    n_out, k_in = w_f32.shape
    if k_in % group_size != 0:
        raise ValueError(f"in_features={k_in} not divisible by group_size={group_size}")
    grouped = w_f32.view(n_out, k_in // group_size, group_size)
    amax = grouped.abs().amax(dim=-1)
    base_scale = (amax / float(INT4_QMAX)).clamp(min=min_scale)
    best_scale = quantize_litert_block_scale(amax / float(abs(INT4_QMIN)), min_scale=min_scale)
    best_err = torch.full_like(base_scale, float("inf"))
    for ratio in torch.linspace(0.20, 1.05, grid_points, device=w_f32.device, dtype=torch.float32):
        cand = quantize_litert_block_scale(base_scale * ratio, min_scale=min_scale)
        codes = torch.clamp(torch.round(grouped / cand.unsqueeze(-1)), INT4_QMIN, INT4_QMAX)
        deq = codes * cand.unsqueeze(-1)
        err = (grouped - deq).square().sum(dim=-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, cand, best_scale)
    return best_scale


def _cholesky_hinv_upper(h_p: torch.Tensor, dampening_frac: float) -> torch.Tensor:
    """Compute upper Cholesky factor of H^{-1} in FP64 via NumPy with automatic dampening backoff.

    Uses NumPy LAPACK (`np.linalg.cholesky` + `np.linalg.solve`) instead of `torch.linalg.cholesky` /
    `torch.cholesky_inverse`: PyTorch 2.14.1's macOS Accelerate LAPACK wrappers exhibit intermittent
    heap corruption (SIGBUS/SIGSEGV/SIGILL) on repeated 2048x2048 factorizations.
    """
    # Move to CPU before casting to float64: combined .to(device="cpu", dtype=torch.float64)
    # silently zeroes MPS tensors on PyTorch 2.14.1 because MPS has no float64 support.
    h_np = h_p.detach().cpu().numpy().astype(np.float64, copy=True)
    if not np.isfinite(h_np).all():
        raise RuntimeError("Hessian contains non-finite values (NaN/Inf)")
    h_np = 0.5 * (h_np + h_np.T)
    diag = np.diag(h_np)
    diag_mean = max(float(np.mean(diag)), 1e-12)
    damp = max(float(dampening_frac), 1e-6)
    eye = np.eye(h_np.shape[0], dtype=np.float64)
    last_err: Exception | None = None
    for _ in range(8):
        h_damped = h_np.copy()
        h_damped[np.diag_indices_from(h_damped)] += damp * diag_mean
        try:
            chol_l = np.linalg.cholesky(h_damped)
            l_inv = np.linalg.solve(chol_l, eye)
            h_inv = l_inv.T @ l_inv
            h_inv = 0.5 * (h_inv + h_inv.T)
            u_np = np.linalg.cholesky(h_inv).T.astype(np.float32, copy=True)
            if not np.isfinite(u_np).all():
                raise np.linalg.LinAlgError("Non-finite Cholesky inverse factor")
            return torch.from_numpy(np.ascontiguousarray(u_np))
        except np.linalg.LinAlgError as exc:
            last_err = exc
            damp *= 4.0
    raise RuntimeError(
        f"Failed to Cholesky-factorize damped Hessian (diag_min={float(diag.min()):.4e}, "
        f"diag_max={float(diag.max()):.4e}, diag_mean={diag_mean:.4e}): {last_err}"
    ) from last_err


@torch.no_grad()
def cholesky_gptq_int4(
    w_f32: torch.Tensor,
    norm_h: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
    actorder: str = "static",
    blocksize: int = 128,
    dampening_frac: float = 0.01,
) -> tuple[torch.Tensor, float]:
    """Exact block Cholesky GPTQ for symmetric INT4 [-8, 7] with static group scales.

    When `actorder == "static"`, columns are eliminated in descending diag(H) order while
    keeping the precomputed contiguous group scales fixed, then unpermuted back to original
    column order — so zero column permutation (`g_idx`) is required at inference time.
    """
    n_out, k_in = w_f32.shape
    device = w_f32.device
    h_f = norm_h.detach().float()
    if actorder == "static":
        perm = torch.argsort(h_f.diagonal(), descending=True)
    elif actorder == "weight":
        perm = torch.argsort(w_f32.square().sum(dim=0), descending=True)
    elif actorder in ("none", "", None):
        perm = torch.arange(k_in, device=device)
    else:
        raise ValueError(f"Unsupported actorder={actorder!r}")
    inv_perm = torch.argsort(perm)

    h_inv_u = _cholesky_hinv_upper(h_f[perm][:, perm], dampening_frac).to(device=device)
    w_p = w_f32[:, perm].clone()
    s_exp_p = scales.float().repeat_interleave(group_size, dim=1)[:, perm]
    c_p = torch.empty((n_out, k_in), dtype=torch.int8, device=device)
    loss_acc = torch.zeros(1, dtype=torch.float32, device=device)

    for b0 in range(0, k_in, blocksize):
        b1 = min(b0 + blocksize, k_in)
        w_blk = w_p[:, b0:b1].clone()
        err_blk = torch.empty_like(w_blk)
        hinv_blk = h_inv_u[b0:b1, b0:b1]
        for j in range(b1 - b0):
            col = w_blk[:, j]
            sj = s_exp_p[:, b0 + j]
            d_jj = hinv_blk[j, j]
            c_col = torch.clamp(torch.round(col / sj), INT4_QMIN, INT4_QMAX)
            c_col = torch.nan_to_num(c_col, nan=0.0, posinf=float(INT4_QMAX), neginf=float(INT4_QMIN))
            c_p[:, b0 + j] = c_col.to(torch.int8)
            q_col = c_col * sj
            err = (col - q_col) / d_jj
            if not torch.isfinite(err).all():
                return c_p[:, inv_perm].contiguous(), float("nan")
            err_blk[:, j] = err
            loss_acc.add_(err.square().sum() * 0.5)
            if j + 1 < b1 - b0:
                w_blk[:, j + 1 :] -= err.unsqueeze(1) * hinv_blk[j, j + 1 :].unsqueeze(0)
        if b1 < k_in:
            w_p[:, b1:] -= err_blk @ h_inv_u[b0:b1, b1:]

    return c_p[:, inv_perm].contiguous(), float(loss_acc.item())


@torch.no_grad()
def refine_hessian_cd_int4(
    w_f32: torch.Tensor,
    norm_h: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
    passes: int = 4,
    dampening_frac: float = 0.01,
    min_scale: float = DEFAULT_MIN_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Seidel coordinate descent over group scales and INT4 codes on Tr((W - Q) H (W - Q)^T)."""
    if passes <= 0:
        return codes.to(torch.int8), quantize_litert_block_scale(scales, min_scale=min_scale)

    w_f = w_f32.detach().float()
    h_f = norm_h.detach().float().clone()
    h_f = 0.5 * (h_f + h_f.T)
    diag_mean = h_f.diagonal().mean().clamp(min=1e-12)
    h_f.diagonal().add_(dampening_frac * diag_mean)
    h_diag = h_f.diagonal().clamp(min=1e-12)

    c_mat = codes.to(device=w_f.device, dtype=torch.float32).clone()
    s_mat = quantize_litert_block_scale(scales.to(device=w_f.device), min_scale=min_scale).clone()
    n_out, k_in = w_f.shape
    num_groups = k_in // group_size
    idx = torch.arange(k_in, device=w_f.device).view(num_groups, group_size)
    h_blocks = h_f[idx.unsqueeze(2), idx.unsqueeze(1)].contiguous()

    q_mat = (c_mat.view(n_out, num_groups, group_size) * s_mat.unsqueeze(-1)).view(n_out, k_in)

    for _ in range(passes):
        # Recompute exact residual product at the start of each pass to prevent FP32 drift.
        rh = (w_f - q_mat) @ h_f
        for g in range(num_groups):
            g0, g1 = g * group_size, (g + 1) * group_size
            s_g = s_mat[:, g]
            c_g = c_mat[:, g0:g1]
            q_g_old = q_mat[:, g0:g1].clone()
            q_g = q_g_old.clone()
            rh_g = rh[:, g0:g1].clone()
            h_gg = h_blocks[g]
            hd_g = h_diag[g0:g1]

            # 1. Exact 1-group Newton scale step with strict monotonic quadratic-loss acceptance
            ch_g = c_g @ h_gg
            num = (c_g * rh_g).sum(dim=-1)
            den = (c_g * ch_g).sum(dim=-1)
            valid = den > 1e-8
            step = torch.where(valid, num / den.clamp(min=1e-8), torch.zeros_like(num))
            s_cand = quantize_litert_block_scale(s_g + step, min_scale=min_scale)
            ds_cand = s_cand - s_g
            delta_l = ds_cand * (ds_cand * den - 2.0 * num)
            accept = valid & (delta_l < 0.0)
            s_new = torch.where(accept, s_cand, s_g)
            ds = s_new - s_g
            s_g = s_new
            q_g = c_g * s_g.unsqueeze(-1)
            rh_g = rh_g - ch_g * ds.unsqueeze(-1)

            # 2. Exact coordinate-descent code sweep within group g (monotonic acceptance)
            for j in range(group_size):
                q_j = q_g[:, j]
                q_star = q_j + rh_g[:, j] / hd_g[j]
                c_cand = torch.clamp(torch.round(q_star / s_g), INT4_QMIN, INT4_QMAX)
                q_cand = c_cand * s_g
                dq_cand = q_cand - q_j
                delta_j = dq_cand * (dq_cand * hd_g[j] - 2.0 * rh_g[:, j])
                accept_j = delta_j < 0.0
                c_new = torch.where(accept_j, c_cand, c_g[:, j])
                q_new = torch.where(accept_j, q_cand, q_j)
                dq = q_new - q_j
                c_g[:, j] = c_new
                q_g[:, j] = q_new
                rh_g.addmm_(dq.unsqueeze(1), h_gg[j : j + 1, :], alpha=-1.0)

            # 3. Exact 1-group Newton scale polish after column code sweep (monotonic acceptance)
            ch_g = c_g @ h_gg
            num = (c_g * rh_g).sum(dim=-1)
            den = (c_g * ch_g).sum(dim=-1)
            valid = den > 1e-8
            step = torch.where(valid, num / den.clamp(min=1e-8), torch.zeros_like(num))
            s_cand = quantize_litert_block_scale(s_g + step, min_scale=min_scale)
            ds_cand = s_cand - s_g
            delta_l = ds_cand * (ds_cand * den - 2.0 * num)
            accept = valid & (delta_l < 0.0)
            s_g = torch.where(accept, s_cand, s_g)
            q_g = c_g * s_g.unsqueeze(-1)

            s_mat[:, g] = s_g
            q_mat[:, g0:g1] = q_g
            dq_g = q_g - q_g_old
            rh.addmm_(dq_g, h_f[g0:g1, :], alpha=-1.0)

    return c_mat.to(torch.int8), s_mat


def _min_nonzero_abs_diff(abs_codes: np.ndarray) -> int:
    """Compute min positive diff of sorted(unique(|c| U {0})), matching dequantized_weight_recovery."""
    vals = np.unique(np.append(abs_codes, 0))
    if vals.size <= 1:
        return 0
    return int(np.min(np.diff(vals)))


@torch.no_grad()
def ensure_exact_recovery_codes(
    w_f32: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    h_diag: torch.Tensor | None = None,
    *,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, int]:
    """Ensure every block recovers identically under ai_edge_quantizer's dequantized_weight_recovery.

    dequantized_weight_recovery estimates each block's scale as:
      s_rec = min(diff(sort(abs(w_block) U {0}))) = d * s
    where d = min(diff(sort(abs(c_block) U {0}))).
    If d == 0 (all zeros) or d == 1 (two magnitudes differ by 1, including 0 and 1), or if every
    |c| in the block is divisible by d (and d * s is representable), recovery is exact.
    Only when d >= 2 AND some |c| % d != 0 (e.g. |c| = {0, 2, 5}) would s_rec fail to divide |c|
    and cause re-rounding. For those rare blocks, we nudge the single least-sensitive element
    by +/-1 code (staying in [-8, 7]) so that d == 1.
    """
    n_out, k_in = codes.shape
    num_groups = k_in // group_size
    c_np = codes.detach().cpu().to(dtype=torch.int16).numpy().reshape(n_out * num_groups, group_size)
    abs_c = np.abs(c_np)

    # Vectorized fast path: append 0, sort, compute min positive diff across all blocks at once.
    with_zero = np.concatenate([abs_c, np.zeros((abs_c.shape[0], 1), dtype=np.int16)], axis=1)
    sorted_abs = np.sort(with_zero, axis=1)
    diffs = np.diff(sorted_abs, axis=1)
    masked = np.where(diffs > 0, diffs, 99)
    min_d = masked.min(axis=1)

    # Blocks needing inspection: min_d >= 2 (and < 99) AND either min_d is not a power of 2
    # (so min_d * s may not lie on the bf16->fp16 scale grid) or some |c| is not divisible by min_d.
    cand_mask = (min_d >= 2) & (min_d < 99)
    if not np.any(cand_mask):
        return codes.to(torch.int8), 0

    cand_indices = np.flatnonzero(cand_mask)
    bad_blocks = [
        int(bi)
        for bi in cand_indices
        if (int(min_d[bi]) not in (2, 4)) or np.any((abs_c[bi] % min_d[bi]) != 0)
    ]
    if not bad_blocks:
        return codes.to(torch.int8), 0

    w_np = w_f32.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(n_out * num_groups, group_size)
    s_np = scales.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(n_out * num_groups)
    hd_np = (
        h_diag.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(num_groups, group_size)
        if h_diag is not None
        else np.ones((num_groups, group_size), dtype=np.float32)
    )

    fixed = 0
    for bi in bad_blocks:
        g_idx = bi % num_groups
        c_row = c_np[bi]
        w_row = w_np[bi]
        s_val = float(s_np[bi])
        hd_row = hd_np[g_idx]

        best_delta_err = float("inf")
        best_j = -1
        best_new_c = 0
        for j in range(group_size):
            old_c = int(c_row[j])
            old_err = float(hd_row[j]) * ((float(w_row[j]) - old_c * s_val) ** 2)
            for new_c in (old_c - 1, old_c + 1):
                if new_c < INT4_QMIN or new_c > INT4_QMAX:
                    continue
                trial = c_row.copy()
                trial[j] = new_c
                d_new = _min_nonzero_abs_diff(np.abs(trial))
                if d_new <= 1 or (d_new in (2, 4) and np.all((np.abs(trial) % d_new) == 0)):
                    new_err = float(hd_row[j]) * ((float(w_row[j]) - new_c * s_val) ** 2)
                    delta = new_err - old_err
                    if delta < best_delta_err:
                        best_delta_err = delta
                        best_j = j
                        best_new_c = new_c
        if best_j < 0:
            # Fallback: force element closest to +/-s to +/-1 so min_d == 1 unconditionally.
            err_to_1 = hd_row * np.minimum((w_row - s_val) ** 2, (w_row + s_val) ** 2)
            best_j = int(np.argmin(err_to_1))
            best_new_c = 1 if w_row[best_j] >= 0 else -1
        c_row[best_j] = best_new_c
        fixed += 1

    fixed_codes = torch.from_numpy(c_np.reshape(n_out, k_in).astype(np.int8)).to(device=codes.device)
    return fixed_codes, fixed


def decode_int4_blocks(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Decode signed INT4 codes [N, K] and block scales [N, K//group_size] to exact FP32 weights."""
    n_out, k_in = codes.shape
    return (
        codes.to(torch.float32).view(n_out, k_in // group_size, group_size)
        * scales.to(torch.float32).unsqueeze(-1)
    ).view(n_out, k_in)


def verify_dequantized_weight_recovery(
    w_deq_f32: torch.Tensor,
    *,
    block_size: int = GROUP_SIZE,
    tol: float = 1e-4,
    use_ai_edge_quantizer: bool = False,
) -> float:
    """Verify exact recovery matching ai_edge_quantizer's dequantized_weight_recovery logic.

    Checks both the float32 recovery tolerance (`_validate_recovered_weights`) and the
    flatbuffer bf16->fp16 blockwise scale round-trip used by LiteRT.

    Defaults to a pure-NumPy implementation of `get_zp_scale_from_dequantized_symmetric_weights`
    to avoid loading `ai_edge_quantizer`/`scipy` C++ extensions inside PyTorch MPS calibration
    processes (which can segfault on macOS); set `use_ai_edge_quantizer=True` in unit tests.
    """
    import ml_dtypes

    arr = w_deq_f32.detach().cpu().to(dtype=torch.float32).numpy()
    if use_ai_edge_quantizer:
        from ai_edge_quantizer import qtyping
        from ai_edge_quantizer.algorithms.uniform_quantize import (
            dequantized_weight_recovery,
            uniform_quantize_tensor,
        )

        zp, scale = dequantized_weight_recovery.get_zp_scale_from_dequantized_symmetric_weights(
            dequant_vals=arr,
            quantized_dimension=1,
            block_size=block_size,
        )
        qparams = qtyping.UniformQuantParams(
            scale=scale,
            zero_point=zp,
            num_bits=4,
            symmetric=True,
            quantized_dimension=1,
            block_size=block_size,
        )
        qvals = uniform_quantize_tensor.uniform_quantize(arr, qparams, is_blockwise_quant=True)
        dequantized_weight_recovery._validate_recovered_weights(arr, qvals, qparams, tol=tol)
        rec = uniform_quantize_tensor.uniform_dequantize(qvals, qparams)
        max_diff = float(np.max(np.abs(rec - arr)))

        # Also verify LiteRT's serialized bf16->fp16 blockwise scale round-trip error.
        scale_rt = scale.astype(ml_dtypes.bfloat16).astype(np.float16).astype(np.float32)
        qparams_rt = qtyping.UniformQuantParams(
            scale=scale_rt,
            zero_point=zp,
            num_bits=4,
            symmetric=True,
            quantized_dimension=1,
            block_size=block_size,
        )
        rec_rt = uniform_quantize_tensor.uniform_dequantize(qvals, qparams_rt)
        max_diff_rt = float(np.max(np.abs(rec_rt - arr)))
        if max_diff_rt > tol:
            raise RuntimeError(
                f"bf16->fp16 block scale round-trip exceeded tolerance: {max_diff_rt:.6e} > {tol:.6e}"
            )
        return max(max_diff, max_diff_rt)

    # Pure-NumPy exact replica of dequantized_weight_recovery.get_zp_scale_from_dequantized_symmetric_weights
    n_out, k_in = arr.shape
    num_blocks = k_in // block_size
    blocks = arr.reshape(n_out, num_blocks, block_size)
    abs_sorted = np.sort(
        np.concatenate([np.zeros((n_out, num_blocks, 1), dtype=np.float32), np.abs(blocks)], axis=-1),
        axis=-1,
    )
    diffs = np.diff(abs_sorted, axis=-1)
    pos_diffs = np.where(diffs > 1e-12, diffs, np.inf)
    scale = np.min(pos_diffs, axis=-1, keepdims=True)
    scale = np.where(np.isfinite(scale), scale, 0.0).astype(np.float32)

    safe_scale = np.where(scale > 0.0, scale, 1.0)
    qvals = np.where(
        scale > 0.0, np.clip(np.round(blocks / safe_scale), INT4_QMIN, INT4_QMAX), 0.0
    ).astype(np.int8)
    rec = qvals.astype(np.float32) * scale
    max_diff = float(np.max(np.abs(rec - blocks)))
    if max_diff > tol:
        raise RuntimeError(f"DEQUANTIZED_WEIGHT_RECOVERY float32 diff {max_diff:.6e} > {tol:.6e}")

    scale_rt = scale.astype(ml_dtypes.bfloat16).astype(np.float16).astype(np.float32)
    rec_rt = qvals.astype(np.float32) * scale_rt
    max_diff_rt = float(np.max(np.abs(rec_rt - blocks)))
    if max_diff_rt > tol:
        raise RuntimeError(
            f"bf16->fp16 block scale round-trip exceeded tolerance: {max_diff_rt:.6e} > {tol:.6e}"
        )
    return max(max_diff, max_diff_rt)


def hidden_kl_scaled(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head: nn.Module,
    *,
    logits_scaling: float = 1.0,
    token_chunk_size: int = 128,
    checkpointed: bool = True,
) -> torch.Tensor:
    """Token-chunked KL(teacher || student) over lm_head logits with optional Granite logits_scaling."""
    s_flat = student_hidden.reshape(-1, student_hidden.shape[-1])
    t_flat = teacher_hidden.to(device=student_hidden.device, dtype=student_hidden.dtype).reshape(
        -1, teacher_hidden.shape[-1]
    )
    n_tok = s_flat.shape[0]
    inv_scale = 1.0 / float(logits_scaling)

    def _chunk_kl(s_chunk: torch.Tensor, t_chunk: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t_logits = lm_head(t_chunk).float() * inv_scale
            log_pt = F.log_softmax(t_logits, dim=-1)
            pt = log_pt.exp()
        s_logits = lm_head(s_chunk).float() * inv_scale
        log_ps = F.log_softmax(s_logits, dim=-1)
        return (pt * (log_pt - log_ps)).sum(dim=-1).sum()

    total_kl = s_flat.new_zeros((), dtype=torch.float32)
    for start in range(0, n_tok, token_chunk_size):
        end = min(start + token_chunk_size, n_tok)
        sc = s_flat[start:end]
        tc = t_flat[start:end]
        if checkpointed and torch.is_grad_enabled():
            chunk_sum = torch.utils.checkpoint.checkpoint(_chunk_kl, sc, tc, use_reentrant=False)
        else:
            chunk_sum = _chunk_kl(sc, tc)
        total_kl = total_kl + chunk_sum
    return total_kl / float(max(n_tok, 1))


def collect_eval_ids(paths: Iterable[Path]) -> set[str]:
    """Load article IDs from any existing evaluation split files to guard against data leakage."""
    ids: set[str] = set()
    for p in paths:
        if p.exists():
            for row in load_jsonl(p):
                ids.add(str(row["id"]))
    return ids


def build_calibration_sequences(
    tokenizer: Any,
    *,
    num_train: int = 128,
    num_val: int = 16,
    max_seq_len: int = 1024,
    seed: int = 20260401,
    eval_split_paths: Iterable[Path] = (),
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    """Build non-leaking calibration token sequences strictly from cnn_dailymail train split.

    Half the sequences use the exact summary-v3 chat template + reference highlights;
    half use raw news article text so activations cover both chat summarization and
    general English language modeling without overfitting to prompt formatting.
    """
    from datasets import load_dataset

    forbidden_ids = collect_eval_ids(eval_split_paths)
    # Sample ~2.2x articles relative to target packed chunks so we always fill all 1024-token chunks.
    total_needed = max(int((num_train + num_val) * 2.2) + 32, num_train + num_val + 32)

    orig_template = getattr(tokenizer, "chat_template", None)
    tokenizer.chat_template = GRANITE4_MINIMAL_CHAT_TEMPLATE
    try:
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
        rng = random.Random(seed)
        # Reservoir sample over first 20k train rows for fast deterministic diversity.
        pool_limit = max(total_needed * 40, 6000)
        sample: list[dict[str, str]] = []
        seen = 0
        for rec in ds:
            rec_id = str(rec.get("id") or "")
            article = rec.get("article") or ""
            highlights = rec.get("highlights") or ""
            if not rec_id or not article.strip() or not highlights.strip():
                continue
            if rec_id in forbidden_ids:
                raise RuntimeError(f"Calibration leakage detected: train ID {rec_id!r} overlaps eval split")
            seen += 1
            item = {"id": rec_id, "article": article, "highlights": highlights}
            if len(sample) < total_needed:
                sample.append(item)
            else:
                slot = rng.randrange(seen)
                if slot < total_needed:
                    sample[slot] = item
            if seen >= pool_limit:
                break

        if len(sample) < total_needed:
            raise RuntimeError(f"Requested {total_needed} calibration rows, only got {len(sample)}")
        rng.shuffle(sample)

        # Split articles first so train and val chunks never share a document, then pack into
        # exact fixed-length `max_seq_len` windows (standard GPTQ calibration packing; also avoids
        # PyTorch MPSGraph shape-cache bloat from hundreds of distinct sequence lengths).
        eos_id = tokenizer.eos_token_id
        val_article_count = max(num_val * 2, 32)
        val_articles = sample[:val_article_count]
        train_articles = sample[val_article_count:]

        def _pack_articles(articles: list[dict[str, str]], target_chunks: int) -> tuple[list[torch.Tensor], list[str]]:
            buf: list[int] = []
            chunks: list[torch.Tensor] = []
            doc_ids: list[str] = []
            for idx, item in enumerate(articles):
                if idx % 2 == 0:
                    text = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(document=item["article"])},
                            {"role": "assistant", "content": item["highlights"]},
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    ids = tokenizer.encode(text, add_special_tokens=False)
                else:
                    ids = tokenizer.encode(item["article"], add_special_tokens=False)
                    if eos_id is not None:
                        ids.append(int(eos_id))
                buf.extend(ids)
                doc_ids.append(item["id"])
                while len(buf) >= max_seq_len and len(chunks) < target_chunks:
                    chunks.append(torch.tensor(buf[:max_seq_len], dtype=torch.long))
                    buf = buf[max_seq_len:]
                if len(chunks) >= target_chunks:
                    break
            return chunks, doc_ids

        val_rows, val_ids = _pack_articles(val_articles, num_val)
        train_rows, train_ids = _pack_articles(train_articles, num_train)
        used_ids = val_ids + train_ids
    finally:
        if orig_template is not None:
            tokenizer.chat_template = orig_template

    if len(train_rows) < num_train or len(val_rows) < num_val:
        raise RuntimeError(
            f"Insufficient packed calibration chunks: got train={len(train_rows)}/{num_train}, "
            f"val={len(val_rows)}/{num_val}"
        )
    return train_rows, val_rows, used_ids


class _StopAfterCapture(Exception):
    pass


@torch.no_grad()
def _capture_layer0_inputs(
    model: nn.Module,
    sequences: list[torch.Tensor],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]]:
    """Capture exact (hidden_states, attention_mask, position_embeddings) entering decoder layer 0."""
    captured: list[tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]] = []
    holder: dict[str, Any] = {}

    def _pre_hook(_mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        h = args[0] if args else kwargs["hidden_states"]
        attn_mask = kwargs.get("attention_mask")
        pos_emb = kwargs.get("position_embeddings")
        holder["h"] = h.detach().to(device="cpu")
        holder["mask"] = attn_mask.detach().to(device="cpu") if attn_mask is not None else None
        holder["pos"] = (
            (pos_emb[0].detach().to(device="cpu"), pos_emb[1].detach().to(device="cpu"))
            if pos_emb is not None
            else None
        )
        raise _StopAfterCapture()

    handle = model.model.layers[0].register_forward_pre_hook(_pre_hook, with_kwargs=True)
    try:
        for seq_idx, seq in enumerate(sequences):
            holder.clear()
            inp = seq.to(dtype=torch.long).to(device=device).unsqueeze(0)
            try:
                model.model(input_ids=inp, use_cache=False, return_dict=True)
            except _StopAfterCapture:
                pass
            if "h" not in holder:
                raise RuntimeError(
                    f"Layer-0 forward pre-hook did not capture hidden_states for sequence {seq_idx}; "
                    f"verify model.model.layers[0] is invoked by model.model(input_ids=...)."
                )
            captured.append((holder["h"], holder["mask"], holder["pos"]))
            holder.clear()
    finally:
        handle.remove()
    return captured


@torch.no_grad()
def _quantize_linear_with_hessian(
    name: str,
    linear: nn.Linear,
    hessian: torch.Tensor,
    num_tokens: int,
    *,
    group_size: int = GROUP_SIZE,
    actorder: str = "static",
    cd_passes: int = 4,
    dampening_frac: float = 0.01,
    compute_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LinearQuantStats]:
    """Quantize one nn.Linear via MSE-scale + Cholesky GPTQ + H-CD + recovery guarantee."""
    # Never combine device + dtype conversion in one .to() call on MPS (PyTorch 2.14.1 MPS bug).
    w_f32 = linear.weight.detach().cpu().to(dtype=torch.float32).to(device=compute_device)
    norm_h_f64 = hessian.detach().cpu().to(dtype=torch.float64) / float(max(num_tokens, 1))
    norm_h = norm_h_f64.to(dtype=torch.float32).to(device=compute_device).contiguous()
    if not torch.isfinite(w_f32).all():
        raise RuntimeError(f"{name}: input weight contains non-finite values")
    if not torch.isfinite(norm_h).all():
        raise RuntimeError(f"{name}: normalized Hessian contains non-finite values")

    # Baseline RTN Hessian loss for diagnostics
    init_scales = find_mse_scale_int4(w_f32, group_size=group_size)
    rtn_codes = torch.clamp(
        torch.round(w_f32 / init_scales.repeat_interleave(group_size, dim=1)),
        INT4_QMIN,
        INT4_QMAX,
    ).to(torch.int8)
    rtn_diff = w_f32 - decode_int4_blocks(rtn_codes, init_scales, group_size=group_size)
    rtn_hloss = float((rtn_diff @ norm_h * rtn_diff).sum().item())

    # Step 1: Cholesky GPTQ (actorder="static", no column permutation)
    gptq_codes, gptq_diag_loss = cholesky_gptq_int4(
        w_f32,
        norm_h,
        init_scales,
        group_size=group_size,
        actorder=actorder,
        dampening_frac=dampening_frac,
    )
    if math.isfinite(gptq_diag_loss):
        gptq_diff = w_f32 - decode_int4_blocks(gptq_codes, init_scales, group_size=group_size)
        gptq_hloss = float((gptq_diff @ norm_h * gptq_diff).sum().item())
    else:
        gptq_hloss = float("nan")

    # Guard against ill-conditioned Hessians (e.g., Granite's high-dynamic-range attention/MLP
    # layers) where small-dampening Cholesky GPTQ can overflow or overshoot clipped INT4 bounds.
    best_codes = gptq_codes
    best_hloss = gptq_hloss
    if (not math.isfinite(best_hloss)) or (best_hloss > rtn_hloss):
        for retry_damp in (0.1, 0.5, 1.0):
            if retry_damp <= dampening_frac:
                continue
            cand_codes, cand_diag_loss = cholesky_gptq_int4(
                w_f32,
                norm_h,
                init_scales,
                group_size=group_size,
                actorder=actorder,
                dampening_frac=retry_damp,
            )
            if not math.isfinite(cand_diag_loss):
                continue
            cand_diff = w_f32 - decode_int4_blocks(cand_codes, init_scales, group_size=group_size)
            cand_hloss = float((cand_diff @ norm_h * cand_diff).sum().item())
            if math.isfinite(cand_hloss) and ((not math.isfinite(best_hloss)) or cand_hloss < best_hloss):
                best_codes = cand_codes
                best_hloss = cand_hloss
        if (not math.isfinite(best_hloss)) or (rtn_hloss < best_hloss):
            best_codes = rtn_codes
            best_hloss = rtn_hloss
    gptq_codes = best_codes
    gptq_hloss = best_hloss

    # Step 2: Gauss-Seidel Hessian Coordinate Descent (H-CD)
    hcd_codes, hcd_scales = refine_hessian_cd_int4(
        w_f32,
        norm_h,
        gptq_codes,
        init_scales,
        group_size=group_size,
        passes=cd_passes,
        dampening_frac=dampening_frac,
    )
    hcd_pre_diff = w_f32 - decode_int4_blocks(hcd_codes, hcd_scales, group_size=group_size)
    hcd_pre_loss = float((hcd_pre_diff @ norm_h * hcd_pre_diff).sum().item())
    if (not math.isfinite(hcd_pre_loss)) or (hcd_pre_loss > gptq_hloss * 1.001):
        hcd_codes = gptq_codes
        hcd_scales = init_scales

    # Step 3: Exact DEQUANTIZED_WEIGHT_RECOVERY block fixup (rare blocks only)
    fixed_codes, num_fixed = ensure_exact_recovery_codes(
        w_f32,
        hcd_codes,
        hcd_scales,
        h_diag=norm_h.diagonal(),
        group_size=group_size,
    )

    w_deq = decode_int4_blocks(fixed_codes, hcd_scales, group_size=group_size)
    if not torch.isfinite(w_deq).all():
        n_nan = int(torch.isnan(w_deq).sum().item())
        n_inf = int(torch.isinf(w_deq).sum().item())
        s_fin = bool(torch.isfinite(hcd_scales).all().item())
        s_nan = int(torch.isnan(hcd_scales).sum().item())
        s_inf = int(torch.isinf(hcd_scales).sum().item())
        raise RuntimeError(
            f"{name}: dequantized weight contains non-finite values "
            f"(w_deq nan={n_nan} inf={n_inf} device={w_deq.device} dtype={w_deq.dtype}; "
            f"hcd_scales finite={s_fin} nan={s_nan} inf={s_inf} min={float(hcd_scales.min()):.3e} max={float(hcd_scales.max()):.3e}; "
            f"fixed_codes dtype={fixed_codes.dtype} min={int(fixed_codes.min())} max={int(fixed_codes.max())}; "
            f"init_scales finite={bool(torch.isfinite(init_scales).all().item())})"
        )
    final_diff = w_f32 - w_deq
    hcd_hloss = float((final_diff @ norm_h * final_diff).sum().item())
    rel_mse = float(final_diff.square().sum().item() / max(float(w_f32.square().sum().item()), 1e-12))

    # Step 4: Verify exact ai_edge_quantizer recovery
    max_diff = verify_dequantized_weight_recovery(w_deq, block_size=group_size)

    stats = LinearQuantStats(
        name=name,
        shape=(int(w_f32.shape[0]), int(w_f32.shape[1])),
        rtn_hessian_loss=rtn_hloss,
        gptq_hessian_loss=gptq_hloss,
        hcd_hessian_loss=hcd_hloss,
        rel_mse=rel_mse,
        fixed_blocks=num_fixed,
        recovery_max_diff=max_diff,
    )
    return fixed_codes.cpu(), hcd_scales.cpu(), w_deq.cpu(), stats


@torch.no_grad()
def calibrate_granite_decoder_linears(
    model: nn.Module,
    sequences: list[torch.Tensor],
    *,
    device: torch.device,
    group_size: int = GROUP_SIZE,
    actorder: str = "static",
    cd_passes: int = 4,
    dampening_frac: float = 0.01,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[dict[str, torch.Tensor], list[LinearQuantStats]]:
    """Sequentially calibrate all 40 Granite 4.0 decoder layers via GPTQ + H-CD.

    Returns:
      - `dequantized_weights`: map of `module_name -> exact FP32 dequantized weight tensor`
      - `stats`: per-linear quantization diagnostics
    """
    model.eval()
    t0 = time.time()
    if progress_cb:
        progress_cb(f"Capturing layer-0 inputs for {len(sequences)} calibration sequences on {device}...")
    layer_inputs = _capture_layer0_inputs(model, sequences, device)

    dequantized_weights: dict[str, torch.Tensor] = {}
    all_stats: list[LinearQuantStats] = []
    num_layers = len(model.model.layers)
    # Run per-column GPTQ + H-CD solvers on CPU when on Apple MPS: avoids PyTorch 2.14.1 MPS
    # 2D-indexing SIGBUS bugs and eliminates ~10k tiny per-column Metal kernel launches/layer.
    solver_device = torch.device("cpu") if device.type == "mps" else device

    for layer_idx, layer in enumerate(model.model.layers):
        lt0 = time.time()
        prefix = f"model.layers.{layer_idx}"

        # Stage A: Accumulate H_qkv = X_in^T X_in in FP64 on CPU (avoid MPS matmul/hook aliasing)
        k_qkv = layer.self_attn.q_proj.in_features
        h_qkv = torch.zeros((k_qkv, k_qkv), dtype=torch.float64, device="cpu")
        n_tok = 0
        for h_cpu, _, _ in layer_inputs:
            x = layer.input_layernorm(h_cpu.to(device=device))
            x2d = x.detach().cpu().to(dtype=torch.float32).reshape(-1, k_qkv)
            h_qkv.add_((x2d.T @ x2d).to(dtype=torch.float64))
            n_tok += x2d.shape[0]

        for proj_name in ("q_proj", "k_proj", "v_proj"):
            mod = getattr(layer.self_attn, proj_name)
            full_name = f"{prefix}.self_attn.{proj_name}"
            _, _, w_deq, st = _quantize_linear_with_hessian(
                full_name,
                mod,
                h_qkv,
                n_tok,
                group_size=group_size,
                actorder=actorder,
                cd_passes=cd_passes,
                dampening_frac=dampening_frac,
                compute_device=solver_device,
            )
            mod.weight.data.copy_(w_deq.to(dtype=mod.weight.dtype).to(device=mod.weight.device))
            dequantized_weights[full_name] = w_deq
            all_stats.append(st)
        del h_qkv

        # Stage B: Accumulate H_o for o_proj using the quantized q/k/v projections
        k_o = layer.self_attn.o_proj.in_features
        h_o = torch.zeros((k_o, k_o), dtype=torch.float64, device="cpu")

        def _o_hook(_m: nn.Module, args: tuple[Any, ...]) -> None:
            x2d = args[0].detach().cpu().to(dtype=torch.float32).reshape(-1, k_o)
            h_o.add_((x2d.T @ x2d).to(dtype=torch.float64))

        hook_o = layer.self_attn.o_proj.register_forward_pre_hook(_o_hook)
        try:
            for h_cpu, mask_cpu, pos_cpu in layer_inputs:
                x = layer.input_layernorm(h_cpu.to(device=device))
                mask = mask_cpu.to(device=device) if mask_cpu is not None else None
                pos = (pos_cpu[0].to(device=device), pos_cpu[1].to(device=device)) if pos_cpu is not None else None
                layer.self_attn(hidden_states=x, attention_mask=mask, use_cache=False, position_embeddings=pos)
        finally:
            hook_o.remove()

        o_name = f"{prefix}.self_attn.o_proj"
        _, _, w_deq, st = _quantize_linear_with_hessian(
            o_name,
            layer.self_attn.o_proj,
            h_o,
            n_tok,
            group_size=group_size,
            actorder=actorder,
            cd_passes=cd_passes,
            dampening_frac=dampening_frac,
            compute_device=solver_device,
        )
        layer.self_attn.o_proj.weight.data.copy_(
            w_deq.to(dtype=layer.self_attn.o_proj.weight.dtype).to(device=layer.self_attn.o_proj.weight.device)
        )
        dequantized_weights[o_name] = w_deq
        all_stats.append(st)
        del h_o

        # Compute post-attention hidden states with quantized self_attn and accumulate H_mlp_in
        k_mlp_in = layer.shared_mlp.input_linear.in_features
        h_mlp_in = torch.zeros((k_mlp_in, k_mlp_in), dtype=torch.float64, device="cpu")
        post_attn_states: list[torch.Tensor] = []
        for h_cpu, mask_cpu, pos_cpu in layer_inputs:
            h_dev = h_cpu.to(device=device)
            x = layer.input_layernorm(h_dev)
            mask = mask_cpu.to(device=device) if mask_cpu is not None else None
            pos = (pos_cpu[0].to(device=device), pos_cpu[1].to(device=device)) if pos_cpu is not None else None
            attn_out, _ = layer.self_attn(
                hidden_states=x, attention_mask=mask, use_cache=False, position_embeddings=pos
            )
            h_mid = h_dev + attn_out * layer.residual_multiplier
            post_attn_states.append(h_mid.to(device="cpu"))
            x_mlp = layer.post_attention_layernorm(h_mid).detach().cpu().to(dtype=torch.float32).reshape(-1, k_mlp_in)
            h_mlp_in.add_((x_mlp.T @ x_mlp).to(dtype=torch.float64))

        # Stage C: Quantize shared_mlp.input_linear (fused SwiGLU gate+up)
        in_name = f"{prefix}.shared_mlp.input_linear"
        _, _, w_deq, st = _quantize_linear_with_hessian(
            in_name,
            layer.shared_mlp.input_linear,
            h_mlp_in,
            n_tok,
            group_size=group_size,
            actorder=actorder,
            cd_passes=cd_passes,
            dampening_frac=dampening_frac,
            compute_device=solver_device,
        )
        layer.shared_mlp.input_linear.weight.data.copy_(
            w_deq.to(dtype=layer.shared_mlp.input_linear.weight.dtype).to(
                device=layer.shared_mlp.input_linear.weight.device
            )
        )
        dequantized_weights[in_name] = w_deq
        all_stats.append(st)
        del h_mlp_in

        # Stage D: Accumulate H_mlp_out using quantized input_linear and quantize output_linear
        k_mlp_out = layer.shared_mlp.output_linear.in_features
        h_mlp_out = torch.zeros((k_mlp_out, k_mlp_out), dtype=torch.float64, device="cpu")

        def _out_hook(_m: nn.Module, args: tuple[Any, ...]) -> None:
            x2d = args[0].detach().cpu().to(dtype=torch.float32).reshape(-1, k_mlp_out)
            h_mlp_out.add_((x2d.T @ x2d).to(dtype=torch.float64))

        hook_out = layer.shared_mlp.output_linear.register_forward_pre_hook(_out_hook)
        try:
            for h_mid_cpu in post_attn_states:
                x_mlp = layer.post_attention_layernorm(h_mid_cpu.to(device=device))
                layer.shared_mlp(x_mlp)
        finally:
            hook_out.remove()

        out_name = f"{prefix}.shared_mlp.output_linear"
        _, _, w_deq, st = _quantize_linear_with_hessian(
            out_name,
            layer.shared_mlp.output_linear,
            h_mlp_out,
            n_tok,
            group_size=group_size,
            actorder=actorder,
            cd_passes=cd_passes,
            dampening_frac=dampening_frac,
            compute_device=solver_device,
        )
        layer.shared_mlp.output_linear.weight.data.copy_(
            w_deq.to(dtype=layer.shared_mlp.output_linear.weight.dtype).to(
                device=layer.shared_mlp.output_linear.weight.device
            )
        )
        dequantized_weights[out_name] = w_deq
        all_stats.append(st)
        del h_mlp_out

        # Stage E: Propagate exact quantized layer output to layer_idx + 1
        next_inputs: list[tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]] = []
        for ( _, mask_cpu, pos_cpu), h_mid_cpu in zip(layer_inputs, post_attn_states, strict=True):
            h_mid = h_mid_cpu.to(device=device)
            mlp_out = layer.shared_mlp(layer.post_attention_layernorm(h_mid))
            h_next = h_mid + mlp_out * layer.residual_multiplier
            next_inputs.append((h_next.to(device="cpu"), mask_cpu, pos_cpu))
        layer_inputs = next_inputs
        if device.type == "mps":
            torch.mps.empty_cache()

        if progress_cb:
            layer_stats = all_stats[-6:]
            mean_rel = sum(s.rel_mse for s in layer_stats) / len(layer_stats)
            rtn_tot = sum(s.rtn_hessian_loss for s in layer_stats)
            hcd_tot = sum(s.hcd_hessian_loss for s in layer_stats)
            fixed_tot = sum(s.fixed_blocks for s in layer_stats)
            max_rec = max(s.recovery_max_diff for s in layer_stats)
            progress_cb(
                f"Layer {layer_idx + 1:2d}/{num_layers} ({time.time() - lt0:.1f}s, total {time.time() - t0:.1f}s): "
                f"H-loss {rtn_tot:.4f} -> {hcd_tot:.4f} | rel_mse={mean_rel:.5f} | "
                f"fixed_blocks={fixed_tot} | max_rec_diff={max_rec:.2e}"
            )

    return dequantized_weights, all_stats


class _FP32MasterBF16Weight(nn.Module):
    """Parametrizes a BF16 RMSNorm weight with an FP32 master parameter and BF16 STE."""

    def __init__(self, init_weight: torch.Tensor) -> None:
        super().__init__()
        w_f32 = init_weight.detach().to(dtype=torch.float32)
        self.register_buffer("init_master", w_f32.clone())
        self.master = nn.Parameter(w_f32.clone())

    def forward(self, _orig: torch.Tensor) -> torch.Tensor:
        target_dtype = _orig.dtype
        if target_dtype == torch.float32:
            return self.master
        return self.master.to(target_dtype) + (self.master - self.master.detach()).to(target_dtype)


@torch.no_grad()
def compute_teacher_hidden_states(
    teacher_model: nn.Module,
    sequences: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    """Compute final decoder hidden states (post model.norm) from the unquantized BF16 teacher."""
    teacher_model.eval()
    states: list[torch.Tensor] = []
    for idx, seq in enumerate(sequences):
        inp = seq.to(dtype=torch.long).to(device=device).unsqueeze(0)
        h = teacher_model.model(input_ids=inp, use_cache=False, return_dict=True).last_hidden_state
        states.append(h.detach().cpu())
        if device.type == "mps" and (idx + 1) % 32 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return states


def run_rmsnorm_kl_qat(
    student_model: nn.Module,
    train_rows: list[torch.Tensor],
    train_teacher_states: list[torch.Tensor],
    val_rows: list[torch.Tensor],
    val_teacher_states: list[torch.Tensor],
    *,
    device: torch.device,
    num_epochs: int = 2,
    batch_size: int = 4,
    norm_lr: float = 1.5e-4,
    anchor_wd: float = 0.02,
    kl_chunk_size: int = 128,
    seed: int = 42,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """End-to-end FP32-master RMSNorm KL QAT over frozen quantized INT4 decoder weights."""
    student_model.eval()
    student_model.requires_grad_(False)
    logits_scaling = float(getattr(student_model.config, "logits_scaling", 1.0))

    # Enable gradient checkpointing across decoder layers to keep memory low during backward
    orig_layer_fwds = []
    for lyr in student_model.model.layers:
        orig_fn = lyr.forward
        orig_layer_fwds.append((lyr, orig_fn))

        def _ckpt_fwd(*args: Any, _orig: Any = orig_fn, **kwargs: Any) -> Any:
            if torch.is_grad_enabled():
                return torch.utils.checkpoint.checkpoint(
                    functools.partial(_orig, **kwargs), *args, use_reentrant=False
                )
            return _orig(*args, **kwargs)

        lyr.forward = _ckpt_fwd

    norm_modules: dict[str, tuple[nn.Module, _FP32MasterBF16Weight]] = {}
    norm_params: list[nn.Parameter] = []
    norm_inits: list[torch.Tensor] = []
    for mod_name, mod in list(student_model.named_modules()):
        if not (mod_name.startswith("model.layers.") or mod_name == "model.norm"):
            continue
        if "norm" not in mod_name.lower():
            continue
        if not hasattr(mod, "weight") or not isinstance(mod.weight, torch.Tensor) or mod.weight.ndim != 1:
            continue
        wrap = _FP32MasterBF16Weight(mod.weight).to(device=mod.weight.device)
        parametrize.register_parametrization(mod, "weight", wrap)
        norm_modules[f"{mod_name}.weight"] = (mod, wrap)
        norm_params.append(wrap.master)
        norm_inits.append(wrap.init_master)

    if not norm_params:
        raise RuntimeError("No RMSNorm parameters found for KL QAT")

    @torch.no_grad()
    def _eval_val_kl() -> float:
        kls = []
        for seq, h_t in zip(val_rows, val_teacher_states, strict=True):
            inp = seq.to(device=device, dtype=torch.long).unsqueeze(0)
            h_s = student_model.model(input_ids=inp, use_cache=False, return_dict=True).last_hidden_state
            kl = hidden_kl_scaled(
                h_s,
                h_t,
                student_model.lm_head,
                logits_scaling=logits_scaling,
                token_chunk_size=kl_chunk_size,
                checkpointed=False,
            )
            kls.append(float(kl.item()))
        return sum(kls) / max(len(kls), 1)

    try:
        init_val_kl = _eval_val_kl()
        if progress_cb:
            progress_cb(
                f"RMSNorm KL QAT start: {len(norm_modules)} norms "
                f"({sum(p.numel() for p in norm_params):,} params), "
                f"logits_scaling={logits_scaling}, init_val_kl={init_val_kl:.6f}"
            )

        optimizer = torch.optim.AdamW(norm_params, lr=norm_lr, betas=(0.9, 0.95), weight_decay=0.0)
        rng = torch.Generator().manual_seed(seed)
        steps_per_epoch = math.ceil(len(train_rows) / batch_size)
        total_steps = num_epochs * steps_per_epoch
        step = 0
        t0 = time.time()

        best_val_kl = init_val_kl
        best_norms: dict[str, torch.Tensor] = {
            k: wrap.master.detach().cpu().clone() for k, (_, wrap) in norm_modules.items()
        }
        history: list[dict[str, float]] = []

        for epoch in range(1, num_epochs + 1):
            perm = torch.randperm(len(train_rows), generator=rng).tolist()
            ep_losses: list[float] = []
            for b0 in range(0, len(train_rows), batch_size):
                batch_indices = perm[b0 : b0 + batch_size]
                if step < 0.08 * total_steps:
                    lr_mult = (step + 1) / max(1.0, 0.08 * total_steps)
                else:
                    prog = (step - 0.08 * total_steps) / max(1.0, 0.92 * total_steps)
                    lr_mult = 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog))
                cur_lr = norm_lr * lr_mult
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr

                optimizer.zero_grad()
                b_loss = 0.0
                for idx in batch_indices:
                    inp = train_rows[idx].to(device=device, dtype=torch.long).unsqueeze(0)
                    h_s = student_model.model(
                        input_ids=inp, use_cache=False, return_dict=True
                    ).last_hidden_state
                    loss = hidden_kl_scaled(
                        h_s,
                        train_teacher_states[idx],
                        student_model.lm_head,
                        logits_scaling=logits_scaling,
                        token_chunk_size=kl_chunk_size,
                        checkpointed=True,
                    ) / float(len(batch_indices))
                    loss.backward()
                    b_loss += float(loss.item())

                torch.nn.utils.clip_grad_norm_(norm_params, 1.0)
                optimizer.step()

                if anchor_wd > 0:
                    with torch.no_grad():
                        for p, p0 in zip(norm_params, norm_inits, strict=True):
                            p.sub_(p - p0, alpha=cur_lr * anchor_wd)

                ep_losses.append(b_loss)
                step += 1
                if progress_cb and (step == 1 or step % 8 == 0 or step == total_steps):
                    recent = sum(ep_losses[-min(len(ep_losses), 8) :]) / min(len(ep_losses), 8)
                    progress_cb(
                        f"  KL QAT epoch {epoch}/{num_epochs} step {step}/{total_steps} "
                        f"lr={cur_lr:.2e} train_kl={recent:.6f} ({time.time() - t0:.1f}s)"
                    )

            val_kl = _eval_val_kl()
            rec = {
                "epoch": float(epoch),
                "train_kl": sum(ep_losses) / max(len(ep_losses), 1),
                "val_kl": val_kl,
                "elapsed_s": round(time.time() - t0, 2),
            }
            history.append(rec)
            if progress_cb:
                progress_cb(
                    f"  KL QAT epoch {epoch} done: train_kl={rec['train_kl']:.6f} "
                    f"val_kl={val_kl:.6f} (best={min(val_kl, best_val_kl):.6f})"
                )
            if val_kl <= best_val_kl:
                best_val_kl = val_kl
                best_norms = {
                    k: wrap.master.detach().cpu().clone() for k, (_, wrap) in norm_modules.items()
                }

        return {
            "init_val_kl": init_val_kl,
            "best_val_kl": best_val_kl,
            "history": history,
            "norm_overrides": best_norms,
        }
    finally:
        for lyr, orig_fn in orig_layer_fwds:
            lyr.forward = orig_fn
        for _, (mod, _) in norm_modules.items():
            if parametrize.is_parametrized(mod, "weight"):
                parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)


def save_dequantized_hf_checkpoint(
    source_model_dir: Path,
    output_dir: Path,
    dequantized_weights: dict[str, torch.Tensor],
    norm_overrides: dict[str, torch.Tensor],
    *,
    report: dict[str, Any] | None = None,
) -> Path:
    """Save an FP32 HF checkpoint with exact dequantized INT4 linears and tuned RMSNorms.

    Tied `embed_tokens` / `lm_head` and all non-decoder tensors are preserved untouched (upcast to FP32
    so litert-torch loads exact values without BF16 rounding of the fake-quantized INT4 grid).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(source_model_dir, torch_dtype=torch.float32)

    for mod_name, w_f32 in dequantized_weights.items():
        mod = model.get_submodule(mod_name)
        mod.weight.data.copy_(w_f32.to(dtype=torch.float32))

    for param_name, w_f32 in norm_overrides.items():
        mod_name = param_name.removesuffix(".weight")
        mod = model.get_submodule(mod_name)
        mod.weight.data.copy_(w_f32.to(dtype=torch.float32))

    model.config.torch_dtype = torch.float32
    model.config.dtype = "float32"
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(source_model_dir)
    tokenizer.save_pretrained(output_dir)

    if report is not None:
        (output_dir / "calibration_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return output_dir
