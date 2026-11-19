"""Unit tests for calibrated INT4 (GPTQ + H-CD + DEQUANTIZED_WEIGHT_RECOVERY) pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ml_dtypes = pytest.importorskip("ml_dtypes")

from litert_vs_llamacpp.calibrate_int4 import (
    INT4_QMAX,
    INT4_QMIN,
    cholesky_gptq_int4,
    collect_eval_ids,
    decode_int4_blocks,
    ensure_exact_recovery_codes,
    find_mse_scale_int4,
    hidden_kl_scaled,
    quantize_litert_block_scale,
    refine_hessian_cd_int4,
    verify_dequantized_weight_recovery,
)


def test_quantize_litert_block_scale_matches_flatbuffer_roundtrip() -> None:
    scales = torch.tensor([1e-8, 1e-5, 0.003712, 0.0419, 1.337], dtype=torch.float32)
    q = quantize_litert_block_scale(scales)
    np_rt = q.numpy().astype(ml_dtypes.bfloat16).astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(q.numpy(), np_rt, rtol=0.0, atol=0.0)
    assert float(q.min()) >= 1e-6


def test_gptq_and_hcd_reduce_hessian_loss_and_recover_exactly() -> None:
    rng = torch.Generator().manual_seed(1234)
    n_out, k_in = 64, 128
    w = torch.randn((n_out, k_in), generator=rng, dtype=torch.float32) * 0.05

    # Correlated activations so GPTQ + H-CD have non-trivial off-diagonal Hessian structure.
    z = torch.randn((256, k_in), generator=rng, dtype=torch.float32)
    mixing = torch.randn((k_in, k_in), generator=rng, dtype=torch.float32) / (k_in**0.5)
    x = z @ mixing
    h = (x.T @ x) / x.shape[0]

    scales = find_mse_scale_int4(w, group_size=32)
    rtn_codes = torch.clamp(torch.round(w / scales.repeat_interleave(32, dim=1)), INT4_QMIN, INT4_QMAX).to(
        torch.int8
    )
    rtn_diff = w - decode_int4_blocks(rtn_codes, scales, group_size=32)
    rtn_loss = float((rtn_diff @ h * rtn_diff).sum().item())

    gptq_codes, _ = cholesky_gptq_int4(w, h, scales, group_size=32, actorder="static")
    gptq_diff = w - decode_int4_blocks(gptq_codes, scales, group_size=32)
    gptq_loss = float((gptq_diff @ h * gptq_diff).sum().item())

    hcd_codes, hcd_scales = refine_hessian_cd_int4(
        w, h, gptq_codes, scales, group_size=32, passes=4
    )
    fixed_codes, _ = ensure_exact_recovery_codes(
        w, hcd_codes, hcd_scales, h_diag=h.diagonal(), group_size=32
    )
    w_deq = decode_int4_blocks(fixed_codes, hcd_scales, group_size=32)
    hcd_diff = w - w_deq
    hcd_loss = float((hcd_diff @ h * hcd_diff).sum().item())

    assert gptq_loss < rtn_loss
    assert hcd_loss < gptq_loss
    assert int(fixed_codes.min()) >= INT4_QMIN
    assert int(fixed_codes.max()) <= INT4_QMAX

    max_rec_diff = verify_dequantized_weight_recovery(w_deq, block_size=32, tol=1e-4)
    assert max_rec_diff < 1e-6


def test_ensure_exact_recovery_fixes_pathological_gap_blocks() -> None:
    # Construct blocks with non-unit min_diff that fail naive recovery:
    #   block 0: magnitudes {0, 2, 5} -> min_diff=2, does not divide 5
    #   block 1: magnitudes {0, 3, 6} -> min_diff=3, 3*s may not sit on bf16->fp16 grid
    #   block 2: magnitudes {0, 3, 5} -> min_diff=2, does not divide 3 or 5
    #   block 3: all zeros -> already valid
    codes = torch.zeros((4, 32), dtype=torch.int8)
    codes[0, 0] = 2
    codes[0, 1] = -5
    codes[1, 0] = 3
    codes[1, 1] = -6
    codes[2, 0] = 3
    codes[2, 1] = 5

    scales = quantize_litert_block_scale(torch.full((4, 1), 0.03125, dtype=torch.float32))
    w = decode_int4_blocks(codes, scales, group_size=32)

    fixed_codes, num_fixed = ensure_exact_recovery_codes(w, codes, scales, group_size=32)
    assert num_fixed == 3

    w_fixed = decode_int4_blocks(fixed_codes, scales, group_size=32)
    max_diff = verify_dequantized_weight_recovery(w_fixed, block_size=32, tol=1e-4)
    max_diff_aeq = verify_dequantized_weight_recovery(
        w_fixed, block_size=32, tol=1e-4, use_ai_edge_quantizer=True
    )
    assert max_diff < 1e-6
    assert max_diff_aeq < 1e-6


def test_quantize_linear_recovers_cleanly_on_ill_conditioned_hessian() -> None:
    """Ill-conditioned Hessians (like Granite's spike layers) must never emit NaN or beat RTN worse."""
    import math
    from litert_vs_llamacpp.calibrate_int4 import _quantize_linear_with_hessian

    rng = torch.Generator().manual_seed(7)
    lin = torch.nn.Linear(64, 32, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(32, 64, generator=rng) * 0.1)

    # Extreme 1e8 dynamic-range correlated Hessian
    x = torch.randn(128, 64, generator=rng)
    x[:, :4] *= 1e4
    x[:, 4:8] = x[:, :4] + 1e-3 * torch.randn(128, 4, generator=rng)
    h = (x.T @ x).double()

    codes, scales, w_deq, st = _quantize_linear_with_hessian(
        "test.ill_cond", lin, h, 128, group_size=32, cd_passes=2, compute_device=torch.device("cpu")
    )
    assert torch.isfinite(w_deq).all()
    assert math.isfinite(st.gptq_hessian_loss)
    assert math.isfinite(st.hcd_hessian_loss)
    assert int(codes.min().item()) >= -8
    assert int(codes.max().item()) <= 7
    assert st.hcd_hessian_loss <= st.rtn_hessian_loss * 1.001
    assert st.recovery_max_diff < 1e-5


def test_capture_layer0_inputs_raises_clear_error_when_hook_never_fires() -> None:
    """If model.model forward bypasses layers[0], raise RuntimeError instead of bare KeyError."""
    import pytest
    from litert_vs_llamacpp.calibrate_int4 import _capture_layer0_inputs

    class _NoopInner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
            return input_ids

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _NoopInner()

    with pytest.raises(RuntimeError, match="Layer-0 forward pre-hook did not capture"):
        _capture_layer0_inputs(_DummyModel(), [torch.arange(8, dtype=torch.long)], torch.device("cpu"))


def test_hidden_kl_scaled_zero_on_identity_and_respects_logits_scaling() -> None:
    rng = torch.Generator().manual_seed(7)
    lm_head = torch.nn.Linear(16, 64, bias=False)
    h_t = torch.randn((1, 10, 16), generator=rng)
    h_s = h_t + 0.2 * torch.randn((1, 10, 16), generator=rng)

    kl_zero = hidden_kl_scaled(h_t, h_t, lm_head, logits_scaling=8.0, token_chunk_size=4, checkpointed=False)
    assert abs(float(kl_zero.item())) < 1e-6

    kl_scale1 = hidden_kl_scaled(h_s, h_t, lm_head, logits_scaling=1.0, token_chunk_size=4, checkpointed=False)
    kl_scale8 = hidden_kl_scaled(h_s, h_t, lm_head, logits_scaling=8.0, token_chunk_size=4, checkpointed=False)
    assert float(kl_scale8.item()) > 0.0
    assert float(kl_scale8.item()) < float(kl_scale1.item())


def test_collect_eval_ids_reads_existing_and_ignores_missing(tmp_path: Path) -> None:
    f = tmp_path / "pilot.jsonl"
    f.write_text(
        json.dumps({"id": "abc-1", "document": "doc", "reference": "ref"}) + "\n",
        encoding="utf-8",
    )
    ids = collect_eval_ids([f, tmp_path / "does_not_exist.jsonl"])
    assert ids == {"abc-1"}
