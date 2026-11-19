# Full Benchmark Results & Methodology

Hardware: **Apple M4 Max (16-core CPU / 12 P-cores, 40-core Metal GPU, 128 GiB unified memory)**, macOS `arm64`.  
Model: [`ibm-granite/granite-4.0-1b`](https://huggingface.co/ibm-granite/granite-4.0-1b).  
Runtimes: `litert-lm==0.19.2`, `litert-torch==0.10.0`, `llama.cpp` (`7fce543`, Metal + Accelerate).

## 50-Article CNN/DailyMail Summarization

Greedy decoding (`temperature=0`, `top_k=1`, `seed=42`, `ctx=4096`, `max_output_tokens=1024`) on the frozen 50-article pilot split (`data/cnn_dailymail_test_pilot.jsonl`, 51,429 total prompt tokens, `summary-v3` prompt).

| Engine | Backend | Quant | Size (MiB) | Init (s) | Prefill med (tok/s) | Decode med (tok/s) | TTFT med (ms) | E2E med (s) | Wall time (s) | ROUGE-1 / 2 / L | Cap stops |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LiteRT-LM (GPU composites) | gpu | `int4_block32_cal` | 992.5 | 1.57 | 2,457.9 | 151.0 | 363 | 1.828 | 96.31 | 0.319 / 0.134 / 0.203 | 0/50 |
| llama.cpp | gpu | `Q4_K_M` | 995.1 | 0.78 | 2,666.3 | 102.6 | 348 | 2.574 | 146.64 | 0.298 / 0.124 / 0.192 | 1/50 |
| LiteRT-LM (no GPU composites) | gpu | `int4_block32_cal` | 1,006.7 | 3.44 | 1,442.4 | 81.3 | 598 | 3.324 | 170.75 | 0.319 / 0.134 / 0.203 | 0/50 |
| LiteRT-LM (no GPU composites) | gpu | `int4_block32` (OCTAV) | 1,117.0 | 3.77 | 1,204.8 | 76.8 | 756 | 3.644 | 191.77 | 0.313 / 0.130 / 0.200 | 0/50 |
| LiteRT-LM (no GPU composites) | gpu | `dynamic_wi8` | 1,681.5 | 4.18 | 1,589.8 | 83.7 | 557 | 3.122 | 162.00 | 0.330 / 0.138 / 0.216 | 0/50 |
| llama.cpp | gpu | `Q8_0` | 1,672.3 | 1.04 | 2,726.6 | 75.7 | 351 | 3.068 | 163.27 | 0.332 / 0.141 / 0.216 | 0/50 |
| llama.cpp | cpu | `Q4_K_M` | 995.1 | 1.05 | 433.4 | 84.9 | 2,114 | 4.983 | 267.21 | 0.301 / 0.119 / 0.195 | 0/50 |
| LiteRT-LM | cpu | `int4_block32_cal` | 1,006.7 | 1.95 | 396.0 | 28.3 | 2,328 | 9.798 | 514.55 | 0.326 / 0.141 / 0.209 | 0/50 |
| LiteRT-LM | cpu | `int4_block32` (OCTAV) | 1,117.0 | 2.47 | 402.0 | 28.4 | 2,442 | 9.791 | 510.75 | 0.315 / 0.125 / 0.198 | 0/50 |
| llama.cpp | cpu | `Q8_0` | 1,672.3 | 0.78 | 546.2 | 79.4 | 1,702 | 4.401 | 234.96 | 0.338 / 0.144 / 0.220 | 0/50 |
| LiteRT-LM | cpu | `dynamic_wi8` | 1,681.5 | 1.65 | 628.0 | 27.2 | 1,513 | 8.825 | 453.00 | 0.333 / 0.138 / 0.213 | 0/50 |

## Synthetic Microbenchmark (`litert-lm benchmark` vs `llama-bench`, `ctx=4096`, 3 runs)

| Engine | Backend | Quant | `pp512` (tok/s) | `pp1024` (tok/s) | `tg128` (tok/s) | `tg256` (tok/s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| LiteRT-LM (GPU composites, cold) | gpu | `int4_block32_cal` | 3,313.4 | 3,437.0 | 162.1 | 157.9 |
| LiteRT-LM (GPU composites, post-50-article warm) | gpu | `int4_block32_cal` | 2,741.3 | 2,148.8 | 129.2 | 106.5 |
| llama.cpp | gpu | `Q4_K_M` | 3,023.4 | 1,784.7 | 129.9 | 120.5 |
| llama.cpp | gpu | `Q8_0` | 3,541.7 | 2,252.4 | 97.2 | 90.7 |
| llama.cpp | cpu | `Q4_K_M` | 397.9 | 410.3 | 103.8 | 89.8 |
| llama.cpp | cpu | `Q8_0` | 484.0 | 506.7 | 80.5 | 69.0 |
| LiteRT-LM | cpu | `int4_block32_cal` | 377.0 | 348.5 | 27.9 | 28.1 |
| LiteRT-LM | cpu | `dynamic_wi8` | 721.3 | 767.9 | 27.3 | 27.1 |

## Methodology & Implementation Notes

- **Granite 4.0 1B FP16 overflow:** Granite 4.0 1B produces a ~184k SwiGLU intermediate activation in layer 4 (~43k in layer 39), overflowing FP16 (`65,504`) on Metal in both runtimes.
  - **LiteRT-LM:** bundles set `prefer_activation_type="fp32"` (`scripts/convert_litertlm.py`).
  - **llama.cpp:** GGUFs are converted with `--outtype bf16` and `--tensor-type 'blk\.(4|39)\.ffn_down\.weight=bf16'` (`scripts/setup_and_convert_gguf.sh`).
- **LiteRT GPU composites:** `litert-torch==0.10.0` defaults `apply_gpu_composites=False`. Enabling `--apply_gpu_composites True --use_qkv_norm_rope_composite False --prefill_logits False` enables dynamic KV-cache context (`Is Dynamic Context: YES`) and fused GPU attention, cutting 50-article GPU wall time from `170.75 s` to `96.31 s` (`1.77×`) with identical ROUGE.
- **Offline INT4 calibration (`int4_block32_cal`):** Cholesky GPTQ (`actorder="static"`) + monotonic Gauss–Seidel Hessian coordinate descent on LiteRT's `bf16 -> fp16` scale grid + `DEQUANTIZED_WEIGHT_RECOVERY` block fixup + FP32-master `RMSNorm` KL QAT (`src/litert_vs_llamacpp/calibrate_int4.py`). Calibrated on 128 train + 16 validation sequences (`max_seq_len=1024`) from `cnn_dailymail` `train` split only, with asserted zero overlap against `data/cnn_dailymail_test_*.jsonl`.
- **CPU delegation on macOS:** In `litert-lm==0.19.2`, XNNPACK does not delegate `odml.rms_norm`, `odml.sdpa_transposed`, `odml.cache_update`, or `DYNAMIC_UPDATE_SLICE` (splitting 40-layer decode into 244 partitions), and `enable_ynnpack=True` is not linked in the macOS wheel.
- **Fairness controls:** Cross-request prompt caching disabled (`--cache-ram 0` / fresh conversation per article), warmup excluded from timed runs, exact chat template parity verified against `AutoTokenizer`, and CPU threads pinned to 12 Apple Silicon P-cores.
