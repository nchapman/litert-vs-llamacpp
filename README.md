# LiteRT-LM vs llama.cpp — IBM Granite 4.0 1B Benchmark

Controlled head-to-head benchmark comparing [Google LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (`0.19.2`) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (`7fce543`) on [`ibm-granite/granite-4.0-1b`](https://huggingface.co/ibm-granite/granite-4.0-1b) for single-document summarization (50 frozen CNN/DailyMail test articles, greedy decoding, 4,096-token context).

The calibrated 4-bit `.litertlm` artifact produced by this repo is published on Hugging Face: **[`nchapman/granite-4.0-1b-litert-lm`](https://huggingface.co/nchapman/granite-4.0-1b-litert-lm)**.

## Summary of Findings

All measurements below were collected on an **Apple M4 Max (16-core CPU / 12 P-cores, 40-core Metal GPU, 128 GiB unified memory)** running macOS.

- **4-bit GPU (fastest configuration overall):** With LiteRT GPU composites enabled (`--apply_gpu_composites True`) and offline calibrated INT4 quantization (`int4_block32_cal`), LiteRT-LM completed the 50-article summarization workload in **96.3 s** (`151.0 tok/s` median decode, `2,457.9 tok/s` median prefill, ROUGE-1/2/L `0.319 / 0.134 / 0.203`, `0/50` cap stops) compared to `llama.cpp` `Q4_K_M` in **146.6 s** (`102.6 tok/s` decode, `2,666.3 tok/s` prefill, ROUGE `0.298 / 0.124 / 0.192`, `1/50` cap stops).
- **8-bit GPU:** End-to-end wall time was essentially tied (**162.0 s** LiteRT-LM vs. **163.3 s** `llama.cpp` `Q8_0`) with matched ROUGE (`0.330/0.138/0.216` vs. `0.332/0.141/0.216`). `llama.cpp` was faster on prefill (`2,727` vs. `1,590 tok/s`; note the 8-bit `.litertlm` run predates enabling GPU composites), while LiteRT-LM had slightly faster decode (`83.7` vs. `75.7 tok/s`).
- **CPU (12 P-cores):** `llama.cpp` (Apple Accelerate) was **2.9–3.0× faster on decode** (`79.4–84.9` vs. `27.2–28.4 tok/s`) and **~2× faster end-to-end** (`235–267 s` vs. `453–515 s`). In `litert-lm==0.19.2` on macOS, XNNPACK does not delegate `odml.*` composites or `DYNAMIC_UPDATE_SLICE` (fragmenting 40-layer decode into 244 partitions + single-threaded reference ops), and `enable_ynnpack=True` is not linked in the macOS wheel.
- **Configuration sensitivity on LiteRT-LM:** `litert-torch==0.10.0` defaults `apply_gpu_composites=False`. Without it, LiteRT-LM GPU decode dropped to `81.3 tok/s` (`170.7 s` wall time) because every decode token scanned the full static 4,096-slot FP32 KV cache with unfused attention. Enabling `--apply_gpu_composites True` cut 50-article wall time from `170.7 s` to `96.3 s` (`1.77×`) with identical ROUGE.
- **4-bit calibration impact:** Offline Cholesky GPTQ + Hessian coordinate descent + `RMSNorm` KL QAT + `DEQUANTIZED_WEIGHT_RECOVERY` (`int4_block32_cal`) improved LiteRT-LM 4-bit ROUGE over both stock LiteRT-LM OCTAV (`0.319/0.134/0.203` vs. `0.313/0.130/0.200` GPU; `0.326/0.141/0.209` vs. `0.315/0.125/0.198` CPU) and `llama.cpp Q4_K_M` (`0.298/0.124/0.192` GPU), while reducing bundle size to `992.5 MiB` via unified INT8 tied embeddings/`lm_head`.

## Context for iOS & Android Deployments

These desktop M4 Max measurements do not replace on-device profiling on target phones, where memory bandwidth, RAM ceilings, and thermal throttling differ significantly. Key architectural differences observed between the two runtimes:

| Dimension | LiteRT-LM (`0.19.2`) | `llama.cpp` (`7fce543`) |
| --- | --- | --- |
| **Apple GPU backend** | C++ Dawn WebGPU compiling to Metal 3 (not browser/JS) | Native Metal kernels + FlashAttention |
| **Android acceleration** | GPU (OpenCL/Vulkan/WebGPU), CPU (XNNPACK/YNNPACK), and AOT NPU (Qualcomm Hexagon, MediaTek NeuroPilot, Google Tensor) | CPU (NEON/KleidiAI) + GPU (Vulkan/OpenCL); limited NPU support |
| **CPU fallback** | Weak on macOS `0.19.2` (244 XNNPACK partitions; YNNPACK unlinked); verify YNNPACK/XNNPACK behavior on Android/iOS builds | Consistently strong and predictable across CPU targets |
| **Packaging** | Single `.litertlm` bundle containing model, tokenizer, chat template, and sampler config; dynamic context up to exported max | Single `.gguf` file with embedded metadata/tokenizer |
| **Model conversion** | Heavier PyTorch export (`litert-torch`) for non-Google models; requires explicit composite flags and custom calibration for competitive INT4 quality | Mature, fast `convert_hf_to_gguf.py` + `llama-quantize` across most HF architectures |

## Benchmark Results

### 50-Article CNN/DailyMail Summarization (Greedy, `ctx=4096`, `max_output_tokens=1024`)

All runs used the same 50 frozen articles (`data/cnn_dailymail_test_pilot.jsonl`, 51,429 total prompt tokens, `summary-v3` prompt).

| Engine | Backend | Quant | Size (MiB) | Init (s) | Prefill med (tok/s) | Decode med (tok/s) | TTFT med (ms) | E2E med (s) | Total wall (s) | ROUGE-1 / 2 / L | Cap stops |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| **LiteRT-LM (GPU composites)** | **gpu** | **`int4_block32_cal`** | **992.5** | **1.57** | **2,457.9** | **151.0** | **363** | **1.828** | **96.31** | **0.319 / 0.134 / 0.203** | **0/50** |
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

### Synthetic Microbenchmark (`litert-lm benchmark` vs `llama-bench`, `ctx=4096`, 3 runs)

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

## Methodology & Fairness Controls

- **Single source checkpoint:** Both runtimes are converted from the same `ibm-granite/granite-4.0-1b` BF16 checkpoint.
- **Granite 4.0 1B FP16 overflow handling:** Granite 4.0 1B uses muP scaling (`embedding_multiplier=12`, `logits_scaling=8`) and produces a ~184k SwiGLU intermediate activation in layer 4 (and ~43k in layer 39), overflowing FP16 (`65,504`) on Metal in both runtimes. To ensure numerically valid outputs on both sides:
  - **LiteRT-LM:** bundles set `prefer_activation_type="fp32"` in `scripts/convert_litertlm.py`.
  - **llama.cpp:** GGUFs are converted from BF16 (`--outtype bf16`) with `--tensor-type 'blk\.(4|39)\.ffn_down\.weight=bf16'` in `scripts/setup_and_convert_gguf.sh`.
- **Prompt & decoding parity:** Both engines run greedy decoding (`temperature=0`, `top_k=1`, `seed=42`, `max_output_tokens=1024`, `max_num_tokens=4096`) with cross-request prompt caching disabled (`--cache-ram 0` / fresh conversation per article) and warmup excluded from timed runs. The exported `.litertlm` chat template is verified for exact string parity against Hugging Face `AutoTokenizer`.
- **Zero evaluation leakage in INT4 calibration:** `scripts/calibrate_int4.py` samples 128 train + 16 validation sequences (`max_seq_len=1024`) strictly from the `cnn_dailymail` `train` split and asserts zero ID overlap against `data/cnn_dailymail_test_*.jsonl`.

## Quickstart

### Try the published `.litertlm` model directly

```bash
pip install "litert-lm>=0.19.0"

litert-lm benchmark \
  --from-huggingface-repo nchapman/granite-4.0-1b-litert-lm \
  granite-4.0-1b_dynamic_wi4b32_afp32_cal.litertlm \
  --backend gpu -p 512 -d 128 --max-num-tokens 4096 --runs 3
```

### Reproduce the full pipeline locally (macOS Apple Silicon)

```bash
# 1. Install dependencies
uv sync --extra convert --extra dev

# 2. Prepare the frozen 50-article CNN/DailyMail test set
uv run python scripts/prepare_dataset.py

# 3. Build llama.cpp (Metal + Accelerate) and convert BF16 -> Q8_0 / Q4_K_M / Q4_0 GGUFs
bash scripts/setup_and_convert_gguf.sh

# 4. Run offline INT4 calibration (pre-tokenize cache first to isolate pyarrow from PyTorch)
uv run --extra convert python scripts/calibrate_int4.py \
  --calib-cache data/calib_cnn_dailymail_train_128x1024.pt --save-calib-only
uv run --extra convert python scripts/calibrate_int4.py \
  --calib-cache data/calib_cnn_dailymail_train_128x1024.pt

# 5. Export calibrated INT4 .litertlm with GPU composites and verify on GPU + CPU
uv run --extra convert python scripts/convert_litertlm.py \
  --model models/hf/granite-4.0-1b-int4-cal \
  --quant int4-cal \
  --verify-backends gpu,cpu

# 6. Run the head-to-head benchmark (results saved to results/<timestamp>/)
uv run python scripts/benchmark.py --backends gpu,cpu --quants int8,int4,int4-cal --run-synthetic
```

### Tests

```bash
uv run --extra convert --extra dev pytest -q
```

## Repository Layout

- `src/litert_vs_llamacpp/calibrate_int4.py` — Cholesky GPTQ (`actorder="static"`), monotonic Gauss–Seidel H-CD on LiteRT's `bf16 -> fp16` scale grid, `DEQUANTIZED_WEIGHT_RECOVERY` block fixup, and FP32-master `RMSNorm` KL QAT.
- `src/litert_vs_llamacpp/prompts.py` — `summary-v3` prompt contract, minimal Granite 4.0 chat template, and 8-question floor gate.
- `scripts/calibrate_int4.py` — CLI driver for offline INT4 calibration.
- `scripts/convert_litertlm.py` — `litert-torch` export (`int8`, `int4`, `int4-cal`), GPU composite flags, FP32 activation metadata, and verification.
- `scripts/setup_and_convert_gguf.sh` — `llama.cpp` Metal + Accelerate build and Granite 4.0 BF16-safe GGUF conversion.
- `scripts/benchmark.py` — Head-to-head summarization and synthetic microbenchmark runner (`report.json`, `summary.md`, `*.jsonl`).
- `tests/` — Unit and regression tests for benchmark helpers and calibrated INT4 quantization.
