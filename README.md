# LiteRT-LM vs llama.cpp — Granite 4.0 1B (4-bit)

Head-to-head 4-bit summarization benchmark comparing [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (`0.19.2`) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (`7fce543`) on [`ibm-granite/granite-4.0-1b`](https://huggingface.co/ibm-granite/granite-4.0-1b) (50 frozen CNN/DailyMail test articles, greedy decoding, `ctx=4096`).

Prebuilt 4-bit `.litertlm` model: **[`nchapman/granite-4.0-1b-litert-lm`](https://huggingface.co/nchapman/granite-4.0-1b-litert-lm)**  
Full results (8-bit, stock OCTAV, unoptimized baselines, synthetic benchmarks, methodology): **[`RESULTS.md`](RESULTS.md)**

## 4-bit Results (Apple M4 Max, 128 GiB)

| Backend | Engine | Quant | Size (MiB) | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | Wall time (s) | ROUGE-1 / 2 / L |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **GPU** | **LiteRT-LM** | `int4-cal` | 992.5 | 2,458 | **151.0** | 363 | **96.3** | **0.319 / 0.134 / 0.203** |
| **GPU** | **llama.cpp** | `Q4_K_M` | 995.1 | **2,666** | 102.6 | **348** | 146.6 | 0.298 / 0.124 / 0.192 |
| **CPU** | **llama.cpp** | `Q4_K_M` | 995.1 | **433** | **84.9** | **2,114** | **267.2** | 0.301 / 0.119 / 0.195 |
| **CPU** | **LiteRT-LM** | `int4-cal`¹ | 1,006.7 | 396 | 28.3 | 2,328 | 514.6 | **0.326 / 0.141 / 0.209** |

*Median prefill, decode, and TTFT across 50 articles; wall time is total time for all 50 articles. CPU runs pinned to 12 P-cores.*  
*¹ LiteRT-LM `0.19.2` on macOS runs CPU via XNNPACK (~240 partitions/token; Google's newer [YNNPACK](https://developers.google.com/edge/litert/releases/litert_aug) CPU delegate is currently Linux ARM64 only). This ~28 tok/s decode ceiling matches Google's own M4 Max CPU benchmarks for LiteRT-LM (`24–31 tok/s` across 2B–3B models).*

## Key Takeaways

- **GPU:** LiteRT-LM (`int4-cal` with `--apply_gpu_composites True`) was **1.52× faster end-to-end** (`96.3 s` vs `146.6 s`, `151` vs `103 tok/s` decode) with higher ROUGE than `llama.cpp Q4_K_M`.
- **CPU:** `llama.cpp` was **3.0× faster on decode** (`84.9` vs `28.3 tok/s`) and **1.9× faster overall** (`267.2 s` vs `514.6 s`).
- **Setup note:** Both runtimes required FP16-overflow workarounds for Granite 4.0 1B (`prefer_activation_type="fp32"` in LiteRT-LM; BF16 `ffn_down` on layers 4 and 39 in `llama.cpp`). LiteRT-LM also required non-default `--apply_gpu_composites True` and offline INT4 calibration (`scripts/calibrate_int4.py`) to achieve the GPU numbers above.

## Quickstart

### Run the prebuilt `.litertlm` model

```bash
pip install "litert-lm>=0.19.0"

litert-lm benchmark \
  --from-huggingface-repo nchapman/granite-4.0-1b-litert-lm \
  granite-4.0-1b_dynamic_wi4b32_afp32_cal.litertlm \
  --backend gpu -p 512 -d 128 --max-num-tokens 4096 --runs 3
```

### Reproduce locally (macOS Apple Silicon)

```bash
uv sync --extra convert --extra dev

# 1. Prepare frozen 50-article test set
uv run python scripts/prepare_dataset.py

# 2. Build llama.cpp (Metal + Accelerate) and convert GGUFs
bash scripts/setup_and_convert_gguf.sh

# 3. Calibrate INT4 weights and export .litertlm
uv run --extra convert python scripts/calibrate_int4.py \
  --calib-cache data/calib_cnn_dailymail_train_128x1024.pt --save-calib-only
uv run --extra convert python scripts/calibrate_int4.py \
  --calib-cache data/calib_cnn_dailymail_train_128x1024.pt
uv run --extra convert python scripts/convert_litertlm.py \
  --model models/hf/granite-4.0-1b-int4-cal --quant int4-cal --verify-backends gpu,cpu

# 4. Run 4-bit benchmark
uv run python scripts/benchmark.py --backends gpu,cpu --quants int4-cal
```

### Tests

```bash
uv run --extra convert --extra dev pytest -q
```
