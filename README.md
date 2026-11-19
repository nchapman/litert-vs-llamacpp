# LiteRT-LM vs llama.cpp — Granite 4.0 1B Benchmark

Head-to-head summarization benchmark comparing [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (`0.19.2`) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (`7fce543`) on [`ibm-granite/granite-4.0-1b`](https://huggingface.co/ibm-granite/granite-4.0-1b) (50 frozen CNN/DailyMail test articles, greedy decoding, `ctx=4096`, `max_output_tokens=1024`).

Prebuilt calibrated 4-bit `.litertlm` model: **[`nchapman/granite-4.0-1b-litert-lm`](https://huggingface.co/nchapman/granite-4.0-1b-litert-lm)**.

## Results (Apple M4 Max, 128 GiB)

| Backend | Engine | Quant | Size (MiB) | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | Wall time (s) | ROUGE-1 / 2 / L |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **GPU** | **LiteRT-LM** | `int4-cal`¹ | 992.5 | 2,458 | **151.0** | 363 | **96.3** | **0.319 / 0.134 / 0.203** |
| **GPU** | **llama.cpp** | `Q4_K_M` | 995.1 | **2,666** | 102.6 | **348** | 146.6 | 0.298 / 0.124 / 0.192 |
| **GPU** | **LiteRT-LM** | `int8` | 1,681.5 | 1,590 | **83.7** | 557 | **162.0** | 0.330 / 0.138 / 0.216 |
| **GPU** | **llama.cpp** | `Q8_0` | 1,672.3 | **2,727** | 75.7 | **351** | 163.3 | **0.332 / 0.141 / 0.216** |
| **CPU** | **llama.cpp** | `Q4_K_M` | 995.1 | **433** | **84.9** | **2,114** | **267.2** | 0.301 / 0.119 / 0.195 |
| **CPU** | **LiteRT-LM** | `int4-cal` | 1,006.7 | 396 | 28.3 | 2,328 | 514.6 | **0.326 / 0.141 / 0.209** |
| **CPU** | **llama.cpp** | `Q8_0` | 1,672.3 | 546 | **79.4** | 1,702 | **235.0** | **0.338 / 0.144 / 0.220** |
| **CPU** | **LiteRT-LM** | `int8` | 1,681.5 | **628** | 27.2 | **1,513** | 453.0 | 0.333 / 0.138 / 0.213 |

*Median prefill, decode, and TTFT across 50 articles; wall time is total end-to-end time for all 50 articles. CPU runs pinned to 12 P-cores.*  
*¹ Exported with `--apply_gpu_composites True` and offline INT4 calibration (`scripts/calibrate_int4.py`, CNN/DailyMail train split only). Without GPU composites, LiteRT-LM GPU `int4-cal` decoded at `81.3 tok/s` (`170.7 s` wall time).*

## Key Takeaways

- **4-bit GPU:** LiteRT-LM (`int4-cal` with GPU composites) was **1.52× faster end-to-end** (`96.3 s` vs `146.6 s`, `151` vs `103 tok/s` decode) with higher ROUGE (`0.319` vs `0.298` ROUGE-1) than `llama.cpp Q4_K_M`.
- **8-bit GPU:** Essentially tied on wall time (`162.0 s` vs `163.3 s`) and ROUGE (`0.330` vs `0.332`).
- **CPU:** `llama.cpp` was **~3× faster on decode** (`79–85` vs `27–28 tok/s`) and **~2× faster overall**. On macOS `0.19.2`, LiteRT-LM's XNNPACK backend does not delegate several attention/KV-cache ops and YNNPACK is not linked.
- **Setup caveats:** Both runtimes needed FP16-overflow workarounds for Granite 4.0 1B (`prefer_activation_type="fp32"` in LiteRT-LM; `--outtype bf16` + `blk.(4|39).ffn_down.weight=bf16` in `llama.cpp`). LiteRT-LM also required non-default `--apply_gpu_composites True` and offline INT4 calibration to reach the GPU numbers above.

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

# 4. Run benchmark (writes results/<timestamp>/summary.md and report.json)
uv run python scripts/benchmark.py --backends gpu,cpu --quants int8,int4-cal
```

### Tests

```bash
uv run --extra convert --extra dev pytest -q
```
