# LiteRT-LM vs llama.cpp — Granite 4.0 1B Summarization Benchmark

Head-to-head performance and quality-parity benchmark comparing [Google LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and [llama.cpp](https://github.com/ggml-org/llama.cpp) on [`ibm-granite/granite-4.0-1b`](https://huggingface.co/ibm-granite/granite-4.0-1b), using the CNN/DailyMail `summary-v3` evaluation setup from [`nchapman/lillm-summarizer`](https://github.com/nchapman/lillm-summarizer).

## Why Granite 4.0 1B needs custom `.litertlm` conversion

- `ibm-granite/granite-4.0-1b` is a **dense 40-layer decoder-only transformer** (`GraniteMoeHybridForCausalLM` with all 40 `layer_types` set to `"attention"`, GQA 16Q/4KV, SwiGLU, tied embeddings, 128K context) — there is no prebuilt `.litertlm` on Hugging Face.
- Its tokenizer sets `bos_token_id == eos_token_id == 100257` (`<|end_of_text|>`) with `add_bos_token=False`. Exporting with a clean minimal Granite 4.0 template (`--use_jinja_template False`) avoids minijinja template errors and prevents spurious BOS prepending.
- Both runtimes are converted from the **exact same Hugging Face checkpoint** and compared at matched quantization tiers:
  - **8-bit:** LiteRT-LM `dynamic_wi8_afp32` vs llama.cpp `Q8_0`
  - **4-bit:** LiteRT-LM `int4` blockwise-32 OCTAV on linears + `int8` embedding vs llama.cpp `Q4_K_M` (and `Q4_0`)

## Apple Silicon (macOS) Quickstart

Everything below runs natively on Apple Silicon (`arm64`) — `litert-converter`, `litert-lm` (Metal GPU + XNNPACK CPU), and `llama.cpp` (Metal GPU + Apple Accelerate) all ship/build natively for macOS `arm64`.

### 1. Install Python dependencies (with `uv`)

```bash
uv sync --extra convert --extra dev
```

### 2. Prepare the frozen CNN/DailyMail evaluation set

Uses the exact `lillm-summarizer` reservoir sampling (`abisee/cnn_dailymail` 3.0.0 test split, 50-example pilot seed `20261028`, plus optional disjoint 250-example confirmation seed `20261029`):

```bash
uv run python scripts/prepare_dataset.py
# Optional: also generate the 250-example confirmation split
uv run python scripts/prepare_dataset.py --include-confirmation
```

Every article in both splits fits cleanly inside the 4,096-token KV cache budget (max prompt length is 2,610 tokens on pilot and 2,385 tokens on confirmation, leaving >1,024 tokens for generation).

### 3. Convert Granite 4.0 1B to `.litertlm` (INT8 + INT4 blockwise-32)

Exports both quantization tiers with a full prefill ladder (`1024..1`) and `cache_length=4096`, then verifies bundle metadata, exact chat-template parity against `AutoTokenizer`, and an 8-question greedy floor gate on both Metal (`gpu`) and `cpu`:

```bash
uv run --extra convert python scripts/convert_litertlm.py --quant all --verify-backends gpu,cpu
```

### 4. Build `llama.cpp` (Metal + Accelerate) and convert Granite 4.0 1B to GGUF

Compiles `llama.cpp` with `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_ACCELERATE=ON` on macOS and quantizes the same HF checkpoint to `Q8_0`, `Q4_K_M`, and `Q4_0`:

```bash
bash scripts/setup_and_convert_gguf.sh
```

### 5. Run the head-to-head benchmark

Runs greedy summarization (`temperature=0`, `top_k=1`, `seed=42`, `max_output_tokens=1024`, `max_num_tokens=4096`) on both Metal GPU and Apple Silicon P-cores (auto-detected via `sysctl hw.perflevel0.physicalcpu`), plus optional synthetic microbenchmarks (`litert-lm benchmark` vs `llama-bench`):

```bash
uv run python scripts/benchmark.py --backends gpu,cpu --quants int8,int4 --run-synthetic
```

Quick smoke run on the first 5 articles:

```bash
uv run python scripts/benchmark.py --limit 5 --backends gpu --quants int4
```

Outputs are written to `results/<timestamp>/`:
- `summary.md` — side-by-side Markdown table (prefill tok/s, decode tok/s, median TTFT, median latency, total wall time, ROUGE-1/2/L, cap stops)
- `report.json` — full hardware metadata, per-run stats, and synthetic benchmark outputs
- `*.jsonl` — per-example generated summaries and timings

## Hardware Utilization Notes (macOS Apple Silicon)

- **GPU (Metal):**
  - **LiteRT-LM:** `Backend.GPU()` with warmed cache directory (shaders compiled during warmup, excluded from timed runs).
  - **llama.cpp:** `llama-server -ngl 99 -fa on -c 4096 -b 1024 -ub 1024 --cache-ram 0` (full Metal offload, Flash Attention enabled, cross-request prompt cache disabled so every article is freshly prefilled).
- **CPU:**
  - Both engines are pinned to Apple Silicon performance cores (`hw.perflevel0.physicalcpu`) by default to avoid E-core straggler slowdown; override with `--threads N`.

## Tests

```bash
uv run --extra dev pytest -q
```
