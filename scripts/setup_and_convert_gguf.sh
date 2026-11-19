#!/usr/bin/env bash
# Clone/build llama.cpp with full hardware acceleration (Metal + Accelerate on macOS,
# CUDA on Linux if available) and convert ibm-granite/granite-4.0-1b to GGUF (Q8_0, Q4_K_M, Q4_0).

set -euo pipefail

MODEL_ID="${MODEL_ID:-ibm-granite/granite-4.0-1b}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-third_party/llama.cpp}"
OUT_DIR="${OUT_DIR:-models/gguf}"
HF_LOCAL_DIR="${HF_LOCAL_DIR:-models/hf/granite-4.0-1b}"

mkdir -p "$(dirname "$LLAMA_CPP_DIR")" "$OUT_DIR"

# 1. Clone or reuse llama.cpp
if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  echo "=== Cloning ggml-org/llama.cpp into $LLAMA_CPP_DIR ==="
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
else
  echo "=== Using existing llama.cpp checkout at $LLAMA_CPP_DIR ($(git -C "$LLAMA_CPP_DIR" rev-parse --short HEAD)) ==="
fi

# 2. Build llama.cpp with optimal platform acceleration
BUILD_DIR="$LLAMA_CPP_DIR/build"
OS_NAME="$(uname -s)"
ARCH="$(uname -m)"

CMAKE_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
)

if [[ "$OS_NAME" == "Darwin" ]]; then
  echo "=== Configuring llama.cpp for macOS ($ARCH) with Metal + Accelerate ==="
  CMAKE_FLAGS+=(
    -DGGML_METAL=ON
    -DGGML_METAL_EMBED_LIBRARY=ON
    -DGGML_ACCELERATE=ON
  )
  NPROC="$(sysctl -n hw.ncpu)"
else
  NPROC="$(nproc)"
  if command -v nvcc >/dev/null 2>&1; then
    echo "=== Configuring llama.cpp for Linux ($ARCH) with CUDA ==="
    CMAKE_FLAGS+=(-DGGML_CUDA=ON)
  else
    echo "=== Configuring llama.cpp for Linux ($ARCH) CPU ==="
  fi
fi

cmake -S "$LLAMA_CPP_DIR" -B "$BUILD_DIR" "${CMAKE_FLAGS[@]}"
cmake --build "$BUILD_DIR" --config Release --target llama-server llama-bench llama-quantize llama-cli -j "$NPROC"

# 3. Download HF checkpoint locally (exact same source weights as LiteRT-LM conversion)
echo "=== Ensuring Hugging Face checkpoint $MODEL_ID is downloaded to $HF_LOCAL_DIR ==="
uv run python -c '
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
' "$MODEL_ID" "$HF_LOCAL_DIR"

# 4. Convert HF -> BF16 GGUF (matching Granite 4.0's native bfloat16 weights), then quantize.
# Granite 4.0 1B has massive SwiGLU activations (~184k in blk.4.ffn_down, ~43k in blk.39.ffn_down)
# that overflow llama.cpp Metal's FP16 simdgroup_half8x8 mul_mm kernels (> 65504 -> inf/NaN).
# Keeping blk.(4|39).ffn_down.weight as BF16 dispatches Metal's simdgroup_bfloat8x8 kernel
# at full GPU speed with zero overflow.
BF16_GGUF="$OUT_DIR/granite-4.0-1b-bf16.gguf"
if [[ ! -f "$BF16_GGUF" ]]; then
  echo "=== Converting HF checkpoint -> $BF16_GGUF ==="
  uv run --extra convert python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$HF_LOCAL_DIR" \
    --outfile "$BF16_GGUF" \
    --outtype bf16
fi

QUANT_BIN="$BUILD_DIR/bin/llama-quantize"
for QTYPE in Q8_0 Q4_K_M Q4_0; do
  TARGET="$OUT_DIR/granite-4.0-1b-${QTYPE}.gguf"
  if [[ ! -f "$TARGET" ]]; then
    echo "=== Quantizing -> $TARGET ==="
    "$QUANT_BIN" --tensor-type 'blk\.(4|39)\.ffn_down\.weight=bf16' "$BF16_GGUF" "$TARGET" "$QTYPE"
  else
    echo "Already exists: $TARGET"
  fi
done

echo "=== Done! GGUF files in $OUT_DIR ==="
ls -lh "$OUT_DIR"/*.gguf
