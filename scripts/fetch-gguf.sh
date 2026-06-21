#!/usr/bin/env bash
# fetch-gguf.sh — download a GGUF model and tokenizer for the simple demo
#
# Usage:
#   scripts/fetch-gguf.sh                    # download default (1.5B Qwen2.5)
#   scripts/fetch-gguf.sh 3b                # download 3B model
#   scripts/fetch-gguf.sh 7b                # download 7B model
#   scripts/fetch-gguf.sh custom <url>       # download custom GGUF
#
# Output: ~/.cache/fak-models/gguf/<model-file>.gguf
#         ~/.cache/fak-models/tokenizers/<tokenizer-dir>/

set -euo pipefail

# Model configurations
declare -A MODELS=(
    ["1.5b"]="Qwen2.5-1.5B-Instruct.Q8_0"
    ["3b"]="Qwen2.5-3B-Instruct.Q8_0"
    ["7b"]="Qwen2.5-7B-Instruct.Q4_K_M"
)

declare -A SIZES=(
    ["1.5b"]="1.6GB"
    ["3b"]="3.2GB"
    ["7b"]="4.7GB"
)

BASE_URL="https://huggingface.co/mradermacher/Qwen2.5"

# Parse arguments
SIZE="${1:-1.5b}"
CUSTOM_URL=""

if [[ "$SIZE" == "custom" ]]; then
    CUSTOM_URL="$2"
    MODEL_NAME=$(basename "$CUSTOM_URL" .gguf)
    TOK_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json"
else
    # Normalize size key
    SIZE=$(echo "$SIZE" | tr '[:upper:]' '[:lower:]')
    if [[ -z "${MODELS[$SIZE]}" ]]; then
        echo "Error: unknown size '$SIZE'. Choose from: 1.5b, 3b, 7b, or 'custom <url>'" >&2
        exit 1
    fi
    MODEL_NAME="${MODELS[$SIZE]}"
    TOK_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json"
fi

# Output directories
CACHE_DIR="$HOME/.cache/fak-models"
GGUF_DIR="$CACHE_DIR/gguf"
TOK_DIR="$CACHE_DIR/tokenizers/qwen2.5"

mkdir -p "$GGUF_DIR" "$TOK_DIR"

# Download URLs
if [[ -n "$CUSTOM_URL" ]]; then
    GGUF_URL="$CUSTOM_URL"
    GGUF_FILE="$MODEL_NAME.gguf"
else
    # Convert size to URL format: 1.5b → 1.5B, 3b → 3B, 7b → 7B
    SIZE_UPPER=$(echo "$SIZE" | tr '[:lower:]' '[:upper:]')
    if [[ "$SIZE" == "1.5b" ]]; then SIZE_UPPER="1.5B"; fi
    GGUF_URL="$BASE_URL-${SIZE_UPPER}-Instruct-GGUF/resolve/main/$MODEL_NAME.gguf"
    GGUF_FILE="$MODEL_NAME.gguf"
fi

GGUF_PATH="$GGUF_DIR/$GGUF_FILE"
TOK_PATH="$TOK_DIR/tokenizer.json"

echo "📥 Fetching GGUF model..."
echo "   Model: $MODEL_NAME"
echo "   Size: ${SIZES[$SIZE]:-unknown}"
echo "   Target: $GGUF_PATH"
echo ""

# Check if already downloaded
if [[ -f "$GGUF_PATH" ]]; then
    echo "✅ Model already exists at $GGUF_PATH"
    read -p "Re-download? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping model download."
    else
        rm -f "$GGUF_PATH"
        curl -L --progress-bar "$GGUF_URL" -o "$GGUF_PATH"
    fi
else
    curl -L --progress-bar "$GGUF_URL" -o "$GGUF_PATH"
fi

echo ""
echo "📥 Fetching tokenizer..."

if [[ -f "$TOK_PATH" ]]; then
    echo "✅ Tokenizer already exists at $TOK_PATH"
else
    curl -L --progress-bar "$TOK_URL" -o "$TOK_PATH"
fi

echo ""
echo "✅ Done!"
echo ""
echo "To run the demo:"
echo "  go run ./cmd/simpledemo -gguf $GGUF_PATH -tok $TOK_DIR"
echo ""
echo "Or set environment variables:"
echo "  export FAK_GGUF=\"$GGUF_PATH\""
echo "  export FAK_TOKENIZER=\"$TOK_DIR\""
