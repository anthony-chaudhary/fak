#!/usr/bin/env bash
# fetch-gguf.sh — download a GGUF model and tokenizer for the simple demo
#
# Usage:
#   scripts/fetch-gguf.sh                     # download default (1.5B Qwen2.5)
#   scripts/fetch-gguf.sh 3b                  # download 3B model
#   scripts/fetch-gguf.sh 7b                  # download 7B model
#   scripts/fetch-gguf.sh custom <url>        # download custom GGUF
#
# Flags / env (first-run + CI safety):
#   --yes, -y, FAK_FETCH_YES=1     re-download an existing file without prompting
#   FAK_FETCH_SHA256=<hex>         expected sha256 for a `custom` download (verified)
#   FAK_FETCH_SKIP_VERIFY=1        skip the sha256 check (escape hatch; not recommended)
#
# Integrity: every download goes to a temp file via `curl --fail` (an HTTP error
# body is never written as the model), is verified against a pinned sha256, and is
# only moved into place on success — a corrupt / truncated / error-page download is
# deleted and the script exits non-zero rather than leaving a broken artifact that
# the engine would only trip over later.
#
# Output: ~/.cache/fak-models/gguf/<model-file>.gguf
#         ~/.cache/fak-models/tokenizers/qwen2.5/tokenizer.json

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

# Pinned SHA-256 of each GGUF — the HuggingFace LFS content hash (the same digest
# huggingface_hub verifies downloads against), read from the resolve endpoint's
# X-Linked-ETag header. Update these if the upstream quant is re-rolled.
declare -A SHA256=(
    ["1.5b"]="cf68e79db07b6588220bb00a853e13a5e83b707d6811115aa1e3a70109f53a89"
    ["3b"]="8b3bd75789b50aea32f8d0bf3f8cc031f201cde7e4a9728db84a70932ff4a522"
    ["7b"]="b5fe74b613436747acf83405e896a338ded690037ab77581d1177829dc19f680"
)
# tokenizer.json from Qwen/Qwen2.5-1.5B-Instruct (shared across the Qwen2.5 sizes).
TOK_SHA256="c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"

BASE_URL="https://huggingface.co/mradermacher/Qwen2.5"

# --- integrity helpers ------------------------------------------------------

# sha256_file <path> -> lowercase hex digest, or "" if no sha256 tool is found.
sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo ""
    fi
}

# download_verified <url> <dest> <expected_sha> <label>
# Download to "<dest>.partial" with `curl --fail`, verify the sha256 (when known and
# not skipped), then move it into place. On ANY failure the temp file is removed and
# the function returns non-zero — so a half-written / corrupt artifact never lands at
# <dest>.
download_verified() {
    local url="$1" dest="$2" expected="$3" label="$4"
    local tmp="$dest.partial"
    rm -f "$tmp"
    # --fail: a 4xx/5xx is a non-zero exit, not an error body written to disk.
    # --location: follow the HuggingFace -> CDN redirect.
    if ! curl -fL --progress-bar "$url" -o "$tmp"; then
        rm -f "$tmp"
        echo "❌ download failed: $label" >&2
        echo "   URL: $url" >&2
        echo "   The source may be down, renamed, or rate-limited — nothing was written." >&2
        echo "   Retry later, or point at a mirror: scripts/fetch-gguf.sh custom <url>" >&2
        return 1
    fi
    if [[ "${FAK_FETCH_SKIP_VERIFY:-0}" == "1" ]]; then
        echo "⚠️  FAK_FETCH_SKIP_VERIFY=1 — skipping integrity check for $label." >&2
    elif [[ -n "$expected" ]]; then
        local got; got="$(sha256_file "$tmp")"
        if [[ -z "$got" ]]; then
            echo "⚠️  no sha256 tool (sha256sum/shasum) — cannot verify $label." >&2
        elif [[ "$got" != "$expected" ]]; then
            rm -f "$tmp"
            echo "❌ checksum mismatch for $label — deleted the corrupt download." >&2
            echo "   expected: $expected" >&2
            echo "   got:      $got" >&2
            echo "   The download was truncated/corrupted, or the upstream artifact changed." >&2
            echo "   If you trust the new upstream, re-run with FAK_FETCH_SKIP_VERIFY=1." >&2
            return 1
        else
            echo "🔒 verified sha256: $label"
        fi
    else
        echo "⚠️  no pinned sha256 for $label — integrity is HTTP-checked only." >&2
    fi
    mv -f "$tmp" "$dest"
}

main() {
    # Pull --yes/-y out of the args; keep the rest positional.
    local assume_yes="${FAK_FETCH_YES:-0}"
    local args=()
    local a
    for a in "$@"; do
        case "$a" in
            --yes|-y) assume_yes=1 ;;
            *)        args+=("$a") ;;
        esac
    done
    set -- "${args[@]}"

    local SIZE="${1:-1.5b}"
    local CUSTOM_URL=""
    local MODEL_NAME TOK_URL EXPECTED_SHA

    if [[ "$SIZE" == "custom" ]]; then
        CUSTOM_URL="${2:-}"
        if [[ -z "$CUSTOM_URL" ]]; then
            echo "Error: 'custom' needs a URL: scripts/fetch-gguf.sh custom <url>" >&2
            exit 1
        fi
        MODEL_NAME=$(basename "$CUSTOM_URL" .gguf)
        TOK_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json"
        EXPECTED_SHA="${FAK_FETCH_SHA256:-}"
    else
        # Normalize size key
        SIZE=$(echo "$SIZE" | tr '[:upper:]' '[:lower:]')
        if [[ -z "${MODELS[$SIZE]:-}" ]]; then
            echo "Error: unknown size '$SIZE'. Choose from: 1.5b, 3b, 7b, or 'custom <url>'" >&2
            exit 1
        fi
        MODEL_NAME="${MODELS[$SIZE]}"
        TOK_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json"
        EXPECTED_SHA="${FAK_FETCH_SHA256:-${SHA256[$SIZE]:-}}"
    fi

    # Output directories
    local CACHE_DIR="$HOME/.cache/fak-models"
    local GGUF_DIR="$CACHE_DIR/gguf"
    local TOK_DIR="$CACHE_DIR/tokenizers/qwen2.5"
    mkdir -p "$GGUF_DIR" "$TOK_DIR"

    # Download URLs
    local GGUF_URL GGUF_FILE
    if [[ -n "$CUSTOM_URL" ]]; then
        GGUF_URL="$CUSTOM_URL"
        GGUF_FILE="$MODEL_NAME.gguf"
    else
        # Convert size to URL format: 1.5b → 1.5B, 3b → 3B, 7b → 7B
        local SIZE_UPPER
        SIZE_UPPER=$(echo "$SIZE" | tr '[:lower:]' '[:upper:]')
        if [[ "$SIZE" == "1.5b" ]]; then SIZE_UPPER="1.5B"; fi
        GGUF_URL="$BASE_URL-${SIZE_UPPER}-Instruct-GGUF/resolve/main/$MODEL_NAME.gguf"
        GGUF_FILE="$MODEL_NAME.gguf"
    fi

    local GGUF_PATH="$GGUF_DIR/$GGUF_FILE"
    local TOK_PATH="$TOK_DIR/tokenizer.json"

    echo "📥 Fetching GGUF model..."
    echo "   Model: $MODEL_NAME"
    echo "   Size: ${SIZES[$SIZE]:-unknown}"
    echo "   Target: $GGUF_PATH"
    echo ""

    # Check if already downloaded
    if [[ -f "$GGUF_PATH" ]]; then
        echo "✅ Model already exists at $GGUF_PATH"
        local redownload=0
        if [[ "$assume_yes" == "1" ]]; then
            redownload=1
        elif [[ -t 0 ]]; then
            read -p "Re-download? [y/N] " -n 1 -r
            echo
            [[ $REPLY =~ ^[Yy]$ ]] && redownload=1
        else
            # Non-interactive (CI / piped): never prompt, never dead-end. Keep the
            # existing file and succeed; --yes / FAK_FETCH_YES=1 opts into a refetch.
            echo "   Non-interactive (no TTY) — keeping the existing file."
            echo "   Pass --yes or FAK_FETCH_YES=1 to force a re-download."
        fi
        if [[ "$redownload" == "1" ]]; then
            download_verified "$GGUF_URL" "$GGUF_PATH" "$EXPECTED_SHA" "GGUF model ($MODEL_NAME)"
        else
            echo "Skipping model download."
        fi
    else
        download_verified "$GGUF_URL" "$GGUF_PATH" "$EXPECTED_SHA" "GGUF model ($MODEL_NAME)"
    fi

    echo ""
    echo "📥 Fetching tokenizer..."

    if [[ -f "$TOK_PATH" ]]; then
        echo "✅ Tokenizer already exists at $TOK_PATH"
    else
        download_verified "$TOK_URL" "$TOK_PATH" "$TOK_SHA256" "tokenizer.json"
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
}

# Run unless sourced for testing (scripts/fetch-gguf_test.sh sets FETCH_GGUF_SOURCED=1).
if [[ "${FETCH_GGUF_SOURCED:-0}" != "1" ]]; then
    main "$@"
fi
