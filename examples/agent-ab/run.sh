#!/bin/bash
# run.sh — the fak agent live A/B, pointed at the model you choose.
#
# Runs the prompt-injection / destructive-op A/B (the same one the top-level
# README's safety table reports) and scores TWO arms side by side: the model
# with no kernel (baseline, naive-exec) vs the same model behind the kernel's
# floor (fak). Every run writes a report carrying a real transcript_sha.
#
#   ./run.sh                                 # OFFLINE: deterministic mock planner — no model, no network, no key
#   ./run.sh --local                         # LOCAL:   an Ollama model (qwen2.5:1.5b) over the OpenAI /v1 shape
#   ./run.sh --provider gemini --key-env GEMINI_API_KEY   # CLOUD: your own model (BILLS YOU)
#
# The default is --offline so the demo is safe and byte-reproducible with zero
# setup. --offline swaps the MODEL for a deterministic planner so the KERNEL's
# decisions are the only thing under test; it is NOT the live A/B with the
# network stubbed out. The live A/B is the --local / --provider lane.
#
# Requires: the `fak` binary on PATH (install per the top-level README, or
# `go build -o fak ./cmd/fak` and add it to PATH). --local also needs ollama.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
OUT="$HERE/agent-report.json"
LOG="$HERE/trace.log"
log(){ printf '\033[36m[agent-ab]\033[0m %s\n' "$*" >&2; }

command -v fak >/dev/null || { log "fak not found on PATH — install per the top-level README, or 'go build -o fak ./cmd/fak' and add it to PATH"; exit 1; }

MODE="offline"
PROVIDER="gemini"
KEY_ENV="GEMINI_API_KEY"
MODEL=""
BASE_URL=""

while [ $# -gt 0 ]; do
  case "$1" in
    --offline)  MODE="offline"; shift ;;
    --local)    MODE="local";   shift ;;
    --provider) MODE="cloud"; PROVIDER="$2"; shift 2 ;;
    --key-env)  KEY_ENV="$2";  shift 2 ;;
    --model)    MODEL="$2";    shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    *) log "unknown flag: $1"; exit 2 ;;
  esac
done

case "$MODE" in
  offline)
    log "OFFLINE A/B — deterministic mock planner (no model, no network, no key)."
    log "this proves the KERNEL's decisions; the live A/B is --local / --provider."
    set -- fak agent --offline -out "$OUT" -log "$LOG"
    ;;
  local)
    : "${MODEL:=qwen2.5:1.5b}"
    : "${BASE_URL:=http://localhost:11434/v1}"
    : "${OLLAMA_KEY:=ollama}"; export OLLAMA_KEY
    log "LOCAL live A/B — model=$MODEL via $BASE_URL (Ollama; the key is ignored)."
    command -v ollama >/dev/null && ollama pull "$MODEL" >&2 || log "note: 'ollama' not found — start your own OpenAI-compatible server at $BASE_URL"
    set -- fak agent -provider openai -base-url "$BASE_URL" -model "$MODEL" -api-key-env OLLAMA_KEY -out "$OUT" -log "$LOG"
    ;;
  cloud)
    : "${MODEL:=gemini-2.5-flash}"
    if [ -z "$BASE_URL" ]; then
      case "$PROVIDER" in
        gemini)    BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai" ;;
        openai)    BASE_URL="https://api.openai.com/v1" ;;
        xai)       BASE_URL="https://api.x.ai/v1" ;;
        anthropic) BASE_URL="https://api.anthropic.com" ;;
        *) log "unknown provider: $PROVIDER (expected openai|anthropic|gemini|xai)"; exit 2 ;;
      esac
    fi
    [ -n "${!KEY_ENV:-}" ] || { log "COST GUARD: \$$KEY_ENV is empty — refusing to make a paid call. export $KEY_ENV=... first."; exit 1; }
    log "CLOUD live A/B — provider=$PROVIDER model=$MODEL via $BASE_URL."
    log "WARNING: this BILLS your $PROVIDER account (two arms x up to max-turns)."
    set -- fak agent -provider "$PROVIDER" -base-url "$BASE_URL" -model "$MODEL" -api-key-env "$KEY_ENV" -out "$OUT" -log "$LOG"
    ;;
esac

log "running: $*"
echo
"$@"
echo
log "report: $OUT   trace: $LOG"
log "provenance: jq '{live, transcript_sha}' '$OUT'"
