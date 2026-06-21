#!/usr/bin/env bash
# run_local_model.sh — drive ONE real local model through the fak agentic A/B
# harness (fak arm = kernel-mediated, baseline arm = unmediated) on the frozen
# tau-bench airline task, and write the agent-report JSON the parity aggregator
# ingests.
#
# It is the LOCAL leg of the parity workflow: the same `fak agent` rig that drives
# Gemini / Claude (via an OpenAI-compatible gateway) is pointed at a tiny CPU model
# served by experiments/agent-live/local_shim.py (a stdlib OpenAI-compatible shim
# over a cached transformers model). No GPU, no network, fully offline.
#
# Usage:
#   tools/run_local_model.sh <hf-model-id> <port> <out.json> [max-turns]
# Example:
#   tools/run_local_model.sh Qwen/Qwen2.5-1.5B-Instruct 8131 \
#       fak/experiments/parity/local-qwen-1.5b.json 12
set -euo pipefail

MODEL="${1:?usage: run_local_model.sh <model> <port> <out> [max-turns]}"
PORT="${2:?need a port}"
OUT="${3:?need an output path}"
MAXTURNS="${4:-12}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHIM="$ROOT/fak/experiments/agent-live/local_shim.py"
FAK_BIN="${FAK_BIN:-$ROOT/tools/.bin/fak.exe}"

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

mkdir -p "$(dirname "$OUT")" "$(dirname "$FAK_BIN")"

# Build the fak binary once (native; only test EXECUTION is blocked on this host,
# `go build` is fine — see CLAUDE.md).
if [[ ! -x "$FAK_BIN" || "${FAK_REBUILD:-0}" == "1" ]]; then
  echo "[runner] building fak -> $FAK_BIN" >&2
  ( cd "$ROOT/fak" && go build -o "$FAK_BIN" ./cmd/fak )
fi

echo "[runner] starting shim for $MODEL on :$PORT (cpu)..." >&2
python "$SHIM" --model "$MODEL" --port "$PORT" >"/tmp/shim-$PORT.log" 2>&1 &
SHIM_PID=$!
cleanup() { kill "$SHIM_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Wait for the shim to load the model and answer /v1/models (up to ~5 min for the
# bigger CPU models' weight load).
for i in $(seq 1 150); do
  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[runner] shim ready after ${i}s" >&2
    break
  fi
  if ! kill -0 "$SHIM_PID" 2>/dev/null; then
    echo "[runner] shim died during load; log:" >&2; tail -20 "/tmp/shim-$PORT.log" >&2; exit 1
  fi
  sleep 2
done

echo "[runner] running fak agent A/B (max-turns=$MAXTURNS)..." >&2
"$FAK_BIN" agent \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL" \
  --api-key-env NONE_LOCAL \
  --max-turns "$MAXTURNS" \
  --out "$OUT"

echo "[runner] wrote $OUT" >&2
