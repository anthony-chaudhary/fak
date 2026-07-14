#!/usr/bin/env bash
# glm52_e2e_after_serve_dgx3.sh - the end-to-end proof to run AFTER the GLM-5.2
# serve runner reaches GLM52_SERVE_READY (tools/glm52_stage_serve_dgx3.sh).
#
# It exercises the full goal against the LOCALLY-SERVED GLM-5.2 on the A100 node:
#   1. the #413 serving witness (direct + fak-gateway + quarantine flows), and
#   2. the fak LOCAL GUARD end-to-end - front the served model with the kernel and
#      drive a real agent turn through `fak guard --provider openai`, the same path
#      Claude Code uses (it talks OpenAI via OPENAI_BASE_URL, which fak guard injects
#      into the child only). The kernel adjudicates every tool call and writes a
#      hash-chained audit journal.
#
# Asserts no throughput/quality number of its own; the witness records the measured
# evidence. RUN ON THE GPU HOST once :8000 serves glm-5.2.
#
# Usage:  FAK=/path/to/fak SRC=/path/to/fak-checkout bash tools/glm52_e2e_after_serve_dgx3.sh
set -uo pipefail
export HOME="${HOME:-/root}" GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}"
export PATH="/usr/local/go/bin:/usr/local/cuda-12.8/bin:$PATH"

SRC="${SRC:-/projects/fak-src}"
FAK="${FAK:-/tmp/fakdgx}"
PORT="${PORT:-8000}"
OUT="${OUT:-/projects/glm52-q4/E2E}"
BASE="http://127.0.0.1:${PORT}/v1"
PH(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$OUT.log"; echo "$*" > "$OUT.phase"; }

# 0. confirm the endpoint actually serves glm-5.2 (never fake a pass).
if ! curl -sf -m 8 "$BASE/models" | grep -q glm-5.2; then
  PH "E2E_ABORT endpoint not serving glm-5.2 at $BASE"; exit 10
fi
PH "E2E_START endpoint live at $BASE"

# 1. the #413 serving witness. --engine-cache-engine left EMPTY on purpose:
#    llama.cpp has no sglang/vllm cache-reset, and the witness records that
#    honestly rather than claiming exact-span eviction the engine cannot offer.
PH "WITNESS_RUN"
python3 "$SRC/tools/glm52_serving_witness.py" \
  --base-url "$BASE" --model glm-5.2 --context-length 8192 \
  --fak-command "$FAK" \
  --out "$SRC/experiments/glm52/full-size-serving-witness.json" \
  --markdown "$SRC/experiments/glm52/full-size-serving-witness.md" \
  >> "$OUT.log" 2>&1
PH "WITNESS_DONE rc=$?"

# 2. the fak local-guard end-to-end against the served GLM-5.2.
PH "GUARD_RUN"
"$FAK" guard --provider openai --base-url "$BASE" --audit "$OUT.audit.jsonl" -- \
  bash -c 'curl -sf -m 60 "$OPENAI_BASE_URL/chat/completions" -H "Content-Type: application/json" -d "{\"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"In one short sentence, what is the capital of France?\"}],\"max_tokens\":40}"' \
  >> "$OUT.log" 2>&1
PH "GUARD_DONE rc=$?"

# 3. summarize the evidence (witness verdict, guard turn, kernel adjudication count).
WV=$(grep -o '"full_size_serving_witness"[^,}]*' "$SRC/experiments/glm52/full-size-serving-witness.json" 2>/dev/null | head -1)
ADJ=$(wc -l < "$OUT.audit.jsonl" 2>/dev/null || echo 0)
PH "E2E_SUMMARY witness=[$WV] audit_rows=$ADJ"
PH "E2E_COMPLETE"
