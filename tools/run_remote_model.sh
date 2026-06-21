#!/usr/bin/env bash
# run_remote_model.sh -- drive the fak agentic A/B harness against a model served
# on a REMOTE private endpoint, writing
# the agent-report JSON the parity aggregator ingests.
#
# This is the remote sibling of run_local_model.sh: instead of starting a local
# shim, the model server runs ON the endpoint (started by its owner, bound to the
# private network, and we point
# `fak agent --base-url` at it over Tailscale. The endpoint is resolved + reachability-
# gated by fleet_endpoints.sh, so an endpoint whose server is not up is reported and
# skipped, never silently hammered.
#
# Usage:
#   tools/run_remote_model.sh <endpoint-name> <model-id> <out.json> [max-turns]
# Example:
#   tools/run_remote_model.sh worker-a Qwen/Qwen2.5-1.5B-Instruct \
#       fak/experiments/parity/remote-worker-a-qwen.json 12
set -euo pipefail

EP="${1:?usage: run_remote_model.sh <endpoint-name> <model-id> <out> [max-turns]}"
MODEL="${2:?need a model id (must match what the remote server serves)}"
OUT="${3:?need an output path}"
MAXTURNS="${4:-12}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAK_BIN="${FAK_BIN:-$ROOT/tools/.bin/fak.exe}"

# Reachability gate (hard signal: the serve port accepts a connection). On failure
# this prints the reason to stderr and exits non-zero -- the endpoint is OPTIONAL,
# so a batch driver should treat a non-zero exit here as "skip this endpoint".
echo "[remote] resolving endpoint '$EP' ..." >&2
if ! BASE_URL="$("$ROOT/tools/fleet_endpoints.sh" --resolve "$EP")"; then
  echo "[remote] endpoint '$EP' not available -- start its server (see the runbook) or pick another." >&2
  exit 3
fi
echo "[remote] base-url: $BASE_URL  (model=$MODEL, max-turns=$MAXTURNS)" >&2

mkdir -p "$(dirname "$OUT")" "$(dirname "$FAK_BIN")"

# Build fak once (native build is fine on this host; only test EXECUTION is blocked).
if [[ ! -x "$FAK_BIN" || "${FAK_REBUILD:-0}" == "1" ]]; then
  echo "[remote] building fak -> $FAK_BIN" >&2
  ( cd "$ROOT/fak" && go build -o "$FAK_BIN" ./cmd/fak )
fi

echo "[remote] running fak agent A/B against $EP ..." >&2
"$FAK_BIN" agent \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --api-key-env NONE_LOCAL \
  --max-turns "$MAXTURNS" \
  --out "$OUT"

echo "[remote] wrote $OUT" >&2
