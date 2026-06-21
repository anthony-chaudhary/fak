#!/usr/bin/env bash
# run_phase1_gate.sh -- collect and verify the production-readiness Phase 1 gate.
#
# Phase 1 needs two independent witnesses:
#   1. a named non-reference compute backend accepted by modelbench, and
#   2. a fresh live local-gpu/non-CPU 7-9B fak-agent report that reaches parity.
#
# This script composes the existing tools and fails closed. It never reuses stale
# remote evidence: the remote report is written with a run-specific prefix, and
# paritybench is pointed only at that run's local-gpu glob.
#
# Usage:
#   tools/run_phase1_gate.sh \
#     --backend <compute-backend-name> \
#     --endpoint <fleet-endpoint-name> \
#     --model <7-9B-model-id> \
#     [--max-turns 12]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="${FAK_PHASE1_BACKEND:-}"
ENDPOINT="${FAK_PHASE1_ENDPOINT:-worker-a}"
MODEL="${FAK_PHASE1_MODEL:-}"
MAXTURNS="${FAK_PHASE1_MAX_TURNS:-12}"
OUT_DIR="${FAK_PHASE1_OUT_DIR:-$ROOT/fak/experiments/parity}"
LOCAL_GLOB="${FAK_PHASE1_LOCAL_GLOB:-experiments/parity/local-*.json}"
REFERENCE_CARDS="${FAK_PHASE1_REFERENCE_CARDS:-experiments/parity/reference-frontier.json}"
REFERENCE="${FAK_PHASE1_REFERENCE:-claude-sonnet}"
RUN_ID="${FAK_PHASE1_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

usage() {
  sed -n '2,24p' "$0" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="${2:?--backend needs a value}"; shift 2 ;;
    --endpoint) ENDPOINT="${2:?--endpoint needs a value}"; shift 2 ;;
    --model) MODEL="${2:?--model needs a value}"; shift 2 ;;
    --max-turns) MAXTURNS="${2:?--max-turns needs a value}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
    --local) LOCAL_GLOB="${2:?--local needs a value}"; shift 2 ;;
    --reference-cards) REFERENCE_CARDS="${2:?--reference-cards needs a value}"; shift 2 ;;
    --reference) REFERENCE="${2:?--reference needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "phase1: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

slug() {
  printf '%s' "$1" | tr '/:_ .' '-----' | tr -cd 'A-Za-z0-9.-' | tr '[:upper:]' '[:lower:]'
}

mkdir -p "$OUT_DIR"
FAK_ROOT="$(cd "$ROOT/fak" && pwd)"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
case "$OUT_DIR" in
  "$FAK_ROOT") OUT_DIR_REL="." ;;
  "$FAK_ROOT"/*) OUT_DIR_REL="${OUT_DIR#"$FAK_ROOT"/}" ;;
  *)
    echo "phase1: --out-dir must live under $FAK_ROOT so paritybench can read the fresh report" >&2
    exit 2
    ;;
esac
MODEL_SLUG="$(slug "${MODEL:-missing-model}")"
BACKEND_SLUG="$(slug "${BACKEND:-missing-backend}")"
REMOTE_OUT="$OUT_DIR/phase1-${RUN_ID}-remote-${ENDPOINT}-${MODEL_SLUG}.json"
MODELBENCH_OUT="$OUT_DIR/phase1-${RUN_ID}-modelbench-${BACKEND_SLUG}.json"
LOCAL_GPU_GLOB="${OUT_DIR_REL}/phase1-${RUN_ID}-remote-*.json"

echo "[phase1] run-id=$RUN_ID endpoint=$ENDPOINT model=${MODEL:-<missing>} backend=${BACKEND:-<missing>}" >&2

backend_rc=0
if [[ -z "$BACKEND" ]]; then
  echo "[phase1] backend gate: FAIL -- pass --backend <non-reference compute backend>" >&2
  backend_rc=2
else
  echo "[phase1] backend gate: modelbench -backend $BACKEND -require-non-reference" >&2
  set +e
  ( cd "$ROOT/fak" && go run ./cmd/modelbench -backend "$BACKEND" -require-non-reference -out "$MODELBENCH_OUT" )
  backend_rc=$?
  set -e
  if [[ $backend_rc -eq 0 ]]; then
    echo "[phase1] backend gate: PASS -> $MODELBENCH_OUT" >&2
  else
    echo "[phase1] backend gate: FAIL (exit $backend_rc)" >&2
  fi
fi

remote_rc=0
if [[ -z "$MODEL" ]]; then
  echo "[phase1] remote 7-9B rung: FAIL -- pass --model <7-9B model id served by the endpoint>" >&2
  remote_rc=2
else
  echo "[phase1] remote 7-9B rung: $ENDPOINT -> $REMOTE_OUT" >&2
  set +e
  "$ROOT/tools/run_remote_model.sh" "$ENDPOINT" "$MODEL" "$REMOTE_OUT" "$MAXTURNS"
  remote_rc=$?
  set -e
  if [[ $remote_rc -eq 0 ]]; then
    echo "[phase1] remote 7-9B rung: PASS -> $REMOTE_OUT" >&2
  else
    echo "[phase1] remote 7-9B rung: FAIL (exit $remote_rc)" >&2
  fi
fi

echo "[phase1] capability gate: paritybench --local-gpu '$LOCAL_GPU_GLOB' --require-phase1" >&2
set +e
( cd "$ROOT/fak" && go run ./cmd/paritybench \
    --local "$LOCAL_GLOB" \
    --local-gpu "$LOCAL_GPU_GLOB" \
    --reference-cards "$REFERENCE_CARDS" \
    --reference "$REFERENCE" \
    --out-json "experiments/parity/parity.json" \
    --out-md "experiments/parity/PARITY.md" \
    --require-phase1 )
parity_rc=$?
set -e
if [[ $parity_rc -eq 0 ]]; then
  echo "[phase1] capability gate: PASS" >&2
else
  echo "[phase1] capability gate: FAIL (exit $parity_rc)" >&2
fi

if [[ $backend_rc -eq 0 && $remote_rc -eq 0 && $parity_rc -eq 0 ]]; then
  echo "[phase1] PASS: non-reference backend and live local-gpu 7-9B parity evidence are both present." >&2
  exit 0
fi

echo "[phase1] FAIL: backend_rc=$backend_rc remote_rc=$remote_rc parity_rc=$parity_rc" >&2
exit 1
