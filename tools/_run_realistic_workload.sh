#!/usr/bin/env bash
# Drive the transcript-derived realistic-workload benchmark: a headline replay on the REAL
# prefix + a tool-call-fraction sensitivity sweep. Sized for the 135M CPU model.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${FAK_FLEETSERVE_BIN:-$ROOT/fak/.benchbin/fleetserve}"
if [[ ! -x "$BIN" && -x "$BIN.exe" ]]; then
  BIN="$BIN.exe"
fi
DIR="${FAK_MODEL_DIR:-$ROOT/fak/internal/model/.cache/smollm2-135m}"
OUTD="$ROOT/fak/experiments/agent-live/realistic-workload"
PROF="$OUTD/profile.json"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

if [[ ! -x "$BIN" ]]; then
  mkdir -p "$(dirname "$BIN")"
  log "building fleetserve -> $BIN"
  (cd "$ROOT/fak" && go build -o "$BIN" ./cmd/fleetserve)
fi

log "HEADLINE replay (real prefix, p50 track, C=1,4, turn-cap 12, decode x0.2)"
"$BIN" -dir "$DIR" -quant -workload "$PROF" -track-pct 50 -turn-cap 12 -tune-decode 0.2 \
  -concurrency 1,4 -reps 1 -out "$OUTD/replay-real.json"
log "headline rc=$?"

log "SWEEP tool-call fraction (prefix x0.1 for tractable wall-clock, C=8, turn-cap 12, decode x0.2)"
python "$ROOT/tools/workload_tune_sweep.py" \
  --bin "$BIN" --model-dir "$DIR" --profile "$PROF" \
  --knob toolfrac --grid 0.25,0.5,1,1.5,2,3 \
  --concurrency 8 --track-pct 50 --turn-cap 12 --tune-decode 0.2 --tune-result 1 --tune-prefix 0.1 --reps 1 \
  --out "$OUTD/tune-sweep.json" --md "$OUTD/TUNE-SWEEP.md"
log "sweep rc=$?"
log "DONE"
