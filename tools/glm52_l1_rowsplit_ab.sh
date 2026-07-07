#!/usr/bin/env bash
# glm52_l1_rowsplit_ab.sh - L1 (#3075): the one-command A/B that measures the dominant
# single-stream lever - llama.cpp split-mode=layer (ONE GPU per token, the other 7 idle) vs
# split-mode=row (every GPU's bandwidth on every token). It stands the resident serve up twice
# (once per split mode, a cold load each - the unavoidable cost of the A/B), benches each through
# the L8 harness (tools/glm52_bench_lever.sh), and prints the decode-tok/s delta plus the
# gpus-busy delta that IS the lever's thesis.
#
# WHY TWO COLD LOADS: split-mode is a launch flag, so row vs layer needs a fresh llama-server each.
# Budget ~15-25 min total on the resident node. Both artifacts are kept for the #3075 witness.
#
# RUN ON THE RESIDENT GPU HOST (server 3, 80GB cards - GLM-5.2 UD-Q4_K_M must fit resident).
# It refuses to run where the model cannot be resident (that is the layer-split degenerate case,
# not this lever). Delegates the serve to glm52_mgpu_serve.sh and the measurement to
# glm52_bench_lever.sh; asserts nothing it did not measure.
#
# Usage:
#   DEVICES=0,1,2,3,4,5,6,7 bash tools/glm52_l1_rowsplit_ab.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-8002}"
ALIAS="${ALIAS:-glm-5.2}"
RUN="${RUN:-/tmp/glm52_mgpu}"
OUTDIR="${OUTDIR:-/projects/glm52-q4/L1-ROWSPLIT}"
ITERS="${ITERS:-5}"
mkdir -p "$OUTDIR" 2>/dev/null || true
PH(){ echo "$(date -u +%H:%M:%S) L1AB $*"; }

stop_serve(){
  # kill any llama-server on our port so the next split-mode starts clean.
  pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
  for i in $(seq 1 20); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
}

run_one(){
  local mode="$1" art="$OUTDIR/bench-$1"
  PH "SERVE split=$mode (cold resident load)"
  stop_serve
  SPLIT_MODE="$mode" DEVICES="$DEVICES" PORT="$PORT" ALIAS="$ALIAS" RUN="$RUN" \
    bash "$HERE/glm52_mgpu_serve.sh"
  # glm52_mgpu_serve.sh exits 0 on GLM52_MGPU_READY (or PORT_BUSY); confirm the endpoint really serves.
  if ! curl -sf -m 8 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$ALIAS"; then
    PH "SERVE_FAILED split=$mode (phase=$(cat "$RUN/PHASE" 2>/dev/null)) - aborting A/B"; return 20
  fi
  PH "BENCH split=$mode"
  LEVER="L1-$mode" SPLIT_MODE="$mode" PORT="$PORT" ALIAS="$ALIAS" DEVICES="$DEVICES" \
    ITERS="$ITERS" OUT="$art" bash "$HERE/glm52_bench_lever.sh"
  echo "$art.json"
}

PH "START devices=$DEVICES port=$PORT iters=$ITERS"
LAYER_JSON="$(run_one layer | tail -1)" || { PH "LAYER_LEG_FAILED"; exit 21; }
ROW_JSON="$(run_one row | tail -1)"       || { PH "ROW_LEG_FAILED"; exit 22; }
stop_serve

# fold the two artifacts into the A/B verdict.
python3 - "$LAYER_JSON" "$ROW_JSON" "$OUTDIR/l1-ab-verdict.json" <<'PYEOF'
import json,sys
layer=json.load(open(sys.argv[1])); row=json.load(open(sys.argv[2]))
ld=layer.get("decode_toks_median",0); rd=row.get("decode_toks_median",0)
verdict={
 "schema":"fak.glm52-l1-rowsplit-ab.v1",
 "layer":{"decode_toks":ld,"gpus_busy":layer.get("gpus_busy_ge10pct"),"gpus_sampled":layer.get("gpus_sampled")},
 "row":{"decode_toks":rd,"gpus_busy":row.get("gpus_busy_ge10pct"),"gpus_sampled":row.get("gpus_sampled")},
 "decode_speedup_x": round(rd/ld,3) if ld else None,
 "decode_delta_toks": round(rd-ld,3),
}
json.dump(verdict,open(sys.argv[3],"w"),indent=2)
print(f"L1AB VERDICT layer={ld} tok/s (busy {verdict['layer']['gpus_busy']}/{verdict['layer']['gpus_sampled']}) "
      f"-> row={rd} tok/s (busy {verdict['row']['gpus_busy']}/{verdict['row']['gpus_sampled']}) "
      f"speedup={verdict['decode_speedup_x']}x  verdict={sys.argv[3]}")
PYEOF
PH "DONE verdict=$OUTDIR/l1-ab-verdict.json"
