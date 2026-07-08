#!/usr/bin/env bash
# glm52_l4_fa_cudagraph_ab.sh - L4 (#3076): the one-command 2x2 that measures the flash-attention +
# CUDA-graph decode lever (~1.2-1.8x, the launch/attention-overhead lever). It stands the resident
# serve up four ways - fa{off,on} x graph{off,on} - benches each through the L8 harness
# (tools/glm52_bench_lever.sh), captures the greedy first-token text per cell for the parity gate,
# and folds the four artifacts into a verdict JSON: best decode-tok/s cell + its multiplier over the
# baseline, plus the cross-cell first-token PARITY check (a flag that changes the greedy first token
# is a correctness regression, not a speedup - it is failed, not averaged away).
#
# THE 2x2 (each cell is a fresh cold llama-server - fa is a launch flag, graphs an env, so both need
# a fresh serve; the L1 A/B has the same shape and cost):
#   C1 fa=off graph=off   (re-pins the WITNESSED baseline's true cell)
#   C2 fa=on  graph=off
#   C3 fa=on  graph=on
#   C4 fa=off graph=on
# Budget ~30-45 min total on the resident node (four cold loads). All artifacts are kept for #3076.
#
# ORDER NOTE (#3076 is ordered after #3075): llama.cpp SKIPS CUDA-graph capture under -sm layer
# multi-device split, so the graph axis is expected to read ~inert until L1 switches to a row split.
# Run this AFTER L1 with SPLIT_MODE=row for the lever's real test; a near-1.0x graph column under the
# default layer split is an informative negative, not a failure.
#
# RUN ON THE RESIDENT GPU HOST (server 3, 80GB cards - GLM-5.2 UD-Q4_K_M must fit resident).
# Delegates the serve to glm52_mgpu_serve.sh and the measurement to glm52_bench_lever.sh; asserts
# nothing it did not measure and never fakes a pass (a cell whose serve/endpoint fails aborts the A/B).
#
# Usage:
#   DEVICES=0,1,2,3,4,5,6,7 bash tools/glm52_l4_fa_cudagraph_ab.sh              # default layer split
#   SPLIT_MODE=row DEVICES=0,1,2,3,4,5,6,7 bash tools/glm52_l4_fa_cudagraph_ab.sh  # post-L1 real test
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-8002}"
ALIAS="${ALIAS:-glm-5.2}"
RUN="${RUN:-/tmp/glm52_mgpu}"
SPLIT_MODE="${SPLIT_MODE:-layer}"            # passthrough to the serve; set row for the post-L1 test
OUTDIR="${OUTDIR:-/projects/glm52-q4/L4-FA-CUDAGRAPH}"
ITERS="${ITERS:-5}"
# Fixed temperature-0 prompt for the greedy first-token parity gate; the first generated token must
# match across all four cells (a flag/graph change that alters it is a correctness regression).
PARITY_PROMPT="${PARITY_PROMPT:-The capital of France is}"
mkdir -p "$OUTDIR" 2>/dev/null || true
PH(){ echo "$(date -u +%H:%M:%S) L4AB $*"; }

stop_serve(){
  # kill any llama-server on our port so the next cell starts from a clean cold load.
  pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
  for i in $(seq 1 20); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
}

# greedy first token (temperature 0, n_predict 1) -> the parity witness text for the current serve.
first_token(){
  curl -s -m 60 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PARITY_PROMPT\",\"n_predict\":1,\"temperature\":0,\"cache_prompt\":false}" \
    | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("content",""))
except Exception: print("")' 2>/dev/null
}

# run_one <cell> <fa on|off> <graph on|off> -> prints the bench artifact path as its LAST line.
run_one(){
  local cell="$1" fa="$2" graph="$3" art="$OUTDIR/bench-$cell"
  PH "SERVE cell=$cell fa=$fa graph=$graph split=$SPLIT_MODE (cold resident load)"
  stop_serve
  FLASH_ATTN="$fa" CUDA_GRAPHS="$graph" SPLIT_MODE="$SPLIT_MODE" \
    DEVICES="$DEVICES" PORT="$PORT" ALIAS="$ALIAS" RUN="$RUN" \
    bash "$HERE/glm52_mgpu_serve.sh"
  # glm52_mgpu_serve.sh exits 0 on GLM52_MGPU_READY (or PORT_BUSY); confirm the endpoint really serves.
  if ! curl -sf -m 8 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$ALIAS"; then
    PH "SERVE_FAILED cell=$cell (phase=$(cat "$RUN/PHASE" 2>/dev/null)) - aborting A/B"; return 20
  fi
  # capture the greedy first token BEFORE the timed decode; stash it beside the perf artifact.
  first_token > "$art.firsttok.txt"
  PH "BENCH cell=$cell first_token=$(cat "$art.firsttok.txt")"
  LEVER="L4-$cell-fa$fa-graph$graph" SPLIT_MODE="$SPLIT_MODE" PORT="$PORT" ALIAS="$ALIAS" \
    DEVICES="$DEVICES" ITERS="$ITERS" OUT="$art" bash "$HERE/glm52_bench_lever.sh"
  echo "$art.json"
}

PH "START devices=$DEVICES port=$PORT split=$SPLIT_MODE iters=$ITERS outdir=$OUTDIR"
run_one C1 off off >/dev/null || { PH "C1_FAILED"; exit 21; }
run_one C2 on  off >/dev/null || { PH "C2_FAILED"; exit 22; }
run_one C3 on  on  >/dev/null || { PH "C3_FAILED"; exit 23; }
run_one C4 off on  >/dev/null || { PH "C4_FAILED"; exit 24; }
stop_serve

# fold the four artifacts + first-token witnesses into the 2x2 verdict.
python3 - "$OUTDIR" "$SPLIT_MODE" "$OUTDIR/l4-ab-verdict.json" <<'PYEOF'
import json,sys,os
outdir,split_mode,outp = sys.argv[1:4]
cells=[("C1","off","off"),("C2","on","off"),("C3","on","on"),("C4","off","on")]
rows=[]
for name,fa,graph in cells:
    bj=os.path.join(outdir,f"bench-{name}.json")
    ft=os.path.join(outdir,f"bench-{name}.firsttok.txt")
    d=json.load(open(bj)) if os.path.exists(bj) else {}
    tok=open(ft).read() if os.path.exists(ft) else None
    rows.append({"cell":name,"fa":fa,"graph":graph,
      "decode_toks":d.get("decode_toks_median"),
      "prompt_toks":d.get("prompt_toks_median"),
      "gpus_busy":d.get("gpus_busy_ge10pct"),"gpus_sampled":d.get("gpus_sampled"),
      "first_token_text":tok})
base=next((r for r in rows if r["cell"]=="C1"),None)
bd=base.get("decode_toks") if base else None
measured=[r for r in rows if isinstance(r["decode_toks"],(int,float)) and r["decode_toks"]]
best=max(measured,key=lambda r:r["decode_toks"]) if measured else None
# parity: every cell's greedy first token must match the C1 baseline (temp 0, fixed prompt).
seen=[r["first_token_text"] for r in rows if r["first_token_text"] is not None]
mism=[r["cell"] for r in rows if base and r["first_token_text"] is not None
      and r["first_token_text"]!=base.get("first_token_text")]
verdict={
  "schema":"fak.glm52-l4-fa-cudagraph-ab.v1",
  "split_mode":split_mode,
  "cells":rows,
  "baseline_cell":"C1",
  "best_cell":best["cell"] if best else None,
  "best_decode_toks":best["decode_toks"] if best else None,
  "speedup_over_baseline_x": round(best["decode_toks"]/bd,3) if best and bd else None,
  "parity":{
    "all_cells_match": len(set(seen))<=1 if seen else None,
    "first_token_text": base.get("first_token_text") if base else None,
    "distinct_tokens": sorted(set(seen)),
    "mismatch_cells": mism,
  },
}
json.dump(verdict,open(outp,"w"),indent=2)
par=verdict["parity"]["all_cells_match"]
print(f"L4AB VERDICT split={split_mode} baseline(C1)={bd} tok/s -> best={verdict['best_cell']}="
      f"{verdict['best_decode_toks']} tok/s  speedup={verdict['speedup_over_baseline_x']}x  "
      f"parity_ok={par} (mismatch={mism})  verdict={outp}")
if par is False:
    print("L4AB PARITY_FAIL: a flag/graph cell changed the greedy first token - a correctness "
          "regression, not a speedup. Fail the offending cell(s); do not report their tok/s as a win.")
PYEOF
PH "DONE verdict=$OUTDIR/l4-ab-verdict.json"
