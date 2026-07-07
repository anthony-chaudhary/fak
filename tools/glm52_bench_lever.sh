#!/usr/bin/env bash
# glm52_bench_lever.sh - L8 (#3082): the ONE-COMMAND decode benchmark every GLM-5.2 lever
# records its artifact through. It turns the serialized resident lane's per-lever tax from
# "hand-run a serve + eyeball a curl" into a single reproducible measurement + artifact, so an
# operator (or a fak-guarded agent) can measure any lever (L1 split-mode, L4 FA/graphs, L5 quant,
# L3 spec-decode) in one call and diff the artifacts.
#
# WHAT IT MEASURES (asserts NOTHING it did not measure):
#   Given a LIVE llama.cpp serve on $PORT serving $ALIAS, it
#     1. WARMS the serve with one throwaway generation, so the ~500 s one-time backend warmup
#        (cold first-turn tax, #3051) is paid BEFORE the timed turns -> steady-state numbers;
#     2. runs $ITERS decode-timed /completion calls and records llama.cpp's OWN timings
#        (predicted_per_second = decode tok/s, prompt_per_second = prefill tok/s) plus an
#        independent wall-clock tok/s as a cross-check;
#     3. samples per-GPU sm-utilization via `nvidia-smi dmon -s u` DURING one sustained decode
#        -- the L1 (#3075) proof that split-mode=row lights ALL GPUs where split-mode=layer lights
#        one (idle-GPU count is the lever's whole thesis).
#   It emits a JSON artifact tagged with $LEVER + the serve context, and prints the one-line fold.
#
# It does NOT stand the serve up (that is glm52_mgpu_serve.sh, whose SPLIT_MODE toggle is the L1
# knob). Benchmark whatever is already serving on $PORT; if nothing is, it aborts (never fakes a
# pass). RUN ON THE GPU HOST after GLM52_MGPU_READY.
#
# Usage (server 3, after the serve is READY on :8002):
#   LEVER=L1-row PORT=8002 DEVICES=0,1,2,3,4,5,6,7 bash tools/glm52_bench_lever.sh
#   # then re-run with the layer-split serve and diff the two artifacts.
set -uo pipefail

LEVER="${LEVER:-unnamed}"                 # artifact tag, e.g. L1-layer / L1-row / L4-fa
SPLIT_MODE_CTX="${SPLIT_MODE:-unknown}"   # recorded serve context (set SPLIT_MODE to match the serve)
PORT="${PORT:-8002}"
ALIAS="${ALIAS:-glm-5.2}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"     # GPUs to sample util on (should match the serve's set)
ITERS="${ITERS:-5}"                       # timed decode calls (median is reported)
NPREDICT="${NPREDICT:-128}"               # tokens per timed call
UTIL_NPREDICT="${UTIL_NPREDICT:-512}"     # tokens for the sustained-decode util capture
PROMPT="${PROMPT:-Count slowly from one to twenty, one number per line.}"
OUT="${OUT:-/projects/glm52-q4/BENCH}"    # artifact path prefix (.json/.log)
BASE="http://127.0.0.1:${PORT}"
mkdir -p "$(dirname "$OUT")" 2>/dev/null || true
LOG="$OUT.log"
PH(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }

command -v python3 >/dev/null 2>&1 || { PH "NO_PYTHON3 (needed to parse timings)"; exit 20; }

# 0. never fake a pass: the endpoint must actually serve $ALIAS.
if ! curl -sf -m 8 "$BASE/v1/models" 2>/dev/null | grep -q "$ALIAS"; then
  PH "BENCH_ABORT endpoint not serving $ALIAS at $BASE (stand the serve up first)"; exit 10
fi
PH "BENCH_START lever=$LEVER endpoint=$BASE iters=$ITERS n_predict=$NPREDICT"

# one /completion call -> "<decode_toks> <prompt_toks> <predicted_n> <wall_s>" (all numeric).
one_call(){
  local np="$1" t0 t1 resp
  t0=$(date +%s.%N)
  resp=$(curl -s -m 180 "$BASE/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$np,\"cache_prompt\":false,\"temperature\":0}")
  t1=$(date +%s.%N)
  # python emits the three llama.cpp timings; awk appends the independent wall-clock seconds.
  printf '%s' "$resp" | python3 -c '
import sys,json
try:
    t=json.load(sys.stdin).get("timings",{})
    print(t.get("predicted_per_second",0), t.get("prompt_per_second",0), t.get("predicted_n",0))
except Exception:
    print("0 0 0")
' 2>/dev/null | awk -v t0="$t0" -v t1="$t1" '{wall=t1-t0; print $1, $2, $3, wall}'
}

# 1. WARM (pay the one-time cold tax before the timed turns).
PH "WARM (paying the one-time backend warmup so timed turns are steady-state)"
one_call 8 >/dev/null 2>&1 || true

# 2. timed decode calls -> collect decode tok/s.
PH "TIMED_DECODE x$ITERS"
DEC_LIST=""; PRE_LIST=""; WALL_LIST=""
for i in $(seq 1 "$ITERS"); do
  read -r dec pre pn wall < <(one_call "$NPREDICT")
  DEC_LIST="$DEC_LIST $dec"; PRE_LIST="$PRE_LIST $pre"; WALL_LIST="$WALL_LIST $wall"
  PH "  iter=$i decode_toks=$dec prompt_toks=$pre wall_s=$wall"
done

# 3. per-GPU util during ONE sustained decode (the L1 idle-GPU proof).
PH "UTIL_CAPTURE dmon during a $UTIL_NPREDICT-token decode on GPUs $DEVICES"
DMON="$OUT.dmon.txt"; : > "$DMON"
( nvidia-smi dmon -s u -d 1 -c 30 -i "$DEVICES" > "$DMON" 2>/dev/null ) &
DMON_PID=$!
one_call "$UTIL_NPREDICT" >/dev/null 2>&1 || true
wait "$DMON_PID" 2>/dev/null || true

# 4. fold the samples + emit the artifact (median decode tok/s; per-GPU peak/mean sm%).
python3 - "$LEVER" "$SPLIT_MODE_CTX" "$PORT" "$DEVICES" "$NPREDICT" "$OUT.json" "$DMON" <<PYEOF
import json,sys,statistics
lever,split_ctx,port,devices,npred,outp,dmon = sys.argv[1:8]
dec=[float(x) for x in """$DEC_LIST""".split() if x]
pre=[float(x) for x in """$PRE_LIST""".split() if x]
wall=[float(x) for x in """$WALL_LIST""".split() if x]
def med(xs): return round(statistics.median(xs),3) if xs else 0.0
# parse dmon: columns "# gpu ... sm mem ..."; grab per-gpu sm% samples.
per={}
try:
    for ln in open(dmon):
        ln=ln.strip()
        if not ln or ln.startswith("#"): continue
        f=ln.split()
        if len(f)<4: continue
        try: g=int(f[0]); sm=int(f[1])
        except ValueError: continue
        per.setdefault(g,[]).append(sm)
except FileNotFoundError:
    pass
gpu_sm={str(g):{"peak":max(v),"mean":round(sum(v)/len(v),1)} for g,v in sorted(per.items()) if v}
busy=sum(1 for g in gpu_sm.values() if g["peak"]>=10)
wall_toks=[round(npred_i/w,3) for npred_i,w in [(int(npred),w) for w in wall] if w>0]
art={
  "schema":"fak.glm52-bench-lever.v1",
  "lever":lever,"serve_split_mode":split_ctx,"port":int(port),"devices":devices,
  "n_predict":int(npred),"iters":len(dec),
  "decode_toks_median":med(dec),"decode_toks_samples":dec,
  "prompt_toks_median":med(pre),
  "wall_toks_median":med(wall_toks),
  "gpu_sm_util":gpu_sm,"gpus_busy_ge10pct":busy,"gpus_sampled":len(gpu_sm),
}
json.dump(art,open(outp,"w"),indent=2)
print(f"BENCH_FOLD lever={lever} decode_toks_median={art['decode_toks_median']} "
      f"gpus_busy={busy}/{art['gpus_sampled']} artifact={outp}")
PYEOF
PH "BENCH_DONE artifact=$OUT.json"
