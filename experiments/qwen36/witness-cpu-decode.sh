#!/usr/bin/env bash
# witness-cpu-decode.sh — turnkey CPU-decode witness for the Qwen3.6-27B → 10 tok/s
# campaign (epic #4623). ONE command produces the campaign's core witness set on the
# target box (dual EPYC-7742, 8 NUMA), with no serve/curl/slope harness:
#
#   * memory-bandwidth roofline  (STREAM-Triad, dependency-free Go)      -> C3 #4626
#   * header/active-set + decode ceiling  (q4kdiag -plan-only -membw)    -> C3 #4626
#   * first-token int8-vs-f32 agreement   (q4kdiag -decode, id 248068)   -> C1 #4624
#   * decode tok/s A/B over {placement × int8 × workers}                 -> C1/C2 #4625
#
# The int8 Q4_K reducer is env-selected (FAK_KQ_INT8) and worker count is env-selected
# (FAK_WORKERS), so the whole A/B is just this script wrapping q4kdiag in numactl+env.
#
# Usage:   experiments/qwen36/witness-cpu-decode.sh <Qwen3.6-27B-Q4_K_M.gguf> [outdir]
# Env:     DECODE_N=64  WARMUP=3  CELL_TIMEOUT=1800  WORKERS_SWEEP="32 64"
#          FAST=1 (skip the placement matrix; run only the first-token agreement cells)
#
# Nothing here mutates the repo except writing the dated result dir under experiments/.
set -u

# ---- args ------------------------------------------------------------------
GGUF="${1:-}"
if [ -z "$GGUF" ] || [ ! -f "$GGUF" ]; then
  echo "usage: $0 <Qwen3.6-27B-Q4_K_M.gguf> [outdir]" >&2
  echo "  (GGUF path required and must exist)" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "cannot cd to repo root" >&2; exit 1; }

DECODE_N="${DECODE_N:-64}"
WARMUP="${WARMUP:-3}"
CELL_TIMEOUT="${CELL_TIMEOUT:-1800}"
WORKERS_SWEEP="${WORKERS_SWEEP:-32 64}"
FAST="${FAST:-0}"

# Timestamp without Date.now() flakiness concerns — plain date on the box.
TS="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname 2>/dev/null || echo unknown-host)"
OUTDIR="${2:-experiments/qwen36/witness-${HOST}-${TS}}"
mkdir -p "$OUTDIR"
RESULTS="$OUTDIR/result.jsonl"   # one JSON object per stage/cell, appended live
: > "$RESULTS"
LOG="$OUTDIR/run.log"

log(){ echo "[witness $(date -u +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }
emit(){ echo "$1" >> "$RESULTS"; }   # append a JSON line

log "repo=$REPO_ROOT gguf=$GGUF out=$OUTDIR decode_n=$DECODE_N workers_sweep='$WORKERS_SWEEP' fast=$FAST"

# ---- preflight -------------------------------------------------------------
command -v go >/dev/null 2>&1 || { log "FATAL: go not found on PATH"; exit 1; }
GOVER="$(go version 2>/dev/null | awk '{print $3}')"
GITREV="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
NPROC="$(nproc 2>/dev/null || echo 0)"
HAVE_NUMACTL=0; command -v numactl >/dev/null 2>&1 && HAVE_NUMACTL=1
NUMA_NODES=0
if [ "$HAVE_NUMACTL" = 1 ]; then
  NUMA_NODES="$(numactl -H 2>/dev/null | awk '/available:/{print $2}')"
fi
GGUF_BYTES="$(stat -c %s "$GGUF" 2>/dev/null || stat -f %z "$GGUF" 2>/dev/null || echo 0)"
log "host=$HOST nproc=$NPROC numactl=$HAVE_NUMACTL numa_nodes=${NUMA_NODES:-0} go=$GOVER git=$GITREV gguf_bytes=$GGUF_BYTES"

# ---- build the diagnostic once ---------------------------------------------
BIN="$OUTDIR/q4kdiag"
log "building q4kdiag -> $BIN"
if ! go build -o "$BIN" ./cmd/q4kdiag/ 2>>"$LOG"; then
  log "FATAL: q4kdiag build failed (see $LOG)"; exit 1
fi

# ---- stage 1: STREAM-Triad memory bandwidth (dependency-free Go) -----------
# Triad a[i] = b[i] + q*c[i] moves 24 B/elem (read b, read c, write a). Reports
# single-thread and all-core aggregate GB/s. This is the roofline the decode tok/s
# is graded against (C3). Written to a temp .go and `go run`.
TRIAD_SRC="$OUTDIR/streambw.go"
cat > "$TRIAD_SRC" <<'GOEOF'
package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"
)

func triad(a, b, c []float64, q float64, lo, hi int) {
	for i := lo; i < hi; i++ {
		a[i] = b[i] + q*c[i]
	}
}

func run(n, workers, iters int) float64 {
	a := make([]float64, n)
	b := make([]float64, n)
	c := make([]float64, n)
	for i := range b {
		b[i] = float64(i%7) + 1
		c[i] = float64(i%5) + 1
	}
	// warm
	triad(a, b, c, 3.0, 0, n)
	t0 := time.Now()
	for it := 0; it < iters; it++ {
		if workers <= 1 {
			triad(a, b, c, 3.0, 0, n)
			continue
		}
		var wg sync.WaitGroup
		chunk := (n + workers - 1) / workers
		for w := 0; w < workers; w++ {
			lo := w * chunk
			hi := lo + chunk
			if lo >= n {
				break
			}
			if hi > n {
				hi = n
			}
			wg.Add(1)
			go func(lo, hi int) { defer wg.Done(); triad(a, b, c, 3.0, lo, hi) }(lo, hi)
		}
		wg.Wait()
	}
	dt := time.Since(t0).Seconds()
	bytes := float64(24) * float64(n) * float64(iters)
	return bytes / dt / 1e9
}

func main() {
	// ~1 GiB per array (128Mi float64) so we blow past all caches into DRAM.
	n := 128 << 20
	iters := 6
	if len(os.Args) > 1 {
		if v, err := strconv.Atoi(os.Args[1]); err == nil && v > 0 {
			n = v
		}
	}
	single := run(n, 1, iters)
	all := run(n, runtime.NumCPU(), iters)
	fmt.Printf("STREAM_TRIAD single_gbps=%.2f allcore_gbps=%.2f ncpu=%d elems=%d iters=%d\n",
		single, all, runtime.NumCPU(), n, iters)
}
GOEOF
log "stage1: STREAM-Triad bandwidth probe"
BW_LINE="$(timeout 300 go run "$TRIAD_SRC" 2>>"$LOG")"
log "stage1: $BW_LINE"
ALLCORE_GBPS="$(echo "$BW_LINE" | grep -oE 'allcore_gbps=[0-9.]+' | cut -d= -f2)"
[ -z "$ALLCORE_GBPS" ] && ALLCORE_GBPS=0
# interleaved bandwidth under numactl --interleave=all (the aggregate NUMA number)
BW_INTERLEAVE=""
if [ "$HAVE_NUMACTL" = 1 ]; then
  BW_INTERLEAVE="$(timeout 300 numactl --interleave=all go run "$TRIAD_SRC" 2>>"$LOG" | grep -oE 'allcore_gbps=[0-9.]+' | cut -d= -f2)"
  log "stage1: interleave allcore_gbps=$BW_INTERLEAVE"
fi
emit "{\"stage\":\"bandwidth\",\"stream_line\":\"$BW_LINE\",\"allcore_gbps\":$ALLCORE_GBPS,\"interleave_allcore_gbps\":\"${BW_INTERLEAVE:-}\"}"
# The roofline number fed to -membw: prefer interleaved aggregate if measured.
MEMBW="$ALLCORE_GBPS"
[ -n "$BW_INTERLEAVE" ] && MEMBW="$BW_INTERLEAVE"

# ---- stage 2: header plan + decode ceiling (no full load if -plan-only) ----
log "stage2: q4kdiag -plan-only (+ -membw $MEMBW ceiling)"
PLAN_OUT="$OUTDIR/plan-only.txt"
timeout 120 env FAK_Q4K=1 "$BIN" -gguf "$GGUF" -plan-only -membw "$MEMBW" >"$PLAN_OUT" 2>&1
emit "{\"stage\":\"plan_only\",\"membw_gbps\":$MEMBW,\"file\":\"plan-only.txt\"}"
log "stage2: plan-only written to $PLAN_OUT"

# ---- helper: run one decode cell -------------------------------------------
# args: label placement int8 workers
run_cell(){
  local label="$1" placement="$2" int8="$3" workers="$4"
  local nc=()
  case "$placement" in
    default)      nc=() ;;
    interleave)   [ "$HAVE_NUMACTL" = 1 ] && nc=(numactl --interleave=all) ;;
    node0local)   [ "$HAVE_NUMACTL" = 1 ] && nc=(numactl --cpunodebind=0 --membind=0) ;;
  esac
  local wenv=()
  [ "$workers" != "default" ] && wenv=("FAK_WORKERS=$workers")
  log "cell[$label]: placement=$placement int8=$int8 workers=$workers"
  local out
  out="$(timeout "$CELL_TIMEOUT" "${nc[@]}" env FAK_Q4K=1 "FAK_KQ_INT8=$int8" "${wenv[@]}" \
        "$BIN" -gguf "$GGUF" -decode "$DECODE_N" -warmup "$WARMUP" -membw "$MEMBW" 2>>"$LOG")"
  local rc=$?
  local rline
  rline="$(echo "$out" | grep -E '^RESULT ' | head -1)"
  if [ -z "$rline" ]; then
    log "cell[$label]: NO RESULT (rc=$rc; likely load OOM/timeout — see $LOG)"
    emit "{\"stage\":\"decode_cell\",\"label\":\"$label\",\"placement\":\"$placement\",\"int8\":\"$int8\",\"workers\":\"$workers\",\"rc\":$rc,\"ok\":false}"
    return
  fi
  local tokS ftid
  tokS="$(echo "$rline" | grep -oE 'decode_tok_s=[0-9.]+' | cut -d= -f2)"
  ftid="$(echo "$rline" | grep -oE 'first_token_id=[0-9]+' | cut -d= -f2)"
  log "cell[$label]: tok/s=$tokS first_token_id=$ftid"
  emit "{\"stage\":\"decode_cell\",\"label\":\"$label\",\"placement\":\"$placement\",\"int8\":\"$int8\",\"workers\":\"$workers\",\"decode_tok_s\":$tokS,\"first_token_id\":$ftid,\"result\":\"$(echo "$rline" | sed 's/\"/\\\"/g')\",\"ok\":true}"
}

# ---- stage 3: first-token int8-vs-f32 agreement (C1 blocker) ---------------
# Two cells, default placement, minimal decode: confirm BOTH the int8 and f32 Q4_K
# paths put id 248068 first. (This is the correctness rung; tok/s here is incidental.)
log "stage3: first-token agreement (int8=0 vs int8=1, default placement)"
run_cell "ftok-f32"  default 0 default
run_cell "ftok-int8" default 1 default

# ---- stage 4: placement × workers decode sweep (C1/C2) ---------------------
if [ "$FAST" != 1 ]; then
  log "stage4: decode tok/s sweep over {placement × int8 × workers}"
  # Baseline: today's uncapped path (default placement, all workers).
  run_cell "int8-default-allworkers" default 1 default
  # Placement lever: spread weights across nodes.
  run_cell "int8-interleave-allworkers" interleave 1 default
  # Single-node local (the ~40GB/s / ~2.6 tok/s node-0 ceiling probe).
  run_cell "int8-node0local-w32" node0local 1 32
  # Worker sweep under interleave (find the barrier knee; predicted best).
  for w in $WORKERS_SWEEP; do
    run_cell "int8-interleave-w$w" interleave 1 "$w"
  done
  # f32 reference at sane workers (contrast the int8 speedup at fixed placement).
  run_cell "f32-interleave-w32" interleave 0 32
else
  log "stage4: SKIPPED (FAST=1)"
fi

# ---- assemble manifest.json ------------------------------------------------
MANIFEST="$OUTDIR/manifest.json"
cat > "$MANIFEST" <<JSONEOF
{
  "\$schema": "benchmark/run-manifest.v1",
  "run_id": "${HOST}-qwen36-cpu-decode-witness-${TS}",
  "campaign": "epic-4623-qwen36-cpu-decode-10toks",
  "machine_id": "${HOST}",
  "timestamp": "${TS}",
  "git": { "rev": "${GITREV}", "branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)" },
  "harness": { "name": "q4kdiag-decode-witness", "version": "1", "tool": "cmd/q4kdiag -decode" },
  "box": { "nproc": ${NPROC:-0}, "numa_nodes": ${NUMA_NODES:-0}, "numactl": ${HAVE_NUMACTL}, "go": "${GOVER}" },
  "gguf": { "path": "${GGUF}", "bytes": ${GGUF_BYTES:-0} },
  "bandwidth": { "allcore_gbps": ${ALLCORE_GBPS:-0}, "interleave_allcore_gbps": "${BW_INTERLEAVE:-}", "membw_fed_to_ceiling": ${MEMBW:-0} },
  "method": "In-process greedy decode via cmd/q4kdiag -decode N (Prefill the 22-token 'Say OK.' oracle, then time N Step() calls after warmup). decode_tok_s = N / timed_wall. int8 Q4_K reducer selected by FAK_KQ_INT8; worker count by FAK_WORKERS; NUMA placement by numactl wrapper. first_token_id must be 248068 (oracle argmax) on every cell.",
  "config": { "decode_n": ${DECODE_N}, "warmup": ${WARMUP}, "cell_timeout_s": ${CELL_TIMEOUT}, "workers_sweep": "${WORKERS_SWEEP}", "fast": ${FAST} },
  "results_file": "result.jsonl",
  "tags": ["cpu-only","no-gpu","qwen3.6-27b","q4_k","int8-q4k","numa","decode-witness","epic-4623"]
}
JSONEOF

# ---- summary ---------------------------------------------------------------
log "DONE. manifest=$MANIFEST results=$RESULTS"
echo "" >&2
echo "==== WITNESS SUMMARY ($OUTDIR) ====" >&2
if command -v jq >/dev/null 2>&1; then
  echo "-- first-token agreement (want first_token_id=248068 on both) --" >&2
  jq -r 'select(.stage=="decode_cell" and (.label|startswith("ftok"))) | "  \(.label): first_token_id=\(.first_token_id // "n/a") tok/s=\(.decode_tok_s // "n/a")"' "$RESULTS" 2>/dev/null >&2
  echo "-- decode tok/s sweep --" >&2
  jq -r 'select(.stage=="decode_cell" and (.label|startswith("ftok")|not)) | "  \(.label): tok/s=\(.decode_tok_s // "FAIL") (placement=\(.placement) int8=\(.int8) workers=\(.workers))"' "$RESULTS" 2>/dev/null >&2
else
  grep -E 'decode_cell' "$RESULTS" >&2
fi
echo "===================================" >&2
