#!/usr/bin/env bash
# fak_node_bench.sh -- run fak's self-contained kernel benchmarks on THIS machine,
# as a benchmark NODE, tagged by host/OS/arch, for cross-hardware comparison.
#
# The point: fak's in-kernel forward pass + GEMV/GEMM/quant kernels are pure Go, so
# their performance is a property of the HARDWARE they run on. Running the same
# benchmarks natively on multiple private machines yields a real cross-ISA
# comparison. These are
# COMPUTE nodes, not model-serving clients -- nothing here listens on a port.
#
# Nodes already have the repo, so the node workflow is just:
#     git pull && bash tools/fak_node_bench.sh            # (--pull does the pull for you)
#
# Results -> fak/experiments/fleet-nodes/<host>/  (per-node subdir; no collisions).
# Only `go build` is used (never `go test`), so the WSL/Application-Control test
# limitation in CLAUDE.md does not apply. A bench that the OS blocks (e.g. desktop
# WDAC) is reported and skipped, never fatal.
set -uo pipefail

PULL=0; SHORT=0; HOST_OVERRIDE=""; OUT_OVERRIDE=""
for a in "$@"; do case "$a" in
  --pull) PULL=1;;
  --short) SHORT=1;;
  --host=*) HOST_OVERRIDE="${a#--host=}";;
  --out=*) OUT_OVERRIDE="${a#--out=}";;
esac; done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $PULL == 1 ]]; then
  echo "[node] git pull --ff-only ..."
  git pull --ff-only || echo "[node] pull failed/skipped; continuing with the local tree"
fi

# --- node identity (hard facts, recorded with the results) ---
HOST_SRC="${HOST_OVERRIDE:-${FAK_NODE_HOST:-$(hostname 2>/dev/null || echo node)}}"
HOST="$(printf '%s' "$HOST_SRC" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9._-' '-' | sed 's/-*$//')"
[[ -z "$HOST" ]] && HOST="node"
OS="$(uname -s 2>/dev/null || echo unknown)"; ARCH="$(uname -m 2>/dev/null || echo unknown)"
case "$OS" in
  MINGW*|MSYS*|CYGWIN*) EXT=".exe"; OSN="windows"; CPU="${PROCESSOR_IDENTIFIER:-?}"; CORES="${NUMBER_OF_PROCESSORS:-?}";;
  Darwin)               EXT="";     OSN="darwin";  CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo ?)"; CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo ?)";;
  *)                    EXT="";     OSN="linux";   CPU="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ //' || echo ?)"; CORES="$(nproc 2>/dev/null || echo ?)";;
esac
# Report from inside the module so GOTOOLCHAIN=auto resolves the toolchain that
# actually builds the benches (go.mod requires go 1.26), not the base go on PATH.
GOV="$( (cd "$(dirname "${BASH_SOURCE[0]}")/../fak" 2>/dev/null && go version) 2>/dev/null | awk '{print $3" "$4}' || echo 'go:NOT-FOUND')"
GITREV="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"

OUT="${OUT_OVERRIDE:-${FAK_NODE_OUT:-$ROOT/fak/experiments/fleet-nodes/$HOST}}"
case "$OUT" in
  /*|[A-Za-z]:*) ;;
  *) OUT="$ROOT/$OUT" ;;
esac
mkdir -p "$OUT"
BIN="$ROOT/fak/.benchbin"; mkdir -p "$BIN"

cat > "$OUT/node-info.json" <<EOF
{
  "host": "$HOST", "os": "$OSN", "arch": "$ARCH",
  "cpu": "$CPU", "cores": "$CORES", "go": "$GOV", "git": "$GITREV"
}
EOF
echo "[node] $HOST  $OSN/$ARCH  cores=$CORES  $GOV  git=$GITREV"
echo "[node] cpu: $CPU"
if [[ "$GOV" == go:NOT-FOUND ]]; then
  echo "[node] ERROR: Go toolchain not found on PATH -- install Go (the module needs go 1.26) and re-run." >&2
  exit 1
fi

# --- build the self-contained benches (also a 'does fak compile here' check) ---
echo "[node] building (go build; never go test) ..."
BENCHES=(q8kernel modelprof modelbench batchbench fleetbench)
( cd "$ROOT/fak" && go build -o "$BIN/fak$EXT" ./cmd/fak ) && echo "  built fak (kernel compiles on $OSN/$ARCH)" || echo "  WARN: fak build failed"
for b in "${BENCHES[@]}"; do
  ( cd "$ROOT/fak" && go build -o "$BIN/$b$EXT" "./cmd/$b" ) && echo "  built $b" || echo "  WARN: build $b failed"
done

# --- run them (per-bench failure is reported, never fatal) ---
run() { # run <name> <logfile> <cmd...>
  local name="$1" log="$2"; shift 2
  if [[ ! -x "$1" && ! -f "$1" ]]; then echo "  $name SKIP (not built)"; return; fi
  # Run from the module dir: modelprof/batchbench resolve the SmolLM2-135M export
  # at the CWD-relative path internal/model/.cache/smollm2-135m, which only exists
  # under $ROOT/fak. The binary path ($1) and $log are absolute, so the cd is safe.
  ( cd "$ROOT/fak" && "$@" ) > "$log" 2>&1; local rc=$?
  if [[ $rc -eq 0 ]]; then echo "  $name ok -> $(basename "$log")"; return; fi
  local hint
  # Diagnose from the log, not a one-size-fits-all guess: a missing model export
  # (modelprof/batchbench load SmolLM2-135M) reads very differently from the
  # Windows/WDAC case where the OS refuses to exec the unsigned binary at all.
  if grep -qiE 'smollm2-135m|\.cache/.*config\.json|^load: config' "$log" 2>/dev/null; then
    hint="needs the SmolLM2-135M export: run internal/model/export_oracle.py into internal/model/.cache/smollm2-135m"
  else
    hint="OS may block unsigned binaries (e.g. Windows WDAC)"
  fi
  echo "  $name FAIL (exit $rc) -- see $(basename "$log") ($hint; not fatal)"
}
echo "[node] running benchmarks ..."
WORKLOAD="$ROOT/fak/experiments/agent-live/production-workload.json"
PREFILL_REPS="${FAK_PREFILL_REPS:-3}"
DECODE_REPS="${FAK_DECODE_REPS:-3}"
DECODE_STEPS="${FAK_DECODE_STEPS:-32}"
WORKLOAD_PREFILL_CAP="${FAK_WORKLOAD_PREFILL_CAP:-0}"
WORKLOAD_PROMPT_CAP="${FAK_WORKLOAD_PROMPT_CAP:-0}"
BATCH_REPS="${FAK_BATCH_REPS:-3}"
BATCHES="${FAK_BATCHES:-1,4,8,16,32,64,128,256}"
BATCH_WORKLOAD="${FAK_BATCH_WORKLOAD:-0}"
if [[ $SHORT == 1 ]]; then
  PREFILL_REPS="${FAK_PREFILL_REPS:-1}"
  DECODE_REPS="${FAK_DECODE_REPS:-1}"
  DECODE_STEPS="${FAK_DECODE_STEPS:-4}"
  WORKLOAD_PREFILL_CAP="${FAK_WORKLOAD_PREFILL_CAP:-128}"
  WORKLOAD_PROMPT_CAP="${FAK_WORKLOAD_PROMPT_CAP:-16}"
  BATCH_REPS="${FAK_BATCH_REPS:-1}"
  BATCHES="${FAK_BATCHES:-1,4,8}"
  BATCH_WORKLOAD="${FAK_BATCH_WORKLOAD:-1}"
fi
run q8kernel   "$OUT/q8kernel.txt"   "$BIN/q8kernel$EXT"
run modelprof  "$OUT/modelprof.txt"  "$BIN/modelprof$EXT" -out "$OUT/modelprof.json"
if [[ -f "$WORKLOAD" ]]; then
  run modelbench-q8 "$OUT/modelbench-q8.txt" "$BIN/modelbench$EXT" -quant \
    -prefill-reps "$PREFILL_REPS" -decode-reps "$DECODE_REPS" -decode-steps "$DECODE_STEPS" \
    -workload "$WORKLOAD" -workload-prefill-cap "$WORKLOAD_PREFILL_CAP" \
    -out "$OUT/modelbench-q8.json"
else
  run modelbench-q8 "$OUT/modelbench-q8.txt" "$BIN/modelbench$EXT" -quant \
    -prefill-reps "$PREFILL_REPS" -decode-reps "$DECODE_REPS" -decode-steps "$DECODE_STEPS" \
    -out "$OUT/modelbench-q8.json"
fi
if [[ $SHORT == 0 ]]; then
  if [[ -f "$WORKLOAD" && "$BATCH_WORKLOAD" == "1" ]]; then
    run batchbench "$OUT/batchbench.txt" "$BIN/batchbench$EXT" -quant \
      -reps "$BATCH_REPS" -decode-steps "$DECODE_STEPS" -batches "$BATCHES" \
      -workload "$WORKLOAD" -workload-prompt-cap "$WORKLOAD_PROMPT_CAP" \
      -out "$OUT/batchbench-q8.json"
  else
    run batchbench "$OUT/batchbench.txt" "$BIN/batchbench$EXT" -quant \
      -reps "$BATCH_REPS" -decode-steps "$DECODE_STEPS" -batches "$BATCHES" \
      -out "$OUT/batchbench-q8.json"
  fi
else
  if [[ -f "$WORKLOAD" && "$BATCH_WORKLOAD" == "1" ]]; then
    run batchbench-short "$OUT/batchbench.txt" "$BIN/batchbench$EXT" -quant \
      -reps "$BATCH_REPS" -decode-steps "$DECODE_STEPS" -batches "$BATCHES" \
      -workload "$WORKLOAD" -workload-prompt-cap "$WORKLOAD_PROMPT_CAP" \
      -out "$OUT/batchbench-q8.json"
  else
    run batchbench-short "$OUT/batchbench.txt" "$BIN/batchbench$EXT" -quant \
      -reps "$BATCH_REPS" -decode-steps "$DECODE_STEPS" -batches "$BATCHES" \
      -out "$OUT/batchbench-q8.json"
  fi
fi
FG_ARGS=(--grid log --turn-max 16 --agent-max 16 --trials 8 --out "$OUT/fleetbench.json" --csv "$OUT/fleetbench.csv")
run fleetbench "$OUT/fleetbench.txt" "$BIN/fleetbench$EXT" "${FG_ARGS[@]}"

cat > "$OUT/production-readiness-manifest.json" <<EOF
{
  "schema": "fak.production-readiness-node.v1",
  "host": "$HOST",
  "os": "$OSN",
  "arch": "$ARCH",
  "go": "$GOV",
  "git": "$GITREV",
  "workload": "fak/experiments/agent-live/production-workload.json",
  "workload_prefill_cap": "$WORKLOAD_PREFILL_CAP",
  "workload_prompt_cap": "$WORKLOAD_PROMPT_CAP",
  "batch_workload": "$BATCH_WORKLOAD",
  "decode_steps": "$DECODE_STEPS",
  "artifacts": [
    "node-info.json",
    "q8kernel.txt",
    "modelprof.json",
    "modelbench-q8.json",
    "batchbench-q8.json",
    "fleetbench.json",
    "fleetbench.csv"
  ]
}
EOF

echo "[node] done -> $OUT"
echo "[node] ---- q8kernel (cross-ISA GEMV) ----"; sed -n '1,6p' "$OUT/q8kernel.txt" 2>/dev/null || true
echo "[node] collect: copy fak/experiments/fleet-nodes/$HOST/ back to the driver,"
echo "       or commit it, then run python tools/fak_node_compare.py on the driver."
