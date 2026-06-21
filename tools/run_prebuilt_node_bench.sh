#!/usr/bin/env bash
# Run a prebuilt fak benchmark payload on a machine with no Go checkout.
#
# Payload layout expected:
#   bin/{q8kernel,modelprof,modelbench,batchbench,fleetbench}
#   fak/internal/model/.cache/smollm2-135m/...
#   fak/experiments/agent-live/production-workload.json
set -uo pipefail

HOST_OVERRIDE=""
OUT_OVERRIDE=""
SHORT=0
SEND_TO="${FAK_NODE_SEND_TO:-your-bench-driver}"

for a in "$@"; do
  case "$a" in
    --host=*) HOST_OVERRIDE="${a#--host=}" ;;
    --out=*) OUT_OVERRIDE="${a#--out=}" ;;
    --short) SHORT=1 ;;
    --send-to=*) SEND_TO="${a#--send-to=}" ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $a" >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_SRC="${HOST_OVERRIDE:-${FAK_NODE_HOST:-$(hostname 2>/dev/null || echo node)}}"
HOST="$(printf '%s' "$HOST_SRC" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9._-' '-' | sed 's/-*$//')"
[[ -z "$HOST" ]] && HOST="node"

OS="$(uname -s 2>/dev/null || echo unknown)"
ARCH="$(uname -m 2>/dev/null || echo unknown)"
case "$OS" in
  Darwin) OSN="darwin"; CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model 2>/dev/null || echo ?)"; CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo ?)" ;;
  *) OSN="$(printf '%s' "$OS" | tr 'A-Z' 'a-z')"; CPU="?"; CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo ?)" ;;
esac

BUILD_GO="$(cat "$ROOT/BUILD-GO.txt" 2>/dev/null || echo prebuilt-go)"
BUILD_GIT="$(cat "$ROOT/BUILD-GIT.txt" 2>/dev/null || echo ?)"
OUT="${OUT_OVERRIDE:-$ROOT/results/$HOST}"
case "$OUT" in
  /*|[A-Za-z]:*) ;;
  *) OUT="$ROOT/$OUT" ;;
esac
mkdir -p "$OUT" "$ROOT/results"

cat > "$OUT/node-info.json" <<EOF
{
  "host": "$HOST", "os": "$OSN", "arch": "$ARCH",
  "cpu": "$CPU", "cores": "$CORES", "go": "$BUILD_GO", "git": "$BUILD_GIT"
}
EOF

run() {
  local name="$1" log="$2"; shift 2
  if [[ ! -x "$1" ]]; then
    echo "  $name SKIP (missing executable: $1)"
    return
  fi
  if ( cd "$ROOT/fak" && "$@" > "$log" 2>&1 ); then
    echo "  $name ok -> $(basename "$log")"
  else
    echo "  $name FAIL (exit $?) -- see $(basename "$log")"
  fi
}

echo "[prebuilt-node] $HOST  $OSN/$ARCH  cores=$CORES  $BUILD_GO  git=$BUILD_GIT"
echo "[prebuilt-node] cpu: $CPU"
echo "[prebuilt-node] running benchmarks ..."

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

run q8kernel "$OUT/q8kernel.txt" "$ROOT/bin/q8kernel"
run modelprof "$OUT/modelprof.txt" "$ROOT/bin/modelprof" -out "$OUT/modelprof.json"
run modelbench-q8 "$OUT/modelbench-q8.txt" "$ROOT/bin/modelbench" -quant \
  -prefill-reps "$PREFILL_REPS" -decode-reps "$DECODE_REPS" -decode-steps "$DECODE_STEPS" \
  -workload "$WORKLOAD" -workload-prefill-cap "$WORKLOAD_PREFILL_CAP" \
  -out "$OUT/modelbench-q8.json"
if [[ "$BATCH_WORKLOAD" == "1" ]]; then
  run batchbench "$OUT/batchbench.txt" "$ROOT/bin/batchbench" -quant \
    -reps "$BATCH_REPS" -decode-steps "$DECODE_STEPS" -batches "$BATCHES" \
    -workload "$WORKLOAD" -workload-prompt-cap "$WORKLOAD_PROMPT_CAP" \
    -out "$OUT/batchbench-q8.json"
else
  run batchbench "$OUT/batchbench.txt" "$ROOT/bin/batchbench" -quant \
    -reps "$BATCH_REPS" -decode-steps "$DECODE_STEPS" -batches "$BATCHES" \
    -out "$OUT/batchbench-q8.json"
fi
run fleetbench "$OUT/fleetbench.txt" "$ROOT/bin/fleetbench" \
  --grid log --turn-max 16 --agent-max 16 --trials 8 \
  --out "$OUT/fleetbench.json" --csv "$OUT/fleetbench.csv"

cat > "$OUT/production-readiness-manifest.json" <<EOF
{
  "schema": "fak.production-readiness-node.v1",
  "host": "$HOST",
  "os": "$OSN",
  "arch": "$ARCH",
  "go": "$BUILD_GO",
  "git": "$BUILD_GIT",
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

archive="$ROOT/results/${HOST}-$(date +%Y%m%d-%H%M%S).tgz"
tar -czf "$archive" -C "$ROOT/results" "$HOST"
echo "[prebuilt-node] archive: $archive"
if command -v tailscale >/dev/null 2>&1; then
  echo "[prebuilt-node] Taildropping archive to $SEND_TO ..."
  tailscale file cp "$archive" "$SEND_TO:" || echo "[prebuilt-node] Taildrop failed; send $archive manually" >&2
fi
echo "[prebuilt-node] done"
