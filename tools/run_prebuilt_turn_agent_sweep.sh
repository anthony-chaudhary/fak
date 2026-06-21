#!/usr/bin/env bash
# Run a prebuilt fak turn-agent sweep on a machine with no Go checkout.
#
# Payload layout expected:
#   bin/fleetserve
#   fak/internal/model/.cache/smollm2-135m/...
set -uo pipefail

HOST_OVERRIDE=""
OUT_OVERRIDE=""
PROFILE="long"
SEND_TO="${FAK_NODE_SEND_TO:-your-bench-driver}"

usage() {
  sed -n '2,8p' "$0"
  cat <<'EOF'

Options:
  --host=NAME          result host label
  --out=PATH           output directory
  --profile=NAME       long | mac-longer | mac-stress
  --send-to=HOST       Taildrop target for the result archive

Environment overrides:
  FAK_TURN_PREFIX, FAK_TURN_TURNS, FAK_TURN_AGENTS, FAK_TURN_DECODE,
  FAK_TURN_RESULT, FAK_TURN_REPS, FAK_TURN_ABLATION
EOF
}

for a in "$@"; do
  case "$a" in
    --host=*) HOST_OVERRIDE="${a#--host=}" ;;
    --out=*) OUT_OVERRIDE="${a#--out=}" ;;
    --profile=*) PROFILE="${a#--profile=}" ;;
    --send-to=*) SEND_TO="${a#--send-to=}" ;;
    -h|--help)
      usage
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

case "$PROFILE" in
  long)
    PREFIX="${FAK_TURN_PREFIX:-1024}"
    TURNS="${FAK_TURN_TURNS:-40,80}"
    AGENTS="${FAK_TURN_AGENTS:-8,12,16,20}"
    DECODE="${FAK_TURN_DECODE:-32}"
    RESULT="${FAK_TURN_RESULT:-48}"
    REPS="${FAK_TURN_REPS:-3}"
    ABLATION="${FAK_TURN_ABLATION:-true}"
    PROFILE_NOTE="canonical local long profile"
    ;;
  mac-longer)
    PREFIX="${FAK_TURN_PREFIX:-2048}"
    TURNS="${FAK_TURN_TURNS:-80,120}"
    AGENTS="${FAK_TURN_AGENTS:-8,12,16,20}"
    DECODE="${FAK_TURN_DECODE:-32}"
    RESULT="${FAK_TURN_RESULT:-16}"
    REPS="${FAK_TURN_REPS:-1}"
    ABLATION="${FAK_TURN_ABLATION:-false}"
    PROFILE_NOTE="longer turns/context while staying under 8k per agent"
    ;;
  mac-stress)
    PREFIX="${FAK_TURN_PREFIX:-4096}"
    TURNS="${FAK_TURN_TURNS:-120,160}"
    AGENTS="${FAK_TURN_AGENTS:-8,12,16}"
    DECODE="${FAK_TURN_DECODE:-32}"
    RESULT="${FAK_TURN_RESULT:-16}"
    REPS="${FAK_TURN_REPS:-1}"
    ABLATION="${FAK_TURN_ABLATION:-false}"
    PROFILE_NOTE="beyond-8k capacity stress; not model-context-valid"
    ;;
  *)
    echo "unknown profile: $PROFILE" >&2
    exit 2
    ;;
esac

BUILD_GO="$(cat "$ROOT/BUILD-GO.txt" 2>/dev/null || echo prebuilt-go)"
BUILD_GIT="$(cat "$ROOT/BUILD-GIT.txt" 2>/dev/null || echo ?)"
OUT="${OUT_OVERRIDE:-$ROOT/results/$HOST/turn-agent-$PROFILE}"
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

cat > "$OUT/turn-agent-sweep-manifest.json" <<EOF
{
  "schema": "fak.prebuilt-turn-agent-sweep.v1",
  "host": "$HOST",
  "profile": "$PROFILE",
  "profile_note": "$PROFILE_NOTE",
  "prefix": "$PREFIX",
  "turns": "$TURNS",
  "agents": "$AGENTS",
  "decode": "$DECODE",
  "result": "$RESULT",
  "reps": "$REPS",
  "ablation": "$ABLATION",
  "engine": "fak fleetserve Q8"
}
EOF

if [[ ! -x "$ROOT/bin/fleetserve" ]]; then
  echo "missing executable: $ROOT/bin/fleetserve" >&2
  exit 1
fi

echo "[turn-agent] $HOST  $OSN/$ARCH  cores=$CORES  $BUILD_GO  git=$BUILD_GIT"
echo "[turn-agent] profile=$PROFILE  P=$PREFIX T=$TURNS A=$AGENTS D=$DECODE R=$RESULT reps=$REPS ablation=$ABLATION"
if ( cd "$ROOT/fak" && "$ROOT/bin/fleetserve" -quant -prefix "$PREFIX" -turns "$TURNS" \
  -decode "$DECODE" -result "$RESULT" -concurrency "$AGENTS" -reps "$REPS" \
  -ablation="$ABLATION" -out "$OUT/turn-agent-fak-q8.json" > "$OUT/turn-agent-fak-q8.out.txt" 2> "$OUT/turn-agent-fak-q8.err.txt" ); then
  echo "[turn-agent] ok -> $OUT/turn-agent-fak-q8.json"
else
  rc=$?
  echo "[turn-agent] FAIL (exit $rc) -- see $OUT/turn-agent-fak-q8.err.txt" >&2
  exit $rc
fi

archive="$ROOT/results/${HOST}-turn-agent-${PROFILE}-$(date +%Y%m%d-%H%M%S).tgz"
tar -czf "$archive" -C "$ROOT/results" "$HOST"
echo "[turn-agent] archive: $archive"
if command -v tailscale >/dev/null 2>&1; then
  echo "[turn-agent] Taildropping archive to $SEND_TO ..."
  tailscale file cp "$archive" "$SEND_TO:" || echo "[turn-agent] Taildrop failed; send $archive manually" >&2
fi
echo "[turn-agent] done"
