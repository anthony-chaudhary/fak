#!/bin/bash
# run.sh — one command to run the fak IFC taint-flow / provenance demo (#332).
#
# It builds the fak kernel (or honors a prebuilt FAK_BIN), starts `fak serve` with the
# demo's capability-floor policy (which ALLOW-LISTS the send_email sink so ONLY the IFC
# flow rule can refuse it, and declares the source taint of fetch_url / read_corp_kb),
# drives demo.py through the two native wires (/v1/fak/admit + /v1/fak/adjudicate), and
# tears down everything IT started on exit.
#
# NO MODEL is needed: the IFC floor is a pure function of the bytes' PROVENANCE, so the
# load-bearing verdict (a tainted->egress refusal) fires with the deterministic kernel
# alone. The demo's exit code gates on that kernel verdict.
#
#   ./run.sh                       # build, serve, run the demo, teardown
#   ./run.sh --no-color            # plain output
#   FAK_BIN=./fak ./run.sh         # use a prebuilt binary instead of building
#
# Env knobs:
#   FAK_DEMO_PORT    kernel port                          (default 8080)
#   FAK_BIN          prebuilt fak binary to use           (default: build ./cmd/fak)
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/ifc-taint-flow -> fak/
PORT="${FAK_DEMO_PORT:-8080}"
POLICY="$HERE/research-sink-policy.json"
TMP="${TMPDIR:-/tmp}"; TMP="${TMP%/}"
log(){ printf '\033[36m[demo]\033[0m %s\n' "$*" >&2; }

KPID=""; BIN_DIR=""
cleanup(){
  [ -n "$KPID" ] && kill "$KPID" 2>/dev/null || true
  [ -n "$BIN_DIR" ] && rm -rf "$BIN_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1) the kernel binary (build it, or honor a prebuilt FAK_BIN)
BIN="${FAK_BIN:-}"
if [ -z "$BIN" ]; then
  command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl (or set FAK_BIN to a prebuilt fak)"; exit 1; }
  BIN_DIR="$(mktemp -d)"; BIN="$BIN_DIR/fak"
  log "building fak kernel -> $BIN"
  ( cd "$FAK_DIR" && go build -o "$BIN" ./cmd/fak )
fi

# 2) start the kernel with the demo policy (fail loud if the port is taken)
if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  log "port $PORT already has a kernel — stop it or set FAK_DEMO_PORT"; exit 1
fi
FAKLOG="$TMP/fak-ifc-kernel.log"
log "starting kernel: fak serve :$PORT  (policy=$(basename "$POLICY"))"
"$BIN" serve --addr "127.0.0.1:$PORT" --policy "$POLICY" >"$FAKLOG" 2>&1 & KPID=$!
tries=0
until curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; do
  if ! kill -0 "$KPID" 2>/dev/null; then
    log "kernel died on startup (addr=127.0.0.1:$PORT) — last log lines:"; tail -20 "$FAKLOG" >&2 || true; exit 1
  fi
  tries=$((tries + 1)); if [ "$tries" -ge 200 ]; then log "kernel did not become healthy within ~60s — last log lines:"; tail -20 "$FAKLOG" >&2 || true; exit 1; fi
  sleep 0.3
done
log "kernel healthy: $(curl -s "http://127.0.0.1:$PORT/healthz")"
echo

# 3) run the demo through the kernel's native IFC wires
FAK_DEMO_KERNEL="http://127.0.0.1:$PORT" python3 "$HERE/demo.py" "$@"
