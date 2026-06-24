#!/bin/bash
# run.sh — one command to run the fak wire-side result-quarantine demo (issue #334).
#
# Builds the fak kernel, starts `fak serve` with NO upstream model (the offline mock
# planner — the result-side floor needs no model, key, or GPU), runs demo.py against it
# over the wire (POST /v1/fak/admit), and tears down everything IT started on exit.
#
#   ./run.sh                 # build, serve, run the demo, teardown
#   ./run.sh --no-color      # plain output
#   FAK_BIN=/path/to/fak ./run.sh   # use a prebuilt binary instead of building
#
# Requires: Go (to build fak) OR a prebuilt FAK_BIN, and Python 3 (stdlib only).
# Env knobs:
#   FAK_DEMO_PORT    kernel port                  (default 8080)
#   FAK_BIN          prebuilt fak binary to use   (default: build ./cmd/fak)
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/wire-quarantine-demo -> fak/
PORT="${FAK_DEMO_PORT:-8080}"
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

# 2) start the kernel — no --base-url means the offline mock planner; the result-side
#    admission floor (context-MMU / normgate quarantine + IFC source-stamp) is armed
#    regardless, so /v1/fak/admit screens results with no model in the loop.
if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  log "port $PORT already has a server — stop it or set FAK_DEMO_PORT"; exit 1
fi
FAKLOG="$(mktemp)"
log "starting kernel: fak serve :$PORT  (offline mock planner — no model, key, or GPU)"
"$BIN" serve --addr "127.0.0.1:$PORT" >"$FAKLOG" 2>&1 & KPID=$!
tries=0
until curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; do
  if ! kill -0 "$KPID" 2>/dev/null; then
    log "kernel died on startup — last log lines:"; tail -20 "$FAKLOG" >&2 || true; exit 1
  fi
  tries=$((tries + 1)); if [ "$tries" -ge 100 ]; then log "kernel did not become healthy within ~30s:"; tail -20 "$FAKLOG" >&2 || true; exit 1; fi
  sleep 0.3
done
log "kernel healthy: $(curl -s "http://127.0.0.1:$PORT/healthz")"
echo

# 3) run the demo against the kernel over the wire
FAK_DEMO_KERNEL="http://127.0.0.1:$PORT" python3 "$HERE/demo.py" "$@"
