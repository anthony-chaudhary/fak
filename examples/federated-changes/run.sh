#!/bin/bash
# run.sh — one command to run the fak federated "what changed" feed + revoke demo (#340).
#
# It builds the fak kernel (or honors a prebuilt FAK_BIN), starts TWO `fak serve`
# instances on different ports (A and B) with a tiny capability-floor policy that
# allow-lists the write-shaped tools the demo drives (a write only lands on
# /v1/fak/changes once the kernel lets it COMPLETE), drives demo.py across both
# instances, and tears down everything IT started on exit.
#
# NO MODEL is needed: the change feed and the refutation are pure kernel state, so
# the load-bearing observations fire with `--engine mock` alone. The demo's exit
# code gates on those observations.
#
#   ./run.sh                       # build, serve x2, run the demo, teardown
#   ./run.sh --no-color            # plain output
#   FAK_BIN=./fak ./run.sh         # use a prebuilt binary instead of building
#
# Env knobs:
#   FAK_PORT_A   instance A port                            (default 8431)
#   FAK_PORT_B   instance B port                            (default 8432)
#   FAK_BIN      prebuilt fak binary to use                 (default: build ./cmd/fak)
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/federated-changes -> fak/
PORT_A="${FAK_PORT_A:-8431}"
PORT_B="${FAK_PORT_B:-8432}"
POLICY="$HERE/federated-policy.json"
TMP="${TMPDIR:-/tmp}"; TMP="${TMP%/}"
log(){ printf '\033[36m[demo]\033[0m %s\n' "$*" >&2; }

PID_A=""; PID_B=""; BIN_DIR=""
cleanup(){
  [ -n "$PID_A" ] && kill "$PID_A" 2>/dev/null || true
  [ -n "$PID_B" ] && kill "$PID_B" 2>/dev/null || true
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

# start_one PORT NAME LOGFILE PIDVAR — start a kernel and wait for /healthz.
start_one(){
  local port="$1" name="$2" logf="$3"
  if curl -sf "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
    log "port $port already has a kernel — stop it or set FAK_PORT_$name"; exit 1
  fi
  log "starting kernel $name: fak serve :$port  (policy=$(basename "$POLICY"))"
  "$BIN" serve --addr "127.0.0.1:$port" --engine mock --policy "$POLICY" >"$logf" 2>&1 &
  local pid=$!
  local tries=0
  until curl -sf "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "kernel $name died on startup (addr=127.0.0.1:$port) — last log lines:"; tail -20 "$logf" >&2 || true; exit 1
    fi
    tries=$((tries + 1)); if [ "$tries" -ge 200 ]; then log "kernel $name did not become healthy within ~60s"; tail -20 "$logf" >&2 || true; exit 1; fi
    sleep 0.3
  done
  echo "$pid"
}

# 2) start BOTH instances (the federated pair)
PID_A="$(start_one "$PORT_A" A "$TMP/fak-fed-a.log")"
PID_B="$(start_one "$PORT_B" B "$TMP/fak-fed-b.log")"
log "A healthy: $(curl -s "http://127.0.0.1:$PORT_A/healthz")"
log "B healthy: $(curl -s "http://127.0.0.1:$PORT_B/healthz")"
echo

# 3) run the demo across the two kernels' native coherence wires
FAK_A="http://127.0.0.1:$PORT_A" FAK_B="http://127.0.0.1:$PORT_B" python3 "$HERE/demo.py" "$@"
