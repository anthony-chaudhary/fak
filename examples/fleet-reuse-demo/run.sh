#!/usr/bin/env bash
# run.sh — one command to see the fak fleet shared-prompt reuse curve.
#
# Builds the fak kernel, stands up `fak serve` in front of the OFFLINE MOCK planner
# (no model, no key, no GPU — the reuse curve is an exact accounting of prompt bytes,
# not a model measurement), drives N = 1, 2, 5 workers that share ONE prompt prefix
# through that one kernel, prints the before/after reuse table, and tears down
# everything IT started on exit.
#
#   ./run.sh                    # build, serve (offline mock), run the demo, teardown
#   ./run.sh --offline          # skip the live kernel; print the accounting only
#   FAK_DEMO_N=1,2,5,10 ./run.sh  # choose the worker counts (the reuse curve)
#
# Requires: Go (to build fak) and Python 3 (stdlib only) — a STRICT SUBSET of the
# adjudication-demo prerequisites (no ollama, no model download). On Windows run this
# from WSL or Git Bash; the demo itself is `fak serve` plus stdlib Python.
#
# Env knobs:
#   FAK_DEMO_PORT   kernel port                       (default 8080)
#   FAK_DEMO_N      worker counts, comma-separated     (default 1,2,5)
#   FAK_BIN         prebuilt fak binary to use         (default: build ./cmd/fak)
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/fleet-reuse-demo -> fak/
PORT="${FAK_DEMO_PORT:-8080}"
NLIST="${FAK_DEMO_N:-1,2,5}"
TMP="${TMPDIR:-/tmp}"; TMP="${TMP%/}"
log(){ printf '\033[36m[fleet]\033[0m %s\n' "$*" >&2; }

KPID=""; BIN_DIR=""
cleanup(){
  [ -n "$KPID" ] && kill "$KPID" 2>/dev/null || true
  [ -n "$BIN_DIR" ] && rm -rf "$BIN_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --offline short-circuit: no kernel needed, just the accounting (runs anywhere).
for arg in "$@"; do
  if [ "$arg" = "--offline" ]; then
    log "offline accounting only (no kernel) — the bytes/turns curve is exact arithmetic"
    FAK_DEMO_KERNEL="" python3 "$HERE/demo.py" --offline --n "$NLIST" "$@"
    exit $?
  fi
done

# 1) the kernel binary (build it, or honor a prebuilt FAK_BIN)
BIN="${FAK_BIN:-}"
if [ -z "$BIN" ]; then
  command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl (or set FAK_BIN to a prebuilt fak)"; exit 1; }
  BIN_DIR="$(mktemp -d)"; BIN="$BIN_DIR/fak"
  log "building fak kernel -> $BIN"
  ( cd "$FAK_DIR" && go build -o "$BIN" ./cmd/fak )
fi

# 2) start the kernel in front of the OFFLINE MOCK planner (no --base-url / --gguf).
#    Fail loud if the port is already taken so we never drive a stranger's process.
if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  log "port $PORT already has a kernel — stop it or set FAK_DEMO_PORT"; exit 1
fi
FAKLOG="$TMP/fak-fleet-kernel.log"
log "starting kernel: fak serve :$PORT  (offline mock planner — no model, no GPU)"
"$BIN" serve --addr "127.0.0.1:$PORT" >"$FAKLOG" 2>&1 & KPID=$!
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

# 3) drive the fleet through the one kernel and print the reuse curve.
FAK_DEMO_KERNEL="http://127.0.0.1:$PORT" python3 "$HERE/demo.py" --n "$NLIST" "$@"
