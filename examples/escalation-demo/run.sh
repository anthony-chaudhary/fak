#!/bin/bash
# run.sh — one command to run the fak human-in-the-loop escalation demo.
#
# Builds the fak kernel, starts `fak serve` enforcing the customer-support policy (the same
# capability floor that DECLARES `safe_sinks`), runs demo.py against it, and tears down
# everything IT started on exit. The harness proposes a denied call; the kernel DECIDES;
# the harness ESCALATES the deny to the policy's declared safe_sink.
#
#   ./run.sh                       # build, serve, run the demo, teardown
#
# Prerequisites are a STRICT SUBSET of adjudication-demo/: Go (to build fak) and Python 3.
# No model, no API key, no GPU, no ollama — the deny is a pure function of (policy, call), so
# the escalation path is exercised deterministically (adjudication-demo/ covers the real-model
# proposal side; this covers what happens AFTER a deny).
#
# Env knobs:
#   FAK_DEMO_PORT    kernel port                            (default 8080)
#   FAK_BIN          prebuilt fak binary to use             (default: build ./cmd/fak)
#   FAK_DEMO_POLICY  policy manifest to enforce             (default customer-support-readonly-policy.json)
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/escalation-demo -> fak/
PORT="${FAK_DEMO_PORT:-8080}"
POLICY="${FAK_DEMO_POLICY:-$FAK_DIR/examples/customer-support-readonly-policy.json}"
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

# 2) start the kernel enforcing the policy (no --base-url → no model needed; we hit the
#    fak-native /v1/fak/adjudicate surface). Fail loud if the port is already taken.
if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  log "port $PORT already has a kernel — stop it or set FAK_DEMO_PORT"; exit 1
fi
FAKLOG="$TMP/fak-escalation-kernel.log"
log "starting kernel: fak serve :$PORT  (capability floor = $(basename "$POLICY"))"
"$BIN" serve --addr "127.0.0.1:$PORT" --policy "$POLICY" >"$FAKLOG" 2>&1 & KPID=$!
until curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; do
  if ! kill -0 "$KPID" 2>/dev/null; then
    log "kernel died on startup (addr=127.0.0.1:$PORT) — last log lines:"; tail -20 "$FAKLOG" >&2 || true; exit 1
  fi
  sleep 0.3
done
log "kernel healthy: $(curl -s "http://127.0.0.1:$PORT/healthz")"
echo

# 3) run the demo against the kernel
FAK_DEMO_KERNEL="http://127.0.0.1:$PORT" FAK_DEMO_POLICY="$POLICY" \
  python3 "$HERE/demo.py" "$@"
