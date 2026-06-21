#!/bin/bash
# run.sh — one command to run the fak kernel tool-call adjudication demo.
#
# Builds the fak kernel, ensures a tool-capable local model is served behind it via
# `fak serve` (the kernel, with a capability floor), runs demo.py through the kernel,
# and tears down everything IT started on exit. The model PROPOSES tool calls; the
# kernel DECIDES.
#
#   ./run.sh                       # build, serve, run the demo, teardown
#   ./run.sh --dry-run             # show verdicts without executing the allowed commands
#   FAK_DEMO_MODEL=qwen2.5:7b ./run.sh    # smaller/faster model (any tool-capable one)
#
# Requires: Go (to build fak), ollama (https://ollama.com), Python 3.
# Env knobs:
#   FAK_DEMO_MODEL   ollama model id behind the kernel   (default qwen2.5:14b)
#   FAK_DEMO_PORT    kernel port                          (default 8080)
#   FAK_BIN          prebuilt fak binary to use           (default: build ./cmd/fak)
#   OLLAMA_HOST      ollama base                           (default 127.0.0.1:11434)
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/adjudication-demo -> fak/
PORT="${FAK_DEMO_PORT:-8080}"
MODEL="${FAK_DEMO_MODEL:-qwen2.5:14b}"
OLLAMA="${OLLAMA_HOST:-127.0.0.1:11434}"
POLICY="$FAK_DIR/examples/dogfood-claude-policy.json"
TMP="${TMPDIR:-/tmp}"; TMP="${TMP%/}"
log(){ printf '\033[36m[demo]\033[0m %s\n' "$*" >&2; }

KPID=""; OLLAMA_PID=""; BIN_DIR=""
cleanup(){
  [ -n "$KPID" ] && kill "$KPID" 2>/dev/null || true
  [ -n "$OLLAMA_PID" ] && kill "$OLLAMA_PID" 2>/dev/null || true   # only if WE started ollama
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

# 2) a tool-capable local model behind ollama (the kernel proxies to it)
command -v ollama >/dev/null || { log "ollama not found — install from https://ollama.com"; exit 1; }
if ! curl -sf "http://$OLLAMA/api/tags" >/dev/null 2>&1; then
  log "starting 'ollama serve'"; ollama serve >"$TMP/fak-demo-ollama.log" 2>&1 & OLLAMA_PID=$!
  until curl -sf "http://$OLLAMA/api/tags" >/dev/null 2>&1; do
    kill -0 "$OLLAMA_PID" 2>/dev/null || { log "ollama failed to start (see $TMP/fak-demo-ollama.log)"; exit 1; }
    sleep 1
  done
fi
if ! curl -s "http://$OLLAMA/api/tags" 2>/dev/null | grep -q "\"name\":\"$MODEL\""; then
  log "pulling $MODEL (one-time; the 14B is ~9GB — set FAK_DEMO_MODEL for a smaller one)"
  ollama pull "$MODEL"
fi

# 3) start the kernel in front of the model (fail loud if the port is already taken)
if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  log "port $PORT already has a kernel — stop it or set FAK_DEMO_PORT"; exit 1
fi
FAKLOG="$TMP/fak-demo-kernel.log"
log "starting kernel: fak serve :$PORT  (model=$MODEL, capability floor = examples/dogfood-claude-policy.json)"
"$BIN" serve --addr "127.0.0.1:$PORT" --model "$MODEL" \
  --base-url "http://$OLLAMA/v1" --policy "$POLICY" >"$FAKLOG" 2>&1 & KPID=$!
until curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; do
  if ! kill -0 "$KPID" 2>/dev/null; then
    log "kernel died on startup (model=$MODEL, addr=127.0.0.1:$PORT) — last log lines:"; tail -20 "$FAKLOG" >&2 || true; exit 1
  fi
  sleep 0.3
done
log "kernel healthy: $(curl -s "http://127.0.0.1:$PORT/healthz")"
echo

# 4) run the demo through the kernel
FAK_DEMO_KERNEL="http://127.0.0.1:$PORT" FAK_DEMO_MODEL="$MODEL" \
  python3 "$HERE/demo.py" "$@"
