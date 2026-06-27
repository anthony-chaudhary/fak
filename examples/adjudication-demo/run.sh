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
#   FAK_DEMO_MODEL=qwen2.5:14b ./run.sh   # stronger/larger model (any tool-capable one)
#
# Requires: Go (to build fak), Python 3. ollama is OPTIONAL — when it is absent the
# demo automatically falls back to fak's own in-kernel gguf forward (a cached GGUF
# under ~/.cache/fak-models/gguf, else a one-time scripts/fetch-gguf.sh 7b fetch).
# Env knobs:
#   FAK_DEMO_MODEL   ollama model id behind the kernel   (default qwen2.5:7b)
#   FAK_DEMO_PORT    kernel port                          (default 8080)
#   FAK_BIN          prebuilt fak binary to use           (default: build ./cmd/fak)
#   OLLAMA_HOST      ollama base                           (default 127.0.0.1:11434)
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/adjudication-demo -> fak/
PORT="${FAK_DEMO_PORT:-8080}"
MODEL="${FAK_DEMO_MODEL:-qwen2.5:7b}"
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

# 2) pick a backend: the ollama proxy (default when present), or — when ollama is
#    absent — fall back AUTOMATICALLY to fak's OWN in-kernel gguf forward (no ollama,
#    no python model). demo.py is backend-agnostic (it only POSTs to FAK_DEMO_KERNEL),
#    so the swap is invisible to it.
SERVE_ARGS=()              # the model/source flags handed to `fak serve`
SERVE_DESC=""              # human label for the startup log
ASSERT_INKERNEL=0          # in the gguf fall-back, require planner=inkernel on /healthz
if command -v ollama >/dev/null 2>&1; then
  if ! curl -sf "http://$OLLAMA/api/tags" >/dev/null 2>&1; then
    log "starting 'ollama serve'"; ollama serve >"$TMP/fak-demo-ollama.log" 2>&1 & OLLAMA_PID=$!
    tries=0
    until curl -sf "http://$OLLAMA/api/tags" >/dev/null 2>&1; do
      kill -0 "$OLLAMA_PID" 2>/dev/null || { log "ollama failed to start (see $TMP/fak-demo-ollama.log)"; exit 1; }
      tries=$((tries + 1)); [ "$tries" -ge 60 ] && { log "ollama did not answer within ~60s — giving up (see $TMP/fak-demo-ollama.log)"; exit 1; }
      sleep 1
    done
  fi
  if ! curl -s "http://$OLLAMA/api/tags" 2>/dev/null | grep -q "\"name\":\"$MODEL\""; then
    log "pulling $MODEL (one-time; model downloads can be multi-GB — set FAK_DEMO_MODEL to choose)"
    ollama pull "$MODEL"
  fi
  SERVE_ARGS=(--model "$MODEL" --base-url "http://$OLLAMA/v1")
  SERVE_DESC="model=$MODEL via ollama proxy"
else
  log "ollama not found — falling back to fak's in-kernel gguf forward (no ollama needed)."
  GGUF_CACHE="$HOME/.cache/fak-models/gguf"
  GGUF=""
  [ -d "$GGUF_CACHE" ] && GGUF="$(ls -1t "$GGUF_CACHE"/*.gguf 2>/dev/null | head -1 || true)"
  if [ -z "$GGUF" ]; then
    log "no cached GGUF under $GGUF_CACHE — fetching the 7B Qwen2.5 once (scripts/fetch-gguf.sh 7b)"
    # --yes drives fetch-gguf non-interactively so the fall-back never hangs on a prompt.
    "$FAK_DIR/scripts/fetch-gguf.sh" 7b --yes >&2 || true
    [ -d "$GGUF_CACHE" ] && GGUF="$(ls -1t "$GGUF_CACHE"/*.gguf 2>/dev/null | head -1 || true)"
  fi
  if [ -z "$GGUF" ]; then
    log "could not locate or fetch a GGUF — install ollama (https://ollama.com) and re-run, or"
    log "  fetch one by hand:  scripts/fetch-gguf.sh 7b   then re-run."
    exit 1
  fi
  log "using in-kernel gguf: $GGUF"
  SERVE_ARGS=(--gguf "$GGUF")
  SERVE_DESC="in-kernel gguf $(basename "$GGUF")"
  ASSERT_INKERNEL=1
fi

# 3) start the kernel in front of the chosen backend (fail loud if the port is taken)
if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  log "port $PORT already has a kernel — stop it or set FAK_DEMO_PORT"; exit 1
fi
FAKLOG="$TMP/fak-demo-kernel.log"
log "starting kernel: fak serve :$PORT  ($SERVE_DESC, capability floor = examples/dogfood-claude-policy.json)"
"$BIN" serve --addr "127.0.0.1:$PORT" "${SERVE_ARGS[@]}" --policy "$POLICY" >"$FAKLOG" 2>&1 & KPID=$!
tries=0
until curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; do
  if ! kill -0 "$KPID" 2>/dev/null; then
    log "kernel died on startup ($SERVE_DESC, addr=127.0.0.1:$PORT) — last log lines:"; tail -20 "$FAKLOG" >&2 || true; exit 1
  fi
  tries=$((tries + 1)); if [ "$tries" -ge 200 ]; then log "kernel did not become healthy within ~60s — last log lines:"; tail -20 "$FAKLOG" >&2 || true; exit 1; fi
  sleep 0.3
done
HEALTH="$(curl -s "http://127.0.0.1:$PORT/healthz")"
log "kernel healthy: $HEALTH"
# In the gguf fall-back, assert the kernel is really doing its OWN in-kernel forward
# (planner=inkernel) — so a misconfigured fall-back fails loud instead of silently
# proxying nowhere. Field semantics: internal/gateway/gateway_test.go.
if [ "$ASSERT_INKERNEL" = "1" ]; then
  case "$HEALTH" in
    *'"planner":"inkernel"'*|*'"planner": "inkernel"'*) : ;;
    *) log "expected planner=inkernel on /healthz but got: $HEALTH"; exit 1 ;;
  esac
fi
echo

# 4) run the demo through the kernel
FAK_DEMO_KERNEL="http://127.0.0.1:$PORT" FAK_DEMO_MODEL="$MODEL" \
  python3 "$HERE/demo.py" "$@"
