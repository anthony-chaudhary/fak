#!/usr/bin/env bash
# connect-fak-node.sh — connect this shell session to a remote fak gateway.
#
# Exports ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY so that `claude`, `fak guard`,
# and any tool reading those env vars will route through the always-on fak kernel
# on the target Mac node.
#
# Usage (must be sourced so env vars land in the calling shell):
#   source scripts/connect-fak-node.sh <tailscale-ip> <bearer-key> [port]
#   .       scripts/connect-fak-node.sh <tailscale-ip> <bearer-key> [port]
#
#   # Test reachability first:
#   source scripts/connect-fak-node.sh 100.x.y.z sk-fak-... --probe
#
#   # Disconnect (clear / restore):
#   source scripts/connect-fak-node.sh --disconnect
#
# Get the tailscale-ip:  run `tailscale ip -4` on the Mac node.
# Get the bearer-key:    output of `./tools/install-mac-node.sh --bind-all` on the Mac.
#
# TIP: add the two export lines to ~/.zshrc / ~/.bashrc to make the connection
# permanent across new shell sessions.

log()  { printf '\033[36m[connect-fak-node]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[33m[connect-fak-node] WARNING: %s\033[0m\n' "$*" >&2; }

# Guard: refuse to run in exec mode (env vars would be lost on exit)
_sourced=0
# shellcheck disable=SC2128
[ "${BASH_SOURCE[0]}" != "$0" ] 2>/dev/null && _sourced=1
[ "${ZSH_EVAL_CONTEXT:-}" = "toplevel:file" ] && _sourced=1
if [ "$_sourced" = "0" ]; then
  echo "[connect-fak-node] ERROR: this script must be sourced, not executed." >&2
  echo "  source $0 <host> <key>  OR  . $0 <host> <key>" >&2
  exit 1
fi

# --- --disconnect ---
if [ "${1:-}" = "--disconnect" ]; then
  if [ -n "${_FAK_ORIG_ANTHROPIC_BASE_URL:-}" ]; then
    export ANTHROPIC_BASE_URL="$_FAK_ORIG_ANTHROPIC_BASE_URL"
    unset _FAK_ORIG_ANTHROPIC_BASE_URL
  else
    unset ANTHROPIC_BASE_URL
  fi
  unset ANTHROPIC_API_KEY
  log "disconnected"
  return 0
fi

# --- parse args ---
HOST="${1:-}"
KEY="${2:-}"
PORT="8080"
PROBE=""
# Allow port as positional arg 3 or --probe anywhere
shift 2 2>/dev/null || true
for _a in "$@"; do
  case "$_a" in
    --probe) PROBE=1 ;;
    [0-9]*) PORT="$_a" ;;
    *) warn "unknown argument: $_a" ;;
  esac
done

if [ -z "$HOST" ] || [ -z "$KEY" ]; then
  warn "Usage: source scripts/connect-fak-node.sh <tailscale-ip> <bearer-key> [port]"
  warn "       source scripts/connect-fak-node.sh --disconnect"
  return 1
fi

BASE_URL="http://$HOST:$PORT"
HEALTHZ="$BASE_URL/healthz"

# --- probe ---
if [ -n "$PROBE" ]; then
  log "probing $HEALTHZ ..."
  if curl -sf -H "Authorization: Bearer $KEY" "$HEALTHZ"; then
    echo
    log "gateway healthy"
  else
    warn "gateway not reachable at $HEALTHZ"
    warn "  Is the Mac node running?   ./tools/install-mac-node.sh --status (on the Mac)"
    warn "  Is Tailscale connected?    tailscale status"
    warn "  Was it installed with --bind-all?"
    return 1
  fi
fi

# --- save existing ANTHROPIC_BASE_URL so --disconnect can restore it ---
if [ -n "${ANTHROPIC_BASE_URL:-}" ] && [ "$ANTHROPIC_BASE_URL" != "$BASE_URL" ]; then
  export _FAK_ORIG_ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL"
fi

# --- set env vars in the current shell ---
export ANTHROPIC_BASE_URL="$BASE_URL"
export ANTHROPIC_API_KEY="$KEY"

log "connected to fak gateway at $BASE_URL"
log "  ANTHROPIC_BASE_URL = $ANTHROPIC_BASE_URL"
log "  ANTHROPIC_API_KEY  = ${KEY:0:8}... (truncated)"
log ""
log "Run:  claude    — every tool call now crosses the fak kernel on the Mac."
log "Done: source scripts/connect-fak-node.sh --disconnect"
log ""
log "To persist across shells, add to ~/.zshrc or ~/.bashrc:"
log "  export ANTHROPIC_BASE_URL=\"$BASE_URL\""
log "  export ANTHROPIC_API_KEY=\"$KEY\""
