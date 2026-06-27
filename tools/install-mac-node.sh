#!/usr/bin/env bash
# install-mac-node.sh — one-command always-on Mac dogfood node setup.
#
# Wires all three launchd units (serve-gateway, dogfood-fleet, dispatch-supervisor),
# builds the fak binary, and loads them so the Mac runs 24/7. The gateway plist
# wraps fak serve under caffeinate so the machine can't idle-sleep while the fleet
# is running — no separate keep-awake step needed.
#
# Usage:
#   cd <repo>
#   ./tools/install-mac-node.sh              # loopback (127.0.0.1:8080); single-machine
#   ./tools/install-mac-node.sh --bind-all   # 0.0.0.0:8080 + bearer key; off-host via Tailscale
#   ./tools/install-mac-node.sh --status     # show unit states + caffeinate status
#   ./tools/install-mac-node.sh --uninstall  # remove all units and stop caffeinate
#
# After --bind-all the script prints FAK_GATEWAY_KEY and the exact env lines to paste
# on any Tailscale-connected client (PowerShell or bash). Save the key — it is only
# printed once. To rotate: --uninstall then --bind-all again.
#
# Environment:
#   ANTHROPIC_API_KEY     upstream key to set in the login env (needed for the gateway
#                         to reach api.anthropic.com; set once, persists across reboots)
#   FAK_GATEWAY_KEY       bearer key for --bind-all (generated via openssl if unset)
#   FAK_GATEWAY_PORT      gateway port (default 8080)
#   FAK_DISPATCH_ENABLE   set=1 to also arm the dispatch supervisor watchdog (default: 1)
#   PYTHON                python3 interpreter path override
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$REPO/tools"
LOG="$TOOLS/_watchdog"
AGENTS="$HOME/Library/LaunchAgents"
PORT="${FAK_GATEWAY_PORT:-8080}"
DISPATCH_ENABLE="${FAK_DISPATCH_ENABLE:-1}"
PY="${PYTHON:-$(command -v python3 2>/dev/null || true)}"

log()  { printf '\033[36m[install-mac-node]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[install-mac-node] WARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m[install-mac-node] ERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- parse args ---
MODE="install"
BIND_ALL=""
for arg in "$@"; do
  case "$arg" in
    --bind-all)  BIND_ALL=1 ;;
    --uninstall) MODE="uninstall" ;;
    --status)    MODE="status" ;;
    --help|-h)   sed -n '2,42p' "$0"; exit 0 ;;
    *) die "unknown argument: $arg (want --bind-all | --status | --uninstall)" ;;
  esac
done

ALL_LABELS=(
  "com.fak.serve-gateway"
  "com.fak.dogfood-fleet"
  "com.fleet.dispatch-supervisor"
)

# --- --status ---
if [ "$MODE" = "status" ]; then
  log "launchd unit states:"
  for label in "${ALL_LABELS[@]}"; do
    dst="$AGENTS/$label.plist"
    if [ -f "$dst" ]; then
      info="$(launchctl list "$label" 2>/dev/null || echo "installed but not loaded")"
      printf '  %-42s %s\n' "$label" "$info"
    else
      printf '  %-42s not installed\n' "$label"
    fi
  done
  printf '\n'
  "$TOOLS/mac_keep_awake.sh" status
  exit 0
fi

# --- --uninstall ---
if [ "$MODE" = "uninstall" ]; then
  for label in "${ALL_LABELS[@]}"; do
    dst="$AGENTS/$label.plist"
    if [ -f "$dst" ]; then
      launchctl unload -w "$dst" 2>/dev/null || true
      rm -f "$dst"
      log "unloaded + removed: $label"
    else
      log "not installed: $label"
    fi
  done
  "$TOOLS/mac_keep_awake.sh" stop 2>/dev/null || true
  log "uninstall complete"
  exit 0
fi

# --- prereqs ---
uname -s | grep -q Darwin || die "this script only runs on macOS"
command -v go >/dev/null   || die "Go is required — install via: brew install go"
[ -n "$PY" ]               || die "python3 is required — install via: brew install python3"
mkdir -p "$LOG" "$AGENTS"

# --- build fak binary ---
BIN="$TOOLS/.bin/fak"
log "building fak -> $BIN"
mkdir -p "$(dirname "$BIN")"
( cd "$REPO" && go build -o "$BIN" ./cmd/fak ) || die "go build failed — check the working tree"
log "built ok"

# --- gateway address + optional bearer key ---
ADDR="127.0.0.1:$PORT"
GATEWAY_KEY=""
if [ -n "$BIND_ALL" ]; then
  ADDR="0.0.0.0:$PORT"
  GATEWAY_KEY="${FAK_GATEWAY_KEY:-$(openssl rand -hex 32)}"
fi

# --- helper: install one plist template ---
install_plist() {
  local label="$1"
  local src="$TOOLS/$label.plist"
  local dst="$AGENTS/$label.plist"
  [ -f "$src" ] || die "template not found: $src"

  # Unload first so the reload picks up any new binary/path (idempotent)
  launchctl unload -w "$dst" 2>/dev/null || true

  sed \
    -e "s#__FAK__#$BIN#g" \
    -e "s#__REPO__#$REPO#g" \
    -e "s#__PYTHON__#$PY#g" \
    -e "s#__ADDR__#$ADDR#g" \
    "$src" > "$dst"

  # For --bind-all: append --require-key-env + inject the key into EnvironmentVariables.
  # PlistBuddy appends to the array in document order, so these land after --addr.
  # GATEWAY_KEY is a hex string (no special chars), so no quoting needed for PlistBuddy.
  if [ -n "$BIND_ALL" ] && [ "$label" = "com.fak.serve-gateway" ]; then
    /usr/libexec/PlistBuddy -c \
      "Add :ProgramArguments: string --require-key-env" "$dst"
    /usr/libexec/PlistBuddy -c \
      "Add :ProgramArguments: string FAK_GATEWAY_KEY" "$dst"
    /usr/libexec/PlistBuddy -c \
      "Add :EnvironmentVariables:FAK_GATEWAY_KEY string $GATEWAY_KEY" "$dst"
  fi

  launchctl load -w "$dst"
  log "loaded: $label"
}

# --- install gateway + fleet units ---
install_plist "com.fak.serve-gateway"
install_plist "com.fak.dogfood-fleet"

# --- dispatch supervisor (opt-in via FAK_DISPATCH_ENABLE) ---
DSP_DST="$AGENTS/com.fleet.dispatch-supervisor.plist"
if [ "$DISPATCH_ENABLE" = "1" ]; then
  install_plist "com.fleet.dispatch-supervisor"
else
  launchctl unload -w "$DSP_DST" 2>/dev/null || true
  rm -f "$DSP_DST"
  log "skipped: com.fleet.dispatch-supervisor (set FAK_DISPATCH_ENABLE=1 to arm)"
fi

# --- upstream credential: persist in login env so launchd units inherit it ---
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  launchctl setenv ANTHROPIC_API_KEY "$ANTHROPIC_API_KEY"
  log "ANTHROPIC_API_KEY set in login environment (persists across unit restarts)"
else
  warn "ANTHROPIC_API_KEY is not set — the gateway cannot reach api.anthropic.com."
  warn "Set it now and it will persist:"
  warn "  launchctl setenv ANTHROPIC_API_KEY \"sk-ant-...\""
fi

# --- wait briefly for the gateway to become healthy ---
WAIT=0
log "waiting for gateway on $ADDR..."
until curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; do
  WAIT=$((WAIT + 1))
  [ $WAIT -le 20 ] || { warn "gateway did not come up in 20 s — check: tail -f $LOG/launchd_serve.log"; break; }
  sleep 1
done
if [ $WAIT -le 20 ]; then
  HEALTH="$(curl -s "http://127.0.0.1:$PORT/healthz" 2>/dev/null || true)"
  log "gateway healthy: $HEALTH"
fi

# --- print success + connection instructions ---
log ""
log "=== node ready ==="
log ""
log "Logs:"
log "  tail -f $LOG/launchd_serve.log"
log "  tail -f $LOG/launchd_dogfood_fleet.log"
log ""
log "Coverage check (after first guarded tick in ~30 min):"
log "  python3 tools/dogfood_coverage.py"
log ""

if [ -n "$BIND_ALL" ]; then
  TAILSCALE_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
  if [ -z "$TAILSCALE_IP" ]; then
    TAILSCALE_IP="<this-mac-tailscale-ip>"
    warn "Tailscale not running or not connected — get the IP with: tailscale ip -4"
  fi

  log "=== client connection (paste on any Tailscale-connected client) ==="
  log ""
  log "bash / zsh:"
  log "  export ANTHROPIC_BASE_URL=\"http://$TAILSCALE_IP:$PORT\""
  log "  export ANTHROPIC_API_KEY=\"$GATEWAY_KEY\""
  log "  claude"
  log ""
  log "PowerShell (Windows):"
  log "  \$env:ANTHROPIC_BASE_URL = \"http://$TAILSCALE_IP:$PORT\""
  log "  \$env:ANTHROPIC_API_KEY  = \"$GATEWAY_KEY\""
  log "  claude"
  log ""
  log "Or use the client helper (from the repo root on the client machine):"
  log "  ./scripts/connect-fak-node.ps1 -GatewayHost $TAILSCALE_IP -GatewayKey \"$GATEWAY_KEY\""
  log "  ./scripts/connect-fak-node.sh  $TAILSCALE_IP              $GATEWAY_KEY"
  log ""
  log "FAK_GATEWAY_KEY=$GATEWAY_KEY"
  log "(save this — it is only printed once)"
else
  log "=== single-machine use ==="
  log ""
  log "  fak guard -- claude          # guarded interactive session (loopback gateway)"
  log "  ./scripts/dogfood-claude.sh  # full dogfood launcher"
  log ""
  log "For off-host access (e.g. this Mac + a Windows client over Tailscale), re-run:"
  log "  ./tools/install-mac-node.sh --bind-all"
fi
