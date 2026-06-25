#!/usr/bin/env bash
# mac_keep_awake.sh -- stop this mac from sleeping while the fak dogfood fleet runs.
#
# WHY this pairs with the launchd KeepAlive (com.fak.serve-gateway.plist):
#   - launchd KeepAlive=true restarts the `fak serve` PROCESS if it crashes/exits.
#   - It does NOT stop the OS from sleeping the whole BOX. An M-series mac sleeps
#     ~every 25 min when idle, which drops the node (and the gateway) off the
#     fleet even though the process is "alive". caffeinate is the missing half:
#     it holds a power-assertion so macOS never idle-sleeps while the fleet runs.
#   Together: caffeinate keeps the machine awake; KeepAlive keeps the daemon up.
#
# Off-LAN reachability is a separate concern: the gateway stays bound to
# 127.0.0.1:8080 and is reached from other nodes over Tailscale (private overlay),
# NOT by exposing it on the LAN. This script only handles wakefulness.
#
# Usage:
#   ./mac_keep_awake.sh            # start (idempotent: no-op if already running)
#   ./mac_keep_awake.sh stop       # release the assertion
#   ./mac_keep_awake.sh status     # report whether the assertion is held
#
# Idempotent by construction: a PID file guards against stacking caffeinate procs.
# Under launchd, prefer `caffeinate -s` (assert only while a parent job runs);
# run interactively this uses `caffeinate -dimsu` to block display+idle+disk sleep.
set -euo pipefail

WD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNDIR="$WD/_watchdog"
PIDFILE="$RUNDIR/caffeinate.pid"
mkdir -p "$RUNDIR"

# caffeinate flags:
#   -d prevent the display sleeping   -i prevent idle system sleep
#   -m prevent disk idle sleep        -s assert only while a parent runs
#   -u declare user activity (resets the idle timer)
# Under launchd ($XPC_SERVICE_NAME set) use the lighter -s assertion; otherwise the
# full interactive set so a logged-in operator's box never naps mid-fleet.
if [ -n "${XPC_SERVICE_NAME:-}" ]; then
  CAFFEINATE_FLAGS="-s"
else
  CAFFEINATE_FLAGS="-dimsu"
fi

is_running() {
  [ -f "$PIDFILE" ] || return 1
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

case "${1:-start}" in
  start)
    if is_running; then
      echo "mac_keep_awake: already holding wake assertion (pid $(cat "$PIDFILE"))"
      exit 0
    fi
    if ! command -v caffeinate >/dev/null 2>&1; then
      echo "mac_keep_awake: caffeinate not found (this is macOS-only); skipping" >&2
      exit 0
    fi
    # shellcheck disable=SC2086
    caffeinate $CAFFEINATE_FLAGS &
    echo "$!" > "$PIDFILE"
    echo "mac_keep_awake: holding wake assertion (caffeinate $CAFFEINATE_FLAGS, pid $!)"
    ;;
  stop)
    if is_running; then
      kill "$(cat "$PIDFILE")" 2>/dev/null || true
      rm -f "$PIDFILE"
      echo "mac_keep_awake: released wake assertion"
    else
      rm -f "$PIDFILE"
      echo "mac_keep_awake: no wake assertion held"
    fi
    ;;
  status)
    if is_running; then
      echo "mac_keep_awake: ACTIVE (pid $(cat "$PIDFILE"))"
    else
      echo "mac_keep_awake: inactive"
    fi
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
