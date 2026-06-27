#!/usr/bin/env bash
# register_dos_dispatch_watchdog.sh -- install/remove the user-crontab job
# that runs fleet_dos_dispatch_watchdog.py every 5 minutes, so FLEET's own
# generic-DOS dispatch supervisor (`dos loop --enact --target N`) is kept alive
# forever with zero human intervention.
#
# This is the fleet counterpart to register_supervisor_watchdog.sh (which keeps
# the sibling job supervisor alive). Distinct cron comment markers so the two
# never collide.
#
#   ./register_dos_dispatch_watchdog.sh            # install (default)
#   ./register_dos_dispatch_watchdog.sh status     # show installed state
#   ./register_dos_dispatch_watchdog.sh remove     # uninstall
#
# Env overrides:
#   PYTHON=/path/to/python3   interpreter for the cron line (default: `which python3`)
#   FAK_DISPATCH_WATCHDOG_INTERVAL=300  seconds between ticks (default: 300 = 5 min)
#
# The job runs as the current user so per-account Claude auth / CLAUDE_CONFIG_DIR
# the dispatch workers need is present.
set -euo pipefail

WD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG="$WD/fleet_dos_dispatch_watchdog.py"
PY="${PYTHON:-$(command -v python3)}"
LOGDIR="$WD/_watchdog"
LOG="$LOGDIR/dos-dispatch-watchdog.log"
INTERVAL="${FAK_DISPATCH_WATCHDOG_INTERVAL:-300}"
TARGET="${FAK_DISPATCH_WATCHDOG_TARGET:-4}"

COMMENT_START="# >>> DOS dispatch watchdog >>>"
COMMENT_END="# <<< DOS dispatch watchdog <<<"

case "${1:-install}" in
  status)
    if crontab -l 2>/dev/null | grep -q "$COMMENT_START"; then
      echo "INSTALLED (user crontab)"
      crontab -l 2>/dev/null | sed -n "/$COMMENT_START/,/$COMMENT_END/p"
    else
      echo "NOT INSTALLED"
    fi
    exit 0
    ;;
  remove)
    if ! crontab -l 2>/dev/null | grep -q "$COMMENT_START"; then
      echo "NOT INSTALLED (nothing to remove)"
      exit 0
    fi
    new_cron="$(crontab -l 2>/dev/null | sed "/$COMMENT_START/,/$COMMENT_END/d")"
    echo "$new_cron" | crontab -
    echo "removed DOS dispatch watchdog from user crontab"
    exit 0
    ;;
  install)
    ;;
  *)
    echo "Usage: $0 [install|status|remove]" >&2
    exit 1
    ;;
esac

# install -- use cron for the trigger: "*/N * * * *" is the POSIX idiom for
# "every N minutes, indefinitely". Runs in the current user's crontab so
# per-account Claude auth/env is present.
existing="$(crontab -l 2>/dev/null | sed "/$COMMENT_START/,/$COMMENT_END/d")"
{
  [ -n "$existing" ] && printf '%s\n' "$existing"
  cat <<EOF
$COMMENT_START
# Keep the DOS dispatch supervisor (dos loop --enact) alive.
# Remove with: $0 remove
*/$(( INTERVAL / 60 )) * * * * FAK_DISPATCH_ENABLE=1 FAK_DISPATCH_WATCHDOG_TARGET=$TARGET FAK_DISPATCH_WATCHDOG_INTERVAL=$INTERVAL $PY $WATCHDOG --live --workspace "$(cd "$WD/.." && pwd)" --target $TARGET --interval $INTERVAL >> $LOG 2>&1
$COMMENT_END
EOF
} | crontab -

echo "installed DOS dispatch watchdog (every ${INTERVAL}s, Target=$TARGET, log=$LOG)"
echo "status: $0 status"
echo "remove: $0 remove"