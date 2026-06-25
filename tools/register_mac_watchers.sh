#!/usr/bin/env bash
# register_mac_watchers.sh -- install the fleet resume + supervisor watchdogs as
# macOS/Linux user-crontab jobs (the cross-platform analogue of the Windows
# register_resume_watchdog.ps1 / register_supervisor_watchdog.ps1).
#
# Idempotent: rewrites only the marked block, preserving any other crontab lines.
#
#   ./register_mac_watchers.sh            # resume watcher DRY-RUN (safe default)
#   ./register_mac_watchers.sh --live     # resume watcher LIVE (auto-resumes dead sessions)
#
# Env overrides:
#   FAK_SUPERVISOR_ENABLE=1   also arm the supervisor respawn (needs the job repo)
#   PYTHON=/path/to/python3   interpreter for the cron lines (default: `which python3`)
#
# See fleet_resume_watchdog.py / fleet_supervisor_watchdog.py for the per-tick logic.
set -euo pipefail

LIVE=""; RESUME_LABEL="DRY-RUN -- pass --live to auto-resume"
if [ "${1:-}" = "--live" ]; then LIVE="FAK_LIVE=1 "; RESUME_LABEL="LIVE -- actually auto-resumes"; fi

WD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$WD/_watchdog"
PY="${PYTHON:-$(command -v python3)}"
mkdir -p "$LOG"

SUP_ENV=""
[ "${FAK_SUPERVISOR_ENABLE:-}" = "1" ] && SUP_ENV="FAK_SUPERVISOR_ENABLE=1 "

# Dispatch-supervisor keep-alive (issue #566): opt-in, mac/linux analogue of the
# Windows fleet_dos_dispatch_watchdog.ps1. The python watchdog dry-runs unless
# FAK_DISPATCH_ENABLE=1 is on its cron line, so installing without the opt-in is
# a no-op report.
DISP_ENV=""
[ "${FAK_DISPATCH_ENABLE:-}" = "1" ] && DISP_ENV="FAK_DISPATCH_ENABLE=1 "

existing="$(crontab -l 2>/dev/null | sed '/# >>> fak fleet watchers >>>/,/# <<< fak fleet watchers <<</d')"
{
  [ -n "$existing" ] && printf '%s\n' "$existing"
  cat <<EOF
# >>> fak fleet watchers >>>
# Resume watchdog: refresh session registry + ledger-gated resume-once of dead
# AUTO_RESUME sessions. $RESUME_LABEL
*/5 * * * * ${LIVE}/bin/bash $WD/fak_loop_run.sh fleet-resume-watchdog/cron cron -- $PY $WD/fleet_resume_watchdog.py >> $LOG/cron_resume.log 2>&1
# Supervisor watchdog: keep job-fleet supervisor alive. No-op until job repo present
# AND FAK_SUPERVISOR_ENABLE=1 is set on this line.
*/5 * * * * ${SUP_ENV}/bin/bash $WD/fak_loop_run.sh fleet-supervisor-watchdog/cron cron -- $PY $WD/fleet_supervisor_watchdog.py >> $LOG/cron_supervisor.log 2>&1
# Dispatch supervisor watchdog: keep the DOS dispatch supervisor (dos loop --enact)
# alive. Dry-run/no-op unless FAK_DISPATCH_ENABLE=1 is set on this line (#566).
*/5 * * * * ${DISP_ENV}/bin/bash $WD/fak_loop_run.sh fleet-dos-dispatch-watchdog/cron cron -- $PY $WD/fleet_dos_dispatch_watchdog.py >> $LOG/cron_dispatch.log 2>&1
# <<< fak fleet watchers <<<
EOF
} | crontab -

SUP_STATUS=$([ -n "$SUP_ENV" ] && echo ENABLED || echo opt-in)
DISP_STATUS=$([ -n "$DISP_ENV" ] && echo ENABLED || echo opt-in)
echo "installed fak fleet watchers (resume=${RESUME_LABEL%% *}, supervisor=$SUP_STATUS, dispatch=$DISP_STATUS):"
crontab -l | sed -n '/# >>> fak fleet watchers >>>/,/# <<< fak fleet watchers <<</p'
