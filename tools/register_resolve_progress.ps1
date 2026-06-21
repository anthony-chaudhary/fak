<#
register_resolve_progress.ps1 -- install/remove the OS Scheduled Task that runs
the issue-resolve PROGRESS + CLOSE tick on a cadence (the harvesting watchdog).

While register_issue_dispatch.ps1 keeps a worker SPAWNING (it produces #N-citing
commits), this task keeps the loop CLOSING: every few minutes it runs
tools/issue_resolve_progress.py, which snapshots the trajectory toward the target
and -- when --close --live is set -- drives every OPEN_WITNESSED issue to CLOSED
via the witnessed close arm (each close re-verified per-SHA by dos commit-audit).
Together the two tasks are the always-on "solve issues and keep going" loop:
spawn -> ship #N commit -> witness -> close, unattended.

It is read-only/idempotent and DoS-free: no worker is spawned here, only `gh
issue close` on issues a git-ancestry witness already proved resolved. Closing is
trivially reversible (gh issue reopen) and every close cites its witnessing SHA.

SAFE BY DEFAULT: installed WITHOUT -Live, the tick only SNAPSHOTS (records the
progress curve, closes nothing). Add -Live to actually close witnessed issues.

  .\register_resolve_progress.ps1 -Workspace C:\work\fak                 # snapshot-only
  .\register_resolve_progress.ps1 -Workspace C:\work\fak -Live -Target 50  # snapshot + close
  .\register_resolve_progress.ps1 -Action status
  .\register_resolve_progress.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName    = 'FleetResolveProgress',
  [string]$Workspace   = $(Split-Path -Parent $PSScriptRoot),
  [int]$Target         = 50,
  [int]$EveryMinutes   = 15,
  [switch]$Live
)
$ErrorActionPreference = 'Stop'

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  Write-Output "State=$($t.State)  LastRun=$($i.LastRunTime)  LastResult=$($i.LastTaskResult)  NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'remove') {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Output "removed $TaskName"
  return
}

# install -- resolve python and the tick; pick snapshot-only vs live-close.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "python not found on PATH" }
$tick = Join-Path $Workspace 'tools\issue_resolve_progress.py'
if (-not (Test-Path $tick)) { throw "issue_resolve_progress.py not found at $tick" }

# --close always (so the snapshot also reports closeable count); --live gates the
# actual gh closes. Same schtasks /TR quoting discipline as register_issue_dispatch:
# the Program-Files python path's space must survive both parsers, so $py/$tick/
# $Workspace are single-quoted for PowerShell and the whole -Command payload is
# wrapped in \"...\" so schtasks does not truncate at the first inner quote.
$liveFlag = if ($Live) { ' --live' } else { '' }
# End with `; exit $LASTEXITCODE` so the task's LastTaskResult reflects PYTHON's
# exit code, not powershell.exe's own status. Without it, `-Command` returns the
# host's status (which flaps to 1 on a non-terminating warning) and the task looks
# failed even when the tick succeeded (exit 0) — the LastResult=1 red herring.
$inner = "& '$py' '$tick' --workspace '$Workspace' --target $Target --close$liveFlag --json; exit `$LASTEXITCODE"
$tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command \`"$inner\`""

schtasks /Create /TN $TaskName /SC MINUTE /MO $EveryMinutes /TR $tr /RL LIMITED /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /Create failed ($LASTEXITCODE)" }

$mode = if ($Live) { "LIVE (closes witnessed issues)" } else { "SNAPSHOT-ONLY (records the curve, closes nothing)" }
Write-Output "installed $TaskName -- every $EveryMinutes min, target $Target, current-user interactive, $mode"
Write-Output "watch the curve:  python -c `"import json;[print(l.strip()) for l in open(r'$Workspace\.dispatch-runs\progress.jsonl')]`"  (or tools\dispatch_status.py)"
if (-not $Live) {
  Write-Output "to close witnessed issues automatically:  .\tools\register_resolve_progress.ps1 -Live -Target $Target"
}
