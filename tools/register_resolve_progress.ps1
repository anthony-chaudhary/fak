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
# actual gh closes. Register python.exe DIRECTLY via the ScheduledTasks cmdlets, NOT
# a `powershell.exe -Command "..."` wrapper: a Program-Files python path has a SPACE,
# and the nested quotes protecting it did not survive the PowerShell -> schtasks /TR
# handoff -- the stored -Command truncated at "C:\Program", powershell exited 0
# without launching python, and the task logged LastResult=0 while the close arm
# never ran (witnessed issues stayed open). Splitting Execute from Argument sidesteps
# the quoting entirely, and python's exit code becomes LastTaskResult directly.
$liveFlag  = if ($Live) { ' --live' } else { '' }
$pyArgs    = "`"$tick`" --workspace `"$Workspace`" --target $Target --close$liveFlag --json"
$taskAction = New-ScheduledTaskAction -Execute $py -Argument $pyArgs -WorkingDirectory $Workspace
$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
               -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
# S4U (non-interactive, session 0), NOT Interactive: this tick executes python.exe
# DIRECTLY, and a console exe launched in the interactive session flashes a console
# window on EVERY trigger — the "random popup windows". S4U runs the tick windowless
# yet still AS THIS USER (same profile/config/oauth), so the headless tick is unaffected.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null

$mode = if ($Live) { "LIVE (closes witnessed issues)" } else { "SNAPSHOT-ONLY (records the curve, closes nothing)" }
Write-Output "installed $TaskName -- every $EveryMinutes min, target $Target, current-user interactive, $mode"
Write-Output "watch the curve:  python -c `"import json;[print(l.strip()) for l in open(r'$Workspace\.dispatch-runs\progress.jsonl')]`"  (or tools\dispatch_status.py)"
if (-not $Live) {
  Write-Output "to close witnessed issues automatically:  .\tools\register_resolve_progress.ps1 -Live -Target $Target"
}
