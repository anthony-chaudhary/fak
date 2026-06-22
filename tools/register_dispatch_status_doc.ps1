<#
register_dispatch_status_doc.ps1 -- install/remove the OS Scheduled Task that keeps
the committed issue-dispatch STATUS DOC fresh (docs/dispatch-status.md).

The spawn arm (FleetIssueDispatch) produces #N commits and the close arm
(FleetResolveProgress) drives OPEN_WITNESSED issues to CLOSED -- but every signal
lives in gitignored runtime (.dispatch-runs/progress.jsonl). This task renders the
one human-readable surface an operator opens to see WHICH issues are synced to WHICH
lanes, how closure is progressing, and any worker that spawned and produced nothing.

It runs tools/dispatch_status.py --md docs/dispatch-status.md every few minutes. The
tool is a pure read-only FOLD over the existing sub-tools (preflight, lane router,
closure audit) plus a pure-local scan of .dispatch-runs for 0-byte worker logs. It
launches NO worker and is DoS-free.

IMPORTANT: this task only WRITES the working-tree doc; it does NOT git-commit it.
The repo is a shared multi-session tree where commits are by explicit path only --
automating `git add`/`git commit` here would steal a sibling session's in-flight
files. An operator (or a session) commits docs/dispatch-status.md by path when ready;
the task just keeps the working copy current between commits.

  .\register_dispatch_status_doc.ps1 -Workspace C:\work\fak          # install (every 30 min)
  .\register_dispatch_status_doc.ps1 -Workspace C:\work\fak -EveryMinutes 15
  .\register_dispatch_status_doc.ps1 -Action status
  .\register_dispatch_status_doc.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName    = 'FleetDispatchStatusDoc',
  [string]$Workspace   = $(Split-Path -Parent $PSScriptRoot),
  [string]$DocPath     = 'docs\dispatch-status.md',
  [int]$EveryMinutes   = 30
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

# install -- resolve python and the doc-render tick.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "python not found on PATH" }
$tick = Join-Path $Workspace 'tools\dispatch_status.py'
if (-not (Test-Path $tick)) { throw "dispatch_status.py not found at $tick" }

# Register python.exe DIRECTLY via the ScheduledTasks cmdlets, NOT a
# `powershell.exe -Command "..."` wrapper (same fix as register_resolve_progress /
# register_issue_dispatch): a Program-Files python path has a SPACE, and the nested
# quotes protecting it did not survive the PowerShell -> schtasks /TR handoff -- the
# stored -Command truncated at "C:\Program", powershell exited 0 without launching
# python, and the task logged LastResult=0 while the doc was never re-rendered (it
# went stale while every run reported success). Splitting Execute from Argument
# sidesteps the quoting; WorkingDirectory anchors the relative --md path, and python's
# exit code becomes LastTaskResult directly.
$pyArgs    = "`"$tick`" --workspace `"$Workspace`" --md `"$DocPath`" --json"
$taskAction = New-ScheduledTaskAction -Execute $py -Argument $pyArgs -WorkingDirectory $Workspace
$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
               -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null

Write-Output "installed $TaskName -- every $EveryMinutes min, renders $DocPath (read-only fold; commits nothing)"
Write-Output "read the doc:   $Workspace\$DocPath"
Write-Output "commit it by path when ready:   git commit -s -- $DocPath"
