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

# Same schtasks /TR quoting discipline as register_resolve_progress / register_issue_dispatch:
# the Program-Files python path's SPACE must survive both PowerShell's parser and
# schtasks' /TR parser. So $py/$tick/$Workspace are single-quoted for PowerShell (the
# call operator & needs the exe quoted) and the whole -Command payload is wrapped in
# \"...\" so schtasks does not truncate at the first inner quote. End with
# `; exit $LASTEXITCODE` so the task's LastTaskResult reflects PYTHON's exit code, not
# powershell.exe's host status (which flaps to 1 on a non-terminating warning).
$inner = "& '$py' '$tick' --workspace '$Workspace' --md '$DocPath' --json; exit `$LASTEXITCODE"
$tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command \`"$inner\`""

schtasks /Create /TN $TaskName /SC MINUTE /MO $EveryMinutes /TR $tr /RL LIMITED /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /Create failed ($LASTEXITCODE)" }

Write-Output "installed $TaskName -- every $EveryMinutes min, renders $DocPath (read-only fold; commits nothing)"
Write-Output "read the doc:   $Workspace\$DocPath"
Write-Output "commit it by path when ready:   git commit -s -- $DocPath"
