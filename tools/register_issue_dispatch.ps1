<#
register_issue_dispatch.ps1 -- install/remove the OS Scheduled Task that keeps the
DoS-SAFE issue dispatcher always-on.

Unlike register_dos_dispatch_watchdog.ps1 (which respawns the kernel supervisor
`dos loop --enact` and spawns workers with NO host/account preflight), this task
runs ONE guarded tick of tools/issue_dispatch.py every few minutes. Each tick:

  * preflights tools/dispatch_preflight.py  -> host guard clean, an account is
    free, AND live workers < cap, else it REFUSES (the no-DoS guarantee: the live
    worker population can never exceed the cap), and
  * pins the switcher-chosen account, then launches at most one lane worker.

SAFE BY DEFAULT: installed WITHOUT -Live, the tick is DRY-RUN -- the task only
LOGS the plan it would run, spawning nothing. Add -Live to actually spawn workers
(an explicit opt-in to autonomous spawning). -MaxWorkers is the hard ceiling the
preflight enforces (default 2).

  .\register_issue_dispatch.ps1                       # install, DRY-RUN (logs plans, spawns nothing)
  .\register_issue_dispatch.ps1 -Live -MaxWorkers 2   # install, LIVE (bounded autonomous spawning)
  .\register_issue_dispatch.ps1 -Action status
  .\register_issue_dispatch.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName   = 'FleetIssueDispatch',
  [string]$Workspace  = $(Split-Path -Parent $PSScriptRoot),
  [int]$MaxWorkers    = 2,
  [int]$EveryMinutes  = 10,
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

# install -- resolve python and the guarded tick; pick dry-run vs live.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "python not found on PATH" }
$tick = Join-Path $Workspace 'tools\issue_dispatch.py'
if (-not (Test-Path $tick)) { throw "issue_dispatch.py not found at $tick" }

$liveFlag = if ($Live) { ' --live' } else { '' }
$inner = "& '$py' '$tick' --workspace '$Workspace' --max-workers $MaxWorkers$liveFlag --json"
# Wrap in powershell so the JSON plan lands in the task log for the status command.
$tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command `"$inner`""

schtasks /Create /TN $TaskName /SC MINUTE /MO $EveryMinutes /TR $tr /RL LIMITED /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /Create failed ($LASTEXITCODE)" }

$mode = if ($Live) { "LIVE (bounded autonomous spawning, cap=$MaxWorkers)" } else { "DRY-RUN (logs plans, spawns nothing)" }
Write-Output "installed $TaskName -- every $EveryMinutes min, current-user interactive, $mode"
Write-Output "check status any time:  python tools\dispatch_status.py"
if (-not $Live) {
  Write-Output "to go live later:  .\tools\register_issue_dispatch.ps1 -Live -MaxWorkers $MaxWorkers"
}
