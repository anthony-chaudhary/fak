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
  # Which tick the always-on task runs:
  #   resolve (default) -> issue_resolve_dispatch.py: spawns an ISSUE-resolution
  #     worker on one concrete open issue (cites #N so the close path fires). This
  #     is the arm that moves the open-issue counter on a plan-empty repo.
  #   loop -> issue_dispatch.py: spawns the generic /dos-dispatch-loop worker that
  #     resolves units from the PLAN portfolio (use when the repo ships PLAN-*.md).
  [ValidateSet('resolve','loop')] [string]$Mode = 'resolve',
  # Worker backend (resolve mode only): claude = opus (t1); opencode = glm-5.2 (t2,
  # a separate zai-coding-plan quota pool). Route a lane to opencode to relieve the
  # opus weekly-quota ceiling. Pair with -Lane to dedicate a task to one lane.
  [ValidateSet('claude','opencode')] [string]$Backend = 'claude',
  [string]$Lane = '',
  # Comma-separated lanes to drop from the busiest-pick (e.g. the opus task excludes
  # 'docs' so the glm task owns it). Ignored when -Lane is set.
  [string]$ExcludeLane = '',
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
$tickName = if ($Mode -eq 'resolve') { 'issue_resolve_dispatch.py' } else { 'issue_dispatch.py' }
$tick = Join-Path $Workspace ('tools\' + $tickName)
if (-not (Test-Path $tick)) { throw "$tickName not found at $tick" }

$liveFlag = if ($Live) { ' --live' } else { '' }
# Register the tick to run python.exe DIRECTLY via the ScheduledTasks cmdlets, NOT
# through a `powershell.exe -Command "..."` wrapper. The wrapper was a SILENT-NO-OP
# trap: a standard python install lives under "C:\Program Files\Python3xx\python.exe"
# (a SPACE in the path), and the nested quotes needed to protect it did not survive
# the PowerShell -> schtasks /TR handoff -- the stored -Command truncated at
# "C:\Program", powershell.exe exited 0 without ever launching python, and the task
# logged LastResult=0 while the tick never actually ran. Splitting Execute (the
# program) from Argument (its args) sidesteps ALL nested quoting: Task Scheduler keeps
# the program path in its own field (no quoting needed), and python's own exit code
# becomes the task's LastTaskResult directly (so the `; exit $LASTEXITCODE` shim that
# the old -Command form needed is gone too).
$pyArgs    = "`"$tick`" --workspace `"$Workspace`" --max-workers $MaxWorkers$liveFlag --json"
# --backend / --lane are resolve-tick options only (the loop tick has neither).
if ($Mode -eq 'resolve') {
  if ($Backend -ne 'claude') { $pyArgs += " --backend $Backend" }
  if ($Lane)                 { $pyArgs += " --lane $Lane" }
  if ($ExcludeLane)          { $pyArgs += " --exclude-lane $ExcludeLane" }
}
$taskAction = New-ScheduledTaskAction -Execute $py -Argument $pyArgs -WorkingDirectory $Workspace
$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
               -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null

$runMode = if ($Live) { "LIVE (bounded autonomous spawning, cap=$MaxWorkers)" } else { "DRY-RUN (logs plans, spawns nothing)" }
Write-Output "installed $TaskName -- every $EveryMinutes min, arm=$Mode ($tickName), current-user interactive, $runMode"
Write-Output "check status any time:  python tools\dispatch_status.py"
if (-not $Live) {
  Write-Output "to go live later:  .\tools\register_issue_dispatch.ps1 -Live -Mode $Mode -MaxWorkers $MaxWorkers"
}
