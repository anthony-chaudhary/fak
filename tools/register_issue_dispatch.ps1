<#
register_issue_dispatch.ps1 -- install/remove the OS Scheduled Task that keeps the
DoS-SAFE issue dispatcher always-on.

Unlike register_dos_dispatch_watchdog.ps1 (which respawns the kernel supervisor
`dos loop --enact` and spawns workers with NO host/account preflight), this task
runs ONE guarded issue-resolution tick every few minutes. Each tick:

  * preflights `fak dispatch tick`  -> host guard clean, an account is
    free, AND live workers < cap, else it REFUSES (the no-DoS guarantee: the live
    worker population can never exceed the cap), and
  * pins the switcher-chosen account, then launches at most one lane worker.

SAFE BY DEFAULT: installed WITHOUT -Live, the tick is DRY-RUN -- the task only
LOGS the plan it would run, spawning nothing. Add -Live to actually spawn workers
(an explicit opt-in to autonomous spawning). -MaxWorkers is the operator's outer
ceiling (default 4); the preflight's adaptive cap = min(this, host_cap, seats) is
the real DoS bound, so a loaded box or a depleted account pool throttles below it.
NOTE: an already-installed task keeps the -MaxWorkers it was registered with (the
value is baked into the stored argument list) -- re-run install to pick up a new one.

  .\register_issue_dispatch.ps1                       # install, DRY-RUN (logs plans, spawns nothing)
  .\register_issue_dispatch.ps1 -Live -MaxWorkers 4   # install, LIVE (bounded autonomous spawning)
  .\register_issue_dispatch.ps1 -Action preview        # print the task action without installing
  .\register_issue_dispatch.ps1 -Action status
  .\register_issue_dispatch.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status','preview')] [string]$Action = 'install',
  [string]$TaskName   = 'FleetIssueDispatch',
  [string]$Workspace  = $(Split-Path -Parent $PSScriptRoot),
  [int]$MaxWorkers    = 4,
  [int]$EveryMinutes  = 10,
  # Which tick the always-on task runs:
  #   resolve (default) -> fak dispatch tick: spawns an ISSUE-resolution
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
  # Optional path to a fak binary. If unset, the installer probes ./fak.exe, PATH fak,
  # then falls back to `go run ./cmd/fak` so source-tree installs cannot silently use
  # a stale binary that lacks `fak loop`.
  [string]$FakExe = $env:FAK_BIN,
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

. (Join-Path $PSScriptRoot 'fak_loop_task.ps1')

# Register the tick through `fak loop run` via the ScheduledTasks cmdlets, NOT
# through a `powershell.exe -Command "..."` wrapper. The wrapper was a SILENT-NO-OP
# trap: a standard install path can contain spaces, and the nested quotes needed to
# protect it did not survive the PowerShell -> schtasks /TR handoff. Splitting
# Execute (fak/go) from Argument (fak args + child args) keeps Task Scheduler out of
# nested shell quoting while the loop ledger records the child exit code and duration.
if ($Mode -eq 'resolve') {
  $tickName = 'fak dispatch tick'
  $fakChild = Resolve-FakLoopAction -Workspace $Workspace -FakExe $FakExe
  $childArgs = @($fakChild.Execute)
  $childArgs += [string[]]$fakChild.PrefixArgs
  $childArgs += @('dispatch', 'tick', '--workspace', $Workspace, '--max-workers', [string]$MaxWorkers, '--json')
  if ($Live) { $childArgs += '--live' }
  if ($Backend -ne 'claude') { $childArgs += @('--backend', $Backend) }
  if ($Lane)                 { $childArgs += @('--lane', $Lane) }
  if ($ExcludeLane)          { $childArgs += @('--exclude-lane', $ExcludeLane) }
} else {
  $py = (Get-Command python -ErrorAction SilentlyContinue).Source
  if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
  if (-not $py) { throw "python not found on PATH" }
  $tickName = 'issue_dispatch.py'
  $tick = Join-Path $Workspace ('tools\' + $tickName)
  if (-not (Test-Path $tick)) { throw "$tickName not found at $tick" }
  $childArgs = @($py, $tick, '--workspace', $Workspace, '--max-workers', [string]$MaxWorkers, '--json')
  if ($Live) { $childArgs += '--live' }
}
$wrapperLoop = if ($Mode -eq 'resolve') { "issue-resolve-dispatch/task-scheduler/$Backend" } else { 'issue-dispatch/task-scheduler' }
$taskAction = New-FakLoopScheduledTaskAction -Workspace $Workspace -FakExe $FakExe -LoopId $wrapperLoop -ChildArgs $childArgs

if ($Action -eq 'preview') {
  [ordered]@{
    task              = $TaskName
    mode              = $Mode
    tick              = $tickName
    loop              = $wrapperLoop
    execute           = $taskAction.Execute
    arguments         = $taskAction.Arguments
    working_directory = $taskAction.WorkingDirectory
    child_args        = $childArgs
    live              = [bool]$Live
    max_workers       = $MaxWorkers
    backend           = $Backend
    lane              = $Lane
    exclude_lane      = $ExcludeLane
  } | ConvertTo-Json -Depth 6
  return
}

$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
               -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
# S4U (non-interactive, session 0), NOT Interactive: this tick executes fak/go
# directly, and a console exe launched in the interactive session flashes a console
# window on EVERY trigger — the "random popup windows". S4U runs the tick windowless
# yet still AS THIS USER (same profile/config/oauth), so the headless dispatch and the
# workers it spawns are unaffected. (Same pattern as the FleetHeartbeat/S4U tasks.)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null

$runMode = if ($Live) { "LIVE (bounded autonomous spawning, cap=$MaxWorkers)" } else { "DRY-RUN (logs plans, spawns nothing)" }
Write-Output "installed $TaskName -- every $EveryMinutes min, arm=$Mode ($tickName), current-user S4U, $runMode"
Write-Output "loop ledger:  .fak\loops.jsonl via fak loop run ($wrapperLoop)"
Write-Output "check status any time:  python tools\dispatch_status.py"
if (-not $Live) {
  Write-Output "to go live later:  .\tools\register_issue_dispatch.ps1 -Live -Mode $Mode -MaxWorkers $MaxWorkers"
}
