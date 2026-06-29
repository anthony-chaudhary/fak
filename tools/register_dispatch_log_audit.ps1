<#
register_dispatch_log_audit.ps1 -- install/remove the OS Scheduled Task that runs
the daily dispatch-log-audit (tools/dispatch_log_audit.py): scan .dispatch-runs/
worker logs, classify failure signatures (panic/traceback, hook-failure storm,
OFF_TRUNK storm, auth wall, banner-only no-op), and file the genuinely-new ones as
GitHub issues -- deduped against the open backlog + a persistent seen-ledger, and
hard-capped per run. See #1300.

This is the *failure -> ticket* FEEDER, the complement to #1276's status-doc
(docs/dispatch-status.md SHOWS a dead backend; it does not TRACK novel failures).
Unlike register_issue_dispatch.ps1 (a 10-minute SPAWN tick bounded by a worker
cap), this task fires ONCE A DAY and creates at most -MaxIssues issues; it spawns
no worker, so there is no DoS surface to bound -- the only side effect is
`gh issue create`, gated behind -Enact.

SAFE BY DEFAULT: installed WITHOUT -Enact, the run is DRY-RUN -- it only PRINTS the
issues it would file and writes nothing (not even the seen-ledger). Add -Enact to
actually create issues (the explicit opt-in to the side effect), mirroring the
dispatch tools' dry-run-first contract.

  .\register_dispatch_log_audit.ps1                       # install, DRY-RUN daily (files nothing)
  .\register_dispatch_log_audit.ps1 -Enact -MaxIssues 3   # install, LIVE (files <=3 issues/day)
  .\register_dispatch_log_audit.ps1 -Action status
  .\register_dispatch_log_audit.ps1 -Action remove
  .\register_dispatch_log_audit.ps1 -At 09:30 -Enact      # pick the daily fire time
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName  = 'FleetDispatchLogAudit',
  [string]$Workspace = $(Split-Path -Parent $PSScriptRoot),
  # Daily fire time (local), HH:mm. A morning slot lands fresh failure tickets for
  # the day's triage, just after the idea-scout feeder.
  [string]$At        = '09:30',
  # Hard cap on issues filed per daily run -- the anti-storm bound. The tool also
  # enforces this; passing it here keeps the registered command self-documenting.
  [int]$MaxIssues    = 3,
  # `hook: ... Failed` lines in one session to call a storm (the tool's default is 3).
  [int]$HookMin      = 3,
  # Only scan logs modified within this many hours (0 = every resolve log).
  [int]$LookbackHours = 48,
  # Optional path to a fak binary. If unset, the installer probes ./fak.exe, PATH fak,
  # then falls back to `go run ./cmd/fak` so source-tree installs cannot silently use
  # a stale binary that lacks `fak loop`.
  [string]$FakExe = $env:FAK_BIN,
  [switch]$Enact
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

# install -- resolve python and the daily audit; pick dry-run vs enact.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "python not found on PATH" }
$tick = Join-Path $Workspace 'tools\dispatch_log_audit.py'
if (-not (Test-Path $tick)) { throw "dispatch_log_audit.py not found at $tick" }

# Register the tick through `fak loop run` (Execute = fak/go, Argument = fak args +
# child python args), NOT via a `powershell.exe -Command "..."` wrapper -- the wrapper
# is a SILENT-NO-OP trap: a standard python lives under "C:\Program Files\Python3xx\
# python.exe" (a SPACE in the path), and the nested quotes needed to protect it do not
# survive the PowerShell -> schtasks /TR handoff, so the stored command truncates at
# "C:\Program", exits 0, and the tick never runs while LastResult reads 0. Splitting
# Execute from Argument sidesteps all nested quoting, and the loop ledger records the
# child exit code + duration so a no-op daily run is visible in `fak loop status`.
# (Same wiring as register_idea_scout.ps1.)
$childArgs = @($py, $tick, '--workspace', $Workspace, '--max-issues', [string]$MaxIssues,
               '--hook-min', [string]$HookMin, '--lookback-hours', [string]$LookbackHours, '--json')
if ($Enact) { $childArgs += '--enact' }
$wrapperLoop = 'dispatch-log-audit/task-scheduler'
$taskAction = New-FakLoopScheduledTaskAction -Workspace $Workspace -FakExe $FakExe -LoopId $wrapperLoop -ChildArgs $childArgs
$trigger    = New-ScheduledTaskTrigger -Daily -At $At
# S4U (non-interactive, session 0), NOT Interactive: this tick executes python.exe
# DIRECTLY, and a console exe launched in the interactive session flashes a console
# window on EVERY trigger -- the "random popup windows". S4U runs the tick windowless
# yet still AS THIS USER (same profile/config/oauth), so the headless tick is unaffected.
$principal  = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$settings   = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
                -Principal $principal -Settings $settings -Force | Out-Null

$runMode = if ($Enact) { "ENACT (files <=$MaxIssues issues/day)" } else { "DRY-RUN (logs the plan, files nothing)" }
Write-Output "installed $TaskName -- daily at $At, current-user S4U, $runMode"
Write-Output "loop ledger:  .fak\loops.jsonl via fak loop run ($wrapperLoop)"
Write-Output "run it once now (dry-run):  python tools\dispatch_log_audit.py"
if (-not $Enact) {
  Write-Output "to go live later:  .\tools\register_dispatch_log_audit.ps1 -Enact -MaxIssues $MaxIssues -At $At"
}
