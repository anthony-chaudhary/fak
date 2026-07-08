<#
register_dispatch_session_audit.ps1 -- install/remove the OS Scheduled Task that
runs the daily dispatch SESSION audit (`fak dispatch audit`): fold the
.dispatch-runs/ worker sessions into a per-worker OUTCOME classification
(SHIPPED / WASTED_SPAWN / QUOTA_WALLED / RETRY_STORM / NO_OP / ERRORED) + a
per-backend wasted-spawn / wasted-wall-clock rollup, ALSO scan the raw logs for
failure signatures (panic/traceback, hook storm, off-trunk storm, auth wall,
banner-only-noop), and file the genuinely-new findings as GitHub issues --
deduped against a per-runs-dir marker ledger AND the open backlog, ordered
worst-first (signature failures interleaved by severity, never below NO_OP
noise), and hard-capped per run. Spine for epic #3327
(goal 2: automatic processes that separately analyze the fleet and file
improvement tickets).

This is the Go feeder, and it is a SUPERSET of the Python log-signature feeder
(tools/dispatch_log_audit.py + tools/register_dispatch_log_audit.ps1 /
FleetDispatchLogAudit, #1300) -- NOT a disjoint complement. `fak dispatch audit`
does BOTH: (1) classifies each session's economic OUTCOME (was the spawn wasted?
did it hit a provider wall? did it storm?) via Fold(), AND (2) re-runs the SAME
log-signature detectors the Python tool greps for -- panic/traceback, hook
storm, off-trunk storm, auth wall, banner-only-noop -- via ScanDirSignatures
(#3337). The signature keys are Python-parity (cross-tool-stable) precisely so
this feeder dedups against an issue the Python tool already filed; it is the
transition step TOWARD the Go tool superseding the Python one, not a second
disjoint lens. OPERATOR NOTE: do not leave BOTH live long-term -- once this is
enacted, retire the FleetDispatchLogAudit task, or the two will race on the same
signatures (they keep SEPARATE dedup ledgers, and issue titles carry a per-tool
"(N×)" count, so title-dedup alone will not reliably suppress the cross-tool
duplicate -- the stable signature fingerprint is the only reliable seam).

WHY A HOST SCHEDULED TASK (not GitHub Actions): .dispatch-runs/ is host-local and
gitignored -- the real session history lives on the fleet host, not in a CI
checkout -- so the analyzer must run WHERE THE DATA IS, exactly as the sibling
log-audit feeder does. #3335 (the Actions variant) tracks a future feed-based
cross-host path once a committed session feed exists; this task is the working
spine today.

It fires ONCE A DAY and creates at most -MaxIssues issues; it spawns no worker, so
there is no DoS surface to bound -- the only side effect is `gh issue create`,
gated behind -Enact. Dedup (marker + open-title) means a steady failure files
exactly ONE open issue; the worst-first -MaxIssues cap bounds the FIRST pass over
a large historical backlog.

SAFE BY DEFAULT: installed WITHOUT -Enact, the daily run is READ-ONLY -- `fak
dispatch audit` renders the rollup table and writes nothing (no --file-issues, no
markers). Add -Enact to pass --file-issues (the explicit opt-in to the side
effect), mirroring the dispatch tools' dry-run-first contract.

  .\register_dispatch_session_audit.ps1                       # install, READ-ONLY daily (files nothing)
  .\register_dispatch_session_audit.ps1 -Enact -MaxIssues 5   # install, LIVE (files <=5 issues/day, worst-first)
  .\register_dispatch_session_audit.ps1 -Action status
  .\register_dispatch_session_audit.ps1 -Action remove
  .\register_dispatch_session_audit.ps1 -At 09:50 -Enact      # pick the daily fire time
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName  = 'FleetDispatchSessionAudit',
  [string]$Workspace = $(Split-Path -Parent $PSScriptRoot),
  # Daily fire time (local), HH:mm. Staggered 20 min after FleetDispatchLogAudit
  # (09:30) so the two dispatch feeders do not contend on the runs dir or the gh API.
  [string]$At        = '09:50',
  # Hard cap on issues filed per daily run -- the anti-storm bound, applied
  # worst-first. `fak dispatch audit` also enforces it (--max-issues); passing it
  # here keeps the registered command self-documenting.
  [int]$MaxIssues    = 5,
  # Directory of dispatch worker logs. Defaults to the workspace's .dispatch-runs/.
  [string]$RunsDir   = '',
  # Optional path to a fak binary. If unset, the installer probes ./fak.exe, PATH fak,
  # then falls back to `go run ./cmd/fak` so a source-tree install cannot silently use
  # a stale binary that lacks `fak dispatch audit`.
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

# install -- resolve the runs dir and the fak child command; pick read-only vs enact.
if (-not $RunsDir) { $RunsDir = Join-Path $Workspace '.dispatch-runs' }

# The CHILD is fak itself (`fak dispatch audit ...`), run UNDER `fak loop run` so the
# loop ledger records the child exit code + duration -- a no-op daily run is then
# visible in `fak loop status`, the same instrumentation the log-audit sibling gets
# around its python child. Resolve the child fak the same way the wrapper does
# (repo fak.exe / PATH fak / `go run ./cmd/fak`) so both halves use one binary.
$childFak  = Resolve-FakLoopAction -Workspace $Workspace -FakExe $FakExe
$childArgs = @($childFak.Execute) + [string[]]$childFak.PrefixArgs + @(
               'dispatch','audit','--runs-dir', $RunsDir, '--max-issues', [string]$MaxIssues)
# READ-ONLY by default; -Enact adds the side-effecting arm.
if ($Enact) { $childArgs += '--file-issues' }

$wrapperLoop = 'dispatch-session-audit/task-scheduler'
$taskAction = New-FakLoopScheduledTaskAction -Workspace $Workspace -FakExe $FakExe -LoopId $wrapperLoop -ChildArgs $childArgs
$trigger    = New-ScheduledTaskTrigger -Daily -At $At
# S4U (non-interactive, session 0), NOT Interactive: a console exe launched in the
# interactive session flashes a window on every trigger. S4U runs windowless yet
# still AS THIS USER (same profile/oauth), so the headless tick is unaffected.
$principal  = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$settings   = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
                -Principal $principal -Settings $settings -Force | Out-Null

$runMode = if ($Enact) { "ENACT (files <=$MaxIssues issues/day, worst-first)" } else { "READ-ONLY (renders the rollup, files nothing)" }
Write-Output "installed $TaskName -- daily at $At, current-user S4U, $runMode"
Write-Output "runs dir:     $RunsDir"
Write-Output "loop ledger:  .fak\loops.jsonl via fak loop run ($wrapperLoop)"
Write-Output "run it once now (read-only):  fak dispatch audit --runs-dir `"$RunsDir`""
if (-not $Enact) {
  Write-Output "to go live later:  .\tools\register_dispatch_session_audit.ps1 -Enact -MaxIssues $MaxIssues -At $At"
}
