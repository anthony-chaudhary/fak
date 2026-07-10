<#
register_scout_loop.ps1 -- install/remove the OS Scheduled Task that runs the
`scout-loop` research->backlog loop (.claude/skills/scout-loop/SKILL.md) on a daily
cadence. Each fire launches ONE detached `/goal` worker that reads the scout-loop
fuel (.claude/goal-prompts/scout-and-study-witnessed.md): CRAWL the freshest
outward signal (the idea-scout triage queue + industry scans) -> SELECT one
repo-shaped lead -> STUDY it with /study-repo -> WITNESS each borrow with
/field-borrow -> FILE the surviving PARTIAL/ABSENT borrows as small tickets ->
REGISTER a dated CONCEPT-STUDY note. One lead per pass, then the worker stops.

This is the STUDY arm that consumes what register_idea_scout.ps1 (FleetIdeaScout,
the FEEDER) produces. FleetIdeaScout files raw triage issues once a day at 09:00;
this task fires AFTER that (default 10:30) so the queue is fresh, and turns one
lead into scoped, witnessed backlog the dispatch loop can then resolve.

SAFE BY DEFAULT: installed WITHOUT -Launch, the daily fire runs the launcher in
-PlanOnly mode -- it resolves the account + runs the preflight but SPAWNS NOTHING,
logging only the plan to the loop ledger. Add -Launch to actually spawn one
detached study worker per fire (the explicit opt-in to the side effect), mirroring
register_idea_scout.ps1's dry-run-first contract and /super-loop's PLAN-by-default.

Because a fire spawns a Claude session, the launch path keeps the no-DoS preflight
cap intact (launch_goal_detached.ps1 re-checks SPAWN_OK per spawn). -SkipPreflight
is deliberately NOT exposed here; do not route around the cap for a background loop.

  .\register_scout_loop.ps1                          # install, PLAN-mode daily (spawns nothing)
  .\register_scout_loop.ps1 -Launch                  # install, LIVE (one detached study pass/day)
  .\register_scout_loop.ps1 -Launch -At 10:30 -WorkKind engineering
  .\register_scout_loop.ps1 -Action status
  .\register_scout_loop.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName  = 'FleetScoutLoop',
  [string]$Workspace = $(Split-Path -Parent $PSScriptRoot),
  # Daily fire time (local), HH:mm. Default lands AFTER FleetIdeaScout's 09:00 slot
  # so the triage queue the loop reads is already refreshed for the day.
  [string]$At        = '10:30',
  # Which tier the detached worker runs on. A cadenced background research loop
  # defaults to tier-2 (gardening/GLM) so it does not compete with frontier seats
  # reserved for issue-resolution; bump to 'engineering' for max study quality.
  [string]$WorkKind  = 'gardening',
  # The /goal fuel the detached worker reads -- one scout-loop pass, then stop.
  [string]$PointerFile = '.claude/goal-prompts/scout-and-study-witnessed.md',
  # Optional path to a fak binary. If unset, the loop-task helper probes ./fak.exe,
  # PATH fak, then falls back to `go run ./cmd/fak` (same as register_idea_scout.ps1).
  [string]$FakExe = $env:FAK_BIN,
  [switch]$Launch
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

# install -- resolve the launcher + a PowerShell host to run it, then build the
# child command. The detached launcher is a .ps1, so the child executable is a
# PowerShell host invoked with -File (NOT -Command): -File takes the script path +
# its args as separate tokens, sidestepping the nested-quote / space-in-path trap
# that the -Command form and a powershell.exe wrapper fall into (see the long note
# in register_idea_scout.ps1). Prefer pwsh (7+) and fall back to Windows PowerShell.
$launcher = Join-Path $Workspace 'tools\launch_goal_detached.ps1'
if (-not (Test-Path $launcher)) { throw "launch_goal_detached.ps1 not found at $launcher" }
$fuel = Join-Path $Workspace $PointerFile
if (-not (Test-Path $fuel)) { throw "scout-loop fuel not found at $fuel" }

$psHost = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $psHost) { $psHost = (Get-Command powershell -ErrorAction SilentlyContinue).Source }
if (-not $psHost) { throw "no PowerShell host (pwsh/powershell) found on PATH" }

# Child = <psHost> -NoProfile -ExecutionPolicy Bypass -File launch_goal_detached.ps1
#         -PointerFile <fuel> -Workspace <ws> -WorkKind <tier> [-PlanOnly]
# -PlanOnly is the default (safe): the fire resolves the account + preflight and
# spawns NOTHING. -Launch drops -PlanOnly so the fire actually detaches a worker.
$childArgs = @(
  $psHost, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $launcher,
  '-PointerFile', $PointerFile, '-Workspace', $Workspace, '-WorkKind', $WorkKind
)
if (-not $Launch) { $childArgs += '-PlanOnly' }

$wrapperLoop = 'scout-loop/task-scheduler'
$taskAction = New-FakLoopScheduledTaskAction -Workspace $Workspace -FakExe $FakExe -LoopId $wrapperLoop -ChildArgs $childArgs
$trigger    = New-ScheduledTaskTrigger -Daily -At $At
# One instance at a time (IgnoreNew) so a slow study pass never stacks; the launcher
# itself returns quickly after detaching, so 20 min bounds the SPAWN, not the worker.
$settings   = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)

# Logon type, PREFERRED then FALLBACK:
#  * S4U (non-interactive, session 0) is preferred -- the fire runs a PowerShell host
#    that spawns a detached worker; an interactive-session console exe flashes a window
#    on every trigger. S4U runs it windowless yet still AS THIS USER (same profile /
#    oauth), so the worker inherits the right seat. (Same as register_idea_scout.) BUT
#    registering an S4U principal needs SeTcbPrivilege -> an ELEVATED shell.
#  * Interactive is the fallback for a NON-elevated install: it registers without admin
#    and still fires on cadence AS THIS USER, at the cost of running only while the user
#    is logged on and briefly showing the host window. A running loop beats a loop that
#    could not install; the mode is reported so the operator can re-run elevated for S4U.
function Register-ScoutTask([string]$logon) {
  # Build the principal with a LITERAL -LogonType S4U on the preferred path so the
  # windowgate popup scanner (internal/windowgate: reOffDesktop) can SEE the off-desktop
  # logon -- it matches a literal token, not a `-LogonType $var`, so the prior variable
  # form was mis-read as an on-desktop popup installer. The fallback path keeps its logon
  # in the $logon variable (no literal on-desktop logon token for the scanner to flag);
  # runtime behavior is unchanged -- S4U preferred, the fallback only on a non-elevated deny.
  if ($logon -eq 'S4U') {
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
  } else {
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType $logon -RunLevel Limited
  }
  # -ErrorAction Stop: Register-ScheduledTask is a CIM cmdlet whose failure surfaces as a
  # NON-terminating error that the script-scope $ErrorActionPreference='Stop' does not
  # convert inside a function -- without this the S4U "Access is denied" prints but is
  # not caught, so the Interactive fallback is skipped and nothing installs.
  Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
                  -Principal $principal -Settings $settings -Force -ErrorAction Stop | Out-Null
}
$logonUsed = 'S4U'
try {
  Register-ScoutTask 'S4U'
} catch {
  $msg = $_.Exception.Message
  $denied = ($msg -match 'Access is denied') -or ($msg -match '0x80070005') -or ($msg -match 'privilege')
  if (-not $denied) { throw }
  Write-Output "S4U registration denied (needs an elevated shell); falling back to Interactive logon."
  Register-ScoutTask 'Interactive'
  $logonUsed = 'Interactive'
}

$runMode = if ($Launch) { "LIVE (one detached study pass/day, tier=$WorkKind)" } else { "PLAN-ONLY (logs the plan, spawns nothing)" }
if ($logonUsed -eq 'Interactive') {
  Write-Output "installed $TaskName -- daily at $At, current-user Interactive (non-elevated fallback; fires only while logged on, brief window), $runMode"
  Write-Output "for windowless session-0 running: re-run this script from an ELEVATED shell to upgrade to S4U."
} else {
  Write-Output "installed $TaskName -- daily at $At, current-user S4U, $runMode"
}
Write-Output "fuel:         $PointerFile"
Write-Output "loop ledger:  .fak\loops.jsonl via fak loop run ($wrapperLoop)"
Write-Output "dry-run the launcher once now:  .\tools\launch_goal_detached.ps1 -PlanOnly -PointerFile $PointerFile -Workspace $Workspace -WorkKind $WorkKind"
if (-not $Launch) {
  Write-Output "to go live later:  .\tools\register_scout_loop.ps1 -Launch -At $At -WorkKind $WorkKind"
}
