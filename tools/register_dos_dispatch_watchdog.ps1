<#
register_dos_dispatch_watchdog.ps1 -- install/remove the OS-level Scheduled Task
that runs fleet_dos_dispatch_watchdog.ps1 every 5 minutes, so FLEET's own
generic-DOS dispatch supervisor (`dos loop --enact --target N`) is kept alive
forever with zero human intervention.

This is the fleet counterpart to register_supervisor_watchdog.ps1 (which keeps
the SIBLING C:\work\job supervisor alive). Distinct TaskName so the two never
collide.

The task is registered S4U (#3322): `Register-ScheduledTask` with an S4U principal
(`-LogonType S4U -RunLevel Limited`) + `-StartWhenAvailable`. That is the migration
off the old `schtasks /Create ... /RL LIMITED` Interactive default behind the
2026-07-09 `0x800710E0` fleet outage (see register_resume_watchdog.ps1's header) —
an Interactive task must launch into the attached desktop and is refused on a
headless/RDP box, and skips any tick missed while the box slept. S4U runs windowless
in session 0 yet still AS THIS USER (same profile / Claude auth / CLAUDE_CONFIG_DIR
the dispatch workers need), so it does NOT lose the per-account auth, and
StartWhenAvailable fires a missed tick on wake so the loop survives a cold boot.
Setting an S4U principal needs an ELEVATED shell (non-elevated Register-ScheduledTask
-> "Access is denied"); the catch below says so.

  .\register_dos_dispatch_watchdog.ps1            # install (default)
  .\register_dos_dispatch_watchdog.ps1 -Action status
  .\register_dos_dispatch_watchdog.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName  = 'FleetDOSDispatchWatchdog',
  # Default to the sibling watchdog in THIS clone (resolved from $PSScriptRoot) so
  # registering from any checkout schedules that checkout's script. Override with
  # -Watchdog.
  [string]$Watchdog  = '',
  # #6502: a population may NOT be baked into the registered task. -1 means "not
  # passed", which is the only admitted value; anything else is refused below with
  # the reason code, and the operator is pointed at the SSOT instead.
  [int]$Target       = -1,
  [int]$Interval     = 120
)
$ErrorActionPreference = 'Stop'
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Watchdog) { $Watchdog = Join-Path $scriptRoot 'fleet_dos_dispatch_watchdog.ps1' }
. (Join-Path $scriptRoot 'fleet_target.ps1')
# Registration applies the same rule the status audit does: a task that embeds a
# population is
# refused, so re-running an installer cannot recreate the defect #6502 cleaned up.
if ($Target -ge 0) {
  throw ("EMBEDDED_WATCHDOG_TARGET: refusing to register $TaskName with -Target $Target baked into its arguments. " +
    "Task Scheduler replays a baked-in population forever, which is how a target-8 tick could undo an operator's " +
    "target-0 quarantine. Declare the population once instead: .\fleet_target.ps1 -Set $Target -Reason '...'")
}

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  foreach ($taskAction in @($t.Actions)) {
    Assert-FleetTaskHasNoEmbeddedTarget -Arguments "$($taskAction.Arguments)" -TaskName $TaskName | Out-Null
  }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  Write-Output "State=$($t.State)  LastRun=$($i.LastRunTime)  LastResult=$($i.LastTaskResult)  NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'remove') {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Output "removed $TaskName"
  return
}

# install -- Register-ScheduledTask with an S4U principal + StartWhenAvailable
# (#3322). A -Once trigger repeating every 5 min over a 10-year duration is the
# ScheduledTasks-module idiom for "every 5 minutes, indefinitely" (the module
# rejects a truly-unbounded RepetitionDuration). S4U runs windowless in session 0,
# so the old conhost --headless flash-suppression is unnecessary -- powershell.exe
# is launched directly. Same migration register_resume_watchdog.ps1 already made.
# NOTE the missing -Target (#6502): the tick reads the desired population from the
# declared SSOT (fleet_target.ps1). -Interval is a cadence, not a population, so it
# stays.
$act = New-ScheduledTaskAction -Execute 'powershell.exe' `
  -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}" -Interval {1}' -f $Watchdog, $Interval)
$trg = New-ScheduledTaskTrigger -Once -At (Get-Date) `
  -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
$prin = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
# StartWhenAvailable (#3322): a tick missed while the box slept/was off fires on wake.
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -StartWhenAvailable -MultipleInstances IgnoreNew
try {
  Register-ScheduledTask -TaskName $TaskName -Action $act -Trigger $trg `
    -Principal $prin -Settings $set -Force -ErrorAction Stop | Out-Null
} catch {
  throw ("Register-ScheduledTask failed: $($_.Exception.Message). S4U registration needs an elevated shell -- re-run this installer as Administrator (right-click PowerShell -> Run as administrator).")
}
# kick one run now so the fleet supervisor is owned by the task immediately
Start-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Out-Null
Write-Output "installed $TaskName (every 5 min, current-user S4U windowless, Interval=$Interval; population read from the SSOT at tick time)"
Write-Output "log: %LOCALAPPDATA%\Fleet\dos-dispatch-watchdog\dos-dispatch-watchdog.log (override with FLEET_STATE_DIR)"

