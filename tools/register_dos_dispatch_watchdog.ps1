<#
register_dos_dispatch_watchdog.ps1 -- install/remove the OS-level Scheduled Task
that runs fleet_dos_dispatch_watchdog.ps1 every 5 minutes, so FLEET's own
generic-DOS dispatch supervisor (`dos loop --enact --target N`) is kept alive
forever with zero human intervention.

This is the fleet counterpart to register_supervisor_watchdog.ps1 (which keeps
the SIBLING C:\work\job supervisor alive). Distinct TaskName so the two never
collide.

The task is registered with `schtasks /Create ... /RL LIMITED` — the Interactive
logon type. NOTE (owed migration): that is the SAME logon type behind the
2026-07-09 `0x800710E0` fleet outage (see register_resume_watchdog.ps1's header) —
an Interactive task must launch into the attached desktop and is refused on a
headless/RDP box. The correct target is an S4U principal, which runs windowless in
session 0 yet still AS THIS USER (same profile / Claude auth / CLAUDE_CONFIG_DIR
the dispatch workers need), so S4U does NOT lose the per-account auth this header
used to claim Interactive was required for. This installer still owes that S4U
migration that register_resume_watchdog.ps1 and register_issue_dispatch.ps1
already made; until then a no-flag reinstall re-registers Interactive.

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
  [int]$Target       = 4,
  [int]$Interval     = 120
)
$ErrorActionPreference = 'Stop'
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Watchdog) { $Watchdog = Join-Path $scriptRoot 'fleet_dos_dispatch_watchdog.ps1' }

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

# install -- use schtasks.exe for the trigger: /SC MINUTE /MO 5 is the robust
# Windows idiom for "every 5 minutes, indefinitely" (the ScheduledTasks module's
# RepetitionDuration rejects an unbounded TimeSpan). Runs in the current user's
# interactive context so per-account Claude auth/env is present, but through
# conhost --headless so the tick never flashes a console window.
$tr = "conhost.exe --headless powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Watchdog`" -Target $Target -Interval $Interval"
schtasks /Create /TN $TaskName /SC MINUTE /MO 5 /TR $tr /RL LIMITED /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /Create failed ($LASTEXITCODE)" }
# kick one run now so the fleet supervisor is owned by the task immediately
schtasks /Run /TN $TaskName 2>$null | Out-Null
Write-Output "installed $TaskName (every 5 min, current-user interactive headless, Target=$Target Interval=$Interval)"
Write-Output "log: %LOCALAPPDATA%\Fleet\dos-dispatch-watchdog\dos-dispatch-watchdog.log (override with FLEET_STATE_DIR)"
