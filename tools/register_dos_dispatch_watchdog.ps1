<#
register_dos_dispatch_watchdog.ps1 -- install/remove the OS-level Scheduled Task
that runs fleet_dos_dispatch_watchdog.ps1 every 5 minutes, so FLEET's own
generic-DOS dispatch supervisor (`dos loop --enact --target N`) is kept alive
forever with zero human intervention.

This is the fleet counterpart to register_supervisor_watchdog.ps1 (which keeps
the SIBLING C:\work\job supervisor alive). Distinct TaskName so the two never
collide.

The task runs in the CURRENT USER's interactive context (LogonType Interactive,
/RL LIMITED) so the per-account Claude auth / CLAUDE_CONFIG_DIR the dispatch
workers need is present.

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
  [string]$Watchdog  = (Join-Path $PSScriptRoot 'fleet_dos_dispatch_watchdog.ps1'),
  [int]$Target       = 4,
  [int]$Interval     = 120
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

# install -- use schtasks.exe for the trigger: /SC MINUTE /MO 5 is the robust
# Windows idiom for "every 5 minutes, indefinitely" (the ScheduledTasks module's
# RepetitionDuration rejects an unbounded TimeSpan). Runs in the current user's
# interactive context so per-account Claude auth/env is present.
$tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Watchdog`" -Target $Target -Interval $Interval"
schtasks /Create /TN $TaskName /SC MINUTE /MO 5 /TR $tr /RL LIMITED /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /Create failed ($LASTEXITCODE)" }
# kick one run now so the fleet supervisor is owned by the task immediately
schtasks /Run /TN $TaskName 2>$null | Out-Null
Write-Output "installed $TaskName (every 5 min, current-user interactive, Target=$Target Interval=$Interval)"
Write-Output "log: %LOCALAPPDATA%\Fleet\dos-dispatch-watchdog\dos-dispatch-watchdog.log (override with FLEET_STATE_DIR)"
