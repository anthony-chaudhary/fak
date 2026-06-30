<#
register_supervisor_watchdog.ps1 — install/remove the OS-level Scheduled Task
that runs fleet_supervisor_watchdog.ps1 every 5 minutes (plus at logon), so the
job-fleet supervisor is kept alive forever with zero human intervention.

The task runs in the CURRENT USER's interactive context (LogonType Interactive)
so the per-account auth/profile environment the workers need is present, and
uses MultipleInstances=IgnoreNew so a tick is a no-op while a supervisor is up.

  .\register_supervisor_watchdog.ps1            # install (default)
  .\register_supervisor_watchdog.ps1 -Action status
  .\register_supervisor_watchdog.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName  = 'FleetSupervisorWatchdog',
  # Default to the sibling watchdog in THIS clone (resolved from $PSScriptRoot) so
  # registering from any checkout schedules that checkout's script. Override with
  # -Watchdog.
  [string]$Watchdog  = '',
  # Seed from FAK_SUPERVISOR_TARGET so the env knob (laptop_dispatch_config.ps1) is
  # captured into the scheduled task at install; the watchdog also honors it at runtime.
  [int]$Target       = $(if ($env:FAK_SUPERVISOR_TARGET) { [int]$env:FAK_SUPERVISOR_TARGET } else { 4 })
)
$ErrorActionPreference = 'Stop'
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Watchdog) { $Watchdog = Join-Path $scriptRoot 'fleet_supervisor_watchdog.ps1' }

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

# install — use schtasks.exe for the trigger: /SC MINUTE /MO 5 is the robust
# Windows idiom for "every 5 minutes, indefinitely" (the ScheduledTasks module's
# RepetitionDuration rejects an unbounded TimeSpan). Runs in the current user's
# interactive context so per-account auth/env is present, but through conhost
# --headless so the tick never flashes a console window.
$tr = "conhost.exe --headless powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Watchdog`" -Target $Target"
schtasks /Create /TN $TaskName /SC MINUTE /MO 5 /TR $tr /RL LIMITED /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /Create failed ($LASTEXITCODE)" }
# kick one run now so the supervisor is owned by the task immediately
schtasks /Run /TN $TaskName 2>$null | Out-Null
Write-Output "installed $TaskName (every 5 min, current-user interactive headless, Target=$Target)"
Write-Output "log: %LOCALAPPDATA%\Fleet\watchdog\watchdog.log (override with FLEET_STATE_DIR)"
