<#
register_supervisor_watchdog.ps1 — install/remove the OS-level Scheduled Task
that runs fleet_supervisor_watchdog.ps1 every 5 minutes (plus at logon), so the
job-fleet supervisor is kept alive forever with zero human intervention.

The task is registered S4U (#3322): `Register-ScheduledTask` with an S4U principal
(`-LogonType S4U -RunLevel Limited`) + `-StartWhenAvailable`. That is the migration
off the old `schtasks /Create ... /RL LIMITED` Interactive default behind the
2026-07-09 `0x800710E0` fleet outage (see register_resume_watchdog.ps1's header) —
an Interactive task must launch into the attached desktop and is refused on a
headless/RDP box, and skips any tick missed while the box slept. S4U runs windowless
in session 0 yet still AS THIS USER (same profile / auth / CLAUDE_CONFIG_DIR the
workers need), so it does NOT lose the per-account auth, and StartWhenAvailable fires
a missed tick on wake so the loop survives a cold boot. Setting an S4U principal needs
an ELEVATED shell (non-elevated Register-ScheduledTask -> "Access is denied"); the
catch below says so. It uses MultipleInstances=IgnoreNew so a tick is a no-op while a
supervisor is up.

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

# install — Register-ScheduledTask with an S4U principal + StartWhenAvailable
# (#3322). A -Once trigger repeating every 5 min over a 10-year duration is the
# ScheduledTasks-module idiom for "every 5 minutes, indefinitely" (the module
# rejects a truly-unbounded RepetitionDuration). S4U runs windowless in session 0,
# so the old conhost --headless flash-suppression is unnecessary -- powershell.exe
# is launched directly. Same migration register_resume_watchdog.ps1 already made.
$act = New-ScheduledTaskAction -Execute 'powershell.exe' `
  -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}" -Target {1}' -f $Watchdog, $Target)
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
# kick one run now so the supervisor is owned by the task immediately
Start-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Out-Null
Write-Output "installed $TaskName (every 5 min, current-user S4U windowless, Target=$Target)"
Write-Output "log: %LOCALAPPDATA%\Fleet\watchdog\watchdog.log (override with FLEET_STATE_DIR)"
