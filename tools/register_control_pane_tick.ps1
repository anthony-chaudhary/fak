<#
register_control_pane_tick.ps1 - install/remove the one Scheduled Task that runs
the portable fleet control pane tick.

The tick is the durable cross-machine entry point:
  python tools/fleet_control_pane.py tick

That single command refreshes the session registry, invokes the existing
supervisor watchdog, optionally invokes the resume watchdog live, and persists
tools/_registry/control_pane.json plus CONTROL-PANE.txt.

  .\tools\register_control_pane_tick.ps1
  .\tools\register_control_pane_tick.ps1 -LiveResume
  .\tools\register_control_pane_tick.ps1 -Action status
  .\tools\register_control_pane_tick.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName = 'FleetControlPaneTick',
  [string]$Python = 'python',
  [string]$Pane = '',
  [int]$IntervalMin = 5,
  [switch]$LiveResume
)
$ErrorActionPreference = 'Stop'

$toolsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir = Split-Path -Parent $toolsDir
if (-not $Pane) { $Pane = Join-Path $toolsDir 'fleet_control_pane.py' }

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  Write-Output "State=$($t.State) LastRun=$($i.LastRunTime) LastResult=$($i.LastTaskResult) NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'remove') {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Output "removed $TaskName"
  return
}

$liveArg = if ($LiveResume) { ' --live-resume' } else { '' }
$regDir = Join-Path $toolsDir '_registry'
if (-not (Test-Path $regDir)) { New-Item -ItemType Directory -Path $regDir -Force | Out-Null }
$runner = Join-Path $regDir 'control_pane_tick.cmd'
@(
  '@echo off',
  "cd /d `"$repoDir`"",
  "`"$Python`" `"$Pane`" tick$liveArg"
) | Set-Content -Path $runner -Encoding ASCII
# Register-ScheduledTask with an S4U principal + StartWhenAvailable (#3322): the
# migration off the old `schtasks /Create ... /RL LIMITED` Interactive default, which
# did not run at cold boot before logon and skipped any tick missed while the box was
# off. S4U runs windowless in session 0 (no conhost --headless flash-suppression
# needed) yet still AS THIS USER; StartWhenAvailable fires a missed tick on wake.
# Setting an S4U principal needs an ELEVATED shell; the catch below says so.
$act = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument ('/c ""{0}""' -f $runner)
$trg = New-ScheduledTaskTrigger -Once -At (Get-Date) `
  -RepetitionInterval (New-TimeSpan -Minutes $IntervalMin) -RepetitionDuration (New-TimeSpan -Days 3650)
$prin = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -StartWhenAvailable -MultipleInstances IgnoreNew
try {
  Register-ScheduledTask -TaskName $TaskName -Action $act -Trigger $trg `
    -Principal $prin -Settings $set -Force -ErrorAction Stop | Out-Null
} catch {
  throw ("Register-ScheduledTask failed: $($_.Exception.Message). S4U registration needs an elevated shell -- re-run this installer as Administrator (right-click PowerShell -> Run as administrator).")
}
Start-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Out-Null
Write-Output "installed $TaskName (every $IntervalMin min, current-user S4U windowless)"
Write-Output "runner: $runner"
