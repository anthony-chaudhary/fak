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
$tr = "cmd.exe /c `"`"$runner`"`""
schtasks /Create /TN $TaskName /SC MINUTE /MO $IntervalMin /TR $tr /RL LIMITED /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /Create failed ($LASTEXITCODE)" }
schtasks /Change /TN $TaskName /ENABLE | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /Change /ENABLE failed ($LASTEXITCODE)" }
schtasks /Run /TN $TaskName 2>$null | Out-Null
Write-Output "installed $TaskName (every $IntervalMin min, current-user interactive)"
Write-Output "runner: $runner"
Write-Output "command: $tr"
