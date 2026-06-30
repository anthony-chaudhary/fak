<#
register_resume_watchdog.ps1 - install/remove the Scheduled Task that runs the
cross-account resume watchdog every 10 min (refreshes the on-disk session
registry each tick = "extract in advance", and optionally auto-resumes
autonomous dead sessions).

  .\register_resume_watchdog.ps1                 # install in DRY-RUN (safe default)
  .\register_resume_watchdog.ps1 -Live           # install LIVE (actually auto-resumes)
  .\register_resume_watchdog.ps1 -Action status
  .\register_resume_watchdog.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [switch]$Live,
  [string]$TaskName = 'FleetResumeWatchdog',
  # Default to the sibling watchdog in THIS clone (the watchdog itself resolves
  # its paths from $PSScriptRoot), so registering from any checkout schedules that
  # checkout's script — not a hardcoded operator path. Override with -Watchdog.
  [string]$Watchdog = ''
)
$ErrorActionPreference = 'Stop'
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Watchdog) { $Watchdog = Join-Path $scriptRoot 'fleet_resume_watchdog.ps1' }

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  $a = ($t.Actions | Select-Object -First 1).Arguments
  $modeStr = if ($a -match '-Live') { 'LIVE' } else { 'DRY-RUN' }
  Write-Output "State=$($t.State) mode=$modeStr LastRun=$($i.LastRunTime) LastResult=$($i.LastTaskResult) NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'remove') {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Output "removed $TaskName"; return
}

$liveArg = if ($Live) { ' -Live' } else { '' }
$tr = "conhost.exe --headless powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Watchdog`"$liveArg"
schtasks /Create /TN $TaskName /SC MINUTE /MO 10 /TR $tr /RL LIMITED /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /Create failed ($LASTEXITCODE)" }
$mode = if ($Live) { 'LIVE (auto-resumes)' } else { 'DRY-RUN (logs intentions only)' }
Write-Output "installed $TaskName - every 10 min, current-user interactive headless, $mode"
Write-Output "registry: %LOCALAPPDATA%\Fleet\registry\sessions.json (override with FLEET_STATE_DIR)"
Write-Output "log:      %LOCALAPPDATA%\Fleet\watchdog\resume_watchdog.log"
Write-Output "flip to live later:  .\tools\register_resume_watchdog.ps1 -Live"
