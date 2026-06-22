<#
register_runaway_reaper.ps1 - install/remove the Scheduled Task that runs the
runaway search-process reaper every 5 min (kills Git-Bash find/grep that have run
away into /proc/registry or /mnt/c junction loops -- see runaway_process_reaper.ps1).

  .\register_runaway_reaper.ps1                 # install in DRY-RUN (safe default)
  .\register_runaway_reaper.ps1 -Live           # install LIVE (actually kills runaways)
  .\register_runaway_reaper.ps1 -Action status
  .\register_runaway_reaper.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [switch]$Live,
  [int]$EveryMin = 5,
  [string]$TaskName = 'FleetRunawayReaper',
  # Resolve the sibling reaper in THIS clone, so registering from any checkout
  # schedules that checkout's script -- not a hardcoded operator path.
  [string]$Reaper = (Join-Path $PSScriptRoot 'runaway_process_reaper.ps1')
)
$ErrorActionPreference = 'Stop'

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
$tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Reaper`"$liveArg"
schtasks /Create /TN $TaskName /SC MINUTE /MO $EveryMin /TR $tr /RL LIMITED /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /Create failed ($LASTEXITCODE)" }
$mode = if ($Live) { 'LIVE (kills runaways)' } else { 'DRY-RUN (logs intentions only)' }
Write-Output "installed $TaskName - every $EveryMin min, $mode"
Write-Output "log: %LOCALAPPDATA%\Fleet\watchdog\runaway_reaper.log"
Write-Output "flip to live later:  .\tools\register_runaway_reaper.ps1 -Live"
