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
$pwsh = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source
if (-not $pwsh) { $pwsh = 'powershell.exe' }
$psArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Reaper`"$liveArg"
$taskAction = New-ScheduledTaskAction -Execute $pwsh -Argument $psArgs -WorkingDirectory (Split-Path -Parent $PSScriptRoot)
$trigger    = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
                -RepetitionInterval (New-TimeSpan -Minutes $EveryMin) `
                -RepetitionDuration (New-TimeSpan -Days 3650)
# S4U (non-interactive, session 0), NOT the schtasks default Interactive: a console
# powershell.exe launched in the interactive session flashes a window on EVERY 5-min
# trigger — the "random popup windows" (-WindowStyle Hidden does NOT suppress it, the
# flash is the session-1 console host spawning). S4U runs the reaper windowless in
# session 0 yet still AS THIS USER, so it can still enumerate and terminate this user's
# runaway Git-Bash find/grep processes (same-user terminate needs no elevation). Same
# pattern register_issue_dispatch.ps1 uses; -StartWhenAvailable resumes it after a
# reboot that missed a tick (enabled-by-default-on-restart).
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null
$mode = if ($Live) { 'LIVE (kills runaways)' } else { 'DRY-RUN (logs intentions only)' }
Write-Output "installed $TaskName - every $EveryMin min, $mode, S4U (windowless, restart-durable)"
Write-Output "logs: %LOCALAPPDATA%\Fleet\watchdog\runaway_reaper.log   (human, incl. spawner ancestry)"
Write-Output "      %LOCALAPPDATA%\Fleet\watchdog\runaway_reaper.jsonl (structured, one record per event)"
Write-Output "flip to live later:  .\tools\register_runaway_reaper.ps1 -Live"
