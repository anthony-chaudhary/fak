<#
register_session0_sweep.ps1 - install/remove the Scheduled Task that runs the
elevated Session-0 orphan sweep (issue #2338). Unlike the runaway reaper, the
targets are ANOTHER (tombstoned) account's DACL-protected Session-0 processes, so
the task runs as NT AUTHORITY\SYSTEM (RunLevel Highest) -- an interactive
same-user token cannot terminate them (ERROR_ACCESS_DENIED). The kill is always
PID-scoped by tools/session0_orphan_sweep.py; this wrapper only supplies the
SYSTEM token and the schedule.

  .\register_session0_sweep.ps1                       # install DRY-RUN (census only)
  .\register_session0_sweep.ps1 -Enact                # install LIVE (actually kills)
  .\register_session0_sweep.ps1 -Owners jack-barker -CreatedOn 2026-06-29
  .\register_session0_sweep.ps1 -Action status
  .\register_session0_sweep.ps1 -Action remove

Registering the SYSTEM principal itself requires an elevated shell.
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [switch]$Enact,
  [int]$EveryMin = 30,
  [string]$TaskName = 'FleetSession0Sweep',
  [string]$Owners = 'jack-barker',
  [string]$CreatedOn = '',
  [string]$Sweep = ''
)
$ErrorActionPreference = 'Stop'

$ScriptRoot = $PSScriptRoot
if (-not $ScriptRoot) { $ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $ScriptRoot) { $ScriptRoot = (Get-Location).Path }
if (-not $Sweep) { $Sweep = Join-Path $ScriptRoot 'session0_orphan_sweep.py' }
$RepoRoot = Split-Path -Parent $ScriptRoot

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  $a = ($t.Actions | Select-Object -First 1).Arguments
  $modeStr = if ($a -match '--enact') { 'LIVE' } else { 'DRY-RUN' }
  Write-Output "State=$($t.State) mode=$modeStr LastRun=$($i.LastRunTime) LastResult=$($i.LastTaskResult) NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'remove') {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Output "removed $TaskName"; return
}

$py = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = 'python.exe' }
$enactArg = if ($Enact) { ' --enact' } else { '' }
$dateArg  = if ($CreatedOn) { " --created-on $CreatedOn" } else { '' }
$pyArgs = "`"$Sweep`" --owners $Owners$dateArg$enactArg"
$taskAction = New-ScheduledTaskAction -Execute $py -Argument $pyArgs -WorkingDirectory $RepoRoot
$trigger    = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
                -RepetitionInterval (New-TimeSpan -Minutes $EveryMin) `
                -RepetitionDuration (New-TimeSpan -Days 3650)
# SYSTEM / RunLevel Highest: the targets are another account's DACL-protected
# Session-0 processes, so (unlike the same-user runaway reaper) an interactive
# S4U-as-this-user token would still get ACCESS_DENIED. SYSTEM can terminate them.
# The predicate in session0_orphan_sweep.py is what guarantees SYSTEM's power is
# only ever pointed at the tombstoned-account, date-scoped, dead-parent PIDs --
# never a live Session-1 process or a Windows service.
$principal = New-ScheduledTaskPrincipal -UserId 'NT AUTHORITY\SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null
$mode = if ($Enact) { 'LIVE (PID-scoped kills)' } else { 'DRY-RUN (census only)' }
Write-Output "installed $TaskName - every $EveryMin min, $mode, SYSTEM (RunLevel Highest)"
Write-Output "scope: owners=$Owners$(if($CreatedOn){" created-on=$CreatedOn"}) images=opencode.exe,cmd.exe,python.exe"
Write-Output "flip to live later:  .\tools\register_session0_sweep.ps1 -Enact"
