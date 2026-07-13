<#
register_reboot_advisor.ps1 - install/remove the Scheduled Task that runs the
reboot advisor (reboot_advisor.ps1) on a cadence. The advisor is READ-ONLY: it
measures the two reboot-only leaks (WindowsTerminal chokepoint, TermService svchost)
and toasts the operator EARLY when a reboot would pay off. It never kills or reboots,
so unlike the runaway reaper there is no DRY-RUN/LIVE distinction -- installing it is
always safe.

  .\register_reboot_advisor.ps1                 # install (every 30 min, S4U, windowless)
  .\register_reboot_advisor.ps1 -Action status
  .\register_reboot_advisor.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [int]$EveryMin = 30,
  [string]$TaskName = 'FleetRebootAdvisor',
  # Empty by default and resolved in the body: $PSScriptRoot is EMPTY when read in a
  # param-block default under Windows PowerShell 5.1 launched via -File. Resolve the
  # sibling advisor below with the same 3-tier fallback register_runaway_reaper.ps1 uses.
  [string]$Advisor = ''
)
$ErrorActionPreference = 'Stop'

$ScriptRoot = $PSScriptRoot
if (-not $ScriptRoot) { $ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $ScriptRoot) { $ScriptRoot = (Get-Location).Path }
if (-not $Advisor) { $Advisor = Join-Path $ScriptRoot 'reboot_advisor.ps1' }
$RepoRoot = Split-Path -Parent $ScriptRoot

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  Write-Output "State=$($t.State) LastRun=$($i.LastRunTime) LastResult=$($i.LastTaskResult) NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'remove') {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Output "removed $TaskName"; return
}

$pwsh = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source
if (-not $pwsh) { $pwsh = 'powershell.exe' }
$psArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Advisor`""
$taskAction = New-ScheduledTaskAction -Execute $pwsh -Argument $psArgs -WorkingDirectory $RepoRoot
$trigger    = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
                -RepetitionInterval (New-TimeSpan -Minutes $EveryMin) `
                -RepetitionDuration (New-TimeSpan -Days 3650)
# S4U (non-interactive, session 0), NOT the schtasks default Interactive: an interactive
# console powershell.exe flashes a window on every trigger. S4U runs windowless in
# session 0 yet still AS THIS USER, so it can read this user's WindowsTerminal handle
# counts (no elevation needed for read-only Get-Process). The toast is delivered by
# notify.ps1 into the Action Center, which surfaces regardless of session.
# -StartWhenAvailable resumes it after a reboot that missed a tick.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null
Write-Output "installed $TaskName - every $EveryMin min, read-only advisory, S4U (windowless, restart-durable)"
Write-Output "logs: %LOCALAPPDATA%\Fleet\watchdog\reboot_advisor.log   (human)"
Write-Output "      %LOCALAPPDATA%\Fleet\watchdog\reboot_advisor.jsonl (structured, one record per RECOMMEND)"
Write-Output "toast on RECOMMEND is deduped to once per 6h; remove with:  .\tools\register_reboot_advisor.ps1 -Action remove"
