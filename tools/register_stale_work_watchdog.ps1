<#
register_stale_work_watchdog.ps1 - install/remove the Scheduled Task that runs
the Go-native `fak garden watchdog` stale-work janitor.

  .\register_stale_work_watchdog.ps1
  .\register_stale_work_watchdog.ps1 -Live
  .\register_stale_work_watchdog.ps1 -Action status
  .\register_stale_work_watchdog.ps1 -Action remove

The live path deletes only over-age files inside the declared gitignored
ephemeral directories. Shared-tree WIP is report-only. The Go watchdog owns a
45-second hard child bound, typed timeout/progress JSON, a process-tree reap,
and an O_EXCL overlap refusal. Task Scheduler independently carries
MultipleInstances=IgnoreNew as a second overlap fence.
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [switch]$Live,
  [int]$MaxAgeDays = 7,
  [int]$EveryHours = 6,
  [string]$TaskName = 'FleetStaleWorkGarden',
  [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
  [string]$Fak = $env:FAK_BIN
)
$ErrorActionPreference = 'Stop'

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  $a = ($t.Actions | Select-Object -First 1).Arguments
  $modeStr = if ($a -match '--live') { 'LIVE' } else { 'DRY-RUN' }
  Write-Output "State=$($t.State) mode=$modeStr LastRun=$($i.LastRunTime) LastResult=$($i.LastTaskResult) NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'remove') {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Output "removed $TaskName"; return
}

if (-not $Fak) {
  $found = Get-Command fak -ErrorAction SilentlyContinue
  if ($found) { $Fak = $found.Source }
}
if (-not $Fak) {
  $repoBinary = Join-Path $RepoRoot 'fak.exe'
  if (Test-Path -LiteralPath $repoBinary) { $Fak = $repoBinary }
}
if (-not $Fak) {
  throw 'fak binary not found; install fak or pass -Fak <absolute path>'
}

$logDir = Join-Path $env:LOCALAPPDATA 'Fleet\watchdog'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'stale_work_watchdog.log'

$liveArg = if ($Live) { ' --live' } else { '' }
$inner = "& '$Fak' garden watchdog --repo `"$RepoRoot`" --max-age-days $MaxAgeDays" +
         "$liveArg --watchdog-timeout 45 --tick-budget 35 --json"
$cmd = "`"===== `$((Get-Date -Format o)) =====`" | Out-File -FilePath '$log' -Append -Encoding UTF8; " +
       "$inner 2>&1 | Out-File -FilePath '$log' -Append -Encoding UTF8"

$pwsh = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source
if (-not $pwsh) { $pwsh = 'powershell.exe' }
$psArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command `"$cmd`""
$taskAction = New-ScheduledTaskAction -Execute $pwsh -Argument $psArgs -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
             -RepetitionInterval (New-TimeSpan -Hours $EveryHours) `
             -RepetitionDuration (New-TimeSpan -Days 3650)

# The task-level two-minute ceiling is intentionally just above the Go watchdog's
# 45-second hard bound and its reap/flush margin. IgnoreNew prevents Task Scheduler
# from queueing or overlapping a second scheduled instance; the verb's lock emits
# SKIPPED_CONTENDED when a manual or foreign launcher races it.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
              -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
$principalMode = 'S4U (windowless)'
try {
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
  Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
                 -Principal $principal -Settings $settings -Force -ErrorAction Stop | Out-Null
} catch {
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
  $headlessArgs = "--headless `"$pwsh`" $psArgs"
  $headlessAction = New-ScheduledTaskAction -Execute 'conhost.exe' -Argument $headlessArgs -WorkingDirectory $RepoRoot
  Register-ScheduledTask -TaskName $TaskName -Action $headlessAction -Trigger $trigger `
                 -Principal $principal -Settings $settings -Force | Out-Null
  $principalMode = 'Interactive (unelevated fallback; conhost --headless)'
}
$mode = if ($Live) { "LIVE (GCs ephemera > $MaxAgeDays d)" } else { 'DRY-RUN (reports only)' }
Write-Output "installed $TaskName - every $EveryHours h, $mode, $principalMode, restart-durable"
Write-Output "  repo: $RepoRoot"
Write-Output "  fak:  $Fak"
Write-Output "  log:  $log"
Write-Output "flip to live later:  .\tools\register_stale_work_watchdog.ps1 -Live"
