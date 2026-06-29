<#
register_stale_work_watchdog.ps1 - install/remove the Scheduled Task that runs the
stale-work watchdog (tools/stale_work_watchdog.py) on a cadence: it GCs this clone's
own gitignored per-session ephemera (.dos/markers|streams|stop-failures, tools/_watchdog
logs) once they age past the floor, and reports stuck stop-failure sessions + stale
shared-tree WIP. Nothing else prunes THIS clone's .dos (DOS-cleanup-sweep targets
dos-kernel-public; FakFleetJanitor reaps GCP VMs), so without this the dir grows
without bound.

  .\register_stale_work_watchdog.ps1                 # install DRY-RUN (reports, deletes nothing)
  .\register_stale_work_watchdog.ps1 -Live            # install LIVE (actually GCs over-age ephemera)
  .\register_stale_work_watchdog.ps1 -Action status
  .\register_stale_work_watchdog.ps1 -Action remove
  .\register_stale_work_watchdog.ps1 -Live -MaxAgeDays 14 -EveryHours 12

A janitor only ever deletes gitignored ephemera older than -MaxAgeDays, and only files
provably inside the known ephemeral dirs (the python refuses anything else). It NEVER
touches git state -- shared-tree WIP is reported, never committed.
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [switch]$Live,
  [int]$MaxAgeDays = 7,
  [int]$EveryHours = 6,
  [string]$TaskName = 'FleetStaleWorkWatchdog',
  # Resolve the sibling watchdog in THIS clone so registering from any checkout
  # schedules that checkout's script -- not a hardcoded operator path.
  [string]$Watchdog = (Join-Path $PSScriptRoot 'stale_work_watchdog.py'),
  [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
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

# Resolve a python launcher: prefer the fleet convention, then python3, then python.
$py = $env:FLEET_PYTHON
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { $py = (Get-Command python  -ErrorAction SilentlyContinue).Source }
if (-not $py) { $py = 'python' }

$logDir = Join-Path $env:LOCALAPPDATA 'Fleet\watchdog'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'stale_work_watchdog.log'

$liveArg = if ($Live) { ' --live' } else { '' }
# A wrapper that timestamps each run into the watchdog log, then runs the janitor
# against THIS repo. --json so the log line is machine-greppable.
$inner = "& '$py' -X utf8 `"$Watchdog`" --repo `"$RepoRoot`" --max-age-days $MaxAgeDays$liveArg --json"
$cmd = "`"===== `$((Get-Date -Format o)) =====`" | Out-File -FilePath '$log' -Append -Encoding UTF8; " +
       "$inner 2>&1 | Out-File -FilePath '$log' -Append -Encoding UTF8"

$pwsh = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source
if (-not $pwsh) { $pwsh = 'powershell.exe' }
$psArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command `"$cmd`""
$taskAction = New-ScheduledTaskAction -Execute $pwsh -Argument $psArgs -WorkingDirectory $RepoRoot
$trigger    = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
                -RepetitionInterval (New-TimeSpan -Hours $EveryHours) `
                -RepetitionDuration (New-TimeSpan -Days 3650)
# S4U (windowless session 0) so the periodic run never flashes a console window, same
# pattern as register_runaway_reaper.ps1; -StartWhenAvailable resumes a missed tick.
# Registering an S4U principal can require elevation; if that is denied, fall back to a
# current-user Interactive principal, which registers unelevated (a 6h cadence makes the
# occasional console flash a non-issue). Either way the janitor runs as THIS user.
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
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
Write-Output "  log:  $log"
Write-Output "flip to live later:  .\tools\register_stale_work_watchdog.ps1 -Live"
