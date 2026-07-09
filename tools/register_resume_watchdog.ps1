<#
register_resume_watchdog.ps1 - install/remove the Scheduled Task that runs the
cross-account resume watchdog every 10 min (refreshes the on-disk session
registry each tick = "extract in advance", and optionally auto-resumes
autonomous dead sessions).

  .\register_resume_watchdog.ps1                 # install LIVE (the default; #3321)
  .\register_resume_watchdog.ps1 -DryRun         # install DRY-RUN (logs intentions only)
  .\register_resume_watchdog.ps1 -Live           # install LIVE, explicitly
  .\register_resume_watchdog.ps1 -Action status
  .\register_resume_watchdog.ps1 -Action assert-live   # doctor: non-zero unless installed action carries -Live
  .\register_resume_watchdog.ps1 -Action remove

LIVE is the install default (#3321): auto-resume IS this watchdog's job, and the old
DRY-RUN default meant every habitual no-flag reinstall (docs quote the bare command
first) silently downgraded a LIVE task to log-only — the fleet's auto-resume layer
died with no error anywhere. Testing is the exception now, and it asks by name with
-DryRun.
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status','assert-live')] [string]$Action = 'install',
  [switch]$Live,
  [switch]$DryRun,
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
if ($Action -eq 'assert-live') {
  # Doctor/CI probe (#3321): green only when the installed task will actually
  # auto-resume. Catches the downgrade this script's old default caused, and any
  # future path that re-installs the task without -Live.
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "ASSERT-LIVE FAIL: $TaskName not installed"; exit 1 }
  $a = ($t.Actions | Select-Object -First 1).Arguments
  if ($a -notmatch '(^|\s)-Live(\s|$)') {
    Write-Output "ASSERT-LIVE FAIL: $TaskName is DRY-RUN (arguments: $a); reinstall with .\tools\register_resume_watchdog.ps1"
    exit 1
  }
  if (-not $t.Settings.StartWhenAvailable) {
    Write-Output "ASSERT-LIVE WARN: StartWhenAvailable is off; a tick missed while the box slept will not fire on wake"
  }
  Write-Output "ASSERT-LIVE OK: $TaskName action carries -Live"
  exit 0
}

if ($Live -and $DryRun) { throw '-Live and -DryRun are mutually exclusive' }
# No flag = LIVE (#3321). DRY-RUN only when asked for by name.
if (-not $DryRun) { $Live = $true }
$liveArg = if ($Live) { ' -Live' } else { '' }
# Launch powershell.exe DIRECTLY (not via `conhost.exe --headless`). On 2026-07-09 the
# box began refusing every `conhost.exe`-launched task with 0x800710E0 ("operator or
# administrator refused the request") while sibling tasks that launch their payload
# directly (FleetStrandedRecovery=powershell.exe, FleetIssueDispatch=python.exe) kept
# returning 0x0 -- a clean launcher split with battery/other settings ruled out (this
# box is a desktop, no battery). conhost --headless silently wedged this watchdog for
# ~38 min (drained a 65-deep backlog post-boot, then went dark at 11:42). -WindowStyle
# Hidden suppresses the window; a brief per-tick console flash is the accepted trade for
# a launcher that actually runs. (No-flash-without-conhost = a hidden .vbs WScript.Shell
# wrapper; deferred.) See tools/fleet_resume_watchdog.ps1 header for the earlier, DIFFERENT
# conhost incident (masked a param-binding failure as exit 0).
$tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Watchdog`"$liveArg"
schtasks /Create /TN $TaskName /SC MINUTE /MO 10 /TR $tr /RL LIMITED /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /Create failed ($LASTEXITCODE)" }
# schtasks.exe has no switch for StartWhenAvailable; set it via the ScheduledTasks
# module so a tick missed while the box was asleep/off fires on wake instead of
# waiting up to 10 min (#3321). Non-fatal: without it the cadence still holds.
try {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
  $t.Settings.StartWhenAvailable = $true
  Set-ScheduledTask -InputObject $t | Out-Null
} catch {
  Write-Warning "could not set StartWhenAvailable on ${TaskName}: $_"
}
$mode = if ($Live) { 'LIVE (auto-resumes)' } else { 'DRY-RUN (logs intentions only)' }
Write-Output "installed $TaskName - every 10 min, current-user interactive (direct powershell, hidden window), $mode"
Write-Output "registry: %LOCALAPPDATA%\Fleet\registry\sessions.json (override with FLEET_STATE_DIR)"
Write-Output "log:      %LOCALAPPDATA%\Fleet\watchdog\resume_watchdog.log"
Write-Output "dry-run for testing: .\tools\register_resume_watchdog.ps1 -DryRun"
Write-Output "doctor check:        .\tools\register_resume_watchdog.ps1 -Action assert-live"
