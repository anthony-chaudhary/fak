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

$claudeDisableMarker = if ($env:FLEET_CLAUDE_DISABLE_MARKER) {
  $env:FLEET_CLAUDE_DISABLE_MARKER
} elseif ($env:LOCALAPPDATA) {
  Join-Path $env:LOCALAPPDATA 'Fleet\claude.disabled'
} else {
  Join-Path $env:USERPROFILE '.fleet\claude.disabled'
}

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  $a = ($t.Actions | Select-Object -First 1).Arguments
  $modeStr = if ($a -match '-Live') { 'LIVE' } else { 'DRY-RUN' }
  Write-Output "State=$($t.State) mode=$modeStr LastRun=$($i.LastRunTime) LastResult=$($i.LastTaskResult) NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'install' -and (Test-Path -LiteralPath $claudeDisableMarker -PathType Leaf)) {
  $installed = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if ($installed) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false }
  Write-Output "CLAUDE_DISABLED: $TaskName absent; remove $claudeDisableMarker to opt back in"
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
# Register as S4U ("run whether the user is logged on or not", no stored password), NOT
# Interactive ("run only when logged on" -- the schtasks /Create default this script used
# to take). 2026-07-09 incident: as an Interactive task, from ~12:06 EVERY 10-min tick was
# refused with 0x800710E0 ("operator/administrator refused the request"), silently wedging
# the fleet's auto-resume layer. Root cause is the LOGON TYPE, not the launcher (the failure
# persisted after conhost.exe was dropped for a direct powershell.exe launch): an
# Interactive-logon task must launch into the user's attached interactive desktop, and on
# this headless / RDP-accessed box that is unreliable -- it stays refused even while an RDP
# session is Active, and dies outright when the session disconnects. The 17 sibling fleet
# tasks that rode straight through the same window (FleetIssueDispatch, FleetStrandedRecovery,
# FleetHeartbeat, ...) are ALL S4U: session-0 and windowless, yet still AS THIS USER (same
# profile / CLAUDE_CONFIG_DIR / oauth), so the `claude --resume` children it spawns are
# unaffected -- and S4U needs no conhost --headless flash-suppression (no desktop to flash).
# Same migration register_issue_dispatch.ps1 already made. NOTE: setting an S4U principal
# requires an ELEVATED shell on this box (non-elevated Register/Set-ScheduledTask -> "Access
# is denied"); the catch below says so. See tools/fleet_resume_watchdog.ps1 header for the
# earlier, DIFFERENT conhost incident (masked a param-binding failure as exit 0).
$act = New-ScheduledTaskAction -Execute 'powershell.exe' `
  -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"{1}' -f $Watchdog, $liveArg)
$trg = New-ScheduledTaskTrigger -Once -At (Get-Date) `
  -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 3650)
$prin = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
# StartWhenAvailable (#3321): a tick missed while the box slept/was off fires on wake.
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
try {
  Register-ScheduledTask -TaskName $TaskName -Action $act -Trigger $trg `
    -Principal $prin -Settings $set -Force -ErrorAction Stop | Out-Null
} catch {
  throw ("Register-ScheduledTask failed: $($_.Exception.Message). S4U registration needs an elevated shell -- re-run this installer as Administrator (right-click PowerShell -> Run as administrator).")
}
$mode = if ($Live) { 'LIVE (auto-resumes)' } else { 'DRY-RUN (logs intentions only)' }
Write-Output "installed $TaskName - every 10 min, current-user S4U (windowless, session-disconnect-immune), $mode"
Write-Output "registry: %LOCALAPPDATA%\Fleet\registry\sessions.json (override with FLEET_STATE_DIR)"
Write-Output "log:      %LOCALAPPDATA%\Fleet\watchdog\resume_watchdog.log"
Write-Output "dry-run for testing: .\tools\register_resume_watchdog.ps1 -DryRun"
Write-Output "doctor check:        .\tools\register_resume_watchdog.ps1 -Action assert-live"
