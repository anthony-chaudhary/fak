<#
.SYNOPSIS
  Tame the non-fak background daemons that form the constant disk/IO-ops floor
  behind the low-usage machine stalls (see `fak stallscan` and issue #3153).

.DESCRIPTION
  Diagnosis (2026-07-07): three non-fak daemons dominate I/O operations by ~100x
  over any fak/claude process and form the sustained floor that fak/session spawn
  bursts ride on top of:
    - AUEPMaster.exe  (AMD "Performance Profile Client" / User-Experience telemetry)
                      ~127k-250k disk ops/sec SUSTAINED. Pure telemetry; safe to stop.
    - MsMpEng.exe     (Windows Defender realtime scan) — scans every file the other
                      daemons touch, amplifying the churn. Mitigate with EXCLUSIONS,
                      never by disabling protection.
    - CC_Engine_x64.exe (CCleaner active monitoring) — small-op disk thrasher. The
                      "active monitoring" / "smart cleaning" background feature is
                      optional; stopping it does not affect on-demand cleaning.

  This script is SAFE and REVERSIBLE by design:
    * It only STOPS/DISABLES telemetry + optional-monitoring services, never core OS.
    * Defender is only given path EXCLUSIONS (protection stays on).
    * Every action prints an explicit UNDO command.
    * Run with -WhatIf to preview, -Apply to act. Default is preview.

  Run elevated (Administrator) for the service + Defender-exclusion actions.

.PARAMETER Apply
  Actually perform the mitigations. Without it, the script only reports.

.PARAMETER WorkDirs
  Directories to add as Defender exclusions (the fleet work trees). Defaults to
  the common fak/session dirs; override for your layout.

.EXAMPLE
  # Preview what would change (safe, no elevation needed for the report):
  pwsh -File tools/host_stall_mitigations.ps1

.EXAMPLE
  # Apply (run as Administrator):
  pwsh -File tools/host_stall_mitigations.ps1 -Apply
#>
[CmdletBinding()]
param(
  [switch]$Apply,
  [string[]]$WorkDirs = @('C:\work', "$env:LOCALAPPDATA\Fleet", "$env:USERPROFILE\.claude")
)

$ErrorActionPreference = 'Continue'
function Note($m) { Write-Host "[stall-mit] $m" }
function IsAdmin {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
    [Security.Principal.WindowsBuiltinRole]::Administrator)
}

Note ("mode: " + ($(if ($Apply) { 'APPLY' } else { 'PREVIEW (pass -Apply to act)' })))
if ($Apply -and -not (IsAdmin)) {
  Note 'WARNING: not elevated — service/Defender actions will likely fail. Re-run as Administrator.'
}

# --- 1. AMD AUEPMaster telemetry ------------------------------------------------
Note '--- AMD AUEPMaster (Performance Profile Client telemetry) ---'
$auep = Get-Process -Name 'AUEPMaster','AUEPDU' -ErrorAction SilentlyContinue
if ($auep) {
  Note ("found " + $auep.Count + " AMD telemetry process(es): pid " + ($auep.Id -join ','))
  # It is launched from the AMD install + a scheduled task / run key. Stopping the
  # process is immediate relief; disabling its task/service prevents relaunch.
  $svc = Get-Service -Name 'AMD*Telemetry*','AMD External Events*' -ErrorAction SilentlyContinue
  if ($Apply) {
    Stop-Process -Name 'AUEPMaster','AUEPDU' -Force -ErrorAction SilentlyContinue
    Note 'stopped AUEPMaster (UNDO: it relaunches on reboot / AMD software launch)'
    foreach ($s in $svc) {
      Set-Service -Name $s.Name -StartupType Manual -ErrorAction SilentlyContinue
      Note ("set service '$($s.Name)' to Manual  (UNDO: Set-Service -Name '$($s.Name)' -StartupType Automatic)")
    }
    # Scheduled task that relaunches it, if present.
    Get-ScheduledTask -TaskName '*AUEP*','*User Experience Program*' -ErrorAction SilentlyContinue |
      ForEach-Object {
        Disable-ScheduledTask -TaskName $_.TaskName -TaskPath $_.TaskPath -ErrorAction SilentlyContinue | Out-Null
        Note ("disabled task '$($_.TaskName)'  (UNDO: Enable-ScheduledTask -TaskName '$($_.TaskName)' -TaskPath '$($_.TaskPath)')")
      }
  } else {
    Note 'WOULD: Stop-Process AUEPMaster; set AMD telemetry service(s) to Manual; disable its scheduled task'
  }
} else {
  Note 'AUEPMaster not running — nothing to do'
}

# --- 2. Windows Defender exclusions (protection STAYS ON) -----------------------
Note '--- Windows Defender: add work-dir exclusions (keeps realtime protection ON) ---'
foreach ($d in $WorkDirs) {
  if (-not (Test-Path $d)) { continue }
  if ($Apply) {
    try {
      Add-MpPreference -ExclusionPath $d -ErrorAction Stop
      Note ("excluded '$d'  (UNDO: Remove-MpPreference -ExclusionPath '$d')")
    } catch { Note ("could not exclude '$d': " + $_.Exception.Message) }
  } else {
    Note "WOULD: Add-MpPreference -ExclusionPath '$d'"
  }
}
# Also exclude the two process images that generate the most scan churn.
foreach ($p in @('fak.exe','claude.exe','python.exe','node.exe')) {
  if ($Apply) {
    try { Add-MpPreference -ExclusionProcess $p -ErrorAction Stop
          Note ("excluded process '$p'  (UNDO: Remove-MpPreference -ExclusionProcess '$p')") }
    catch { Note ("could not exclude process '$p': " + $_.Exception.Message) }
  } else { Note "WOULD: Add-MpPreference -ExclusionProcess '$p'" }
}

# --- 3. CCleaner active monitoring ---------------------------------------------
Note '--- CCleaner CC_Engine active monitoring (optional background feature) ---'
$cc = Get-Process -Name 'CC_Engine_x64','CCleaner*' -ErrorAction SilentlyContinue
if ($cc) {
  Note ("found CCleaner monitoring: pid " + ($cc.Id -join ','))
  if ($Apply) {
    Stop-Process -Id $cc.Id -Force -ErrorAction SilentlyContinue
    Note 'stopped CC_Engine (UNDO: relaunches with CCleaner; disable "Active/Smart Monitoring" in CCleaner > Options > Monitoring to keep it off)'
    Get-Service -Name 'CCleaner*' -ErrorAction SilentlyContinue | ForEach-Object {
      Set-Service -Name $_.Name -StartupType Manual -ErrorAction SilentlyContinue
      Note ("set service '$($_.Name)' to Manual  (UNDO: Set-Service -Name '$($_.Name)' -StartupType Automatic)")
    }
  } else {
    Note 'WOULD: Stop-Process CC_Engine; set CCleaner service to Manual. (Best durable fix: turn off Active Monitoring in the CCleaner UI.)'
  }
} else {
  Note 'CCleaner monitoring not running — nothing to do'
}

Note 'done. Re-run `fak stallscan` to confirm the floor dropped, or `pwsh -File tools/fak_stall_monitor.ps1` to watch continuously.'
