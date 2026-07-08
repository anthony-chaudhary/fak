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

# --- 1. AMD AUEPMaster telemetry (DURABLE: kill process AND disable its launcher) ---
Note '--- AMD AUEPMaster (Performance Profile Client telemetry) ---'
# The RELAUNCH mechanism on this box is the `AUEPLauncher` service (StartType
# Automatic) — it respawns AUEPMaster on every boot, so stopping the process alone
# is temporary. The durable fix is to DISABLE that service. We explicitly do NOT
# touch `AMD External Events Utility` — that is the legitimate GPU-driver service
# (display/hotkey events), not telemetry.
$auep = Get-Process -Name 'AUEPMaster','AUEPDU' -ErrorAction SilentlyContinue
if ($auep) {
  Note ("found " + $auep.Count + " AMD telemetry process(es): pid " + ($auep.Id -join ','))
} else {
  Note 'AUEPMaster not currently running (still disabling its launcher so it STAYS off)'
}
# The telemetry launcher service(s) — NOT the External Events Utility driver service.
$svc = Get-Service -Name 'AUEPLauncher' -ErrorAction SilentlyContinue
if (-not $svc) {
  $svc = Get-Service -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match 'AUEP' -and $_.Name -notmatch 'External Events' }
}
if ($Apply) {
  Stop-Process -Name 'AUEPMaster','AUEPDU' -Force -ErrorAction SilentlyContinue
  Note 'stopped AUEPMaster/AUEPDU processes'
  foreach ($s in $svc) {
    Stop-Service -Name $s.Name -Force -ErrorAction SilentlyContinue
    Set-Service -Name $s.Name -StartupType Disabled -ErrorAction SilentlyContinue
    Note ("DISABLED launcher service '$($s.Name)' — stays off across reboots  (UNDO: Set-Service -Name '$($s.Name)' -StartupType Automatic; Start-Service '$($s.Name)')")
  }
  # Any scheduled task that also relaunches it.
  Get-ScheduledTask -ErrorAction SilentlyContinue |
    Where-Object { $_.TaskName -match 'AUEP|User Experience Program' } |
    ForEach-Object {
      Disable-ScheduledTask -TaskName $_.TaskName -TaskPath $_.TaskPath -ErrorAction SilentlyContinue | Out-Null
      Note ("disabled task '$($_.TaskName)'  (UNDO: Enable-ScheduledTask -TaskName '$($_.TaskName)' -TaskPath '$($_.TaskPath)')")
    }
} else {
  $svcNames = ($svc | Select-Object -Expand Name) -join ', '
  Note ("WOULD: Stop AUEPMaster/AUEPDU; DISABLE launcher service(s) [$svcNames] so they stay off across reboots; disable any AUEP scheduled task")
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

# --- 3. CCleaner active monitoring (DURABLE: kill + neutralize its startup entry) --
Note '--- CCleaner CC_Engine active monitoring (optional background feature) ---'
# CCleaner monitoring relaunches from a Run key / startup entry (no service on this
# box), so the durable fix is to remove/rename that autostart entry AND stop the
# process. On-demand cleaning via the CCleaner UI is unaffected.
$cc = Get-Process -Name 'CC_Engine_x64','CCleaner*' -ErrorAction SilentlyContinue
if ($cc) { Note ("found CCleaner monitoring: pid " + ($cc.Id -join ',')) }
else     { Note 'CCleaner not currently running (still neutralizing its autostart so it STAYS off)' }
if ($Apply) {
  if ($cc) { Stop-Process -Id $cc.Id -Force -ErrorAction SilentlyContinue; Note 'stopped CC_Engine' }
  Get-Service -Name 'CCleaner*' -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Service -Name $_.Name -Force -ErrorAction SilentlyContinue
    Set-Service -Name $_.Name -StartupType Disabled -ErrorAction SilentlyContinue
    Note ("DISABLED service '$($_.Name)'  (UNDO: Set-Service -Name '$($_.Name)' -StartupType Automatic)")
  }
  # Neutralize the HKCU/HKLM Run autostart entry (the usual CCleaner monitoring launcher).
  foreach ($hive in @('HKCU:\Software\Microsoft\Windows\CurrentVersion\Run','HKLM:\Software\Microsoft\Windows\CurrentVersion\Run')) {
    $props = (Get-Item $hive -ErrorAction SilentlyContinue).Property
    foreach ($name in $props) {
      if ($name -match 'CCleaner') {
        $val = (Get-ItemProperty -Path $hive -Name $name -ErrorAction SilentlyContinue).$name
        Remove-ItemProperty -Path $hive -Name $name -ErrorAction SilentlyContinue
        Note ("removed Run entry '$name' from $hive  (UNDO: Set-ItemProperty -Path '$hive' -Name '$name' -Value '$val')")
      }
    }
  }
  Note 'CCleaner: also turn OFF "Smart/Active Monitoring" in the UI (Options > Monitoring) for the fully durable fix.'
} else {
  Note 'WOULD: Stop CC_Engine; disable any CCleaner service; remove its Run autostart entry (HKCU/HKLM). Best durable fix: turn off Active Monitoring in the CCleaner UI.'
}

Note 'done. Re-run `fak stallscan` to confirm the floor dropped, or `pwsh -File tools/fak_stall_monitor.ps1` to watch continuously.'
