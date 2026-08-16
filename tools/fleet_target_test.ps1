# Regression witness for #6502. Run with:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/fleet_target_test.ps1
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $root 'fleet_target.ps1')

function Assert-Equal {
  param($Want, $Got, [string]$Label)
  if ($Want -ne $Got) { throw "${Label}: want '$Want', got '$Got'" }
}

function Assert-ThrowsLike {
  param([scriptblock]$Run, [string]$Pattern, [string]$Label)
  try {
    & $Run
  } catch {
    if ($_.Exception.Message -match $Pattern) { return }
    throw "${Label}: wrong refusal: $($_.Exception.Message)"
  }
  throw "${Label}: expected refusal matching $Pattern"
}

$oldStateDir = $env:FLEET_STATE_DIR
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ('fak-fleet-target-test-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
$env:FLEET_STATE_DIR = $tmp
try {
  Set-FleetDesiredTarget -Target 0 -Reason 'regression quarantine' -Issue '#6502' | Out-Null
  Assert-Equal 0 (Get-FleetDesiredTarget) 'target-0 survives readback'
  Assert-Equal 0 (Assert-FleetTargetAdmits -Requested 0) 'matching target-0 tick'
  Assert-Equal 0 (Assert-FleetTargetAdmits) 'argument-free scheduled tick'
  Assert-ThrowsLike { Assert-FleetTargetAdmits -Requested 4 | Out-Null } `
    'TARGET_QUARANTINE_OVERRIDE' 'stale target-4 tick'

  Assert-ThrowsLike { Set-FleetDesiredTarget -Target 1 | Out-Null } `
    'POPULATION_LIFT_UNWITNESSED' 'unwitnessed re-enable'
  Set-FleetDesiredTarget -Target 1 -LauncherAdmits -StatusAgrees -Canary 'canary:#6502' | Out-Null
  Assert-Equal 1 (Get-FleetDesiredTarget) 'witnessed re-enable'
  Set-FleetDesiredTarget -Target 0 -Reason 'restore quarantine' | Out-Null

  Set-Content -Path (Get-FleetTargetPath) -Value '{broken' -Encoding UTF8
  Assert-Equal 0 (Get-FleetDesiredTarget) 'malformed SSOT fails closed'

  Assert-ThrowsLike { Assert-FleetTaskHasNoEmbeddedTarget `
      -Arguments '-File watchdog.ps1 -Target 8 -Interval 120' -TaskName 'stale-task' | Out-Null } `
    'EMBEDDED_WATCHDOG_TARGET' 'status audit of stale task'
  Assert-FleetTaskHasNoEmbeddedTarget -Arguments '-File watchdog.ps1 -Interval 120' | Out-Null

  # Both registrars reject explicit install-time populations before touching Task Scheduler.
  foreach ($registrar in @('register_supervisor_watchdog.ps1', 'register_dos_dispatch_watchdog.ps1')) {
    Assert-ThrowsLike { & (Join-Path $root $registrar) -Action install -Target 4 } `
      'EMBEDDED_WATCHDOG_TARGET' "$registrar explicit target"
  }

  # Their status path also audits an already-installed action instead of reporting it healthy.
  function Get-ScheduledTask {
    [pscustomobject]@{ State = 'Ready'; Actions = @([pscustomobject]@{ Arguments = '-File watchdog.ps1 -Target 4' }) }
  }
  function Get-ScheduledTaskInfo { throw 'status audit should refuse before task info' }
  foreach ($registrar in @('register_supervisor_watchdog.ps1', 'register_dos_dispatch_watchdog.ps1')) {
    Assert-ThrowsLike { & (Join-Path $root $registrar) -Action status -TaskName 'stale-task' } `
      'EMBEDDED_WATCHDOG_TARGET' "$registrar stale status"
  }

  Write-Output 'PASS: target-0 quarantine is durable and stale scheduled targets are refused'
} finally {
  $env:FLEET_STATE_DIR = $oldStateDir
  Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

