<#
fleet_target.ps1 -- the fleet's DESIRED WORKER POPULATION, read from the one
declared SSOT instead of baked into a Scheduled Task's arguments (#6502).

THE DEFECT THIS EXISTS FOR. The 2026-08-11 maintenance audit found two enabled
watchdog tasks each carrying a population EMBEDDED IN ITS ACTION ARGUMENTS --
FleetDOSDispatchWatchdog with `-Target 8 -Interval 120`, FleetSupervisorWatchdog
with `-Target 4` -- firing every five minutes against an operator's `dos loop
--target 0` quarantine. A baked-in number is a snapshot of what somebody wanted
when the task was registered, and Task Scheduler replays a snapshot forever: the
quarantine held only while every downstream admission gate kept refusing, and
would have been undone silently the day one of them changed. The two tasks did not
even agree with each other about how many workers the fleet wanted.

THE RULE. One file owns the number: $FLEET_STATE_DIR\fleet-target.json (else
%LOCALAPPDATA%\Fleet\fleet-target.json). It lives in the state dir, so it survives
scheduled ticks and reboots alike. Every watchdog READS it at tick time; nothing
passes a population on a command line. The read is FAIL-CLOSED -- absent,
unreadable, malformed or negative all resolve to 0 (quarantine), never to a
convenient 4, because an unknown operator intent must not spawn.

The schema and every refusal over it are defined ONCE in Go, in
internal/supervisoragent/population.go (DesiredPopulation / ParsePopulation /
AdmitPopulation / SetPopulation), which is where the regression tests live. This
script is the PowerShell reader of the same bytes and mirrors the same fail-closed
defaults.

  . (Join-Path $PSScriptRoot 'fleet_target.ps1')   # dot-source the functions
  $target = Get-FleetDesiredTarget                 # 0 when quarantined
  Assert-FleetTargetAdmits -Requested 8            # throws under quarantine

  # declare a quarantine (always allowed -- an operator may always stop):
  .\fleet_target.ps1 -Set 0 -Reason 'maintenance audit' -Issue '#6502'

  # raise it (needs the same witnesses the Go SetPopulation demands):
  .\fleet_target.ps1 -Set 4 -Reason 'launch admission repaired' -Issue '#6502' `
      -LauncherAdmits -StatusAgrees -Canary 'canary-6502-a'
#>
[CmdletBinding()]
param(
  # Declare a new desired population. Omitted = this script was dot-sourced for
  # its functions and writes nothing.
  [int]$Set = -1,
  [string]$Reason = '',
  [string]$Issue = '',
  [switch]$LauncherAdmits,
  [switch]$StatusAgrees,
  [string]$Canary = ''
)

# Quarantine is the operator's stop posture and every fail-closed answer.
$script:FleetQuarantineTarget = 0

# Closed refusal vocabulary shared by task ticks and registration/status audits.
$script:FleetReasonQuarantineOverride = 'TARGET_QUARANTINE_OVERRIDE'
$script:FleetReasonPopulationUnwitnessed = 'POPULATION_LIFT_UNWITNESSED'

# Get-FleetStateRoot resolves the fleet state dir the same way every fleet
# watchdog already does, so the SSOT sits beside the logs those loops write.
function Get-FleetStateRoot {
  if ($env:FLEET_STATE_DIR) { return $env:FLEET_STATE_DIR }
  if ($env:LOCALAPPDATA) { return (Join-Path $env:LOCALAPPDATA 'Fleet') }
  return (Join-Path ([System.IO.Path]::GetTempPath()) 'Fleet')
}

# Get-FleetTargetPath is the SSOT's full path. The basename is pinned by
# supervisoragent.PopulationFile.
function Get-FleetTargetPath {
  return (Join-Path (Get-FleetStateRoot) 'fleet-target.json')
}

# Get-FleetDesiredTarget returns the declared population, FAIL-CLOSED to 0. It
# never throws: a watchdog tick that cannot read the operator's intent must reach
# quarantine, not an error path that some caller catches and ignores.
function Get-FleetDesiredTarget {
  $path = Get-FleetTargetPath
  if (-not (Test-Path $path)) { return $script:FleetQuarantineTarget }
  try {
    $raw = Get-Content -Raw -Path $path -ErrorAction Stop
  } catch {
    return $script:FleetQuarantineTarget
  }
  if (-not "$raw".Trim()) { return $script:FleetQuarantineTarget }
  try {
    $rec = "$raw" | ConvertFrom-Json -ErrorAction Stop
  } catch {
    return $script:FleetQuarantineTarget
  }
  if ($null -eq $rec -or $null -eq $rec.target) { return $script:FleetQuarantineTarget }
  $t = 0
  if (-not [int]::TryParse("$($rec.target)", [ref]$t)) { return $script:FleetQuarantineTarget }
  if ($t -lt $script:FleetQuarantineTarget) { return $script:FleetQuarantineTarget }
  return $t
}

# Assert-FleetTargetAdmits is the non-bypassable half, mirroring
# It RETURNS the declared population whatever the caller asked for, and throws when
# a request disagrees. A caller that swallows the
# throw still only ever gets the SSOT's number back.
#
# -Requested -1 means "this tick carried no target argument", which is the intended
# steady state and never a disagreement.
function Assert-FleetTargetAdmits {
  param([int]$Requested = -1)
  $target = Get-FleetDesiredTarget
  if ($Requested -eq -1 -or $Requested -eq $target) { return $target }
  if ($target -eq $script:FleetQuarantineTarget -and $Requested -gt $script:FleetQuarantineTarget) {
    throw ("{0}: tick requested {1} worker(s) while {2} declares the operator's target-0 quarantine" -f `
      $script:FleetReasonQuarantineOverride, $Requested, (Get-FleetTargetPath))
  }
  throw ("WATCHDOG_TARGET_CONFLICT: tick requested {0} worker(s) but {1} declares {2}; read the SSOT instead of passing a target" -f `
    $Requested, (Get-FleetTargetPath), $target)
}

# Assert-FleetTaskHasNoEmbeddedTarget refuses scheduled actions that still carry the
# stale population snapshot this SSOT replaces. Registrars call it both before install
# and while reporting status, so an old task cannot look healthy merely because the
# current script no longer writes -Target.
function Assert-FleetTaskHasNoEmbeddedTarget {
  param(
    [Parameter(Mandatory)] [AllowEmptyString()] [string]$Arguments,
    [string]$TaskName = 'fleet watchdog'
  )
  if ($Arguments -match '(?i)(?:^|\s)-Target(?:\s|=|$)') {
    throw ("EMBEDDED_WATCHDOG_TARGET: refusing {0}; scheduled action still embeds a population: {1}" -f `
      $TaskName, $Arguments)
  }
  return $true
}
# Set-FleetDesiredTarget declares a new population. Lowering (including all the way
# to quarantine) is always admitted: an operator must be able to stop the fleet with
# no ceremony. Raising it demands launcher admission, status agreement and a named
# canary, because that
# is the direction that starts spending.
function Set-FleetDesiredTarget {
  param(
    [Parameter(Mandatory)] [int]$Target,
    [string]$Reason = '',
    [string]$Issue = '',
    [switch]$LauncherAdmits,
    [switch]$StatusAgrees,
    [string]$Canary = ''
  )
  if ($Target -lt 0) { throw "refusing to declare target $Target; declare 0 to quarantine" }
  $current = Get-FleetDesiredTarget
  if ($Target -gt $current) {
    $missing = @()
    if (-not $LauncherAdmits) { $missing += 'launcher admission' }
    if (-not $StatusAgrees) { $missing += 'status agreement' }
    if (-not "$Canary".Trim()) { $missing += 'a witnessed canary' }
    if ($missing.Count -gt 0) {
      throw ("{0}: refusing to raise the declared population from {1} to {2} without {3}" -f `
        $script:FleetReasonPopulationUnwitnessed, $current, $Target, ($missing -join ', '))
    }
  }
  $dir = Get-FleetStateRoot
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  $rec = [ordered]@{
    target = $Target
    reason = $Reason
    issue  = $Issue
    set_by = "$env:USERNAME"
    set_at = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
  }
  if ("$Canary".Trim()) { $rec['canary'] = $Canary }
  $path = Get-FleetTargetPath
  Set-Content -Path $path -Value (($rec | ConvertTo-Json -Depth 3) + "`n") -Encoding UTF8
  return $path
}

# Invoked directly with -Set (rather than dot-sourced): declare the population.
if ($PSBoundParameters.ContainsKey('Set')) {
  $written = Set-FleetDesiredTarget -Target $Set -Reason $Reason -Issue $Issue `
    -LauncherAdmits:$LauncherAdmits -StatusAgrees:$StatusAgrees -Canary $Canary
  Write-Output ("declared desired worker population {0} in {1}" -f $Set, $written)
}

