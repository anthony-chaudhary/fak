<#
capture_fleet_task_xml.ps1 - export the fleet's live Windows Scheduled Tasks to
version-controlled, SCRUBBED task XML under tools/scheduled-tasks/ (#3323).

Why this exists: most fleet loops are rebuildable from a versioned
tools/register_*.ps1, but some are not -- they launch a script that lives outside
the repo, or they are a one-shot campaign loop with no installer at all. Those
tasks are invisible to the repo: reimage the host and the loop is simply gone,
with nothing left that even remembers it existed. An exported task XML is the
fallback source of truth for that residual -- it records the schedule, the
principal shape, and the exact command line the task ran, so the loop can be
rebuilt (or at minimum reconstructed) rather than lost.

What an XML capture does NOT do: it does not vendor the script the task launches.
For a task whose action points outside the repo, restoring the XML restores a task
pointing at a file a fresh host does not have. That residual is recorded per-task
in internal/taskvc/inventory.go, and it is why an installer is always preferred
over a capture.

SCRUB: a raw Export-ScheduledTask carries the host's identity -- the account SID,
COMPUTERNAME\user in <Author>, and absolute home paths. None of that belongs in a
public forever-history tree. Every value is read from the LIVE environment and
replaced with a placeholder, then the result is re-scanned and the write is
REFUSED if any of them survived. The scrub is mechanical on purpose: hand-editing
an export is how a SID reaches the public tree.

  %LOCALAPPDATA% / %APPDATA% / %USERPROFILE% / %COMPUTERNAME%  Windows expands these
      itself when the task runs, so they are lossless -- no substitution needed.
  %FLEET_TASK_USER_SID%  MUST be substituted before re-registering. Principal policy
      (S4U vs InteractiveToken) is #3322's concern, not this capture's.
  %GCP_PROJECT%  a cloud project id, redacted; substitute before re-registering.

Usage:
  .\tools\capture_fleet_task_xml.ps1                       # refresh every committed capture
  .\tools\capture_fleet_task_xml.ps1 -TaskName FleetFoo    # add/refresh one task
  .\tools\capture_fleet_task_xml.ps1 -Check                # drift check only, writes nothing

Restore (after substituting the non-expanding placeholders above):
  Register-ScheduledTask -TaskName FleetStrandedRecovery -Xml (Get-Content -Raw tools\scheduled-tasks\FleetStrandedRecovery.xml)
#>
[CmdletBinding()]
param(
  # Tasks to capture. Default: refresh whatever is already committed, so a bare
  # run never silently widens the captured set.
  [string[]]$TaskName = @(),
  [string]$OutDir = (Join-Path $PSScriptRoot 'scheduled-tasks'),
  # Re-export and compare against the committed capture without writing.
  [switch]$Check
)

$ErrorActionPreference = 'Stop'

# ---- the scrub map -------------------------------------------------------------
# Ordered longest-prefix-first: LOCALAPPDATA must be replaced before USERPROFILE,
# and COMPUTERNAME\user before a bare COMPUTERNAME, or the shorter match wins and
# leaves a half-scrubbed path behind.
$sid = ([System.Security.Principal.WindowsIdentity]::GetCurrent()).User.Value
$scrub = [ordered]@{}
$scrub[$env:LOCALAPPDATA]                     = '%LOCALAPPDATA%'
$scrub[$env:APPDATA]                          = '%APPDATA%'
$scrub[$env:USERPROFILE]                      = '%USERPROFILE%'
$scrub["$env:COMPUTERNAME\$env:USERNAME"]     = 'fak-fleet'
$scrub[$env:COMPUTERNAME]                     = '%COMPUTERNAME%'
$scrub[$sid]                                  = '%FLEET_TASK_USER_SID%'

# The bare username is deliberately NOT scrubbed on its own: it can be a short,
# generic word, and a blind replace would corrupt unrelated text. The home-path
# prefixes and the <Author> form above are what actually carry it.

# Secret-shaped values that are not host identity. Narrow by construction -- each
# anchors on the assignment that introduces it, so it cannot eat neighbouring text.
$redactions = @(
  @{ Pattern = '(?<=PROJECT=)[a-z0-9][-a-z0-9]{4,}'; Replacement = '%GCP_PROJECT%'; Label = 'cloud project id' }
)

function Get-ScrubbedTaskXml([string]$Name) {
  $xml = Export-ScheduledTask -TaskName $Name
  if (-not $xml) { throw "Export-ScheduledTask returned nothing for '$Name'" }
  $text = ($xml -join "`n")

  foreach ($k in $scrub.Keys) {
    if ([string]::IsNullOrEmpty($k)) { continue }
    $text = $text.Replace($k, $scrub[$k])
  }
  foreach ($r in $redactions) {
    $text = [regex]::Replace($text, $r.Pattern, $r.Replacement)
  }

  # Export-ScheduledTask hands back a UTF-16 declaration; we store UTF-8 bytes, so
  # the declaration has to agree or a downstream XML reader trips on the mismatch.
  $text = $text -replace '(?i)encoding="UTF-16"', 'encoding="UTF-8"'
  $text = ($text -replace "`r`n", "`n").TrimEnd() + "`n"

  # Fail closed: if any identity value survived the replace, refuse rather than
  # write it. This is the check that keeps a SID out of the forever history.
  foreach ($needle in @($sid, $env:COMPUTERNAME, $env:USERPROFILE)) {
    if ([string]::IsNullOrEmpty($needle)) { continue }
    if ($text -like "*$needle*") {
      throw "SCRUB_INCOMPLETE: '$Name' still contains a host identity value after scrubbing; refusing to write"
    }
  }
  foreach ($r in $redactions) {
    if ([regex]::IsMatch($text, $r.Pattern)) {
      throw "SCRUB_INCOMPLETE: '$Name' still contains a $($r.Label) after redaction; refusing to write"
    }
  }
  return $text
}

if (-not (Test-Path $OutDir)) {
  if ($Check) { Write-Error "no capture dir at $OutDir"; exit 1 }
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

# Default set = the already-committed captures. Explicit -TaskName adds to it, so
# growing the captured set is always a deliberate act.
$names = @($TaskName)
if (-not $names -or $names.Count -eq 0) {
  $names = @(Get-ChildItem -Path $OutDir -Filter '*.xml' -ErrorAction SilentlyContinue |
             ForEach-Object { $_.BaseName })
}
if (-not $names -or $names.Count -eq 0) {
  Write-Output "no tasks to capture (empty $OutDir and no -TaskName given)"
  exit 0
}

$drift = 0
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
foreach ($n in ($names | Sort-Object -Unique)) {
  $dest = Join-Path $OutDir "$n.xml"
  $live = Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue
  if (-not $live) {
    # A committed capture whose task is gone is a real signal, not an error: the
    # loop was retired and its row in internal/taskvc/inventory.go should go too.
    Write-Output "MISSING  $n (no live task; retire its capture + inventory row?)"
    $drift++
    continue
  }

  $text = Get-ScrubbedTaskXml -Name $n
  $prior = if (Test-Path $dest) { [System.IO.File]::ReadAllText($dest) } else { $null }

  if ($Check) {
    if ($null -eq $prior)      { Write-Output "UNCAPTURED  $n"; $drift++ }
    elseif ($prior -ne $text)  { Write-Output "DRIFT       $n (live task differs from the committed capture)"; $drift++ }
    else                       { Write-Output "ok          $n" }
    continue
  }

  if ($prior -eq $text) { Write-Output "unchanged   $n"; continue }
  [System.IO.File]::WriteAllText($dest, $text, $utf8NoBom)
  Write-Output $(if ($null -eq $prior) { "captured    $n" } else { "updated     $n" })
}

if ($Check -and $drift -gt 0) {
  Write-Error "$drift captured fleet task(s) drifted from the live host"
  exit 1
}
exit 0
