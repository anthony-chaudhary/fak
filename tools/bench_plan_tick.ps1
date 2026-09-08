<#
bench_plan_tick.ps1 -- one read-only tick of the hardware bench planner.

Computes a FRESH UTC --now stamp and renders the plan doc
(.fak\bench-plan.md) from experiments/benchmark/catalog.json. The planner itself
(tools/bench_plan.py) reads NO wall-clock -- that is what keeps it deterministic and
unit-testable -- so THIS wrapper is the one place a real clock enters, exactly so the
regression/staleness math advances with wall time even on a quiet repo.

PURE FOLD: it WRITES only the working-tree doc and git-commits NOTHING. The tree is a
shared multi-session checkout where commits are by explicit path; an operator commits
the doc when ready. No benchmark is ever run here -- this box is the
agent-host; a real run is a later action on a remote bench-node. The planner has no
execute mode, so there is nothing to "go live" on.

Invoked every tick by the FleetBenchPlanDoc scheduled task, or run by hand:

  .\tools\bench_plan_tick.ps1 -Workspace C:\work\fak
#>
[CmdletBinding()]
param(
  [string]$Workspace = $(Split-Path -Parent $PSScriptRoot),
  [string]$DocPath   = '.fak\bench-plan.md'
)
$ErrorActionPreference = 'Stop'

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "python not found on PATH" }

$tool = Join-Path $Workspace 'tools\bench_plan.py'
if (-not (Test-Path $tool)) { throw "bench_plan.py not found at $tool" }

$catalog = Join-Path $Workspace 'experiments\benchmark\catalog.json'
$shaFile = Join-Path $Workspace '.fak\bench-plan-last-catalog.sha'
$docFull = if ([System.IO.Path]::IsPathRooted($DocPath)) { $DocPath } else { Join-Path $Workspace $DocPath }

$curHash = $null
if (Test-Path $catalog) {
  $curHash = (Get-FileHash -Algorithm SHA256 -Path $catalog).Hash
  $lastHash = if (Test-Path $shaFile) { (Get-Content -Path $shaFile -Raw -ErrorAction SilentlyContinue).Trim() } else { $null }
  if ($curHash -and ($curHash -eq $lastHash) -and (Test-Path $docFull)) {
    exit 0
  }
}

# Fresh stamp per tick (the only wall-clock read in the whole planner path); the same
# compact UTC format the catalog and bench_plan.py use.
$now = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')

# Call python via the call operator with the resolved path in a variable -- a path under
# "C:\Program Files\..." (a space) is handled cleanly here, never serialized through a
# schtasks /TR string.
& $py $tool --workspace $Workspace --now $now --md $DocPath --json
$code = $LASTEXITCODE
if ($code -eq 0 -and $curHash) {
  $shaDir = Split-Path -Parent $shaFile
  if (-not (Test-Path $shaDir)) { New-Item -ItemType Directory -Force -Path $shaDir | Out-Null }
  Set-Content -Path $shaFile -Value $curHash -NoNewline
}
exit $code
