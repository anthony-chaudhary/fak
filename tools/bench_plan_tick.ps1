<#
bench_plan_tick.ps1 -- one read-only tick of the hardware bench planner.

Computes a FRESH UTC --now stamp and renders the committed plan doc
(docs/bench-plan.md) from experiments/benchmark/catalog.json. The planner itself
(tools/bench_plan.py) reads NO wall-clock -- that is what keeps it deterministic and
unit-testable -- so THIS wrapper is the one place a real clock enters, exactly so the
regression/staleness math advances with wall time even on a quiet repo.

PURE FOLD: it WRITES only the working-tree doc and git-commits NOTHING. The tree is a
shared multi-session checkout where commits are by explicit path; an operator commits
docs/bench-plan.md when ready. No benchmark is ever run here -- this box is the
agent-host; a real run is a later action on a remote bench-node. The planner has no
execute mode, so there is nothing to "go live" on.

Invoked every tick by the FleetBenchPlanDoc scheduled task, or run by hand:

  .\tools\bench_plan_tick.ps1 -Workspace C:\work\fak
#>
[CmdletBinding()]
param(
  [string]$Workspace = $(Split-Path -Parent $PSScriptRoot),
  [string]$DocPath   = 'docs\bench-plan.md'
)
$ErrorActionPreference = 'Stop'

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "python not found on PATH" }

$tool = Join-Path $Workspace 'tools\bench_plan.py'
if (-not (Test-Path $tool)) { throw "bench_plan.py not found at $tool" }

# Fresh stamp per tick (the only wall-clock read in the whole planner path); the same
# compact UTC format the catalog and bench_plan.py use.
$now = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')

# Call python via the call operator with the resolved path in a variable -- a path under
# "C:\Program Files\..." (a space) is handled cleanly here, never serialized through a
# schtasks /TR string.
& $py $tool --workspace $Workspace --now $now --md $DocPath --json
exit $LASTEXITCODE
