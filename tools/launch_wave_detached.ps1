<#
.SYNOPSIS
  Launch a WAVE of detached `/goal` workers -- one per DISTINCT account -- so a
  parallel fan-out draws on N independent rate-limit buckets instead of piling
  every lane onto one. The multi-account twin of launch_goal_detached.ps1.

.WHY
  launch_goal_detached.ps1 resolves ONE account (the best one) and launches ONE
  worker. A fan-out that calls it N times in a burst gets the SAME account N times
  -- no session has registered yet to move the switcher's fewest-live tie-break --
  so all N workers share ONE usage pool and the fan-out serializes (witnessed: 3
  resolves -> the same tag thrice while 3 distinct pools sat free). This launcher
  asks the switcher for N DISTINCT pools in ONE call (`fleet_accounts.py wave`),
  then dispatches one detached worker per pool. Distinctness is by Anthropic
  accountUuid, so two dirs on one account never both get a lane.

  It does NOT re-implement the spawn: the dangerous part (Start-Process wiring,
  CLAUDE_CONFIG_DIR / CLAUDE_CODE_OAUTH_TOKEN pinning, stdin-fed goal) is the
  already-proven launch_goal_detached.ps1, invoked once per lane with -Account
  pinned to that lane's tag. This script owns only the ALLOCATION + ITERATION.

  PLAN BY DEFAULT. With no -Launch it prints the dispatch plan (which account, dir,
  tier, pool each lane would take) and spawns NOTHING -- safe to run anywhere, and
  the witnessable artifact. Pass -Launch to actually dispatch the wave.

.EXAMPLE
  # See the plan (no spawn): up to 8 distinct tier-1 pools for an engineering wave
  .\tools\launch_wave_detached.ps1 -Count 8 -WorkKind engineering

.EXAMPLE
  # Actually dispatch: one detached worker per distinct account
  .\tools\launch_wave_detached.ps1 -Count 8 -WorkKind engineering -Launch
#>
[CmdletBinding()]
param(
  # How many distinct-account lanes to allocate. The wave under-fills honestly if
  # fewer distinct accounts are available (granted < count, with a reported shortfall).
  [int]$Count = 4,
  [string]$PointerFile = ".claude/goal-prompts/resolve-tickets-witnessed.md",
  [string]$Workspace   = "C:\work\fleet",
  [string]$LogDir      = "C:\work\fleet\.goal-runs",
  [ValidateSet('engineering','eng','dev','feature','implementation',
               'gardening','garden','maintenance','maint','cleanup','chore','triage','')]
  [string]$WorkKind    = 'engineering',
  [ValidateSet('auto','t1','t2','t3','1','2','3')]
  [string]$Tier        = 'auto',
  [string]$Product     = 'claude',
  [switch]$AllowTierFallback,
  # Actually spawn the workers. Without it, this is a dry-run that only prints the plan.
  [switch]$Launch
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot   # tools/ -> repo root

# --- Ask the switcher for N DISTINCT pools in ONE call --------------------------------
$py = if (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { 'python3' }
$waveArgs = @((Join-Path $repoRoot 'tools\fleet_accounts.py'), 'wave',
              '--count', "$Count", '--product', $Product)
if ($WorkKind)          { $waveArgs += @('--work-kind', $WorkKind) }
else                    { $waveArgs += @('--tier', $Tier) }
if ($AllowTierFallback) { $waveArgs += '--allow-tier-fallback' }

$tmpOut = Join-Path ([System.IO.Path]::GetTempPath()) ("wave-{0}.json" -f ([Guid]::NewGuid().ToString('N')))
& $py @waveArgs > $tmpOut 2>$null
$rc = $LASTEXITCODE
$w = $null
if (Test-Path $tmpOut) { try { $w = Get-Content -Raw $tmpOut | ConvertFrom-Json } catch { $w = $null }; Remove-Item $tmpOut -ErrorAction SilentlyContinue }
if (-not $w)    { throw "wave allocation produced no JSON (python=$py, rc=$rc) -- cannot dispatch" }
if (-not $w.ok) { throw "account switcher refused the wave: $($w.reason) -- re-login / wait for reset, or pass -AllowTierFallback." }

# --- Print the plan (always) ----------------------------------------------------------
Write-Output ("WAVE PLAN  requested={0}  granted={1}  shortfall={2}  distinct_pools={3}  target_tier=t{4}" -f `
  $w.requested, $w.granted, $w.shortfall, $w.distinct_pools, $w.target_tier)
Write-Output "  (naive burst would give 1 pool; this wave gives $($w.distinct_pools) -> $($w.distinct_pools)x rate-limit headroom)"
$lane = 0
$w.lanes | ForEach-Object {
  $lane++
  Write-Output ("  lane {0}: {1,-18} t{2}  pool={3}  dir={4}" -f $lane, $_.tag, $_.selected_tier, $_.pool, $_.config_dir)
}
if ($w.shortfall -gt 0) {
  Write-Output "  note: $($w.shortfall) lane(s) short -- the roster has no more distinct available pools at the requested tier."
}

if (-not $Launch) {
  Write-Output ""
  Write-Output "DRY RUN -- no workers spawned. Re-run with -Launch to dispatch one detached worker per lane."
  return
}

# --- Dispatch one detached worker per lane, each pinned to its distinct account --------
# Reuse the proven single-worker launcher (Start-Process wiring, env pinning, stdin goal);
# -Account pins this lane's exact pool so the N workers never re-collapse onto one bucket.
$launcher = Join-Path $repoRoot 'tools\launch_goal_detached.ps1'
$results = @()
$lane = 0
foreach ($l in $w.lanes) {
  $lane++
  Write-Output "`n--- dispatching lane $lane/$($w.granted): account '$($l.tag)' (pool $($l.pool)) ---"
  try {
    # Forward by HASHTABLE SPLAT, not an inline @(if...) array: an inline array binds as a
    # single positional arg, so `-AllowTierFallback` is silently DROPPED (a tier-1 lane with
    # no free pool would then be refused instead of falling back). A splat sets the switch.
    $fwd = @{
      PointerFile = $PointerFile
      Workspace   = $Workspace
      LogDir      = $LogDir
      Account     = $l.tag
      WorkKind    = $WorkKind
    }
    if ($AllowTierFallback) { $fwd.AllowTierFallback = $true }
    & $launcher @fwd
    $results += [pscustomobject]@{ lane = $lane; account = $l.tag; pool = $l.pool; dispatched = $true }
  } catch {
    Write-Warning "lane $lane ($($l.tag)) failed to dispatch: $_"
    $results += [pscustomobject]@{ lane = $lane; account = $l.tag; pool = $l.pool; dispatched = $false }
  }
}

$ok = ($results | Where-Object { $_.dispatched }).Count
Write-Output "`nWAVE DISPATCHED  $ok/$($w.granted) lanes live across $ok distinct rate-limit pool(s)."
$results | Format-Table -AutoSize
