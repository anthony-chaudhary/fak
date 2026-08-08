<#
.SYNOPSIS
  Launch a WAVE of detached `/goal` workers -- one per bounded account session
  slot -- so a parallel fan-out uses the available Claude account capacity instead
  of piling every lane onto one. The multi-account twin of launch_goal_detached.ps1.

.WHY
  launch_goal_detached.ps1 resolves ONE account (the best one) and launches ONE
  worker. A fan-out that calls it N times in a burst gets the SAME account N times
  -- no session has registered yet to move the switcher's fewest-live tie-break --
  so all N workers share ONE usage pool and the fan-out serializes (witnessed: 3
  resolves -> the same tag thrice while 3 distinct pools sat free). This launcher
  asks the switcher for N bounded account session slots in ONE call
  (`fak fleet-accounts wave`), then dispatches one detached worker per slot.
  Distinctness is by Anthropic accountUuid, so two dirs on one account never both
  inflate the pool count.

  It does NOT re-implement the spawn: the dangerous part (Start-Process wiring,
  CLAUDE_CONFIG_DIR / CLAUDE_CODE_OAUTH_TOKEN pinning, guarded-session env
  stripping, the dispatch_preflight.py spawn gate, stdin-fed goal) is the
  already-proven launch_goal_detached.ps1, invoked once per lane with -Account
  pinned to that lane's tag. This script owns only the ALLOCATION + ITERATION.
  Before account allocation, this script asks dispatch_preflight.py how many lanes
  the host/seat pool admits and lowers the wave request to that headroom. Because
  every lane still dispatches through the gated single-worker launcher, the
  preflight cap is also re-checked PER SPAWN: a wave honestly under-fills
  mid-flight the moment the host, seat pool, or cap refuses — it never routes
  around a REFUSE_*.

  PLAN BY DEFAULT. With no -Launch it prints the dispatch plan (which account, dir,
  tier, pool each lane would take) and spawns NOTHING -- safe to run anywhere, and
  the witnessable artifact. Pass -Launch to actually dispatch the wave.

.EXAMPLE
  # See the plan (no spawn): up to 20 tier-1 account session slots for an engineering wave
  .\tools\launch_wave_detached.ps1 -Count 20 -WorkKind engineering

.EXAMPLE
  # Actually dispatch: one detached worker per bounded account session slot
  .\tools\launch_wave_detached.ps1 -Count 20 -WorkKind engineering -Launch
#>
[CmdletBinding()]
param(
  # How many bounded account session slots to allocate. The wave under-fills honestly
  # if fewer slots are available (granted < count, with a reported shortfall).
  # Default mirrors the preflight ceiling (built-in 20, FAK_MAX_WORKERS retunes it);
  # the wave preflight and per-spawn preflight gates both bound it below that ceiling.
  [int]$Count = $(if ($env:FAK_MAX_WORKERS -match '^[1-9]\d*$') { [int]$env:FAK_MAX_WORKERS } else { 20 }),
  [string]$PointerFile = ".claude/goal-prompts/resolve-tickets-witnessed.md",
  [string]$Workspace   = "C:\work\fleet",
  [string]$LogDir      = "C:\work\fleet\.goal-runs",
  [ValidateSet('engineering','eng','dev','feature','implementation',
               'gardening','garden','maintenance','maint','cleanup','chore','triage','')]
  [string]$WorkKind    = 'engineering',
  [ValidateSet('auto','t1','t2','t3','1','2','3')]
  [string]$Tier        = 'auto',
  [string]$Product     = 'claude',
  # Optional fak binary. Empty probes this repo's tools\.bin/fak.exe, repo-root fak.exe,
  # then PATH fak.
  [string]$FakExe      = '',
  [switch]$AllowTierFallback,
  # Operator ceiling for the per-spawn preflight gate (0 = use -Count: the wave you
  # asked for IS your aspirational cap; host_cap / dos target still bound it below).
  [int]$PreflightMaxWorkers = 0,
  # Skip the per-spawn dispatch_preflight.py gate in the child launcher. An EXPLICIT
  # operator override that removes the no-DoS floor for the whole wave. Never automate.
  [switch]$SkipPreflight,
  # Actually spawn the workers. Without it, this is a dry-run that only prints the plan.
  [switch]$Launch,
  # Emit a machine-readable dry-run plan and spawn nothing. Refuses with -Launch.
  [switch]$Json
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot   # tools/ -> repo root
if ($Json -and $Launch) {
  throw "-Json is a dry-run plan format; remove -Launch or run the text launcher explicitly."
}

function Resolve-FakExe {
  param([string]$RepoRoot, [string]$Explicit)
  function Test-FakWaveCount {
    param([string]$Candidate)
    $oldErrorAction = $ErrorActionPreference
    $oldNativeError = $null
    $hadNativeError = Test-Path variable:PSNativeCommandUseErrorActionPreference
    try {
      $ErrorActionPreference = 'Continue'
      if ($hadNativeError) {
        $oldNativeError = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
      }
      $help = & $Candidate 'fleet-accounts' 'wave' '-h' 2>&1 | Out-String
    } catch {
      return $false
    } finally {
      if ($hadNativeError) {
        $PSNativeCommandUseErrorActionPreference = $oldNativeError
      }
      $ErrorActionPreference = $oldErrorAction
    }
    return ($help -match '(?m)^\s*-count\b' -or $help -match '(?m)^\s*--count\b')
  }
  function Resolve-FakCandidate {
    param([string]$Candidate)
    if (Test-Path $Candidate) { return (Resolve-Path $Candidate).Path }
    $cmd = Get-Command $Candidate -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return ''
  }
  if ($Explicit) {
    $resolved = Resolve-FakCandidate -Candidate $Explicit
    if (-not $resolved) { throw "fak binary not found: $Explicit" }
    if (-not (Test-FakWaveCount -Candidate $resolved)) {
      throw "fak binary does not support 'fleet-accounts wave --count': $resolved"
    }
    return $resolved
  }
  $candidates = @(
    (Join-Path $RepoRoot 'tools\.bin\fak.exe'),
    (Join-Path $RepoRoot 'tools\.bin\fak'),
    (Join-Path $RepoRoot 'fak.exe'),
    (Join-Path $RepoRoot 'fak')
  )
  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      $resolved = (Resolve-Path $candidate).Path
      if (Test-FakWaveCount -Candidate $resolved) { return $resolved }
    }
  }
  $cmd = Get-Command fak -ErrorAction SilentlyContinue
  if ($cmd -and (Test-FakWaveCount -Candidate $cmd.Source)) { return $cmd.Source }
  throw "no compatible fak binary found with 'fleet-accounts wave --count' (looked in $RepoRoot\tools\.bin, repo root, and PATH; rebuild fak or pass -FakExe)"
}

function Resolve-PythonExe {
  $pyCmd = Get-Command python -ErrorAction SilentlyContinue
  if ($pyCmd) { return @{ Exe = $pyCmd.Source; Prefix = @() } }
  $pyCmd = Get-Command py -ErrorAction SilentlyContinue
  if ($pyCmd) { return @{ Exe = $pyCmd.Source; Prefix = @('-3') } }
  throw "wave preflight needs python on PATH and none was found (fail-safe: REFUSE_INSPECT) -- fix python, or pass -SkipPreflight to explicitly accept an ungated wave"
}

function Invoke-AdmissionGate {
  # The single launch-admission gate (#617/#3552): self-gate BEFORE a spawn so a wave
  # cannot fire N launches onto one throttled/over-ceiling account with no rate check
  # (the 2026-06-24 q-netra storm class). It does NOT launch -- it returns a verdict:
  # exit 0 = ADMIT, exit 3 = a structured DEFER (LAUNCH_RATE_EXCEEDED / GLOBAL_LAUNCH_CAP
  # / ACCOUNT_THROTTLED) with a retry_after. Fail-OPEN by doctrine: if the gate itself
  # cannot run (no python, crash, usage error) we LOG and PROCEED -- the gate exists to
  # stop a self-inflicted storm, never to wedge the whole wave on its own malfunction.
  param([string]$RepoRoot, [string]$Account)
  $result = [ordered]@{ verdict = 'ADMIT'; reason = ''; retry_after = ''; failopen = $false }
  $py = $null
  try { $py = Resolve-PythonExe } catch { $py = $null }
  if (-not $py) {
    $result.failopen = $true; $result.reason = 'no python for admission gate'; return $result
  }
  $gate = Join-Path $RepoRoot 'tools\launch_admission.py'
  if (-not (Test-Path $gate)) {
    $result.failopen = $true; $result.reason = "admission gate missing: $gate"; return $result
  }
  # --record appends the admitted launch to the durable ledger so the i-th spawn in the
  # fan-out sees the i-1 already recorded (the gate is stateful across the wave).
  $gArgs = @($py.Prefix) + @($gate, 'admit', '--account', $Account, '--record')
  $raw = ''
  $rc = $null
  $oldErrorAction = $ErrorActionPreference
  $hadNativeError = Test-Path variable:PSNativeCommandUseErrorActionPreference
  $oldNativeError = $null
  try {
    # A DEFER is a non-zero (3) exit; without pinning these off, Stop + native error
    # action would THROW on the DEFER and misroute it into the fail-open path.
    $ErrorActionPreference = 'Continue'
    if ($hadNativeError) {
      $oldNativeError = $PSNativeCommandUseErrorActionPreference
      $PSNativeCommandUseErrorActionPreference = $false
    }
    $raw = & $py.Exe @gArgs 2>$null | Out-String
    $rc = $LASTEXITCODE
  } catch {
    $result.failopen = $true
    $result.reason = "admission gate error: $_"
    return $result
  } finally {
    if ($hadNativeError) { $PSNativeCommandUseErrorActionPreference = $oldNativeError }
    $ErrorActionPreference = $oldErrorAction
  }
  if ($rc -eq 3) {
    $result.verdict = 'DEFER'
    try {
      $v = $raw | ConvertFrom-Json
      if ($v.PSObject.Properties['reason'] -and $v.reason) { $result.reason = [string]$v.reason }
      if ($v.PSObject.Properties['retry_after'] -and $v.retry_after) { $result.retry_after = [string]$v.retry_after }
    } catch { }
    return $result
  }
  if ($rc -ne 0) {
    # Any other non-zero exit (usage error, crash) is a gate malfunction, not a DEFER.
    $result.failopen = $true
    $result.reason = "admission gate exit $rc"
  }
  return $result
}

function Invoke-SpawnPacing {
  # Inter-spawn pacing: spread the fan-out so a wave cannot hammer one account in a
  # sub-second burst. Base delay is env-overridable (FAK_LAUNCH_SPAWN_PACING_MS,
  # default 300ms); a Get-Random jitter of up to +50% de-synchronizes the spawns.
  # A base of 0 disables pacing (operator override for a hermetic dry-run cadence).
  $baseMs = 300
  if ($env:FAK_LAUNCH_SPAWN_PACING_MS -match '^\d+$') { $baseMs = [int]$env:FAK_LAUNCH_SPAWN_PACING_MS }
  if ($baseMs -le 0) { return }
  $jitterMs = Get-Random -Minimum 0 -Maximum ([Math]::Max(1, [int]($baseMs / 2)))
  Start-Sleep -Milliseconds ($baseMs + $jitterMs)
}

function Invoke-WavePreflight {
  param(
    [string]$RepoRoot,
    [string]$Workspace,
    [int]$MaxWorkers,
    [string]$WorkKind,
    [string]$Product
  )
  $py = Resolve-PythonExe
  $pfArgs = @($py.Prefix) + @((Join-Path $RepoRoot 'tools\dispatch_preflight.py'), '--json', '--workspace', "$Workspace", '--max-workers', "$MaxWorkers")
  if ($WorkKind) { $pfArgs += @('--work-kind', $WorkKind) }
  if ($Product)  { $pfArgs += @('--product', $Product) }
  # A benign advisory on the preflight's stderr (dispatch_preflight's DIRTY_FAK_BIN note is
  # the live one, #5856) is turned into a NativeCommandError ErrorRecord whenever THIS host's
  # own stderr is redirected -- which it is under every non-console launcher: a cron, a CI
  # step, a bash-driven refill loop. Under the script-scope $ErrorActionPreference='Stop' that
  # record is TERMINATING, so the whole wave aborts before a single spawn, on a warning. Scope
  # the preference to Continue for exactly this call: stderr stays discarded, the JSON verdict
  # on stdout still decides, and a preflight that genuinely fails still lands on the
  # no-verdict throw below (fail-safe REFUSE_INSPECT). The gate is not weakened, only its
  # advisory chatter stops being fatal.
  $prevEap = $ErrorActionPreference
  try {
    $ErrorActionPreference = 'Continue'
    $pfRaw = & $py.Exe @pfArgs 2>$null | Out-String
  } finally { $ErrorActionPreference = $prevEap }
  $pf = $null
  try { $pf = $pfRaw | ConvertFrom-Json } catch { $pf = $null }
  if (-not $pf -or -not $pf.verdict) {
    throw "wave preflight produced no verdict (python=$($py.Exe), rc=$LASTEXITCODE; fail-safe: REFUSE_INSPECT) -- not allocating"
  }
  return $pf
}

function Get-ObjectInt {
  param([object]$Object, [string]$Name)
  if ($null -eq $Object) { return $null }
  $prop = $Object.PSObject.Properties[$Name]
  if ($null -eq $prop -or $null -eq $prop.Value) { return $null }
  try { return [int]$prop.Value } catch { return $null }
}

function Get-WaveAllocationCount {
  param([int]$Requested, [object]$Preflight)
  $n = [Math]::Max(0, $Requested)
  if ($null -eq $Preflight) { return $n }
  if ($Preflight.verdict -and $Preflight.verdict -ne 'SPAWN_OK') { return 0 }
  $headroom = Get-ObjectInt -Object $Preflight -Name 'headroom'
  if ($null -ne $headroom) { $n = [Math]::Min($n, [Math]::Max(0, $headroom)) }
  if ($Preflight.seat) {
    $free = Get-ObjectInt -Object $Preflight.seat -Name 'free'
    if ($null -ne $free) { $n = [Math]::Min($n, [Math]::Max(0, $free)) }
  }
  return [Math]::Max(0, $n)
}

function Format-WavePreflightLine {
  param([object]$Preflight)
  if ($null -eq $Preflight) { return "" }
  $seatFree = $null
  if ($Preflight.seat) { $seatFree = Get-ObjectInt -Object $Preflight.seat -Name 'free' }
  $parts = @("  preflight: $($Preflight.verdict)", "live=$($Preflight.live)", "cap=$($Preflight.cap)", "headroom=$($Preflight.headroom)")
  if ($null -ne $seatFree) { $parts += "seat_free=$seatFree" }
  if ($Preflight.reason) { $parts += "reason=$($Preflight.reason)" }
  return ($parts -join "  ")
}

function Add-IfPresent {
  param([System.Collections.IDictionary]$Map, [string]$Name, [object]$Value)
  if ($null -ne $Value) { $Map[$Name] = $Value }
}

function Convert-WavePreflightPublic {
  param([object]$Preflight)
  if ($null -eq $Preflight) { return $null }
  $out = [ordered]@{
    verdict = $Preflight.verdict
    reason  = $Preflight.reason
    cap     = Get-ObjectInt -Object $Preflight -Name 'cap'
    live    = Get-ObjectInt -Object $Preflight -Name 'live'
  }
  Add-IfPresent -Map $out -Name 'headroom' -Value (Get-ObjectInt -Object $Preflight -Name 'headroom')
  Add-IfPresent -Map $out -Name 'max_workers' -Value (Get-ObjectInt -Object $Preflight -Name 'max_workers')
  Add-IfPresent -Map $out -Name 'host_cap' -Value (Get-ObjectInt -Object $Preflight -Name 'host_cap')
  if ($Preflight.capacity_limiter) { $out['capacity_limiter'] = $Preflight.capacity_limiter }
  if ($Preflight.seat) {
    $seat = [ordered]@{}
    foreach ($name in @('total', 'free', 'leased', 'unattributed_live')) {
      Add-IfPresent -Map $seat -Name $name -Value (Get-ObjectInt -Object $Preflight.seat -Name $name)
    }
    if ($null -ne $Preflight.seat.depleted) { $seat['depleted'] = [bool]$Preflight.seat.depleted }
    $out['seat'] = $seat
  }
  return $out
}

function New-WavePlanPayload {
  param(
    [int]$Requested,
    [int]$AllocationRequested,
    [int]$Granted,
    [int]$Shortfall,
    [int]$PreflightShortfall,
    [object]$Wave,
    [object]$Preflight,
    [string]$Workspace,
    [string]$WorkKind,
    [string]$Product,
    [string]$PointerFile,
    [string]$Reason = ''
  )
  $lanes = @()
  $index = 0
  if ($Wave -and $Wave.lanes) {
    foreach ($lane in @($Wave.lanes)) {
      $laneRank = Get-ObjectInt -Object $lane -Name 'rank'
      if ($null -eq $laneRank) { $laneRank = $index }
      $laneEntry = [ordered]@{
        rank = $laneRank
        tag = $lane.tag
        selected_tier = Get-ObjectInt -Object $lane -Name 'selected_tier'
        session_slot = Get-ObjectInt -Object $lane -Name 'session_slot'
        session_cap = Get-ObjectInt -Object $lane -Name 'session_cap'
        pool = $lane.pool
        config_dir = $lane.config_dir
      }
      Add-IfPresent -Map $laneEntry -Name 'wave_id' -Value $lane.wave_id
      Add-IfPresent -Map $laneEntry -Name 'size' -Value (Get-ObjectInt -Object $lane -Name 'size')
      $lanes += $laneEntry
      $index++
    }
  }
  $preflightVerdict = ''
  if ($Preflight -and $Preflight.PSObject.Properties['verdict'] -and $Preflight.verdict) {
    $preflightVerdict = [string]$Preflight.verdict
  }
  $ok = $Granted -gt 0
  $verdict = 'WAVE_EMPTY'
  $action = 'refused'
  if ($ok) {
    $verdict = 'WOULD_WAVE'
    $action = 'would_wave'
  } elseif ($preflightVerdict -and $preflightVerdict -ne 'SPAWN_OK') {
    $verdict = $preflightVerdict
  } elseif ($AllocationRequested -gt 0) {
    $verdict = 'WAVE_NO_SEATS'
    $action = 'no_seats'
  }
  $payload = [ordered]@{
    schema = 'fleet-launch-wave-detached-plan/1'
    ok = $ok
    verdict = $verdict
    action = $action
    workspace = $Workspace
    live = $false
    launch = $false
    requested = $Requested
    allocation_requested = $AllocationRequested
    granted = $Granted
    size = $(if ($Wave) { Get-ObjectInt -Object $Wave -Name 'size' } else { $Granted })
    wave_id = $(if ($Wave -and $Wave.PSObject.Properties['wave_id']) { $Wave.wave_id } else { $null })
    shortfall = $Shortfall
    preflight_shortfall = $PreflightShortfall
    distinct_pools = $(if ($Wave) { Get-ObjectInt -Object $Wave -Name 'distinct_pools' } else { 0 })
    target_tier = $(if ($Wave) { Get-ObjectInt -Object $Wave -Name 'target_tier' } else { $null })
    work_kind = $WorkKind
    product = $Product
    pointer_file = $PointerFile
    preflight = Convert-WavePreflightPublic -Preflight $Preflight
    lanes = $lanes
    reason = $Reason
  }
  return $payload
}

# --- Ask the switcher for N bounded session slots in ONE call -------------------------
$fak = Resolve-FakExe -RepoRoot $repoRoot -Explicit $FakExe
$allocationCount = $Count
$preflightShortfall = 0
$wavePreflight = $null
if (-not $SkipPreflight) {
  $wavePreflightMaxWorkers = $(if ($PreflightMaxWorkers -gt 0) { $PreflightMaxWorkers } else { $Count })
  $wavePreflight = Invoke-WavePreflight -RepoRoot $repoRoot -Workspace $Workspace -MaxWorkers $wavePreflightMaxWorkers -WorkKind $WorkKind -Product $Product
  $allocationCount = Get-WaveAllocationCount -Requested $Count -Preflight $wavePreflight
  $preflightShortfall = [Math]::Max(0, $Count - $allocationCount)
}

if ($allocationCount -le 0) {
  if ($Json) {
    New-WavePlanPayload -Requested $Count -AllocationRequested 0 -Granted 0 `
      -Shortfall $Count -PreflightShortfall $Count -Wave $null `
      -Preflight $wavePreflight -Workspace $Workspace -WorkKind $WorkKind `
      -Product $Product -PointerFile $PointerFile `
      -Reason 'preflight admitted zero lanes; account allocation skipped' |
      ConvertTo-Json -Depth 12
    return
  }
  Write-Output ("WAVE PLAN  requested={0}  allocation_requested=0  granted=0  shortfall={0}  distinct_pools=0  target_tier=unknown" -f $Count)
  $pfLine = Format-WavePreflightLine -Preflight $wavePreflight
  if ($pfLine) { Write-Output $pfLine }
  Write-Output "  note: preflight admitted zero lane(s), so account allocation was skipped."
  if ($Launch) {
    throw "wave preflight admitted zero lane(s) -- not dispatching. The refusal is the no-DoS floor; recover (fix host / re-login / wait for seats), do not route around it."
  }
  Write-Output ""
  Write-Output "DRY RUN -- no workers spawned. Re-run after preflight headroom returns."
  return
}

$waveArgs = @('fleet-accounts', 'wave', '--count', "$allocationCount", '--product', $Product)
if ($WorkKind)          { $waveArgs += @('--work-kind', $WorkKind) }
else {
  switch ($Tier) {
    { $_ -in @('t1','1') } { $waveArgs += '--t1'; break }
    { $_ -in @('t2','2') } { $waveArgs += '--t2'; break }
    { $_ -in @('t3','3') } { $waveArgs += '--t3'; break }
  }
}
if ($AllowTierFallback) { $waveArgs += '--allow-tier-fallback' }

$tmpOut = Join-Path ([System.IO.Path]::GetTempPath()) ("wave-{0}.json" -f ([Guid]::NewGuid().ToString('N')))
Push-Location $Workspace
try {
  & $fak @waveArgs > $tmpOut 2>$null
  $rc = $LASTEXITCODE
} finally {
  Pop-Location
}
$w = $null
if (Test-Path $tmpOut) { try { $w = Get-Content -Raw $tmpOut | ConvertFrom-Json } catch { $w = $null }; Remove-Item $tmpOut -ErrorAction SilentlyContinue }
if (-not $w)    { throw "wave allocation produced no JSON (fak=$fak, rc=$rc) -- cannot dispatch" }
if (-not $w.ok) {
  if ($Json) {
    New-WavePlanPayload -Requested $Count -AllocationRequested $allocationCount `
      -Granted 0 -Shortfall $Count -PreflightShortfall $preflightShortfall `
      -Wave $w -Preflight $wavePreflight -Workspace $Workspace -WorkKind $WorkKind `
      -Product $Product -PointerFile $PointerFile -Reason $w.reason |
      ConvertTo-Json -Depth 12
    return
  }
  throw "account switcher refused the wave: $($w.reason) -- re-login / wait for reset, or pass -AllowTierFallback."
}

# --- Print the plan (always) ----------------------------------------------------------
$totalShortfall = [int]$w.shortfall + $preflightShortfall
if ($Json) {
  New-WavePlanPayload -Requested $Count -AllocationRequested $allocationCount `
    -Granted ([int]$w.granted) -Shortfall $totalShortfall `
    -PreflightShortfall $preflightShortfall -Wave $w -Preflight $wavePreflight `
    -Workspace $Workspace -WorkKind $WorkKind -Product $Product `
    -PointerFile $PointerFile -Reason $w.reason |
    ConvertTo-Json -Depth 12
  return
}
Write-Output ("WAVE PLAN  requested={0}  allocation_requested={1}  granted={2}  shortfall={3}  distinct_pools={4}  target_tier=t{5}" -f `
  $Count, $allocationCount, $w.granted, $totalShortfall, $w.distinct_pools, $w.target_tier)
$pfLine = Format-WavePreflightLine -Preflight $wavePreflight
if ($pfLine) { Write-Output $pfLine }
Write-Output "  (naive burst would give 1 pool; this wave uses $($w.distinct_pools) distinct pool(s) and $($w.granted) bounded session slot(s))"
$lane = 0
$w.lanes | ForEach-Object {
  $lane++
  $slot = if ($_.session_slot) { "$($_.session_slot)/$($_.session_cap)" } else { "1/1" }
  Write-Output ("  lane {0}: {1,-18} t{2}  slot={3}  pool={4}  dir={5}" -f $lane, $_.tag, $_.selected_tier, $slot, $_.pool, $_.config_dir)
}
if ($preflightShortfall -gt 0) {
  Write-Output "  note: $preflightShortfall lane(s) held by preflight headroom/seat limits before account allocation."
}
if ($w.shortfall -gt 0) {
  Write-Output "  note: $($w.shortfall) lane(s) short after preflight -- the roster has no more available session slots at the requested tier."
}

if (-not $Launch) {
  Write-Output ""
  Write-Output "DRY RUN -- no workers spawned. Re-run with -Launch to dispatch one detached worker per lane."
  return
}

# --- Dispatch one detached worker per lane, each pinned to its allocated account slot ---
# Reuse the proven single-worker launcher (Start-Process wiring, env pinning, stdin goal);
# -Account pins this lane's exact pool so the N workers never re-collapse onto one bucket.
$launcher = Join-Path $repoRoot 'tools\launch_goal_detached.ps1'
$results = @()
$lane = 0
foreach ($l in $w.lanes) {
  $lane++
  # Inter-spawn pacing BETWEEN spawns (never before the first) so the wave fires as a
  # spread cadence, not a sub-second burst onto whichever pools it drew.
  if ($lane -gt 1) { Invoke-SpawnPacing }
  # Launch-admission gate: self-gate BEFORE the spawn. A structured DEFER (exit 3)
  # SKIPS this lane and logs the reason rather than firing; a gate malfunction fails
  # OPEN (logs + proceeds) so a broken gate never wedges the whole wave.
  $adm = Invoke-AdmissionGate -RepoRoot $repoRoot -Account $l.tag
  if ($adm.verdict -eq 'DEFER') {
    $ra = if ($adm.retry_after) { " retry_after=$($adm.retry_after)" } else { "" }
    Write-Output "`n--- lane $lane/$($w.granted): account '$($l.tag)' DEFERRED by admission gate [$($adm.reason)]$ra -- skipping spawn ---"
    $results += [pscustomobject]@{ lane = $lane; account = $l.tag; pool = $l.pool; dispatched = $false; deferred = $true; reason = $adm.reason }
    continue
  }
  if ($adm.failopen) {
    Write-Warning "lane $lane ($($l.tag)): admission gate unavailable ($($adm.reason)) -- failing OPEN, proceeding with spawn"
  }
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
      FakExe      = $fak
      # The wave's requested size is the operator ceiling the per-spawn gate enforces;
      # the adaptive gates (host_cap, dos [supervise].target, seats) only lower it.
      PreflightMaxWorkers = $(if ($PreflightMaxWorkers -gt 0) { $PreflightMaxWorkers } else { $Count })
    }
    if ($AllowTierFallback) { $fwd.AllowTierFallback = $true }
    if ($SkipPreflight)     { $fwd.SkipPreflight = $true }
    & $launcher @fwd
    $results += [pscustomobject]@{ lane = $lane; account = $l.tag; pool = $l.pool; dispatched = $true; deferred = $false; reason = '' }
  } catch {
    Write-Warning "lane $lane ($($l.tag)) failed to dispatch: $_"
    $results += [pscustomobject]@{ lane = $lane; account = $l.tag; pool = $l.pool; dispatched = $false; deferred = $false; reason = '' }
  }
}

$ok = ($results | Where-Object { $_.dispatched }).Count
$distinctOk = @($results | Where-Object { $_.dispatched } | Select-Object -ExpandProperty pool -Unique).Count
Write-Output "`nWAVE DISPATCHED  $ok/$($w.granted) lanes live across $distinctOk distinct rate-limit pool(s)."
$results | Format-Table -AutoSize
