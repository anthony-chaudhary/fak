<#
.SYNOPSIS
  Launch a headless `/goal` worker fully DETACHED from the launching shell,
  under a SWITCHER-CHOSEN account at the right model tier.

.WHY
  A `/goal` worker run inline (`claude -p ...` as a child of a tool-call shell, or
  via dispatch_worker.py's blocking subprocess.run) dies the moment that parent shell
  is reaped -- which is exactly why the first resolve-tickets run was cut off mid-loop
  with no `end_turn`. This launcher uses Start-Process to spawn claude as an INDEPENDENT
  process (its own process tree, not a child of this shell), redirects its output to a
  dated log, records the PID, and returns immediately. The worker then survives this
  session ending.

  ACCOUNT SWITCHER + TIER ROUTING (the dispatch integration): historically this script
  grabbed the ambient `claude` and launched under whatever account happened to be the
  default -- no CLAUDE_CONFIG_DIR, no availability check, no tier. A throttled or
  auth-blocked default account silently failed the dispatch. It now resolves an account
  through the SAME switcher front door every other consumer uses
  (`fak fleet-accounts resolve` -- one call returns config_dir + oauth_token + tier),
  pins `CLAUDE_CONFIG_DIR` to it, and picks the model tier by WORK KIND:

    -WorkKind engineering   -> tier 1 (max-quality frontier; the DEFAULT, unchanged)
    -WorkKind gardening     -> tier 2 (GLM/light) for maintenance/cleanup loops
    -Tier t1|t2|t3|auto     -> explicit tier override (work-kind wins if both given)

  Engineering is the default, so a plain `launch_goal_detached.ps1` keeps the old
  max-quality behavior; only an explicit gardening/maintenance dispatch drops to tier 2.
  Gardening is non-strict: if no tier-2 account is free it up-shifts to tier 1 rather
  than stalling. If NO account is available at all, the launch FAILS loudly (the whole
  point of the switcher) instead of silently running on a blocked ambient account.

  SPAWN GATE (no-DoS): every launch first passes tools/dispatch_preflight.py — the
  same gate issue_dispatch.py re-checks per spawn — and refuses on any non-SPAWN_OK
  verdict (dirty host, no account, seat pool depleted, at cap). The refusal is the
  safety floor; -SkipPreflight exists only for an operator who explicitly accepts an
  ungated spawn, never for automation to route around a REFUSE_*.

  SEAT HYGIENE: a parent session running under `fak guard` carries ANTHROPIC_BASE_URL /
  ANTHROPIC_API_KEY pointing at its session-local loopback gateway. A detached child
  inheriting them bills through the PARENT's seat (env precedence beats the seat's
  OAuth login, nullifying CLAUDE_CONFIG_DIR routing) and dies the instant the parent
  gateway exits — the observed whole-wave same-instant crash (2026-07-01). This
  launcher strips ANTHROPIC_* and the session-identity vars before spawning, so the
  worker owns exactly the seat the switcher granted it.

.NOTES
  - The goal condition is read from the launch POINTER file (kept <4000 chars for the
    /goal cap); the worker reads the full spec from disk itself.
  - bypassPermissions is required for an unattended worker (it edits files, runs git).
  - This does NOT modify the tree or commit; it only starts a process. Stop it with
    `Stop-Process -Id <pid>` (the PID is printed and written to the .pid file).
  - The .pid file is ALSO the spawn gate's witness (#2226): dispatch_preflight.py
    counts live pid breadcrumbs under <Workspace>\.goal-runs, so a stdin-fed worker
    (whose command line carries no scannable marker) occupies a cap slot from
    launch, not from its first lane lease. Dead-pid breadcrumbs are ignored by the
    scan and swept by this launcher before each spawn.
  - -PlanOnly resolves the account and runs the preflight but spawns NOTHING — the
    witnessable dry-run for a single dispatch, mirroring the wave launcher's default.
#>
[CmdletBinding()]
param(
  # Both defaults were wrong in the same direction (#5895) and are coupled: the pointer is
  # Test-Path'd AFTER `Set-Location $Workspace` below, so a relative pointer path is read
  # against the workspace. The old pair named a file that is not in the tree AND a sibling
  # checkout, so a bare invocation threw `pointer file not found` before reaching the gate.
  # Keep this name pointing at a file that exists; the guard test pins both its existence
  # and the /goal char cap, since a rename is what deleted the previous default.
  [string]$PointerFile = ".claude/goal-prompts/resolve-top-issue-witnessed.md",
  # Self-locating: tools/ -> repo root, i.e. the checkout this script was invoked from —
  # the same derivation $repoRoot uses below. The old literal named a sibling clone missing
  # tools/proc_resource_guard.py, so dispatch_preflight fail-safed to REFUSE_INSPECT
  # (cap=4, granted=0) and refused the spawn. An explicit -Workspace still overrides.
  [string]$Workspace   = (Split-Path -Parent $PSScriptRoot),
  # Where the worker's logs + pid breadcrumb land. Empty derives <Workspace>\.goal-runs
  # — the SAME dir dispatch_preflight.py scans for live /goal pid breadcrumbs (#2226).
  # Point it elsewhere only if you accept that the spawn gate then cannot see these
  # workers until they lease a lane.
  [string]$LogDir      = '',
  # Work kind drives the tier: engineering (default) -> tier1, gardening -> tier2.
  # Leave empty to fall back to -Tier. See fleet_accounts.GARDENING/ENGINEERING_WORK_KINDS.
  [ValidateSet('engineering','eng','dev','feature','implementation',
               'gardening','garden','maintenance','maint','cleanup','chore','triage','')]
  [string]$WorkKind    = 'engineering',
  # Explicit tier override (only consulted when -WorkKind is empty).
  [ValidateSet('auto','t1','t2','t3','1','2','3')]
  [string]$Tier        = 'auto',
  # Pin a specific account by tag/basename instead of routing (rare; for debugging).
  [string]$Account     = '',
  # Force the worker's model via `claude --model <v>` (e.g. opus, sonnet, or a full id).
  # Empty (default) = pass no --model, so the account's own default model is used
  # (historically Fable 5 on the promo seats). Set it to escape a model-specific usage
  # cap: a Fable-limited seat still serves on opus/sonnet, which the switcher-picked
  # account authenticates for the same way. Additive — empty keeps prior behavior.
  [string]$Model       = '',
  # GATEWAY BUDGET (the token-saver + budget guard for detached workers). When -Guarded,
  # the worker runs under its OWN `fak guard` gateway, so it inherits fak's default-on
  # token savers AND a hard context/wall budget that self-terminates a runaway worker.
  #   -ContextBudgetTokens : prompt/context-token budget handed to `fak guard
  #     --context-budget-tokens`. It is a CUMULATIVE allowance, NOT a per-turn ceiling
  #     (internal/session/usage.go DebitUsage): every turn subtracts that turn's ENTIRE
  #     resident window (prompt + cache_read + cache_creation), so a cached prefix is
  #     re-charged in full on every turn and the session drains with
  #     BUDGET_CONTEXT_EXHAUSTED when the running total hits <=0. Therefore
  #
  #         turns funded per epoch ~= budget / (mean resident tokens per turn)
  #
  #     and since the resident is never below the launch baseline, a budget of
  #     baseline x k funds AT MOST k turns. "Headroom over the turn-1 floor" IS the turn
  #     count -- being born safe (> baseline) says nothing about surviving turn 3.
  #     The old flat 200000 made exactly that dimensional error: it read as "~3x headroom
  #     over the ~62K floor" but bought ~2 turns, and with --restart-limit those hops are
  #     spent in minutes -> 409 -> child exit 1. Witnessed 2026-08-08 on wave fw08081943:
  #     all 9 workers died BUDGET_CONTEXT_EXHAUSTED 6-8 min into a 45m runway (~15%),
  #     residents 73k-294k, compaction healthy (0 anchor_starved, 0 solvency_forced, all
  #     65 bails correct-by-design). Now mirrors the derivation the Go and Python launch
  #     paths already shared -- max(hardCap - outputReserve, baseline) x turnsPerEpoch =
  #     max(200000 - 32000, 62000) x 12 = 2016000 -- so the budget funds a full epoch
  #     instead of two turns. Keep in sync with
  #     cmd/dispatchworker/guard.go:claudeGuardContextBudgetTokens and
  #     tools/dispatch_worker.py:claude_guard_context_budget_tokens; drift is pinned by
  #     TestLaunchGoalDetachedGuardBudgetsMirrorDispatchWorker. Also satisfies guard's
  #     rule that --restart-on-budget needs a positive budget (guard.go).
  [int]$ContextBudgetTokens = 2016000,
  #   -MaxDuration : wall-clock budget handed to `fak guard --max-duration`. An INDEPENDENT
  #     axis from the token budget — a stuck worker that isn't burning tokens still self-
  #     terminates gracefully at this deadline instead of occupying a seat forever.
  [string]$MaxDuration = "45m",
  #   -Guarded : wrap the worker in `fak guard` (default). Set $false to fall back to the
  #     exact legacy raw `claude -p` spawn (no gateway, no budget) — a trivial revert lever.
  [bool]$Guarded = $true,
  # Optional fak binary. Empty probes this repo's tools\.bin/fak.exe, repo-root fak.exe,
  # then PATH fak.
  [string]$FakExe      = '',
  # Let an engineering/tier-1 dispatch fall back to tier 2 when no tier-1 account is
  # free, rather than refusing. Off by default so engineering stays max-quality.
  [switch]$AllowTierFallback,
  # Operator ceiling handed to the spawn-gate preflight (0 = the preflight's own
  # default). The effective cap stays min(host_cap, dos [supervise].target, this).
  [int]$PreflightMaxWorkers = 0,
  # Skip the dispatch_preflight.py spawn gate. An EXPLICIT operator override only:
  # it removes the no-DoS floor for this one spawn. Never set it from automation.
  [switch]$SkipPreflight,
  # Resolve the account + run the preflight + print the plan, but spawn NOTHING.
  [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-FakExe {
  param([string]$RepoRoot, [string]$Explicit)
  if ($Explicit) {
    if (-not (Test-Path $Explicit)) { throw "fak binary not found: $Explicit" }
    return (Resolve-Path $Explicit).Path
  }
  $candidates = @(
    (Join-Path $RepoRoot 'tools\.bin\fak.exe'),
    (Join-Path $RepoRoot 'tools\.bin\fak'),
    (Join-Path $RepoRoot 'fak.exe'),
    (Join-Path $RepoRoot 'fak')
  )
  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
  }
  $cmd = Get-Command fak -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  throw "no fak binary found (looked in $RepoRoot\tools\.bin, repo root, and PATH; pass -FakExe)"
}

Set-Location $Workspace
if (-not $LogDir) { $LogDir = Join-Path $Workspace '.goal-runs' }

# --- Breadcrumb hygiene (#2226): sweep dead-pid breadcrumbs before the gate runs ------
# dispatch_preflight.py folds live `<LogDir>\*.pid` breadcrumbs into its worker count —
# that is how a stdin-fed /goal worker (no cmdline marker until it leases a lane)
# occupies a cap slot from the instant it spawns. The scan already IGNORES a dead pid,
# so a stale breadcrumb can never wedge spawning; this sweep just stops corpses
# accumulating across waves. Only the .pid marker is removed — logs stay for forensics.
if (Test-Path $LogDir) {
  Get-ChildItem -Path $LogDir -Filter '*.pid' -File -ErrorAction SilentlyContinue |
    ForEach-Object {
      $crumbPid = 0
      $raw = Get-Content -Raw $_.FullName -ErrorAction SilentlyContinue
      if ($raw -and [int]::TryParse($raw.Trim(), [ref]$crumbPid) -and $crumbPid -gt 0 -and
          (Get-Process -Id $crumbPid -ErrorAction SilentlyContinue)) { return }
      Remove-Item $_.FullName -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path $PointerFile)) { throw "pointer file not found: $PointerFile" }
$body = Get-Content -Raw $PointerFile
$cond = "/goal $body"
if ($cond.Length -gt 4000) { throw "goal condition is $($cond.Length) chars (>4000 cap) -- shrink the pointer" }

$claude = (Get-Command claude).Source

# --- Spawn gate: dispatch_preflight.py must say SPAWN_OK (the no-DoS floor) -----------
# The same gate issue_dispatch.py re-checks per spawn, now fronting THIS spawn point
# too (the wave launcher dispatches every lane through here, so the whole multi-account
# path inherits the per-spawn re-check). Fail-safe: no python / no parseable verdict is
# treated as REFUSE_INSPECT — refuse, never spawn ungated by accident.
if (-not $SkipPreflight) {
  $pyCmd = Get-Command python -ErrorAction SilentlyContinue
  $pyPre = @()
  if (-not $pyCmd) { $pyCmd = Get-Command py -ErrorAction SilentlyContinue; $pyPre = @('-3') }
  if (-not $pyCmd) {
    throw "spawn gate needs python on PATH and none was found (fail-safe: REFUSE_INSPECT) -- fix python, or pass -SkipPreflight to explicitly accept an ungated spawn"
  }
  $pfArgs = $pyPre + @((Join-Path $repoRoot 'tools\dispatch_preflight.py'), '--json', '--workspace', "$Workspace")
  if ($WorkKind)                  { $pfArgs += @('--work-kind', $WorkKind) }
  if ($PreflightMaxWorkers -gt 0) { $pfArgs += @('--max-workers', "$PreflightMaxWorkers") }
  $pfRaw = & $pyCmd.Source @pfArgs 2>$null | Out-String
  $pf = $null
  try { $pf = $pfRaw | ConvertFrom-Json } catch { $pf = $null }
  if (-not $pf -or -not $pf.verdict) {
    throw "spawn gate produced no verdict (python=$($pyCmd.Source), rc=$LASTEXITCODE; fail-safe: REFUSE_INSPECT) -- not spawning"
  }
  if ($pf.verdict -ne 'SPAWN_OK') {
    throw "spawn gate refused: $($pf.verdict) -- $($pf.reason). The refusal IS the no-DoS floor; recover (fix the host / re-login / wait for a seat), do not route around it."
  }
  Write-Output ("preflight: SPAWN_OK  live={0} cap={1}" -f $pf.live, $pf.cap)
} else {
  Write-Warning "preflight SKIPPED by operator switch -- this spawn is ungated (no host/cap/account check)."
}

# --- Resolve the account + tier through the switcher (the dispatch integration) -------
# ONE call to the switcher's canonical front door (`fak fleet-accounts resolve`): pin OR
# tier/work-kind route, plus the account's oauth token, in a single flat record. Scoped
# to claude (this launches Claude Code, not opencode). Capture JSON to a temp file to
# dodge PS native-exe stdout quirks, then parse. On a refusal we FAIL -- never silently
# run ambient.
$fak = Resolve-FakExe -RepoRoot $repoRoot -Explicit $FakExe
$tmpOut = Join-Path ([System.IO.Path]::GetTempPath()) ("goal-route-{0}.json" -f ([Guid]::NewGuid().ToString('N')))

$resolveArgs = @('fleet-accounts', 'resolve', '--product', 'claude')
if ($Account)          { $resolveArgs += @('--account', $Account) }
elseif ($WorkKind)     { $resolveArgs += @('--work-kind', $WorkKind) }
else {
  switch ($Tier) {
    { $_ -in @('t1','1') } { $resolveArgs += '--t1'; break }
    { $_ -in @('t2','2') } { $resolveArgs += '--t2'; break }
    { $_ -in @('t3','3') } { $resolveArgs += '--t3'; break }
  }
}
if ($AllowTierFallback) { $resolveArgs += '--allow-tier-fallback' }
& $fak @resolveArgs > $tmpOut 2>$null
$resolveRc = $LASTEXITCODE
$r = $null
if (Test-Path $tmpOut) { try { $r = Get-Content -Raw $tmpOut | ConvertFrom-Json } catch { $r = $null }; Remove-Item $tmpOut -ErrorAction SilentlyContinue }
if (-not $r) { throw "account resolve produced no JSON (fak=$fak, rc=$resolveRc) -- cannot dispatch" }
if (-not $r.ok) {
  $reason = if ($r.reason) { $r.reason } else { 'no available account' }
  throw "account switcher refused dispatch: $reason -- fix the account (re-login / wait for reset) or pass -AllowTierFallback."
}
$acct      = $r
$configDir = $r.config_dir
$tierSel   = $r.selected_tier
$fellBack  = [bool]$r.fallback_used
if (-not $configDir) { throw "resolved account $($r.account) has no config dir" }

if ($PlanOnly) {
  [pscustomobject]@{
    plan_only     = $true
    account       = $acct.account
    account_tag   = $acct.tag
    config_dir    = $configDir
    work_kind     = $WorkKind
    tier          = "t$tierSel"
    tier_fallback = $fellBack
    cond_chars    = $cond.Length
    preflight     = if ($SkipPreflight) { 'SKIPPED' } else { 'SPAWN_OK' }
  } | Format-List
  "PLAN ONLY -- no worker spawned. Re-run without -PlanOnly to dispatch."
  return
}

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$tag    = [IO.Path]::GetFileNameWithoutExtension($PointerFile)
$logOut = Join-Path $LogDir "$tag-$stamp.out.log"
$logErr = Join-Path $LogDir "$tag-$stamp.err.log"
$pidF   = Join-Path $LogDir "$tag-$stamp.pid"
$inF    = Join-Path $LogDir "$tag-$stamp.in.txt"

# Feed the goal via STDIN, never as a CLI argument. The condition text contains
# backtick-wrapped commands and `--flags` (e.g. `dos commit-audit --json`); passing it
# through Start-Process -ArgumentList lets CommandLineToArgvW re-split it and claude's own
# arg parser then chokes on a stray `--json` ("unknown option"). `claude -p` with no prompt
# arg reads the prompt from stdin, which is parse-safe. Write the prompt to a UTF-8 file
# (no BOM) and redirect it in.
[IO.File]::WriteAllText($inF, $cond, [Text.UTF8Encoding]::new($false))

# OWN THE SEAT: strip guarded-session wiring BEFORE pinning the account. A parent
# running under `fak guard` exports ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY aimed at
# its session-local loopback gateway; env precedence beats the seat's OAuth login, so
# an inheriting child bills the parent's seat and dies when the parent gateway exits
# (the whole-wave same-instant crash, observed 2026-07-01; the child-stderr tell is
# "claude.ai connectors are disabled because ANTHROPIC_API_KEY ... is set"). Session-
# identity vars are dropped too, so the worker is never mistaken for a child session.
Get-ChildItem Env: | Where-Object { $_.Name -like 'ANTHROPIC_*' } |
  ForEach-Object { Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue }
foreach ($n in @('CLAUDE_CODE_SESSION_ID', 'CLAUDE_CODE_CHILD_SESSION')) {
  Remove-Item "Env:$n" -ErrorAction SilentlyContinue
}

# Pin the chosen account for the detached worker. CLAUDE_CONFIG_DIR is inherited by
# the Start-Process child (it copies the parent env), so the worker runs under the
# switcher-selected account, not the ambient default.
$env:CLAUDE_CONFIG_DIR = $configDir

# Serve on the credential the account actually authenticates with: the resolver already
# applied the switcher's single token rule (prefer the dir's long-lived .oauth-token over
# the interactive .credentials.json, which EXPIRES) and handed back oauth_token. Observed
# 2026-06-21: gem7/gem8 serve via their setup token while their interactive creds report
# "Not logged in" — so a launcher that pins only CLAUDE_CONFIG_DIR false-fails the worker
# on turn 1. Drop any ambient token when the account has none, so a sibling account's
# token never bleeds into this worker.
if ($r.oauth_token) { $env:CLAUDE_CODE_OAUTH_TOKEN = "$($r.oauth_token)" }
else { Remove-Item Env:CLAUDE_CODE_OAUTH_TOKEN -ErrorAction SilentlyContinue }

# Start-Process => detached child in its own process tree; -Redirect* keep the streams.
# --model (when -Model is set) forces the worker off the account's default model, so a
# Fable-usage-capped seat still serves on opus/sonnet under the same OAuth (#Fable-cap).
$claudeArgs = @("-p", "--permission-mode", "bypassPermissions")
if ($Model) { $claudeArgs = @("-p", "--model", $Model, "--permission-mode", "bypassPermissions") }

# GATEWAY WRAP (seat-hygiene-safe): run the worker under its OWN `fak guard` gateway so it
# gets fak's default-on token savers PLUS a context/wall budget guard. This does NOT leak
# the parent's seat: the "OWN THE SEAT" block above already STRIPPED the parent's
# ANTHROPIC_BASE_URL/KEY, and CLAUDE_CONFIG_DIR + CLAUDE_CODE_OAUTH_TOKEN are pinned to the
# switcher-chosen account BEFORE this spawn. So the NEW `fak guard` establishes a fresh
# session-local loopback gateway authenticated by THAT account's OAuth (guard.go: the
# --provider anthropic default is the subscription OAuth path, needs no API key) and injects
# its OWN ANTHROPIC_BASE_URL/KEY for its child claude (guard_provider.go / guard_child.go) —
# exactly how `fak guard -- claude` works normally. Guard forwards its stdin to the child
# (guard_child.go: child.Stdin = os.Stdin), so the stdin-fed goal prompt still reaches
# `claude -p`. Reuse $fak (already resolved, repo-local .bin first, respects -FakExe) as the
# gateway binary rather than a second PATH probe.
# --restart-on-budget is valid because -ContextBudgetTokens is always a positive budget
# (guard.go requires --context-budget-tokens > 0 for it).
if ($Guarded) {
  $spawnFile = $fak
  # #3296: hand guard an explicit per-worker seed dir so budget-restart hops REUSE one
  # directory instead of minting a fresh %TEMP%\fak-guard-reset-* per hop (unbounded and
  # unreaped -- 2260 orphans witnessed live in the fleet). It lands under .goal-runs so it
  # is reaped alongside this worker's logs. --restart-limit bounds a wedged worker: past it
  # the worker is thrashing, not producing, so let it die and free the seat for fresh churn
  # rather than restart-loop a seat forever (guard default is 0=unlimited). 16, mirroring
  # cmd/dispatchworker/guard.go:claudeGuardRestartLimit and dispatch_worker.py's
  # CLAUDE_GUARD_RESTART_LIMIT -- NOT the 3 this used to pass. With a full-epoch budget a
  # relaunch happens every ~12+ turns, so 16 x ~2min comfortably exceeds the -MaxDuration
  # wall clock: wall-clock is the real bound for a healthy-but-slow worker, while a
  # degenerate sub-2-min reset storm still trips here. A low limit is not a safety margin --
  # it is a second, tighter deadline that reaps healthy workers at ~15% of their runway.
  $seedDir = Join-Path $LogDir "seed-$tag-$stamp"
  New-Item -ItemType Directory -Force -Path $seedDir | Out-Null
  $spawnArgs = @(
    "guard",
    "--context-budget-tokens", "$ContextBudgetTokens",
    "--restart-on-budget",
    "--restart-limit", "16",
    "--restart-seed-dir", $seedDir,
    "--max-duration", "$MaxDuration",
    # A detached /goal worker IS a headless dispatch worker: mark it so, exactly as the
    # Go dispatch paths do (internal/dispatchtick/dispatchtick.go, cmd/dispatchworker/guard.go,
    # #3607). Without this, resolveGuardCompactBudget hands it the interactive
    # DefaultCompactHistoryBudget (48000) instead of the floor-aware
    # HeadlessCompactHistoryBudget (96000). This repo's fixed tool+system floor is ~68-83k, so
    # the 48k budget leaves the worker permanently past-compact — it thrash-restarts and dies
    # on BUDGET_CONTEXT_EXHAUSTED after --restart-limit hops (observed 2026-07-16). The headless
    # profile also applies the curated headless tool surface the dispatch paths already use.
    "--expose-profile", "headless",
    "--",
    $claude
  ) + $claudeArgs
} else {
  # Legacy raw `claude -p` spawn (no gateway / no budget) — kept as a one-flag revert lever.
  $spawnFile = $claude
  $spawnArgs = $claudeArgs
}
$p = Start-Process -FilePath $spawnFile `
  -ArgumentList $spawnArgs `
  -WorkingDirectory $Workspace `
  -RedirectStandardInput  $inF `
  -RedirectStandardOutput $logOut `
  -RedirectStandardError  $logErr `
  -WindowStyle Hidden `
  -PassThru

# The pid breadcrumb doubles as the spawn gate's live-count witness (#2226): a
# stdin-fed worker carries no cmdline marker, so dispatch_preflight.py counts this
# file — while its pid is a live claude-image process — from the instant it exists,
# closing the spawn-to-lease blind window the per-host worker cap had.
$p.Id | Out-File -FilePath $pidF -Encoding ascii

# WITNESS MARKER (#4800): a single machine-parseable line emitted ONLY once a live
# process exists (Start-Process -PassThru returned a real $p with a pid). The wave
# launcher captures this to record a WITNESSED launch (phase=launched + pid) -- so a
# spawn that threw before this point (missing pointer, oversized goal, Start-Process
# failure) yields NO marker and is never counted as launched. run_id is the stable
# per-spawn tag+stamp, also the pid-file/log basename, independently readable after.
Write-Output ("LAUNCH_WITNESS pid={0} tag={1} run_id={2}" -f $p.Id, $acct.tag, "$tag-$stamp")

[pscustomobject]@{
  pid         = $p.Id
  account     = $acct.account
  account_tag = $acct.tag
  config_dir  = $configDir
  work_kind   = $WorkKind
  tier        = "t$tierSel"
  tier_fallback = $fellBack
  model       = $acct.model
  cond_chars  = $cond.Length
  out_log     = $logOut
  err_log     = $logErr
  pid_file    = $pidF
} | Format-List
if ($fellBack) {
  "note: requested tier had no free account; up-shifted to t$tierSel (work preserved)."
}
"DETACHED -- worker survives this session, pinned to account '$($acct.tag)' (t$tierSel). Stop with: Stop-Process -Id $($p.Id)"
