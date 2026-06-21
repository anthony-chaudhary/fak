<#
serialize_goal_fleet.ps1 -- run the priced P0 goal fleet SERIALLY through one Claude seat.

WHY serial: the host switcher offers exactly ONE usable Claude engineering seat
(fleet_accounts.py route --product claude == .claude). The dos-plan-price geometry priced a
4-wide safe set, but with SERVING=1 the fan-out collapses to wave-of-1. Serial execution also
trivially satisfies the disjointness floor: no two workers ever co-hold a lane tree, so the
compute-clique (#5/#7/#9/#10) cannot race. `dos arbitrate` remains the floor IF concurrency is
ever added; here it is a no-op and is recorded as such.

WITNESS not narrate: a worker's "done" is never trusted. After each worker exits, this driver
independently checks for a commit on main referencing the issue, grades it with
`dos commit-audit <sha> --json` (diff-witnessed), and reads `gh issue view`. The Stop hook
(dos hook stop, already wired) separately refuses a worker's stop until git backs its phase.

Usage:
  pwsh -NoProfile -File tools\serialize_goal_fleet.ps1 -DryRun     # validate, launch nothing
  pwsh -NoProfile -File tools\serialize_goal_fleet.ps1             # go live (detached chain)
#>
[CmdletBinding()]
param(
  [string]$Workspace      = '',
  [string]$ContractsDir   = '.claude/goal-prompts/p0-fleet',
  [string]$RunRoot        = '',
  [int]$PerWorkerTimeoutMin = 90,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
if (-not $Workspace) {
  $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } elseif ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { (Get-Location).Path }
  $Workspace = (Resolve-Path (Join-Path $scriptDir '..')).Path
}
Set-Location $Workspace

# Priced serial order: host-completable first, GPU/cgo-bound last (their acceptance cannot be
# witnessed on this host -> they honest-block fast on hardware instead of thrashing the seat).
$PLAN = @(
  @{ n=21; lane='gateway'; pointer='1-issue21-openai.md'; htest='./internal/gateway/...'; host='full' }
  @{ n=5;  lane='model';   pointer='2-issue5-awq.md';     htest='./internal/model -run AWQ'; host='partial: perf->GPU node' }
  @{ n=12; lane='ci';      pointer='3-issue12-race.md';   htest='';                          host='partial: -race->cgo node' }
  @{ n=11; lane='bench';   pointer='4-issue11-bench.md';  htest='./internal/bench/...';      host='partial: N=1000->bench node' }
  @{ n=7;  lane='model';   pointer='5-issue7-tp.md';      htest='./internal/model/...';      host='blocked: 2x GPU node' }
  @{ n=9;  lane='compute'; pointer='6-issue9-prefill.md'; htest='./internal/compute/...';    host='blocked: CUDA node' }
  @{ n=10; lane='compute'; pointer='7-issue10-vulkan.md'; htest='./internal/compute/...';    host='blocked: AMD Vulkan node' }
)

if (-not $RunRoot) {
  $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmssZ')
  $RunRoot = Join-Path $Workspace ".goal-runs/p0-fleet-$stamp"
}
if (-not (Test-Path $RunRoot)) { New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null }
$launcher = Join-Path $Workspace 'tools/launch_goal_detached.ps1'
$rollup   = Join-Path $RunRoot 'rollup.md'
$statusF  = Join-Path $RunRoot 'STATUS.txt'

function Stamp { (Get-Date).ToUniversalTime().ToString('o') }
function Note([string]$m) { Write-Host $m; Add-Content -Path $statusF -Value "$(Stamp)  $m" }

# Preflight: every contract present, launcher present, gh+dos reachable.
$missing = @($PLAN | Where-Object { -not (Test-Path (Join-Path (Join-Path $Workspace $ContractsDir) $_.pointer)) })
if ($missing) { throw "missing contract(s): $(@($missing | ForEach-Object { $_.pointer }) -join ', ')" }
if (-not (Test-Path $launcher)) { throw "launcher not found: $launcher" }

"# P0 goal-fleet serial run`n`n- run root: $RunRoot`n- workspace: $Workspace`n- seat pool: SERVING=1 (single .claude engineering seat) -> serial`n- order (host-tractability): $(@($PLAN | ForEach-Object { '#' + $_.n }) -join ' -> ')`n- started: $(Stamp)`n" | Set-Content -Path $rollup -Encoding UTF8
Note "driver start (DryRun=$DryRun) -- $($PLAN.Count) goals, $PerWorkerTimeoutMin min/worker cap"

function Witness-Issue([int]$N, [string]$preSha, [string]$workerLog, [string]$hostFact, [string[]]$preDirty) {
  # independent read-back -- never the worker's narration.
  # Candidate commits: SUBJECT carries the boundary-anchored (#N) ref (matches the COMMIT-SUBJECT
  # RULE). -E + \(#N\) excludes higher-numbered siblings (#52/#58 for #5) and body-only refs that
  # other concurrent agents on the shared account may have landed.
  $rows = @()
  try { $rows = @(& git -C $Workspace log "$preSha..HEAD" -E --grep "\(#$N\)" --format="%H%x1f%s" 2>$null) } catch {}
  $cands = @()
  foreach ($r in $rows) {
    $p = $r -split ([char]0x1f)
    if ($p.Count -ge 2 -and $p[1] -match "#$N(\D|$)") { $cands += [pscustomobject]@{ sha=$p[0]; subj=$p[1] } }
  }
  # Take the newest candidate whose audit actually WITNESSES the claim (verdict OK / diff-witnessed).
  # ABSTAIN and CLAIM_UNWITNESSED do NOT count as a ship -- record the best-seen verdict separately.
  $shipped = $null; $bestSeen = $null
  foreach ($c in $cands) {
    $arr = $null
    try { $arr = (& dos commit-audit --workspace $Workspace $c.sha --json 2>$null | ConvertFrom-Json) } catch {}
    $a = $arr | Select-Object -First 1   # commit-audit emits a JSON array; take this sha's row
    if (-not $a) { continue }
    if (-not $bestSeen) { $bestSeen = [pscustomobject]@{ sha=$c.sha; verdict="$($a.verdict)"; witness="$($a.witness)" } }
    if (("$($a.verdict)" -match '^(?i)ok$') -or ("$($a.witness)" -eq 'diff-witnessed')) {
      $shipped = [pscustomobject]@{ sha=$c.sha; verdict="$($a.verdict)"; witness="$($a.witness)" }; break
    }
  }
  # Swept-WIP guard: did the witnessed commit touch a path that was already dirty BEFORE this goal
  # ran? That is a possible sweep of operator/other-agent WIP -- surface for review (not auto-fail).
  $sweep = ''
  if ($shipped -and $preDirty -and $preDirty.Count -gt 0) {
    $files = @(); try { $files = @(& git -C $Workspace show --name-only --format= $shipped.sha 2>$null | Where-Object { $_ }) } catch {}
    $hit = @($files | Where-Object { $preDirty -contains $_ })
    if ($hit.Count -gt 0) { $sweep = "REVIEW: commit touches $($hit.Count) pre-existing-dirty path(s): " + (($hit | Select-Object -First 5) -join ', ') }
  }
  $issueState = 'unknown'
  try { $iv = (& gh issue view $N --repo Anthony-Chaudhary/fleet-public --json state 2>$null | ConvertFrom-Json); if ($iv) { $issueState = $iv.state } } catch {}
  # Grade on the WITNESS (not issue-closed -- no contract self-closes); blocks grounded in the
  # operator-authored host fact, not the worker's log narration.
  $outcome = if ($shipped -and -not $sweep) { 'met (witnessed ship)' }
             elseif ($shipped) { 'shipped-needs-review (' + $sweep + ')' }
             elseif ($hostFact -match '^(?i)(blocked|partial)') { 'host-precluded-block (host=' + $hostFact + ')' }
             else { 'no-witnessed-effect' }
  return [pscustomobject]@{ outcome=$outcome; shipped=$shipped; bestSeen=$bestSeen; issue_state=$issueState; sweep=$sweep }
}

foreach ($g in $PLAN) {
  $tag = [IO.Path]::GetFileNameWithoutExtension($g.pointer)
  $rel = "$ContractsDir/$($g.pointer)"
  $preSha = (& git -C $Workspace rev-parse HEAD).Trim()
  # snapshot paths already dirty BEFORE this goal runs, so the witness can flag a WIP sweep.
  $preDirty = @(& git -C $Workspace status --porcelain 2>$null | ForEach-Object { if ($_.Length -gt 3) { $_.Substring(3).Trim() } })
  Note "=== goal #$($g.n) [$($g.lane)] host=$($g.host) -- preSha=$preSha preDirty=$($preDirty.Count) ==="

  if ($DryRun) {
    Add-Content $rollup "## #$($g.n)  (DRY-RUN)`n- lane: $($g.lane) | host: $($g.host)`n- would launch: $launcher -PointerFile $rel -Workspace `"$Workspace`" -LogDir `"$RunRoot`" -WorkKind engineering`n- would witness: git log $preSha..HEAD --grep '#$($g.n)' -> dos commit-audit -> gh issue view $($g.n)`n"
    Note "DRY-RUN: skipped launch for #$($g.n)"
    continue
  }

  # --- live launch: fire the launcher DETACHED and never read its streams ---
  # The launcher Start-Process'es claude and writes the worker .pid file itself. We must NOT Tee
  # or -Wait on the launcher's stdout: the detached claude grandchild inherits that handle and
  # would hang the driver for the worker's whole lifetime (bypassing the timeout cap). Fire and
  # poll the .pid file instead -- its presence is the launch-success signal, its absence = refused.
  try {
    Start-Process powershell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$launcher,'-PointerFile',$rel,'-Workspace',$Workspace,'-LogDir',$RunRoot,'-WorkKind','engineering') -WorkingDirectory $Workspace -WindowStyle Hidden | Out-Null
  } catch {
    Note "launch FAILED for #$($g.n): $($_.Exception.Message)"
    Add-Content $rollup "## #$($g.n)`n- outcome: launch-failed -- $($_.Exception.Message)`n"
    continue
  }
  # poll for a pid file that holds a POSITIVE integer. A 0-byte/mid-write file must NOT parse to
  # PID 0 (System Idle) -- that would make the wait loop hang the full cap then Stop-Process Idle.
  $pidFile = $null; $wpid = 0
  $ldl = (Get-Date).AddSeconds(90)
  while ((Get-Date) -lt $ldl) {
    Start-Sleep -Seconds 3
    $pf = (Get-ChildItem -Path $RunRoot -Filter "$tag-*.pid" -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1)
    if ($pf) {
      $raw = (Get-Content $pf.FullName -ErrorAction SilentlyContinue | Select-Object -First 1)
      if ("$raw" -match '^\s*(\d+)\s*$' -and [int]$Matches[1] -gt 0) { $pidFile = $pf; $wpid = [int]$Matches[1]; break }
    }
  }
  if (-not $pidFile -or $wpid -le 0) {
    Note "no valid pid for #$($g.n) within 90s -- switcher refused or pid unwritten. STOPPING chain."
    Add-Content $rollup "## #$($g.n)`n- outcome: not-launched (no valid pid -- switcher refused or pid unwritten). chain halted.`n"
    break
  }
  $wlog = $pidFile.FullName.Replace('.pid', '.out.log')
  Note "#$($g.n) launched pid=$wpid log=$([IO.Path]::GetFileName($wlog)) -- waiting (cap $PerWorkerTimeoutMin min)"

  $deadline = (Get-Date).AddMinutes($PerWorkerTimeoutMin)
  $exited = $false
  while ((Get-Date) -lt $deadline) {
    if (-not (Get-Process -Id $wpid -ErrorAction SilentlyContinue)) { $exited = $true; break }
    Start-Sleep -Seconds 20
  }
  if (-not $exited) {
    Note "#$($g.n) hit $PerWorkerTimeoutMin min cap -- killing pid=$wpid"
    try { Stop-Process -Id $wpid -Force -ErrorAction SilentlyContinue } catch {}
  }

  $w = Witness-Issue -N $g.n -preSha $preSha -workerLog $wlog -hostFact $g.host -preDirty $preDirty
  Note "#$($g.n) outcome=$($w.outcome) issue=$($w.issue_state) sha=$($w.shipped.sha) verdict=$($w.shipped.verdict)"
  Add-Content $rollup "## #$($g.n)`n- lane: $($g.lane) | host: $($g.host) | pid: $wpid | timed_out: $(-not $exited)`n- outcome: **$($w.outcome)**`n- witnessed commit: $($w.shipped.sha)  (verdict=$($w.shipped.verdict), witness=$($w.shipped.witness))`n- best-seen audit: $($w.bestSeen.sha) (verdict=$($w.bestSeen.verdict))`n- issue state: $($w.issue_state)`n- sweep check: $($w.sweep)`n- log: $([IO.Path]::GetFileName($wlog))`n"
}

Note "driver done -- rollup: $rollup"
"`n- finished: $(Stamp)`n" | Add-Content -Path $rollup
if ($DryRun) { Write-Host "`nDRY-RUN complete. Rollup scaffold: $rollup" } else { Write-Host "`nLIVE run complete. Rollup: $rollup" }
