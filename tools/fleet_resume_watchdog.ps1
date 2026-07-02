<#
fleet_resume_watchdog.ps1 -- the cross-account resume layer for ALL autonomous
Claude sessions (not just the supervisor's job workers).

Each tick:
  1. EXTRACT-IN-ADVANCE: refresh the on-disk session registry
     (tools/_registry/sessions.json) and the AUTO_RESUME plan
     (tools/_registry/resume_plan.json) via fleet_sessions.py.
  2. Resume each AUTO_RESUME session under its resume-target account's
     CLAUDE_CONFIG_DIR, up to -MaxAttempts times (default 8). Attempts are counted in
     a durable ledger (tools/_registry/resume_ledger.jsonl); a session that keeps dying
     in place is re-homed onto a fresh seat by the planner, and once the attempt cap is
     hit it is left for a human. Operator-settled / unrecoverable-auth sids are never
     retried.
  3. Notify (Windows Action Center + notifications.log) on relevant actions:
     a resume, and an account that needs human re-login (BLOCKED_AUTH).

Safety rails:
  * DRY-RUN by default (pass -Live to actually resume).
  * Interactive sessions are SURFACE (never auto-resumed); supervisor workers are
    SUPERVISED (left to run_supervise_loop); throttled accounts are deferred.
  * Host-wide source governor: every live launch asks `fak resume admit` before
    spawning, so the box does not burst many `claude --resume` processes across accounts.
  * BOUNDED RETRY: up to -MaxAttempts (default 8) ledger-counted attempts; a repeat
    crasher is re-homed to a fresh seat by the planner and finally left for a human.
    Ledger-gated, survives state-file loss. Operator-settled / auth-wall sids never retry.
  * Per-tick launch cap plus launch spacing.
  * LIVE-DUPLICATE GUARD: a sid with a `claude --resume <sid>` process already running
    is skipped, and a sid is launched at most once per tick even if the plan lists it
    under two accounts (gem7/day30 share one identity, so one transcript can appear twice).
  * A resume target whose config dir is tombstoned (`.DELETED-*`) is skipped.

  .\fleet_resume_watchdog.ps1                 # dry-run: log what it WOULD resume
  .\fleet_resume_watchdog.ps1 -Live           # actually resume (once per session)
#>
[CmdletBinding()]
param(
  [switch]$Live,
  [string]$FleetDir   = 'C:\work\fleet',
  [int]$WindowH       = 6,
  [int]$MaxPerTick    = 4,
  # Ledger-counted resume attempts per session before it is left for a human. Was an
  # implicit 1 ("resume once ever"); raised so a session that keeps dying is retried --
  # and re-homed onto a fresh seat by the planner after repeated in-place failures --
  # instead of stranded on the first re-crash. Override per-invocation with -MaxAttempts.
  [int]$MaxAttempts   = 8,
  # Pace live launches inside a tick; the source governor also enforces spacing across
  # ticks and across launchers from the shared ledger.
  [int]$LaunchSpacingSec = 8,
  [string]$ClaudeExe  = '',
  [string]$FakExe     = '',
  [string]$LogDir     = '',
  [string]$RegistryDir = '',
  # Active account probing on the registry refresh. 'auto' (default) probes blocked
  # workers only on a -Live tick (off for dry-run, which must stay side-effect-free);
  # 'blocked'/'stale'/'all' force it; 'none' disables. The probe spends one tiny haiku
  # 'say pong' per blocked worker and skips accounts whose throttle reset is still future
  # (account_probe's --skip-active-throttle), so a recovered account re-enters the pool
  # without anyone running a real session on it.
  [ValidateSet('auto','none','blocked','stale','all')]
  [string]$Probe      = 'auto',
  # Anti-spam floor: skip probing an account probed within the last N minutes (read from
  # probe_ledger.jsonl). At the default ~hourly tick this just prevents back-to-back ticks
  # from double-probing; raise it to throttle harder.
  [int]$ProbeMinIntervalMin = 20,
  # The registered scheduled task invokes `-File ...fleet_resume_watchdog.ps1 -Live -Slack`,
  # mirroring the .py port's `--slack`. The public .ps1 dropped this switch, so PowerShell
  # failed parameter binding (NamedParameterNotFound) BEFORE any script code ran -- no
  # heartbeat, no resume -- and conhost --headless masked it as exit 0, so the watchdog
  # looked healthy (LastResult=0) while silently dead for hours. Accept the switch so the
  # tick runs. (Slack POSTING itself flows through notify.ps1 / the .py port's slack_post;
  # actually gating those posts on $Slack is a follow-on -- accepting it here is what
  # un-wedges recovery.)
  [switch]$Slack
)
$ErrorActionPreference = 'Stop'
$stateRoot = if ($env:FLEET_STATE_DIR) {
  $env:FLEET_STATE_DIR
} elseif ($env:LOCALAPPDATA) {
  Join-Path $env:LOCALAPPDATA 'Fleet'
} else {
  Join-Path ([System.IO.Path]::GetTempPath()) 'Fleet'
}
if (-not $ClaudeExe) { $ClaudeExe = Join-Path $env:USERPROFILE '.local\bin\claude.exe' }
if (-not $LogDir) { $LogDir = Join-Path $stateRoot 'watchdog' }
if (-not $RegistryDir) { $RegistryDir = Join-Path $stateRoot 'registry' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$log = Join-Path $LogDir 'resume_watchdog.log'
$notify = Join-Path $FleetDir 'tools\notify.ps1'
function Note($m) {
  $line = "{0}  {1}" -f ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')), $m
  Add-Content -Path $log -Value $line; Write-Output $line
}
function Toast($title, $msg, $level = 'info', $key = '', $minIntervalMinutes = 0) {
  $launchRegDir = if ($RegistryDir) { $RegistryDir } else { Join-Path $stateRoot 'registry' }
  $launch = Join-Path $launchRegDir 'STATUS.txt'
  $args = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $notify,
    '-Title', $title, '-Message', $msg, '-Level', $level,
    '-LogDir', $LogDir, '-Launch', $launch
  )
  if ($key) { $args += @('-Key', $key) }
  if ($minIntervalMinutes -gt 0) { $args += @('-MinIntervalMinutes', "$minIntervalMinutes") }
  try { & powershell @args } catch {}
}

# Per-tick liveness heartbeat, written FIRST -- before the registry refresh / account
# probe below, any of which can throw. With ErrorActionPreference=Stop and the conhost
# --headless launcher reporting exit 0, an early throw used to wedge this watchdog
# silently: this log's mtime (which fleet_bottleneck.py reads as recovery freshness)
# went stale for hours while the scheduled task kept showing LastResult=0.
Note ("tick start: Live=$Live probe=$Probe window=${WindowH}h")

$regDir = $RegistryDir
if (-not (Test-Path $regDir)) { New-Item -ItemType Directory -Path $regDir -Force | Out-Null }
$env:FLEET_REG_DIR = $regDir
$py = Join-Path 'C:\work\job' '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$repoRoot = Split-Path -Parent $PSScriptRoot
$sourcePolicyPath = if ($env:FAK_RESUME_SOURCE_POLICY) {
  $env:FAK_RESUME_SOURCE_POLICY
} else {
  Join-Path $repoRoot '.fak\resume-source-policy.json'
}
if (-not $FakExe) {
  $fakCandidates = @(
    $env:FAK_EXE,
    (Join-Path $repoRoot 'fak.exe'),
    (Join-Path $repoRoot 'fak'),
    'fak.exe',
    'fak'
  ) | Where-Object { $_ }
  foreach ($candidate in $fakCandidates) {
    $resolved = $null
    if ([System.IO.Path]::IsPathRooted($candidate) -or $candidate.Contains('\') -or $candidate.Contains('/')) {
      if (Test-Path $candidate) { $resolved = (Resolve-Path $candidate).Path }
    } else {
      $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
      if ($cmd) { $resolved = $cmd.Source }
    }
    if ($resolved) { $FakExe = $resolved; break }
  }
}
function SourceAdmitGate($ledgerPath, $policyPath) {
  # FailOpen marks an admit that happened WITHOUT the governor's verdict (missing
  # binary / gate error) — the caller must surface it durably (#2173): a fail-open
  # removes the source-concurrency rail, so it can never stay silent.
  if (-not $FakExe) {
    return [pscustomobject]@{ Admit = $true; Reason = 'no-fak-binary'; FailOpen = $true }
  }
  $output = @()
  try {
    $output = & $FakExe resume admit --json --ledger $ledgerPath --policy $policyPath 2>&1
    $code = $LASTEXITCODE
  } catch {
    return [pscustomobject]@{ Admit = $true; Reason = "gate-error:$($_.Exception.Message)"; FailOpen = $true }
  }
  $text = ($output | Out-String).Trim()
  $reason = 'SOURCE_DEFER'
  if ($text) {
    try {
      $doc = $text | ConvertFrom-Json
      if ($doc.decision.reason) { $reason = $doc.decision.reason }
    } catch {
      $reason = $text
    }
  }
  if ($code -eq 3) {
    return [pscustomobject]@{ Admit = $false; Reason = $reason; FailOpen = $false }
  }
  if ($code -eq 0) {
    return [pscustomobject]@{ Admit = $true; Reason = 'admitted'; FailOpen = $false }
  }
  # Fail open: a broken governor must not strand all recovery; the tick's cap/spacing
  # still bound launch pressure.
  return [pscustomobject]@{ Admit = $true; Reason = "gate-error:exit-$code $reason"; FailOpen = $true }
}

# 1. refresh registry + plan (extract in advance). On a live tick, also ACTIVELY probe
# blocked accounts so a silently-recovered account (re-login / access re-enabled / throttle
# expired) re-enters the available pool instead of staying latched on a stale verdict.
$probeMode = if ($Probe -eq 'auto') { if ($Live) { 'blocked' } else { 'none' } } else { $Probe }
$regArgs = @('registry', '--window', "$WindowH")
if ($probeMode -ne 'none') {
  $regArgs += @('--probe', $probeMode, '--min-interval-min', "$ProbeMinIntervalMin")
  # make the probe use the SAME claude binary this watchdog resumes with
  if ($ClaudeExe -and (Test-Path $ClaudeExe)) { $env:FLEET_CLAUDE_EXE = $ClaudeExe }
}
try {
  & $py (Join-Path $FleetDir 'tools\fleet_sessions.py') @regArgs | Out-Null
  Note ("  registry refresh: probe=$probeMode")
} catch {
  # A refresh/probe failure (e.g. the blocked-account probe erroring once accounts go
  # auth-blocked) must not abort the whole tick before a single resume is considered.
  # Log it and continue on whatever resume_plan.json already exists on disk.
  Note ("  registry refresh FAILED: $($_.Exception.Message) -- continuing on existing plan")
}
$planPath = Join-Path $regDir 'resume_plan.json'
$plan = if (Test-Path $planPath) { (Get-Content $planPath -Raw | ConvertFrom-Json).plan } else { @() }
$mode = if ($Live) { 'LIVE' } else { 'DRY-RUN' }
Note ("TICK $mode plan={0} window=${WindowH}h cap=$MaxPerTick" -f @($plan).Count)

# defense-in-depth: the set of account dir-basenames that policy still treats as
# workers. fleet_sessions.py already excludes non-workers when it writes the plan,
# but a stale plan file could predate the policy — so re-check each entry here too.
$workerAccts = @{}
try {
  $acctDoc = & $py (Join-Path $FleetDir 'tools\fleet_accounts.py') json 2>$null | ConvertFrom-Json
  foreach ($a in @($acctDoc.accounts | Where-Object { $_.kind -eq 'worker' })) { $workerAccts[$a.account] = $true }
} catch {}

# durable resume ledger: count prior launches per session and flag operator-settled /
# unrecoverable sids, so a session is retried up to -MaxAttempts times (and moved to a
# fresh seat by the planner after repeated in-place failures) instead of once ever.
$ledgerPath = Join-Path $regDir 'resume_ledger.jsonl'
$launchCount = @{}
$ledgerBlocked = @{}
if (Test-Path $ledgerPath) {
  Get-Content $ledgerPath | ForEach-Object {
    try {
      $r = $_ | ConvertFrom-Json
      $s = $r.session
      if (-not $s) { return }
      if ($r.manual_override -or ("$($r.action)").StartsWith('consolidate')) { $ledgerBlocked[$s] = $true }
      if ($r.outcome -eq 'unrecoverable') { $ledgerBlocked[$s] = $true }
      $nonLaunchPhase = ($r.phase -eq 'deferred') -or ($r.phase -eq 'considered') -or ($r.phase -eq 'skipped') -or ($r.phase -eq 'gate_fail_open')
      if (($r.phase -eq 'launched') -or ($r.phase -eq 'resumed') -or ($r.cause -and -not $nonLaunchPhase)) {
        $launchCount[$s] = [int]$launchCount[$s] + 1
      }
    } catch {}
  }
}
$launched = 0
# One durable warning per tick when the source governor is unavailable (#2173): a
# fail-open launch runs WITHOUT the host-wide concurrency rail, so it must be visible
# in the ledger and the Action Center, never silent.
$gateFailOpenWarned = $false

# Live-duplicate guard: `claude --resume` forks the transcript into a NEW sid, so the
# planned (old) sid never goes live again in the registry -- without this check every
# tick re-plans the same dead sid and stacks another resume process on the box (observed
# 2026-07-01: one sid stacked 4 concurrent copies at the watchdog's 10-min cadence).
# One process scan serves the whole tick; the map is also updated after each launch so
# a sid the plan lists twice (duplicate seat identities) launches at most once per tick.
$liveResume = @{}
try {
  Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction Stop | ForEach-Object {
    if ($_.CommandLine -match '--resume\s+([0-9a-fA-F-]{36})') { $liveResume[$Matches[1]] = $_.ProcessId }
  }
} catch {
  Note "  WARN live-process scan failed ($($_.Exception.Message)) -- duplicate guard inactive this tick"
}

foreach ($p in @($plan)) {
  if ($launched -ge $MaxPerTick) { Note "  per-tick cap reached ($MaxPerTick)"; break }
  $sid = $p.session; $sid8 = $sid.Substring(0, 8)
  $acct = ($p.account -replace '\.claude-?', ''); if (-not $acct) { $acct = 'default' }
  if ($workerAccts.Count -and -not $workerAccts.ContainsKey($p.account)) {
    Note "  SKIP $sid8 -- account $($p.account) is not an offered worker (policy/tombstoned)"; continue
  }
  if ($ledgerBlocked.ContainsKey($sid)) { Note "  SKIP $sid8 -- ledger-blocked (operator-settled or unrecoverable auth wall)"; continue }
  $attempts = [int]$launchCount[$sid]
  if ($attempts -ge $MaxAttempts) { Note "  SKIP $sid8 -- attempt cap reached ($attempts/$MaxAttempts) -- left for a human"; continue }
  if ($liveResume.ContainsKey($sid)) {
    Note "  SKIP $sid8 -- already live as pid $($liveResume[$sid]) (no duplicate resume)"
    if ($Live) {
      $rec = @{ ts = ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')); session = $sid; account = $p.account; phase = 'skipped'; cause = 'already_live'; live_pid = $liveResume[$sid] } | ConvertTo-Json -Compress
      Add-Content -Path $ledgerPath -Value $rec
    }
    continue
  }
  $resumeCfg = if ($p.resume_config_dir) { $p.resume_config_dir } else { $p.config_dir }
  # Defense-in-depth like the worker-account re-check above: the planner has offered a
  # tombstoned seat as a re-home target (observed 2026-07-01: resume_account =
  # .claude-gem8-netra.DELETED-2026-06-29; the launch died on arrival and burned an attempt).
  if ($resumeCfg -match '\.DELETED') {
    Note "  SKIP $sid8 -- resume target $resumeCfg is a tombstoned seat"
    if ($Live) {
      $rec = @{ ts = ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')); session = $sid; account = $p.account; resume_account = $p.resume_account; phase = 'skipped'; cause = 'deleted_seat_target' } | ConvertTo-Json -Compress
      Add-Content -Path $ledgerPath -Value $rec
    }
    continue
  }
  if (-not $Live) {
    $rh = if ($p.rehomed) { " -> $($p.resume_account) (re-home)" } else { "" }
    Note "  WOULD RESUME $sid8 acct=$acct proj=$($p.project)$rh"; continue
  }

  # Host-wide source governor (#1341/#1344): this is the dimension the per-tick cap does
  # not see. It counts live `claude --resume` processes and recent launches across every
  # account on the box, then exits 3 to defer one more launch.
  $admit = SourceAdmitGate $ledgerPath $sourcePolicyPath
  if ($admit.FailOpen -and -not $gateFailOpenWarned) {
    $gateFailOpenWarned = $true
    Note "  WARN source governor UNAVAILABLE ($($admit.Reason)) -- failing OPEN; only the per-tick cap/spacing bound launches this tick"
    # Session-less warning row: every ledger reader keys on `session`, so this row is
    # invisible to retry accounting; `gate_fail_open` is also a non-launch phase, so it
    # never counts as launch pressure. It exists for the operator/status surfaces.
    $warnRec = @{
      ts = ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))
      phase = 'gate_fail_open'
      cause = 'source_governor_unavailable'
      reason = $admit.Reason
      fak_exe = "$FakExe"
      launcher = 'fleet_resume_watchdog.ps1'
    } | ConvertTo-Json -Compress
    Add-Content -Path $ledgerPath -Value $warnRec
    Toast "Resume source governor OFFLINE" "$($admit.Reason) -- live resumes are fail-open (no host-wide rail)" 'warn' 'resume-gate-failopen' 720
  }
  if (-not $admit.Admit) {
    Note "  DEFER $sid8 acct=$acct -- per-source gate: $($admit.Reason)"
    $rec = @{
      ts = ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))
      session = $sid
      account = $p.account
      resume_account = $p.resume_account
      phase = 'deferred'
      cause = 'source_concurrency_gate'
      reason = $admit.Reason
    } | ConvertTo-Json -Compress
    Add-Content -Path $ledgerPath -Value $rec
    continue
  }

  # re-home: copy the transcript into the target account first, else
  # `claude --resume` (CLAUDE_CONFIG_DIR + cwd scoped) can't find it there.
  if ($p.rehomed) {
    $srcCfg = if ($p.source_config_dir) { $p.source_config_dir } else { $p.config_dir }
    $srcFile = Join-Path $srcCfg (Join-Path 'projects' (Join-Path $p.project "$sid.jsonl"))
    if (-not (Test-Path $srcFile)) { Note "  SKIP $sid8 -- re-home source transcript missing"; continue }
    $dstDir = Join-Path $resumeCfg (Join-Path 'projects' $p.project)
    if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
    Copy-Item $srcFile (Join-Path $dstDir "$sid.jsonl") -Force
    $srcSide = Join-Path $srcCfg (Join-Path 'projects' (Join-Path $p.project $sid))
    if (Test-Path $srcSide) { Copy-Item $srcSide (Join-Path $dstDir $sid) -Recurse -Force -ErrorAction SilentlyContinue }
    Note "  RE-HOME $sid8 $($p.account) -> $($p.resume_account) (transcript copied; resuming on healthy account)"
  }

  $env:CLAUDE_CONFIG_DIR = $resumeCfg
  $env:JOB_SUPERVISED_WORKER = $null
  $out = Join-Path $LogDir ("resume-{0}-{1}.log" -f $sid8, ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds()))
  $wd = if ($p.cwd -and (Test-Path $p.cwd)) { $p.cwd } else { $FleetDir }
  $proc = Start-Process -FilePath $ClaudeExe `
    -ArgumentList @('--resume', $sid, '-p',
      'Resume where you left off; re-establish any /goal or /loop and continue toward it.',
      '--dangerously-skip-permissions') `
    -WorkingDirectory $wd -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError "$out.err"
  # record in the durable ledger BEFORE anything else, so a crash can't double-resume.
  # ts is computed PER LAUNCH (not once per tick): the source governor's spacing floor
  # and the launch-spacing witness both read these timestamps, so two launches paced
  # $LaunchSpacingSec apart must not share one stale tick-start second (#2172).
  $attempt = [int]$launchCount[$sid] + 1
  $rec = @{ ts = ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')); session = $sid; account = $p.account; resume_account = $p.resume_account; rehomed = [bool]$p.rehomed; project = $p.project; pid = $proc.Id; cause = $p.disp; phase = 'launched'; attempt = $attempt } | ConvertTo-Json -Compress
  Add-Content -Path $ledgerPath -Value $rec
  $launchCount[$sid] = $attempt
  $liveResume[$sid] = $proc.Id
  $launched++
  Note "  RESUMED $sid8 acct=$acct pid=$($proc.Id) (attempt $attempt/$MaxAttempts; re-eligible if it dies again)"
  Toast "Resumed dead session" "$sid8  ($acct / $($p.project))" 'info' "resume:$sid" 1440
  if ($LaunchSpacingSec -gt 0 -and $launched -lt $MaxPerTick) {
    Start-Sleep -Seconds $LaunchSpacingSec
  }
}

# 2. alert on true login-blocked accounts -- once per account blocker.
$notifiedPath = Join-Path $regDir '_notified.json'
$notified = @{}
if (Test-Path $notifiedPath) {
  try { (Get-Content $notifiedPath -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $notified[$_.Name] = $true } } catch {}
}
$sessPath = Join-Path $regDir 'sessions.json'
if (Test-Path $sessPath) {
  $regDoc = Get-Content $sessPath -Raw | ConvertFrom-Json
  $loginBlockedAccounts = @($regDoc.accounts | Where-Object {
    $_.block_kind -eq 'auth' -and -not $_.throttled -and $_.blocked
  })
  foreach ($a in $loginBlockedAccounts) {
    $key = "auth-account:$($a.account):$($a.block_reason)"
    if ($notified.ContainsKey($key)) { continue }
    $acct = if ($a.tag) { $a.tag } else { ($a.account -replace '\.claude-?', '') }
    $reason = if ($a.block_reason) { $a.block_reason } else { 'auth/login required' }
    $sessions = [int]($a.auth_blocked_sessions)
    $sessionText = if ($sessions -gt 0) { " / $sessions stopped session(s)" } else { "" }
    Toast "Account needs re-login" "$acct : $reason$sessionText" 'warn' $key 1440
    Note "  ALERT auth-blocked acct=$acct reason=$reason (notified)"
    $notified[$key] = $true
  }
  ($notified | ConvertTo-Json) | Set-Content -Path $notifiedPath -Encoding UTF8
}

Note "  done: launched=$launched sessions_in_ledger=$($launchCount.Count)"
# refresh the observability card on disk
try { & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $FleetDir 'tools\fleet_status.ps1') -Quiet -RegistryDir $regDir -LogDir $LogDir } catch {}
exit 0
