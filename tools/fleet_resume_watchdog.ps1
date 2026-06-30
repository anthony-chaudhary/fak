<#
fleet_resume_watchdog.ps1 -- the cross-account resume layer for ALL autonomous
Claude sessions (not just the supervisor's job workers).

Each tick:
  1. EXTRACT-IN-ADVANCE: refresh the on-disk session registry
     (tools/_registry/sessions.json) and the AUTO_RESUME plan
     (tools/_registry/resume_plan.json) via fleet_sessions.py.
  2. Resume each AUTO_RESUME session ONCE EVER, under its owning account's
     CLAUDE_CONFIG_DIR. "Once ever" is enforced by a durable ledger
     (tools/_registry/resume_ledger.jsonl) -- a session that dies again after
     being auto-resumed is left for a human, never re-resumed in a loop.
  3. Notify (Windows Action Center + notifications.log) on relevant actions:
     a resume, and an account that needs human re-login (BLOCKED_AUTH).

Safety rails:
  * DRY-RUN by default (pass -Live to actually resume).
  * Interactive sessions are SURFACE (never auto-resumed); supervisor workers are
    SUPERVISED (left to run_supervise_loop); throttled accounts are deferred.
  * RESUME ONCE: ledger-gated, survives state-file loss.
  * Per-tick launch cap.

  .\fleet_resume_watchdog.ps1                 # dry-run: log what it WOULD resume
  .\fleet_resume_watchdog.ps1 -Live           # actually resume (once per session)
#>
[CmdletBinding()]
param(
  [switch]$Live,
  [string]$FleetDir   = 'C:\work\fleet',
  [int]$WindowH       = 6,
  [int]$MaxPerTick    = 2,
  [string]$ClaudeExe  = '',
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

# durable resume-once ledger: a session that appears here was already resumed once.
$ledgerPath = Join-Path $regDir 'resume_ledger.jsonl'
$resumed = @{}
if (Test-Path $ledgerPath) {
  Get-Content $ledgerPath | ForEach-Object { try { $resumed[($_ | ConvertFrom-Json).session] = $true } catch {} }
}
$nowIso = [DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
$launched = 0

foreach ($p in @($plan)) {
  if ($launched -ge $MaxPerTick) { Note "  per-tick cap reached ($MaxPerTick)"; break }
  $sid = $p.session; $sid8 = $sid.Substring(0, 8)
  $acct = ($p.account -replace '\.claude-?', ''); if (-not $acct) { $acct = 'default' }
  if ($workerAccts.Count -and -not $workerAccts.ContainsKey($p.account)) {
    Note "  SKIP $sid8 -- account $($p.account) is not an offered worker (policy/tombstoned)"; continue
  }
  if ($resumed.ContainsKey($sid)) { Note "  SKIP $sid8 -- already resumed once (ledger)"; continue }
  if (-not $Live) {
    $rh = if ($p.rehomed) { " -> $($p.resume_account) (re-home)" } else { "" }
    Note "  WOULD RESUME $sid8 acct=$acct proj=$($p.project)$rh"; continue
  }

  $resumeCfg = if ($p.resume_config_dir) { $p.resume_config_dir } else { $p.config_dir }
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
  # record in the durable ledger BEFORE anything else, so a crash can't double-resume
  $rec = @{ ts = $nowIso; session = $sid; account = $p.account; resume_account = $p.resume_account; rehomed = [bool]$p.rehomed; project = $p.project; pid = $proc.Id; cause = $p.disp } | ConvertTo-Json -Compress
  Add-Content -Path $ledgerPath -Value $rec
  $resumed[$sid] = $true
  $launched++
  Note "  RESUMED $sid8 acct=$acct pid=$($proc.Id) (ledger-recorded; will not resume again)"
  Toast "Resumed dead session" "$sid8  ($acct / $($p.project))" 'info' "resume:$sid" 1440
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

Note "  done: launched=$launched resumed_total=$($resumed.Count)"
# refresh the observability card on disk
try { & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $FleetDir 'tools\fleet_status.ps1') -Quiet -RegistryDir $regDir -LogDir $LogDir } catch {}
exit 0
