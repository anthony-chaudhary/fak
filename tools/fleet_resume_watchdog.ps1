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

Managed-cache posture (#2178; on-by-default 2026-07-10): a resumed child is fronted with its
own `fak guard --managed-cache on --` by default (best-effort managed cache everywhere), so the
resume wave names the SAME cache posture as `fak accounts launch` / the dispatch worker. Set
FAK_GUARD_API_KEY_ENV=<var> to also bill an Anthropic API key; FAK_MANAGED_CACHE=off opts out,
and FAK_MANAGED_CACHE=auto restores the bare `claude --resume` (guard's own billing-gated auto).

  .\fleet_resume_watchdog.ps1                 # dry-run: log what it WOULD resume
  .\fleet_resume_watchdog.ps1 -Live           # actually resume in a visible window
  .\fleet_resume_watchdog.ps1 -Live -Headless # explicit unattended opt-out
#>
[CmdletBinding()]
param(
  [switch]$Live,
  # Live recovery is operator-visible by default. Use -Headless only for an
  # unattended host where opening a recovery window is undesirable.
  [switch]$Headless,
  [string]$FleetDir   = '',
  [int]$WindowH       = 6,
  # Max resumes launched per tick. Raised 4 -> 6 (2026-07-10) to gear up the drain once the
  # managed-cache-1h-TTL-400 root cause was fixed: with the subscription 400 removed, each
  # launched resume now PROVES a turn instead of crash-looping, so the eligible backlog
  # (observed ~150 mid-incident) actually shrinks per tick -- a modestly higher cap drains it
  # faster. This is a soft cap: the shared source governor + LaunchSpacingSec remain the real
  # account-pressure rail, so 6 cannot burst past what the governor admits. Override with -MaxPerTick.
  [int]$MaxPerTick    = 0,
  [int]$MaxPerTickFloor = 4,
  [int]$MaxPerTickCeiling = 64,
  [int]$SeatSessionCap = 6,
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
  # Active account probing on the registry refresh. 'auto' (default) probes STALE
  # workers (blocked OR idle with no live-session evidence) only on a -Live tick (off
  # for dry-run, which must stay side-effect-free); 'blocked'/'stale'/'all' force it;
  # 'none' disables. The probe spends one tiny haiku 'say pong' per stale worker and
  # skips accounts whose throttle reset is still future (account_probe's
  # --skip-active-throttle), so a recovered account re-enters the pool — and an idle
  # seat that quietly hit its limit leaves it — without anyone running a real session.
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
$claudeDisableMarker = if ($env:FLEET_CLAUDE_DISABLE_MARKER) {
  $env:FLEET_CLAUDE_DISABLE_MARKER
} elseif ($env:LOCALAPPDATA) {
  Join-Path $env:LOCALAPPDATA 'Fleet\claude.disabled'
} else {
  Join-Path $env:USERPROFILE '.fleet\claude.disabled'
}
if (Test-Path -LiteralPath $claudeDisableMarker -PathType Leaf) {
  Write-Output "CLAUDE_DISABLED marker=$claudeDisableMarker; watchdog skipped"
  exit 0
}

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
$logMaxBytes = if ($env:FAK_WATCHDOG_LOG_MAX_BYTES) { [int64]$env:FAK_WATCHDOG_LOG_MAX_BYTES } else { 5MB }
$resumeLogRetainDays = if ($env:FAK_RESUME_LOG_RETAIN_DAYS) { [double]$env:FAK_RESUME_LOG_RETAIN_DAYS } else { 14 }
$ledgerRetainDays = if ($env:FAK_RESUME_LEDGER_RETAIN_DAYS) { [double]$env:FAK_RESUME_LEDGER_RETAIN_DAYS } else { 30 }
$ledgerCompactBytes = if ($env:FAK_RESUME_LEDGER_COMPACT_BYTES) { [int64]$env:FAK_RESUME_LEDGER_COMPACT_BYTES } else { 512KB }
function Rotate-BoundedFile([string]$Path, [int64]$MaxBytes) {
  if ($MaxBytes -le 0 -or -not (Test-Path -LiteralPath $Path)) { return }
  if ((Get-Item -LiteralPath $Path).Length -lt $MaxBytes) { return }
  $old = "$Path.1"
  if (Test-Path -LiteralPath $old) { Remove-Item -LiteralPath $old }
  Move-Item -LiteralPath $Path -Destination $old
}
function Prune-ResumeLogs([string]$Dir, [double]$Days) {
  if ($Days -lt 0) { return }
  $cutoff = [DateTime]::UtcNow.AddDays(-$Days)
  Get-ChildItem -LiteralPath $Dir -File | Where-Object {
    $_.Name -like 'resume-*.log' -or $_.Name -like 'resume-*.log.err'
  } | Where-Object { $_.LastWriteTimeUtc -lt $cutoff } | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName
  }
}
Rotate-BoundedFile $log $logMaxBytes
Rotate-BoundedFile (Join-Path $LogDir 'notifications.log') $logMaxBytes
Prune-ResumeLogs $LogDir $resumeLogRetainDays
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $FleetDir) { $FleetDir = $repoRoot }
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

function Get-ManagedCachePosture {
  # #2178 parity for the resume wave: shape the `fak guard` managed-cache flags from the same
  # two fleet env knobs the Go launchers read (FAK_MANAGED_CACHE / FAK_GUARD_API_KEY_ENV) in
  # the same stable order (--api-key-env then --managed-cache). Only an EXPLICIT 'auto' emits
  # NOTHING (guard keeps its own billing-gated auto); 'off' opts out; a malformed mode returns
  # @() plus a Warn string rather than throwing -- a headless launcher warns-and-continues
  # passive instead of stranding the whole resume wave.
  #
  # Subscription-safe UNSET default (2026-07-10, supersedes the blind unset=>on that #2178
  # introduced). Forcing --managed-cache on activates the stable-prefix 1h-TTL cache_control
  # upgrade, which the subscription-OAuth wire rejects as
  #   `API Error: 400 upstream rejected the request as malformed`
  # -> instant CHILD_CRASH, so the resumed session NEVER proves a turn. This was proven by a
  # clean-env ablation on a subscription seat: bare/auto resumes returned a real turn, but
  # `fak guard --managed-cache on` 400'd EVEN with the extended-cache-ttl beta-union fix
  # compiled in (docs/notes/MANAGED-CACHE-1H-TTL-400-FIX-2026-07-09.md). The 1h upgrade only
  # pays off on an API-KEY-billed seat; on flat-rate subscription OAuth it is pure downside.
  # So an UNSET knob is now billing-AWARE: force `on` only when an API-key env is configured,
  # else default to `auto` (guard's billing-gated posture -> passive on subscription, ACTIVE on
  # api-key). This keeps "best-effort managed cache everywhere" but RESOLVES it correctly
  # instead of 400-crashing every subscription resume. An EXPLICIT on|off|auto is still honored
  # verbatim; an explicit `on` with no api-key env is honored but Warn'd (it will 400 a sub seat).
  $raw = ("$env:FAK_MANAGED_CACHE").Trim().ToLowerInvariant()
  $apiKeyEnv = ("$env:FAK_GUARD_API_KEY_ENV").Trim()
  $warn = $null
  if ($raw -eq '') {
    if ($apiKeyEnv -ne '') { $mode = 'on' }   # api-key seat: 1h-TTL upgrade helps and is well-formed (beta-union fix)
    else { $mode = 'auto' }                    # subscription seat: billing-gated auto -> passive (forcing on 400s the OAuth wire)
  } elseif ($raw -eq 'auto' -or $raw -eq 'on' -or $raw -eq 'off') {
    $mode = $raw
    if ($mode -eq 'on' -and $apiKeyEnv -eq '') {
      $warn = "FAK_MANAGED_CACHE=on with no FAK_GUARD_API_KEY_ENV: the forced 1h-TTL upgrade 400s a subscription-OAuth resume ('upstream rejected the request as malformed'); use auto/off on subscription seats or set an api-key env"
    }
  } else {
    return @{ Args = @(); Warn = "FAK_MANAGED_CACHE='$raw': unknown managed-cache mode (auto|on|off) -- ignoring; resuming passive" }
  }
  $postureArgs = @()
  if ($apiKeyEnv -ne '') { $postureArgs += @('--api-key-env', $apiKeyEnv) }
  if ($mode -ne 'auto') { $postureArgs += @('--managed-cache', $mode) }
  return @{ Args = $postureArgs; Warn = $warn }
}

function AppendJsonLine($path, $obj) {
  $dir = Split-Path -Parent $path
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  Add-Content -Path $path -Value ($obj | ConvertTo-Json -Compress)
}

function ParseUnix($ts) {
  if (-not $ts) { return [int64]0 }
  try { return [DateTimeOffset]::Parse("$ts").ToUnixTimeSeconds() } catch { return [int64]0 }
}

function IsoFromUnix([int64]$unix) {
  if ($unix -le 0) { return [DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ') }
  return [DateTimeOffset]::FromUnixTimeSeconds($unix).UtcDateTime.ToString('yyyy-MM-ddTHH:mm:ssZ')
}

function RecordDrainTick($statusLedger, $mode, $planRows) {
  $ts = [DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
  AppendJsonLine $statusLedger @{
    ts = $ts; phase = 'status'; mode = $mode; auto_resume_depth = @($planRows).Count
  }
  $seenQueued = @{}
  if (Test-Path $statusLedger) {
    Get-Content $statusLedger | ForEach-Object {
      try {
        $r = $_ | ConvertFrom-Json
        if ($r.session -and @('queued','detected','auto_resume') -contains "$($r.phase)") {
          $seenQueued[$r.session] = $true
        }
      } catch {}
    }
  }
  foreach ($p in @($planRows)) {
    if (-not $p.session -or $seenQueued.ContainsKey($p.session)) { continue }
    AppendJsonLine $statusLedger @{
      ts = $ts; session = $p.session; account = $p.account; resume_account = $p.resume_account
      project = $p.project; phase = 'queued'; mode = $mode; cause = $p.disp
    }
    $seenQueued[$p.session] = $true
  }
}

function RecordProgressWitness($statusLedger, $mode, $sid, [int64]$lastLaunch, $progress) {
  if (-not $sid -or $lastLaunch -le 0 -or -not $progress -or [int]$progress.NewTurns -le 0 -or [int64]$progress.ProgressUnix -le $lastLaunch) {
    return
  }
  if (Test-Path $statusLedger) {
    foreach ($line in Get-Content $statusLedger) {
      try {
        $r = $line | ConvertFrom-Json
        if ($r.session -ne $sid) { continue }
        $at = [int64](ParseUnix $(if ($r.progress_witnessed_at) { $r.progress_witnessed_at } else { $r.ts }))
        if ($at -gt $lastLaunch -and (($r.phase -eq 'progress') -or [int]$r.new_turns -gt 0 -or $r.progress_witnessed_at)) {
          return
        }
      } catch {}
    }
  }
  $iso = IsoFromUnix ([int64]$progress.ProgressUnix)
  AppendJsonLine $statusLedger @{
    ts = $iso; session = $sid; phase = 'progress'; mode = $mode
    new_turns = [int]$progress.NewTurns
    progress_witnessed_at = $iso
    progress_witness_source = 'transcript_real_turn_after_resume'
  }
}

function RecordSettled($statusLedger, $mode, $sid, $reason) {
  if (-not $sid) { return }
  AppendJsonLine $statusLedger @{
    ts = [DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    session = $sid; phase = 'settled'; mode = $mode; reason = $reason
  }
}

function RecordTerminalOutcome($ledgerPath, $sid, $outcome, $reason) {
  if (-not $sid -or $outcome -ne 'unrecoverable') { return }
  AppendJsonLine $ledgerPath @{
    ts = [DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    session = $sid; phase = 'settled'; outcome = $outcome
    action = 'consolidate-unrecoverable-resume-wall'; reason = $reason
  }
}

function NewestTranscript($sid) {
  if (-not $sid -or -not $env:USERPROFILE) { return $null }
  $roots = Get-ChildItem -Path (Join-Path $env:USERPROFILE '.claude*') -Directory -ErrorAction SilentlyContinue
  $best = $null
  foreach ($root in @($roots)) {
    $projects = Join-Path $root.FullName 'projects'
    if (-not (Test-Path $projects)) { continue }
    $hit = Get-ChildItem -Path $projects -Recurse -Filter "$sid.jsonl" -File -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($hit -and ((-not $best) -or $hit.LastWriteTime -gt $best.LastWriteTime)) { $best = $hit }
  }
  return $best
}

function MessageText($content) {
  if ($null -eq $content) { return '' }
  if ($content -is [string]) { return $content }
  $parts = @()
  foreach ($b in @($content)) {
    if ($b.text) { $parts += "$($b.text)" }
  }
  return ($parts -join ' ')
}

function TranscriptProgress($sid, [int64]$lastLaunch) {
  $out = [pscustomobject]@{ Outcome = 'unknown'; NewTurns = 0; ProgressUnix = [int64]0; TerminalUnix = [int64]0 }
  $file = NewestTranscript $sid
  if (-not $file) { return $out }
  $terminalText = ''
  Get-Content $file.FullName -ErrorAction SilentlyContinue | ForEach-Object {
    try {
      $r = $_ | ConvertFrom-Json
      $ts = ParseUnix $r.timestamp
      $role = if ($r.message -and $r.message.role) { "$($r.message.role)" } else { "$($r.type)" }
      if ($role -eq 'user' -or $role -eq 'assistant') {
        $terminalText = MessageText $r.message.content
        $out.TerminalUnix = $ts
      }
      if ($role -eq 'assistant' -and $r.message -and $r.message.model -and $r.message.model -ne '<synthetic>' -and $r.message.usage) {
        $prompt = [int]$r.message.usage.input_tokens + [int]$r.message.usage.cache_read_input_tokens + [int]$r.message.usage.cache_creation_input_tokens
        if ($prompt -gt 0 -and $ts -gt $lastLaunch) {
          $out.NewTurns = [int]$out.NewTurns + 1
          if ([int64]$out.ProgressUnix -le 0) { $out.ProgressUnix = $ts }
        }
      }
    } catch {}
  }
  $low = $terminalText.ToLowerInvariant()
  $limitWall = $low -match 'limit\s*[·:|.\-]?\s*resets?' -or $low -match '\b(session|weekly|usage|fable\s+\d+)\s+limit\b' -or $low.Contains('/usage-credits')
  $transientWall = $low.Contains('overloaded') -or $terminalText.Contains('529') -or ($low.Contains('api error') -and $low.Contains('rate'))
  if ($low -match 'login interrupted|please run /login|authentication_error|invalid x-api-key|invalid authentication credentials|401|oauth token has expired|credit balance is too low|organization has disabled claude subscription access|use an anthropic api key instead|not logged in') {
    $out.Outcome = 'unrecoverable'
  } elseif ($limitWall -or $transientWall) {
    $out.Outcome = 'recoverable'
  } elseif ($terminalText.Trim()) {
    $out.Outcome = 'progressed'
  }
  return $out
}

function PruneClosedPlanRows($planPath, $closedSids) {
  if (-not (Test-Path $planPath) -or $closedSids.Count -eq 0) { return }
  try {
    $doc = Get-Content $planPath -Raw | ConvertFrom-Json
    $before = @($doc.plan).Count
    $doc.plan = @($doc.plan | Where-Object { -not $closedSids.ContainsKey($_.session) })
    $after = @($doc.plan).Count
    if ($after -lt $before) {
      $doc | ConvertTo-Json -Depth 12 | Set-Content -Path $planPath -Encoding UTF8
      Note ("  pruned {0} closed AUTO_RESUME row(s) from resume_plan.json" -f ($before - $after))
    }
  } catch {
    Note "  WARN could not prune closed resume_plan rows: $($_.Exception.Message)"
  }
}

# 1. refresh registry + plan (extract in advance). On a live tick, also ACTIVELY probe
# blocked accounts so a silently-recovered account (re-login / access re-enabled / throttle
# expired) re-enters the available pool instead of staying latched on a stale verdict.
# auto -> 'stale' (blocked OR idle-with-no-live-session) on a live tick, not 'blocked':
# a passive available verdict only proves the seat was serving at its LAST activity, so
# an idle seat that hit its session limit after going quiet still reads available and
# the planner will re-home a crashed session onto its wall (observed 2026-07-06). One
# paced pong per idle seat keeps rehome-target evidence fresh; the min-interval floor
# and skip-active-throttle keep the spend bounded.
$probeMode = if ($Probe -eq 'auto') { if ($Live) { 'stale' } else { 'none' } } else { $Probe }
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
# Publish the interactive guard-session journal into the machine-owned shared
# registry. The LocalService crash sensor cannot depend on a user-profile path,
# and this atomic mirror survives Windows Terminal/fak/child process death.
if ($IsWindows -or $env:OS -eq 'Windows_NT') {
  $machineRegistry = Join-Path $env:ProgramData 'fak\guard-control\registry'
  $sourceJournal = Join-Path $regDir 'guard_sessions.jsonl'
  $destJournal = Join-Path $machineRegistry 'guard_sessions.jsonl'
  if (Test-Path -LiteralPath $sourceJournal) {
    $tempJournal = $null
    try {
      New-Item -ItemType Directory -Force -Path $machineRegistry | Out-Null
      $tempJournal = Join-Path $machineRegistry ('.guard_sessions-' + [guid]::NewGuid().ToString('N') + '.tmp')
      Copy-Item -LiteralPath $sourceJournal -Destination $tempJournal -Force
      Move-Item -LiteralPath $tempJournal -Destination $destJournal -Force
      Note "  host registry: published interactive crash cohort source"
    } catch {
      Note "  host registry publish FAILED: $($_.Exception.Message)"
      if ($tempJournal -and (Test-Path -LiteralPath $tempJournal)) {
        Remove-Item -LiteralPath $tempJournal -Force -ErrorAction SilentlyContinue
      }
    }
  }
}
$planPath = Join-Path $regDir 'resume_plan.json'
$plan = if (Test-Path $planPath) { (Get-Content $planPath -Raw | ConvertFrom-Json).plan } else { @() }
# Size this tick from the registry snapshot just refreshed above. An explicit
# -MaxPerTick remains the operator override; zero selects the live derivation.
$capSource = 'override'
$capEvidence = [ordered]@{ cap = $MaxPerTick; floor = $MaxPerTickFloor; ceiling = $MaxPerTickCeiling; seat_cap = $SeatSessionCap; healthy_seats = 0; headroom = 0 }
if ($MaxPerTick -le 0) {
  $capSource = 'derived'
  $capJson = & $fakExe resume cap --sessions (Join-Path $regDir 'sessions.json') --floor $MaxPerTickFloor --ceiling $MaxPerTickCeiling --seat-cap $SeatSessionCap
  if ($LASTEXITCODE -ne 0) { throw "resume cap derivation failed (exit $LASTEXITCODE)" }
  $capEvidence = $capJson | ConvertFrom-Json
  $MaxPerTick = [int]$capEvidence.cap
}
# Continuous-drain of the resume backlog (#3587, DEFAULT OFF) -- .py parity
# (fleet_resume_watchdog.py tick_launch_cap). On a LIVE tick with FAK_DRAIN_CONTINUOUS=1 one tick
# drains PAST the per-tick cap: the source governor (`fak resume admit`) + $LaunchSpacingSec become
# the only rate limiter, so recovery LATENCY decouples from the ~5-min cron instead of being
# quantized by it. FAK_DRAIN_MAX is a per-tick backstop (a bounded loop guard, not a rate limiter).
# Two safety rails live in the launch loop below: a governor DEFER ENDS the drain (box saturated),
# and a fail-open governor reverts $drainCap to $MaxPerTick (no continuous drain without an
# enforcing rate limiter). DRAIN_MAX floors at $MaxPerTick so a tiny backstop never resumes fewer.
$drainContinuous = ("$env:FAK_DRAIN_CONTINUOUS").Trim().ToLowerInvariant() -in @('1', 'true', 'yes', 'on')
$drainMax = if ($env:FAK_DRAIN_MAX) { [int]$env:FAK_DRAIN_MAX } else { 500 }
$drainCap = if ($Live -and $drainContinuous) { [Math]::Max($MaxPerTick, $drainMax) } else { $MaxPerTick }
$mode = if ($Live) { 'LIVE' } else { 'DRY-RUN' }
$drainNote = if ($Live -and $drainContinuous) { " drain=continuous(<=$drainCap)" } else { '' }
Note ("TICK $mode plan={0} window=${WindowH}h cap=$MaxPerTick$drainNote" -f @($plan).Count)
$statusLedger = Join-Path $regDir 'resume_watchdog_status.jsonl'
RecordDrainTick $statusLedger $mode @($plan)
AppendJsonLine $statusLedger @{ ts = [DateTime]::UtcNow.ToString('o'); phase = 'cap'; mode = $mode; cap_source = $capSource; cap = [int]$MaxPerTick; floor = [int]$capEvidence.floor; ceiling = [int]$capEvidence.ceiling; seat_cap = [int]$capEvidence.seat_cap; healthy_seats = [int]$capEvidence.healthy_seats; headroom = [int]$capEvidence.headroom }

# defense-in-depth: the set of account dir-basenames that policy still treats as
# workers. fleet_sessions.py already excludes non-workers when it writes the plan,
# but a stale plan file could predate the policy — so re-check each entry here too.
$workerAccts = @{}
$accountRows = @{}
try {
  $acctDoc = & $py (Join-Path $FleetDir 'tools\fleet_accounts.py') json 2>$null | ConvertFrom-Json
  foreach ($a in @($acctDoc.accounts)) {
    $accountRows[$a.account] = $a
    if ($a.kind -eq 'worker') { $workerAccts[$a.account] = $true }
  }
} catch {}

# durable resume ledger: count prior launches per session and flag operator-settled /
# unrecoverable sids, so a session is retried up to -MaxAttempts times (and moved to a
# fresh seat by the planner after repeated in-place failures) instead of once ever.
$ledgerPath = Join-Path $regDir 'resume_ledger.jsonl'
$launchCount = @{}
$lastLaunch = @{}
$launchCauses = @{}
$ledgerBlocked = @{}
if (Test-Path $ledgerPath) {
  Get-Content $ledgerPath | ForEach-Object {
    try {
      $r = $_ | ConvertFrom-Json
      $s = $r.session
      if (-not $s) { return }
      # Re-arm marker (2026-07-10): a self-heal/operator row that reclaims a session which burned
      # its whole attempt budget on a KNOWN-transient infra fault -- e.g. the managed-cache-1h-TTL
      # 400 wave (#2178), where every resume 400'd on a subscription seat and falsely climbed to
      # the 8/8 cap -- rather than a real defect. Processed in append order, so it zeroes the
      # attempts accrued BEFORE it and lifts a prior soft block; any launch appended AFTER it
      # counts again from 0. A later manual_override/unrecoverable row re-blocks (last write wins).
      if ($r.phase -eq 'rearm' -or $r.outcome -eq 'rearm') {
        $launchCount[$s] = 0
        $lastLaunch[$s] = [int64]0
        $launchCauses[$s] = @()
        [void]$ledgerBlocked.Remove($s)
        return
      }
      if ($r.manual_override -or ("$($r.action)").StartsWith('consolidate')) { $ledgerBlocked[$s] = $true }
      if ($r.outcome -eq 'unrecoverable') { $ledgerBlocked[$s] = $true }
      $nonLaunchPhase = ($r.phase -eq 'deferred') -or ($r.phase -eq 'considered') -or ($r.phase -eq 'skipped') -or ($r.phase -eq 'gate_fail_open')
      if (($r.phase -eq 'launched') -or ($r.phase -eq 'resumed') -or ($r.cause -and -not $nonLaunchPhase)) {
        $launchCount[$s] = [int]$launchCount[$s] + 1
        if ($r.cause) { $launchCauses[$s] = @($launchCauses[$s]) + @("$($r.cause)") }
        $u = ParseUnix $r.ts
        if ($u -gt [int64]$lastLaunch[$s]) { $lastLaunch[$s] = $u }
      }
    } catch {}
  }
}
$launched = 0
# One durable warning per tick when the source governor is unavailable (#2173): a
# fail-open launch runs WITHOUT the host-wide concurrency rail, so it must be visible
# in the ledger and the Action Center, never silent.
$gateFailOpenWarned = $false
$closedSids = @{}

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

# Resolve the managed-cache posture ONCE per tick (the env is tick-constant) and warn ONCE.
# On-by-default (2026-07-10): the posture is on unless FAK_MANAGED_CACHE=auto, so each resumed
# child is fronted with its own `fak guard --managed-cache on --` (#2178 parity); an explicit
# `auto` leaves the bare `claude --resume` untouched. A posture that cannot be applied (no fak
# binary) falls back to a direct launch LOUDLY rather than silently dropping the intent.
$posture = Get-ManagedCachePosture
[string[]]$resumePostureArgs = @($posture.Args)
if ($posture.Warn) { Note "  WARN managed-cache: $($posture.Warn)" }
if ($resumePostureArgs.Count -gt 0 -and -not $FakExe) {
  Note "  WARN managed-cache posture configured but ``fak`` is unavailable -- resuming children directly (passive, no posture banner)"
  $resumePostureArgs = @()
} elseif ($resumePostureArgs.Count -gt 0) {
  Note ("  managed-cache posture -> fronting resumed children with ``fak guard {0} --``" -f ($resumePostureArgs -join ' '))
}

$planArr = @($plan)
$planCount = $planArr.Count
for ($idx = 0; $idx -lt $planCount; $idx++) {
  $p = $planArr[$idx]
  if ($launched -ge $drainCap) { Note "  per-tick cap reached ($drainCap)"; break }
  $sid = $p.session; $sid8 = $sid.Substring(0, 8)
  $acct = ($p.account -replace '\.claude-?', ''); if (-not $acct) { $acct = 'default' }
  if ("$($p.disp)".ToUpperInvariant().Contains('AUTH')) {
    $reason = "plan disposition $($p.disp) requires auth/login; automatic resume cannot fix it"
    Note "  SKIP $sid8 -- $reason"
    if ($Live) {
      $closedSids[$sid] = $true
      AppendJsonLine $ledgerPath @{
        ts = ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))
        session = $sid
        account = $p.account
        resume_account = $p.resume_account
        phase = 'settled'
        action = 'consolidate-auth-plan-row'
        outcome = 'unrecoverable'
        cause = $p.disp
        reason = $reason
      }
      RecordSettled $statusLedger $mode $sid $reason
    }
    continue
  }
  if ($workerAccts.Count -and -not $workerAccts.ContainsKey($p.account)) {
    Note "  SKIP $sid8 -- account $($p.account) is not an offered worker (policy/tombstoned)"
    $closedSids[$sid] = $true
    continue
  }
  if ($ledgerBlocked.ContainsKey($sid)) {
    Note "  SKIP $sid8 -- ledger-blocked (operator-settled or unrecoverable auth wall)"
    $closedSids[$sid] = $true
    continue
  }
  $attempts = [int]$launchCount[$sid]
  if ($attempts -ge $MaxAttempts) {
        $causeGroups = @($launchCauses[$sid] | Group-Object | Sort-Object @{Expression='Count';Descending=$true}, @{Expression='Name';Ascending=$true})
    $dominantCause = if ($causeGroups.Count) { $causeGroups[0].Name } else { '' }
    $causeShare = if ($causeGroups.Count) { "{0}/{1}" -f $causeGroups[0].Count, @($launchCauses[$sid]).Count } else { '' }
    AppendJsonLine $statusLedger @{ ts = [DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'); session = $sid; phase = 'settled'; mode = $mode; reason = "attempt cap reached ($attempts/$MaxAttempts)"; dominant_cause = $dominantCause; cause_share = $causeShare }
    Note "  SKIP $sid8 -- attempt cap reached ($attempts/$MaxAttempts) -- left for a human"
    $closedSids[$sid] = $true
    continue
  }
  $last = [int64]$lastLaunch[$sid]
  if ($attempts -gt 0) {
    $progress = TranscriptProgress $sid $last
    RecordProgressWitness $statusLedger $mode $sid $last $progress
    if ([int]$progress.NewTurns -gt 0 -and $progress.Outcome -ne 'recoverable') {
      Note "  SKIP $sid8 -- already resumed once (resume took)"
      $closedSids[$sid] = $true
      continue
    }
    if ($progress.Outcome -eq 'unrecoverable' -and [int64]$progress.TerminalUnix -gt $last) {
      RecordTerminalOutcome $ledgerPath $sid 'unrecoverable' 'terminal auth/access wall after resume'
      RecordSettled $statusLedger $mode $sid 'terminal auth/access wall after resume'
      Note "  SKIP $sid8 -- last resume hit an auth/access wall -- a re-resume cannot fix it"
      $closedSids[$sid] = $true
      continue
    }
  }
  if ($liveResume.ContainsKey($sid)) {
    Note "  SKIP $sid8 -- already live as pid $($liveResume[$sid]) (no duplicate resume)"
    if ($Live) {
      $rec = @{ ts = ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')); session = $sid; account = $p.account; phase = 'skipped'; cause = 'already_live'; live_pid = $liveResume[$sid] } | ConvertTo-Json -Compress
      Add-Content -Path $ledgerPath -Value $rec
    }
    continue
  }
  $resumeCfg = if ($p.resume_config_dir) { $p.resume_config_dir } else { $p.config_dir }
  $resumeAcct = if ($p.resume_account) { "$($p.resume_account)" } else { "$($p.account)" }
  # Defense-in-depth like the worker-account re-check above: the planner has offered a
  # tombstoned seat as a re-home target (observed 2026-07-01: resume_account =
  # .claude-gem8-netra.DELETED-2026-06-29; the launch died on arrival and burned an attempt).
  if ($resumeCfg -match '\.DELETED') {
    Note "  SKIP $sid8 -- resume target $resumeCfg is a tombstoned seat"
    $closedSids[$sid] = $true
    if ($Live) {
      $rec = @{ ts = ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')); session = $sid; account = $p.account; resume_account = $p.resume_account; phase = 'skipped'; cause = 'deleted_seat_target' } | ConvertTo-Json -Compress
      Add-Content -Path $ledgerPath -Value $rec
    }
    continue
  }
  if ($accountRows.Count -and $resumeAcct -and $accountRows.ContainsKey($resumeAcct)) {
    $target = $accountRows[$resumeAcct]
    if ($target.kind -ne 'worker' -or $target.blocked -or -not $target.available) {
      $why = if ($target.block_reason) { "$($target.block_reason)" } else { "kind=$($target.kind) available=$($target.available) blocked=$($target.blocked)" }
      Note "  SKIP $sid8 -- resume target $resumeAcct is not launchable ($why)"
      $closedSids[$sid] = $true
      if ($Live) {
        $rec = @{ ts = ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')); session = $sid; account = $p.account; resume_account = $p.resume_account; phase = 'skipped'; cause = 'blocked_resume_target'; reason = $why } | ConvertTo-Json -Compress
        Add-Content -Path $ledgerPath -Value $rec
      }
      continue
    }
  } elseif ($workerAccts.Count -and $resumeAcct -and -not $workerAccts.ContainsKey($resumeAcct)) {
    Note "  SKIP $sid8 -- resume target $resumeAcct is not an offered worker (policy/tombstoned)"
    $closedSids[$sid] = $true
    if ($Live) {
      $rec = @{ ts = ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')); session = $sid; account = $p.account; resume_account = $p.resume_account; phase = 'skipped'; cause = 'nonworker_resume_target' } | ConvertTo-Json -Compress
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
  if ($admit.FailOpen -and $drainContinuous -and $drainCap -gt $MaxPerTick) {
    # Continuous-drain safety rail (#3587): a fail-open governor cannot bound a storm, so WITHOUT
    # an enforcing rate limiter the drain must not run past the tick-quantized cap. Revert to
    # $MaxPerTick for the rest of this tick (idempotent -- only lowers once).
    $drainCap = $MaxPerTick
    Note "  drain: source governor UNAVAILABLE -> reverting to per-tick cap ($MaxPerTick) this tick (no continuous drain without the rate limiter)"
    if ($launched -ge $drainCap) { Note "  per-tick cap reached ($drainCap)"; break }
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
    # Continuous-drain (#3587): the governor is host-wide, so a DEFER means the box is saturated --
    # END the drain this tick rather than spinning the rest of the plan into a deferred-row storm
    # onto capped seats. The tick-quantized default keeps the old per-session skip.
    if ($drainContinuous) {
      Note "  drain: source governor DEFER -> box saturated, ending drain this tick"
      break
    }
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
  [string[]]$childArgs = @('--resume', $sid, '-p',
    'Resume where you left off; re-establish any /goal or /loop and continue toward it.',
    '--dangerously-skip-permissions')
  # Opted-in + fak resolvable -> front the child with its OWN `fak guard <posture> --`: it
  # binds its own gateway on its own CLAUDE_CONFIG_DIR seat, prints its own posture banner, and
  # reaches the ACTIVE 1h-TTL upgrade when API-key-billed -- inheriting no wire from this
  # watchdog. Guard auto-detects claude -> --provider anthropic (posture flags BEFORE `--`).
  if ($resumePostureArgs.Count -gt 0 -and $FakExe) {
    $launchFile = $FakExe
    [string[]]$launchArgs = @('guard') + $resumePostureArgs + @('--', $ClaudeExe) + $childArgs
  } else {
    $launchFile = $ClaudeExe
    [string[]]$launchArgs = $childArgs
  }
  $launchMode = if ($Headless) { 'headless' } else { 'visible' }
  $startArgs = @{
    FilePath = $launchFile
    ArgumentList = $launchArgs
    WorkingDirectory = $wd
    PassThru = $true
  }
  if ($Headless) {
    $startArgs.WindowStyle = 'Hidden'
    $startArgs.RedirectStandardOutput = $out
    $startArgs.RedirectStandardError = "$out.err"
  }
  # Record every launch outcome durably, including failures before a child PID
  # exists, so visible/headless posture remains inspectable after this tick exits.
  # ts is computed PER LAUNCH (not once per tick): the source governor's spacing floor
  # and the launch-spacing witness both read these timestamps, so two launches paced
  # $LaunchSpacingSec apart must not share one stale tick-start second (#2172).
  $attempt = [int]$launchCount[$sid] + 1
  try {
    $proc = Start-Process @startArgs
  } catch {
    $detail = "Start-Process: $($_.Exception.Message)"
    $rec = @{ ts = ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')); session = $sid; account = $p.account; resume_account = $p.resume_account; rehomed = [bool]$p.rehomed; project = $p.project; cause = $p.disp; phase = 'launch_failed'; attempt = $attempt; launch_mode = $launchMode; detail = $detail } | ConvertTo-Json -Compress
    Add-Content -Path $ledgerPath -Value $rec
    $launchCount[$sid] = $attempt
    Note "  FAILED $sid8 acct=$acct mode=$launchMode -- $detail"
    Toast "Resume launch failed" "$sid8  ($acct / $($p.project))" 'warn' "resume-failed:$sid" 1440
    continue
  }
  $rec = @{ ts = ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')); session = $sid; account = $p.account; resume_account = $p.resume_account; rehomed = [bool]$p.rehomed; project = $p.project; pid = $proc.Id; cause = $p.disp; phase = 'launched'; attempt = $attempt; launch_mode = $launchMode } | ConvertTo-Json -Compress
  Add-Content -Path $ledgerPath -Value $rec
  $launchCount[$sid] = $attempt
  $liveResume[$sid] = $proc.Id
  $launched++
  Note "  RESUMED $sid8 acct=$acct pid=$($proc.Id) mode=$launchMode (attempt $attempt/$MaxAttempts; re-eligible if it dies again)"
  Toast "Resumed dead session" "$sid8  ($acct / $($p.project))" 'info' "resume:$sid" 1440
  # Pace the next spawn (source-governor spacing witness). Skipped at the drain cap (nothing more
  # launches this tick) and after the final plan entry (nothing follows -- no trailing dead time,
  # which matters under continuous-drain where $drainCap >> $planCount).
  if ($LaunchSpacingSec -gt 0 -and $launched -lt $drainCap -and $idx -lt ($planCount - 1)) {
    Start-Sleep -Seconds $LaunchSpacingSec
  }
}

PruneClosedPlanRows $planPath $closedSids

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





