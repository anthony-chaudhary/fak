<#
fleet_status.ps1 -- one-glance observability for the session-resilience layer.

Aggregates everything the operator needs to answer "is the fleet healthy, and if
not, is it infra or the agents?" into a single card, and writes it to
tools/_registry/STATUS.txt so the latest state is always on disk.

  .\fleet_status.ps1            # print the card + refresh STATUS.txt
  .\fleet_status.ps1 -Quiet     # just refresh STATUS.txt (used by the watchdog tick)
#>
[CmdletBinding()]
param(
  [string]$FleetDir = 'C:\work\fleet',
  [string]$RegistryDir = '',
  [string]$LogDir = '',
  [switch]$Quiet
)
$ErrorActionPreference = 'SilentlyContinue'
$stateRoot = if ($env:FLEET_STATE_DIR) {
  $env:FLEET_STATE_DIR
} elseif ($env:LOCALAPPDATA) {
  Join-Path $env:LOCALAPPDATA 'Fleet'
} else {
  Join-Path ([System.IO.Path]::GetTempPath()) 'Fleet'
}
$regDir = if ($RegistryDir) { $RegistryDir } else { Join-Path $stateRoot 'registry' }
$logDir = if ($LogDir) { $LogDir } else { Join-Path $stateRoot 'watchdog' }
$env:FLEET_REG_DIR = $regDir
$out = New-Object System.Collections.Generic.List[string]
function L($s) { $out.Add($s) }

$now = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm')
L "==================== FLEET SESSION STATUS @ ${now}Z ===================="

# registry rollups
$reg = $null
$sp = Join-Path $regDir 'sessions.json'
if (Test-Path $sp) { $reg = Get-Content $sp -Raw | ConvertFrom-Json }
if ($reg) {
  $cats = @{}; $acts = @{}
  foreach ($s in $reg.sessions) {
    $cats[$s.category] = 1 + ($cats[$s.category]); $acts[$s.action] = 1 + ($acts[$s.action])
  }
  L ("sessions: {0}   (registry refreshed {1})" -f $reg.sessions.Count, $reg.generated_utc)
  L ("category: " + (@('INFRA', 'AGENT', 'USER', 'HANGING', 'LIVE' | ForEach-Object { if ($cats[$_]) { "$_=$($cats[$_])" } }) -join '  '))
  $actKeys = $acts.Keys | Sort-Object
  L ("action:   " + (($actKeys | ForEach-Object { "$_=$($acts[$_])" }) -join '  '))
  if ($reg.throttle.PSObject.Properties.Count) {
    L "throttled accounts:"
    $reg.throttle.PSObject.Properties | ForEach-Object { L ("  {0}  resets {1}" -f $_.Name, $_.Value.reset) }
  }
  # ACCOUNTS — which accounts exist, which are offered, which are available now.
  # Workers + live availability come from the registry's `accounts` array
  # (written by fleet_sessions.py); the tombstoned/excluded roster comes from
  # fleet_accounts.py so the operator sees, in one glance, what is NOT offered.
  if ($reg.accounts) {
    $avail = @($reg.accounts | Where-Object { $_.available })
    $blocked = @($reg.accounts | Where-Object { -not $_.available })
    L "ACCOUNTS:"
    L ("  available now: " + (if ($avail.Count) { ($avail | ForEach-Object { $_.tag }) -join ', ' } else { '(none)' }))
    foreach ($b in $blocked) {
      # freshness: a stale carried-forward verdict is the silent-latch hazard, so show it.
      $fresh = ''
      if ($b.PSObject.Properties['verdict_source']) {
        $src = $b.verdict_source
        $age = if ($b.verdict_age_min -ne $null) { "{0:N0}m old" -f [double]$b.verdict_age_min } else { 'age unknown' }
        $fresh = "  [$src, $age]"
      }
      L ("  blocked: {0}  ({1}){2}" -f $b.tag, $b.block_reason, $fresh)
    }
  }
  # excluded/tombstoned accounts (e.g. the break-glass backup) — never offered.
  # AND identity reconciliation: how many real Anthropic accounts back the dirs, and
  # which dirs are duplicates of one account (the bug where one throttled account looked
  # like several healthy workers).
  try {
    $py2 = 'C:\work\job\.venv\Scripts\python.exe'; if (-not (Test-Path $py2)) { $py2 = 'python' }
    $acctDoc = & $py2 (Join-Path $FleetDir 'tools\fleet_accounts.py') json 2>$null | ConvertFrom-Json
    $excl = @($acctDoc.accounts | Where-Object { $_.kind -eq 'excluded' })
    if ($excl.Count) {
      L ("  excluded (tombstoned): " + (($excl | ForEach-Object { "$($_.tag) [$($_.reason)]" }) -join '; '))
    }
    $cw = @($acctDoc.accounts | Where-Object { $_.product -eq 'claude' -and $_.kind -eq 'worker' })
    $logins = @($cw | Where-Object { $_.account_uuid } | Select-Object -ExpandProperty account_uuid -Unique)
    $dups = @($cw | Where-Object { $_.identity_role -eq 'duplicate' })
    if ($cw.Count) {
      L ("IDENTITY: {0} Claude dir(s) -> {1} distinct Anthropic account(s)" -f $cw.Count, $logins.Count)
      if ($dups.Count) {
        L ("  DUPLICATE DIRS (same account as a canonical, not offered): {0}" -f $dups.Count)
        $dups | ForEach-Object {
          L ("    {0}  login={1}  shares: {2}" -f $_.tag, $_.login_email, ($_.identity_peers -join ', '))
        }
      }
      # dirs whose name disagrees with the login inside (the scramble)
      $mismatch = @($cw | Where-Object { $_.login_email -and -not $_.tag_login_match })
      if ($mismatch.Count) {
        L ("  TAG<>LOGIN MISMATCH (dir name != logged-in account): " +
           (($mismatch | ForEach-Object { "$($_.tag)=$($_.login_email)" }) -join '; '))
      }
    }
  } catch {}
  # things needing a human login. Access/subscription walls are blocked, but
  # re-running /login is not the right prompt for those accounts.
  $authBlocked = @($reg.accounts | Where-Object { $_.block_kind -eq 'auth' -and $_.blocked })
  if ($authBlocked.Count) {
    L "NEEDS RE-LOGIN (infra/auth):"
    $authBlocked | ForEach-Object { L ("  {0}  {1}  stopped_sessions={2}" -f $_.account, $_.block_reason, $_.auth_blocked_sessions) }
  }
  $accessBlocked = @($reg.accounts | Where-Object { $_.block_kind -eq 'access' -and $_.blocked })
  if ($accessBlocked.Count) {
    L "ACCESS WALLS (not fixed by /login):"
    $accessBlocked | ForEach-Object { L ("  {0}  {1}" -f $_.account, $_.block_reason) }
  }
  # STALE VERDICTS: a blocked account whose verdict is old AND not from a live probe is
  # the exact silent-latch the probe layer exists to break. Surface each + the one fix.
  $staleMin = if ($env:FLEET_STALE_VERDICT_MIN) { [double]$env:FLEET_STALE_VERDICT_MIN } else { 360 }
  $stale = @($reg.accounts | Where-Object {
    $_.blocked -and $_.PSObject.Properties['verdict_source'] -and
    $_.verdict_source -ne 'probe' -and $_.verdict_age_min -ne $null -and
    [double]$_.verdict_age_min -ge $staleMin
  })
  if ($stale.Count) {
    L ("STALE VERDICTS (blocked >{0}m without a live probe -- confirm with a probe):" -f $staleMin)
    foreach ($s in $stale) {
      L ("  {0}  ({1})  [{2}, {3:N0}m old]  -> python tools/fleet_accounts.py probe --account {0}" -f `
        $s.tag, $s.block_reason, $s.verdict_source, [double]$s.verdict_age_min)
    }
  }
} else { L "registry: (not yet generated -- run fleet_sessions.py registry)" }

# supervisor
try {
  $sup = & 'C:\work\job\.venv\Scripts\python.exe' 'C:\work\job\scripts\supervise_now.py' --json 2>$null | ConvertFrom-Json
  L ("supervisor: verdict={0} alive={1} pid={2} hb_age={3}s" -f $sup.verdict, $sup.process.alive, $sup.process.pid, $sup.process.heartbeat_age_s)
} catch { L "supervisor: (status unavailable)" }

# scheduled tasks + resume mode + ledger
function TaskLine($name) {
  $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
  if (-not $t) { return "$name = NOT INSTALLED" }
  $i = Get-ScheduledTaskInfo -TaskName $name
  $extra = ''
  if ($name -eq 'FleetResumeWatchdog') {
    $a = ($t.Actions | Select-Object -First 1).Arguments
    $extra = if ($a -match '-Live') { ' mode=LIVE' } else { ' mode=DRY-RUN' }
  }
  "$name = $($t.State)$extra  last=$($i.LastRunTime) result=$($i.LastTaskResult) next=$($i.NextRunTime)"
}
L ("watchdog:   " + (TaskLine 'FleetSupervisorWatchdog'))
L ("resume:     " + (TaskLine 'FleetResumeWatchdog'))
$ledger = Join-Path $regDir 'resume_ledger.jsonl'
$nResumed = if (Test-Path $ledger) { (Get-Content $ledger | Measure-Object -Line).Lines } else { 0 }
L ("resumed-once ledger: $nResumed session(s) ever auto-resumed")

# recent account-probe flips: a status change (ACCESS->OK = recovered, OK->AUTH = regressed)
# is the highest-signal account event, so put the last few on the card.
$pl = Join-Path $regDir 'probe_ledger.jsonl'
if (Test-Path $pl) {
  $flips = @()
  Get-Content $pl -Tail 200 | ForEach-Object {
    try { $e = $_ | ConvertFrom-Json } catch { return }
    if ($e.flip) { $flips += $e }
  }
  if ($flips.Count) {
    L "RECENT PROBE FLIPS:"
    $flips | Select-Object -Last 5 | ForEach-Object {
      L ("  {0}  {1}: {2} -> {3}" -f $_.ts, $_.tag, $_.prev_status, $_.status)
    }
  }
  $lastProbe = Get-Content $pl -Tail 1
  if ($lastProbe) {
    try { $lp = $lastProbe | ConvertFrom-Json; L ("last probe: {0}" -f $lp.ts) } catch {}
  }
}

# recent timeline
$tl = Join-Path $regDir 'transitions.log'
if (Test-Path $tl) {
  L "recent transitions:"
  Get-Content $tl -Tail 6 | ForEach-Object { L "  $_" }
}
$nl = Join-Path $logDir 'notifications.log'
if (Test-Path $nl) {
  L "recent notifications:"
  Get-Content $nl -Tail 5 | ForEach-Object { L "  $_" }
}
L "=========================================================================="

$card = $out -join "`n"
if (-not (Test-Path $regDir)) { New-Item -ItemType Directory -Path $regDir -Force | Out-Null }
$card | Set-Content -Path (Join-Path $regDir 'STATUS.txt') -Encoding UTF8
if (-not $Quiet) { $card }
