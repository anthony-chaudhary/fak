<#
reboot_advisor.ps1 -- watch the two reboot-only leaks on the fleet workstation and
tell the operator EARLY (before the lag is felt) when a reboot would pay off.

Why this exists: the desktop-slowness runbook
(docs/notes/DESKTOP-SLOWNESS-MAINTENANCE-2026-06-28.md) established that the machine
"feels slow" not from resource exhaustion but from two long-uptime accumulations that
ONLY clear on reboot and are NOT killable as ordinary processes:
  1. WindowsTerminal single-process chokepoint -- one process renders every fleet pane;
     its handle/thread count climbs monotonically with uptime (measured 14.7k handles at
     ~40h, 31.9k handles / 1.6k threads at ~73h). This is the interactive typing/scroll lag.
  2. TermService (svchost) handle+memory leak on RDP-accessed hosts (resets only on reboot).

The runbook says "reboot when TermService handles climb past ~30k or its working set past
~4 GB, or roughly weekly" -- but nothing WATCHED for that condition. The operator only
found out by feeling the lag, then an agent re-ran the audit by hand. This closes that
loop: a read-only watchdog that measures those exact signals every tick and, when they
cross the reboot-recommend thresholds, emits a structured signal + a deduped operator
toast ("Reboot recommended: ...").

It is ADVISORY ONLY by design:
  * Read-only -- it never kills a process, never reboots, needs no elevation.
  * The reboot itself stays an operator decision for a quiet window (it kills all live
    agent sessions) -- this only surfaces WHEN to schedule it, it never acts.
  * Signals land next to the other watchdogs:
      %LOCALAPPDATA%\Fleet\watchdog\reboot_advisor.log    (human)
      %LOCALAPPDATA%\Fleet\watchdog\reboot_advisor.jsonl  (structured, one record/tick on RECOMMEND)

  .\reboot_advisor.ps1              # measure + advise (safe; no elevation)
  .\reboot_advisor.ps1 -Quiet       # same, but suppress the toast (log/jsonl only)
  .\reboot_advisor.ps1 -SelfTest    # run the decision-logic assertions, exit non-zero on failure
#>
[CmdletBinding()]
param(
  # WindowsTerminal chokepoint (the single busiest WindowsTerminal process):
  [int]$WtHandleWarn = 28000,   # handles at/above this on one WT process == advise reboot.
  [int]$WtThreadWarn = 1200,    # ...a fresh WindowsTerminal holds ~50-150 threads.
  # TermService svchost leak (per the runbook's stated reboot trigger):
  [int]$TermHandleWarn = 30000,
  [int]$TermWsMbWarn   = 4096,
  # Cadence backstop -- "roughly weekly under heavy fleet use":
  [double]$UptimeHoursWarn = 168,
  # Toast dedup: at most one operator toast per this many minutes (6h default).
  [int]$ToastIntervalMinutes = 360,
  [switch]$Quiet,
  [switch]$SelfTest,
  [string]$LogDir = ''
)
$ErrorActionPreference = 'Stop'

# --- Pure decision core: measured values + thresholds in, verdict + reasons out. ---
# Kept side-effect-free so -SelfTest can exercise every branch without touching the host.
# A threshold of 0 disables that signal.
function Get-RebootVerdict {
  param(
    [int]$WtHandles, [int]$WtThreads, [int]$TermHandles, [int]$TermWsMb, [double]$UptimeHours,
    [int]$WtHandleWarn, [int]$WtThreadWarn, [int]$TermHandleWarn, [int]$TermWsMbWarn, [double]$UptimeHoursWarn
  )
  $reasons = @()
  if ($WtHandleWarn   -gt 0 -and $WtHandles   -ge $WtHandleWarn)   { $reasons += "wt_handles=$WtHandles>=$WtHandleWarn" }
  if ($WtThreadWarn   -gt 0 -and $WtThreads   -ge $WtThreadWarn)   { $reasons += "wt_threads=$WtThreads>=$WtThreadWarn" }
  if ($TermHandleWarn -gt 0 -and $TermHandles -ge $TermHandleWarn) { $reasons += "term_handles=$TermHandles>=$TermHandleWarn" }
  if ($TermWsMbWarn   -gt 0 -and $TermWsMb    -ge $TermWsMbWarn)   { $reasons += "term_ws_mb=$TermWsMb>=$TermWsMbWarn" }
  if ($UptimeHoursWarn -gt 0 -and $UptimeHours -ge $UptimeHoursWarn) { $reasons += ("uptime_h=" + [math]::Round($UptimeHours,1) + ">=$UptimeHoursWarn") }
  $verdict = 'OK'
  if ($reasons.Count -gt 0) { $verdict = 'RECOMMEND' }
  return [ordered]@{ verdict = $verdict; reasons = $reasons }
}

# --- SelfTest: assert the decision logic before we trust it against the live host. ---
if ($SelfTest) {
  $fails = 0
  function Check($name, $cond) {
    if ($cond) { Write-Output "  PASS  $name" }
    else       { Write-Output "  FAIL  $name"; $script:fails++ }
  }
  # All clear -> OK, no reasons.
  $v = Get-RebootVerdict -WtHandles 1000 -WtThreads 80 -TermHandles 2000 -TermWsMb 500 -UptimeHours 10 `
        -WtHandleWarn 28000 -WtThreadWarn 1200 -TermHandleWarn 30000 -TermWsMbWarn 4096 -UptimeHoursWarn 168
  Check "clean box is OK"                 ($v.verdict -eq 'OK')
  Check "clean box has no reasons"        ($v.reasons.Count -eq 0)
  # WindowsTerminal handles alone trip it (the real current-box condition).
  $v = Get-RebootVerdict -WtHandles 31954 -WtThreads 900 -TermHandles 2000 -TermWsMb 500 -UptimeHours 72 `
        -WtHandleWarn 28000 -WtThreadWarn 1200 -TermHandleWarn 30000 -TermWsMbWarn 4096 -UptimeHoursWarn 168
  Check "WT handles alone -> RECOMMEND"   ($v.verdict -eq 'RECOMMEND')
  Check "WT handles reason present"       (($v.reasons -join ' ') -match 'wt_handles=31954')
  Check "WT-only trips exactly one"       ($v.reasons.Count -eq 1)
  # Boundary: exactly at threshold trips (>=), one below does not.
  $v = Get-RebootVerdict -WtHandles 28000 -WtThreads 0 -TermHandles 0 -TermWsMb 0 -UptimeHours 0 `
        -WtHandleWarn 28000 -WtThreadWarn 0 -TermHandleWarn 0 -TermWsMbWarn 0 -UptimeHoursWarn 0
  Check "at-threshold trips (>=)"         ($v.verdict -eq 'RECOMMEND')
  $v = Get-RebootVerdict -WtHandles 27999 -WtThreads 0 -TermHandles 0 -TermWsMb 0 -UptimeHours 0 `
        -WtHandleWarn 28000 -WtThreadWarn 0 -TermHandleWarn 0 -TermWsMbWarn 0 -UptimeHoursWarn 0
  Check "one below threshold is OK"       ($v.verdict -eq 'OK')
  # A zero threshold disables its signal (no false trip when TermService is absent).
  $v = Get-RebootVerdict -WtHandles 0 -WtThreads 0 -TermHandles 999999 -TermWsMb 0 -UptimeHours 0 `
        -WtHandleWarn 28000 -WtThreadWarn 0 -TermHandleWarn 0 -TermWsMbWarn 0 -UptimeHoursWarn 0
  Check "zero threshold disables signal" ($v.verdict -eq 'OK')
  # TermService leak trips independently, and multiple signals accumulate.
  $v = Get-RebootVerdict -WtHandles 30000 -WtThreads 1500 -TermHandles 31000 -TermWsMb 5000 -UptimeHours 200 `
        -WtHandleWarn 28000 -WtThreadWarn 1200 -TermHandleWarn 30000 -TermWsMbWarn 4096 -UptimeHoursWarn 168
  Check "all signals -> 5 reasons"        ($v.reasons.Count -eq 5)
  if ($fails -gt 0) { Write-Output "SELFTEST FAILED ($fails)"; exit 1 }
  Write-Output "SELFTEST OK"; exit 0
}

# --- Live measurement (read-only). ---
$stateRoot = if ($env:FLEET_STATE_DIR) { $env:FLEET_STATE_DIR }
  elseif ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'Fleet' }
  else { Join-Path ([System.IO.Path]::GetTempPath()) 'Fleet' }
if (-not $LogDir) { $LogDir = Join-Path $stateRoot 'watchdog' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$log    = Join-Path $LogDir 'reboot_advisor.log'
$jsonl  = Join-Path $LogDir 'reboot_advisor.jsonl'
$notify = Join-Path $PSScriptRoot 'notify.ps1'
$nowIso = [DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')

function Note($m) {
  $line = "{0}  {1}" -f $nowIso, $m
  Add-Content -Path $log -Value $line; Write-Output $line
}

# WindowsTerminal: the chokepoint is a SINGLE process, so grade the busiest one.
$wtHandles = 0; $wtThreads = 0; $wtCount = 0; $wtWsMb = 0
$wt = @(Get-Process WindowsTerminal -ErrorAction SilentlyContinue)
$wtCount = $wt.Count
if ($wtCount -gt 0) {
  $busiest = $wt | Sort-Object HandleCount -Descending | Select-Object -First 1
  $wtHandles = [int]$busiest.HandleCount
  $wtThreads = [int]$busiest.Threads.Count
  $wtWsMb    = [int]([math]::Round($busiest.WorkingSet64/1MB,0))
}

# TermService svchost (may be absent / not started).
$termHandles = 0; $termWsMb = 0
$ts = Get-CimInstance Win32_Service -Filter "Name='TermService'" -ErrorAction SilentlyContinue
if ($ts -and $ts.ProcessId) {
  $tp = Get-Process -Id ([int]$ts.ProcessId) -ErrorAction SilentlyContinue
  if ($tp) { $termHandles = [int]$tp.HandleCount; $termWsMb = [int]([math]::Round($tp.WorkingSet64/1MB,0)) }
}

# Uptime.
$os = Get-CimInstance Win32_OperatingSystem
$uptimeHours = [math]::Round(((Get-Date) - $os.LastBootUpTime).TotalHours, 1)

$res = Get-RebootVerdict -WtHandles $wtHandles -WtThreads $wtThreads -TermHandles $termHandles `
  -TermWsMb $termWsMb -UptimeHours $uptimeHours `
  -WtHandleWarn $WtHandleWarn -WtThreadWarn $WtThreadWarn -TermHandleWarn $TermHandleWarn `
  -TermWsMbWarn $TermWsMbWarn -UptimeHoursWarn $UptimeHoursWarn

$reasonStr = if ($res.reasons.Count) { $res.reasons -join ' ' } else { '(none)' }
Note ("TICK verdict=$($res.verdict) uptime=${uptimeHours}h wt[proc=$wtCount handles=$wtHandles threads=$wtThreads ws=${wtWsMb}MB] term[handles=$termHandles ws=${termWsMb}MB] :: $reasonStr")

if ($res.verdict -eq 'RECOMMEND') {
  $rec = [ordered]@{
    ts = $nowIso; verdict = $res.verdict; reasons = $res.reasons; uptime_hours = $uptimeHours
    wt_proc = $wtCount; wt_handles = $wtHandles; wt_threads = $wtThreads; wt_ws_mb = $wtWsMb
    term_handles = $termHandles; term_ws_mb = $termWsMb
    thresholds = [ordered]@{
      wt_handles = $WtHandleWarn; wt_threads = $WtThreadWarn
      term_handles = $TermHandleWarn; term_ws_mb = $TermWsMbWarn; uptime_hours = $UptimeHoursWarn
    }
  } | ConvertTo-Json -Compress -Depth 6
  Add-Content -Path $jsonl -Value $rec

  if (-not $Quiet -and (Test-Path $notify)) {
    $msg = "Schedule a reboot in a quiet window: $reasonStr. Clears the WindowsTerminal/TermService leaks (see the desktop-slowness runbook)."
    try {
      & powershell -NoProfile -ExecutionPolicy Bypass -File $notify `
        -Title 'Reboot recommended (fleet workstation)' -Message $msg -Level 'warn' `
        -LogDir $LogDir -Key 'reboot-advice' -MinIntervalMinutes $ToastIntervalMinutes
    } catch {}
  }
}
exit 0
