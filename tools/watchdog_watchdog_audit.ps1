<#
watchdog_watchdog_audit.ps1 - the n2/n3 meta-watchdog: prove the resume watchdog
(and the whole supervision tower) is ALIVE, TICKING, and producing PRODUCTIVE
resumes -- versus silently STALLED. READ-ONLY. No side effects, no elevation.

WHY: a watchdog that dies takes the fleet down SILENTLY (every dead session just
stays dead; nothing complains). A stalled watchdog and a healthy-with-nothing-to-do
watchdog look identical from the outside. This pass distinguishes them from
ARTIFACTS -- ledger mtimes, Task Scheduler exit codes + LogonType, and the
watchdog's own launched-vs-proven witness -- never from any "I'm fine" self-report.

CORRECTED ROOT CAUSE (2026-07-09): the discriminator for the 0x800710E0
("operator/administrator refused the request") outage is the task's
Principal.LogonType, NOT the conhost launch shim. Interactive-logon tasks are
refused on this RDP/headless box even with an Active session; S4U tasks are immune.
An earlier theory blamed `conhost.exe --headless` -- it was DISPROVED: the resume
watchdog kept returning 0x800710E0 after conhost was removed. Fix = migrate the
task principal to S4U (needs elevation): tools\migrate_fleet_tasks_to_s4u.ps1.

Exit: 0 GREEN, 2 AMBER, 3 RED (so a /loop or CI gate can branch on it).

THE EXIT CODE IS THE GATE -- PROPAGATE IT (#6509). A wrapper that runs this audit,
appends the output to a log, and then `exit 0` makes Task Scheduler record
LastTaskResult=0 over 47 RED findings: telemetry with no gate, where a red finding
never alters health and the same warning is appended forever. The typed statuses,
the finding-recurrence fold (age / occurrences / resolution / actionable owner), and
the "a RED audit cannot yield scheduler result 0" proof live in
internal/watchdoghealth (auditgate.go); a wrapper's only job is to hand this script's
$LASTEXITCODE straight back to the scheduler.

  .\watchdog_watchdog_audit.ps1            # human-readable audit + verdict
  .\watchdog_watchdog_audit.ps1 -Json      # machine verdict (one JSON object)
#>
[CmdletBinding()]
param([switch]$Json, [int]$StallMinutes = 15)
Set-StrictMode -Off
$ErrorActionPreference = 'Continue'

$reasons = New-Object System.Collections.ArrayList
$rank = @{ green = 0; amber = 1; red = 2 }
$exitOf = @{ green = 0; amber = 2; red = 3 }
$verdict = 'green'
function Bump($v, $why) {
  if ($script:rank[$v] -gt $script:rank[$script:verdict]) { $script:verdict = $v }
  if ($why) { [void]$script:reasons.Add(("[{0}] {1}" -f $v.ToUpper(), $why)) }
}
function Fmt-Hr($v) { if ($null -eq $v) { '-' } else { '0x{0:X}' -f $v } }
$now = [DateTime]::UtcNow
$out = [ordered]@{}

# --- Layer 0: locate the LIVE registry (repo tools/_registry is usually stale) ---
$reg = @($env:FLEET_REG_DIR, "$env:FLEET_STATE_DIR\registry", "$env:LOCALAPPDATA\Fleet\registry", "$env:TEMP\Fleet\registry") |
  Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
$out.registry = "$reg"

# --- Layer 1 (n2): STALL check -- has the watchdog written ANYTHING recently? ---
$freshMin = $null
if ($reg) {
  foreach ($f in 'resume_ledger.jsonl','resume_watchdog_status.jsonl','resume_plan.json') {
    $p = Join-Path $reg $f
    if (Test-Path $p) {
      $age = ($now - (Get-Item $p).LastWriteTime.ToUniversalTime()).TotalMinutes
      if ($null -eq $freshMin -or $age -lt $freshMin) { $freshMin = $age }
    }
  }
}
$out.newest_ledger_age_min = if ($null -ne $freshMin) { [math]::Round($freshMin, 1) } else { $null }
if ($null -eq $freshMin) { Bump 'red' "no resume ledger found under $reg (watchdog may never have run here)" }
elseif ($freshMin -gt $StallMinutes) { Bump 'red' ("STALL: newest ledger write {0:n0} min ago (> {1} min) -- watchdog is not ticking" -f $freshMin, $StallMinutes) }
elseif ($freshMin -gt ($StallMinutes * 1.5)) { Bump 'amber' ("one missed tick: newest ledger write {0:n0} min ago" -f $freshMin) }

# --- Layer 1b (tower liveness): CORRECTED diagnosis = LogonType, not conhost ---
$tower = @()
Get-ScheduledTask -ErrorAction SilentlyContinue |
  Where-Object { $_.TaskName -match 'Resume|Supervisor|Watchdog|Guard|Seat|Stranded|Dispatch' } |
  ForEach-Object {
    $t = $_
    $i = Get-ScheduledTaskInfo -TaskName $t.TaskName -TaskPath $t.TaskPath -ErrorAction SilentlyContinue
    $hr = Fmt-Hr $i.LastTaskResult
    $lt = "$($t.Principal.LogonType)"
    $down = ($hr -ne '0x0' -and $hr -ne '-' -and $hr -ne '0x41303' -and $hr -ne '0x41301')  # exclude "ready"/"running" sentinels
    $latent = ($lt -eq 'Interactive')
    $tower += [pscustomobject]@{ Task = $t.TaskName; LogonType = $lt; LastResult = $hr; Down = $down; Latent = $latent }
  }
$down = @($tower | Where-Object { $_.Down })
$latent = @($tower | Where-Object { $_.Latent -and -not $_.Down })
$out.tower_total = $tower.Count
$out.tower_down = $down.Count
$out.tower_latent_interactive = $latent.Count
foreach ($d in $down) {
  if ($d.LastResult -eq '0x800710E0') { Bump 'red' ("{0} DOWN 0x800710E0 (LogonType={1}) -- migrate to S4U" -f $d.Task, $d.LogonType) }
  else { Bump 'red' ("{0} DOWN {1}" -f $d.Task, $d.LastResult) }
}
if ($latent.Count -gt 0) { Bump 'amber' ("{0} tower task(s) still LogonType=Interactive (latent 0x800710E0 under RDP): {1}" -f $latent.Count, (($latent | Select-Object -First 6 | ForEach-Object { $_.Task }) -join ', ')) }

# --- Layer 2 (productivity): launched != proven. Consume the watchdog's OWN witness ---
$prov = [ordered]@{ status_verdict = $null; auto_resume_depth = $null; launched_unproven = $null; unproven_pids_dead = $null }
try {
  $fak = (Get-Command fak -ErrorAction SilentlyContinue).Source
  if ($fak) {
    $j = & $fak resume watchdog --status --json 2>$null | Out-String
    $o = $j | ConvertFrom-Json
    $prov.status_verdict = "$($o.verdict)"
    $prov.auto_resume_depth = [int]$o.auto_resume_depth
    $un = @($o.mttr_sessions | Where-Object { $_.status -eq 'launched_unproven' })
    $prov.launched_unproven = $un.Count
    # terminal-unproven = launched_unproven whose resume pid is already dead (will never self-prove).
    # Report null (not 0) when --status does not expose pids, so "not measured" never reads as "all alive".
    $deadUnproven = 0; $pidsSeen = 0
    foreach ($u in $un) { if ($u.pid) { $pidsSeen++; if (-not (Get-Process -Id ([int]$u.pid) -ErrorAction SilentlyContinue)) { $deadUnproven++ } } }
    $prov.unproven_pids_dead = if ($pidsSeen -gt 0) { $deadUnproven } else { $null }
    if ($un.Count -gt 0) {
      $tail = if ($pidsSeen -gt 0) { "; {0} have a dead pid = terminally unproven" -f $deadUnproven } else { " (pid liveness not exposed by --status; cross-check transcripts per Layer 2b)" }
      Bump 'amber' ("{0} resume(s) launched_unproven (no real model turn after launch)$tail" -f $un.Count)
    }
    if ($prov.auto_resume_depth -ge 20) { Bump 'amber' ("backlog {0} deep and draining only ~4/tick -- needs a live ticking watchdog" -f $prov.auto_resume_depth) }
  } else { Bump 'amber' "fak not on PATH -- cannot read the launched-vs-proven witness" }
} catch { Bump 'amber' ("could not read fak resume watchdog --status: {0}" -f $_.Exception.Message) }
$out.productivity = $prov

# --- Layer 3 (n3): who watches THIS auditor? single-point-of-failure check ---
$selfTask = Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object { $_.TaskName -match 'WatchdogAudit|MetaWatchdog|WatchdogWatchdog' }
$out.n3_auditor_scheduled = [bool]$selfTask
$out.n3_auditor_logontype = if ($selfTask) { "$(@($selfTask)[0].Principal.LogonType)" } else { $null }
if (-not $selfTask) { Bump 'amber' "n3 GAP: this audit is not itself scheduled/looped -- it only runs when a human remembers. A dead auditor over a dead watchdog is a silent double-fault." }
# RED, not amber (#6509): an auditor sharing the failure mode it detects is not a latent
# risk, it is a DISABLED GATE -- once refused it goes silent, and silence then reads as
# health. Independence from the checked failure mode is a precondition for trusting any
# verdict this script emits, so it fails the audit outright.
elseif ($out.n3_auditor_logontype -match '^Interactive') { Bump 'red' "n3: the auditor's own task is LogonType=Interactive -- it will be refused (0x800710E0) the SAME way the tasks it audits were, so its silence cannot be read as health. Make it S4U (or keep it as an orthogonal /loop, not a scheduled task)." }

$out.verdict = $verdict.ToUpper()
$out.reasons = @($reasons)
$out.action = switch ($verdict) {
  'red'   { 'Migrate down tower tasks to S4U (elevated): tools\migrate_fleet_tasks_to_s4u.ps1 -Apply -VerifyRun, then re-run this audit.' }
  'amber' { 'Address the latent/unproven items above; no hard-down task, but the tower is not fully healthy.' }
  default { 'Tower healthy: watchdog ticking, no down/latent tasks, resumes proven.' }
}

if ($Json) { $out | ConvertTo-Json -Depth 6; exit $exitOf[$verdict] }

Write-Output "=============================================================="
Write-Output " watchdog-watchdog audit (n2/n3)   $($now.ToString('yyyy-MM-ddTHH:mm:ssZ'))"
Write-Output "=============================================================="
Write-Output ("live registry     : {0}" -f $out.registry)
Write-Output ("newest ledger     : {0} min ago  (STALL threshold {1} min)" -f $out.newest_ledger_age_min, $StallMinutes)
Write-Output ("supervision tower : {0} tasks | {1} DOWN | {2} latent-Interactive" -f $out.tower_total, $out.tower_down, $out.tower_latent_interactive)
foreach ($r in ($tower | Sort-Object Down,Latent -Descending)) {
  $flag = if ($r.Down) { 'DOWN ' } elseif ($r.Latent) { 'latent' } else { 'ok   ' }
  Write-Output ("    {0}  {1,-26} {2,-11} {3}" -f $flag, $r.Task, $r.LogonType, $r.LastResult)
}
$deadPidStr = if ($null -ne $prov.unproven_pids_dead) { "$($prov.unproven_pids_dead)" } else { 'n/a' }
Write-Output ("productivity      : status={0} backlog={1} launched_unproven={2} (dead-pid={3})" -f $prov.status_verdict, $prov.auto_resume_depth, $prov.launched_unproven, $deadPidStr)
Write-Output ("n3 auditor        : scheduled={0} logontype={1}" -f $out.n3_auditor_scheduled, $out.n3_auditor_logontype)
Write-Output ""
Write-Output ("VERDICT: {0}" -f $out.verdict)
foreach ($r in $out.reasons) { Write-Output ("  - {0}" -f $r) }
Write-Output ("ACTION : {0}" -f $out.action)
exit $exitOf[$verdict]
