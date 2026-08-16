<#
fleet_supervisor_watchdog.ps1 — keep the job-fleet supervisor alive, forever,
independent of any Claude Code session.

Root cause this fixes: the standing supervisor (run_supervise_loop.py) is the
thing that respawns stopped dispatch-loop workers. When IT dies mid-loop
(crash, host sleep, or the Claude session that launched it ending), nothing
respawns IT — so the whole fleet silently stalls and a human has to notice and
guess. This watchdog is the missing PID-1: one cheap idempotent tick that
re-launches the supervisor as a DETACHED process if (and only if) it is not
already running.

Opt-in by design (it launches autonomous workers): a respawn requires the job repo
present, the DISPATCH_DISABLED kill-switch absent, AND FAK_SUPERVISOR_ENABLE=1,
else it only REPORTS -- matching the .py port
(fleet_supervisor_watchdog.py). register_supervisor_watchdog.ps1 kicks one run on
install, so without this gate merely installing the task would spawn the supervisor.

Designed to be run on a 5-minute schedule (see register-* below). Safe to run
by hand any time. Never starts a second supervisor if one is alive.

The desired population is NOT this loop's to choose (#6502): it comes from the one
declared SSOT (fleet_target.ps1), read at tick time, so an operator's target-0
quarantine survives every scheduled tick and every reboot. This task used to carry
`-Target 4` in its Scheduled Task arguments while a sibling watchdog carried
`-Target 8` and the operator held target 0 -- three answers to one question.

Exit codes: 0 = alive / disabled / quarantined / job repo absent (no-op) | 3 = the
tick's requested population was refused by the SSOT | 10 = respawned it.
#>
[CmdletBinding()]
param(
  [string]$JobDir = 'C:\work\job',
  # -1 means "this tick carried no population", which is the intended steady state.
  # FAK_SUPERVISOR_TARGET (laptop_dispatch_config.ps1) is still honored as a
  # REQUEST, but it is advisory: it goes through the same SSOT admission as a
  # hand-passed -Target and can never raise the population on its own.
  [int]$Target    = $(if ($env:FAK_SUPERVISOR_TARGET) { [int]$env:FAK_SUPERVISOR_TARGET } else { -1 }),
  [string]$LogDir = ''
)

$ErrorActionPreference = 'Stop'
$stateRoot = if ($env:FLEET_STATE_DIR) {
  $env:FLEET_STATE_DIR
} elseif ($env:LOCALAPPDATA) {
  Join-Path $env:LOCALAPPDATA 'Fleet'
} else {
  Join-Path ([System.IO.Path]::GetTempPath()) 'Fleet'
}
if (-not $LogDir) { $LogDir = Join-Path $stateRoot 'watchdog' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$log = Join-Path $LogDir 'watchdog.log'
function Note($m) {
  $line = "{0}  {1}" -f (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'), $m
  Add-Content -Path $log -Value $line
  Write-Output $line
}

# #6502: the declared SSOT owns the population, not this task's arguments and not
# the env knob. Assert-FleetTargetAdmits returns the SSOT's number whatever was
# requested, so even a stale scheduled -Target cannot repopulate a quarantine.
. (Join-Path $PSScriptRoot 'fleet_target.ps1')
try {
  $Target = Assert-FleetTargetAdmits -Requested $Target
} catch {
  Note ("REFUSED {0}" -f $_.Exception.Message)
  exit 3
}
if ($Target -le 0) {
  Note ("QUARANTINE target=0 per {0} -- not respawning the supervisor" -f (Get-FleetTargetPath))
  exit 0
}

# A supervisor is alive iff a run_supervise_loop.py process exists.
$alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'run_supervise_loop\.py' }

if ($alive) {
  Note ("ALIVE   pid(s)={0} target={1} -- no action" -f (($alive.ProcessId) -join ','), $Target)
  exit 0
}

# Not alive. Mirror the .py port's two safety gates (fleet_supervisor_watchdog.py:87-92)
# before launching an autonomous supervisor: (1) the job repo / supervisor script must
# exist, and (2) FAK_SUPERVISOR_ENABLE must be opt-in -- else only REPORT and exit 0.
# (1) also avoids a crash/respawn-loop when $JobDir is missing or the script was moved.
$runLoop = Join-Path $JobDir 'scripts\run_supervise_loop.py'
if (-not (Test-Path $runLoop)) {
  Note "NOOP    job repo / supervisor absent ($runLoop) -- nothing to keep alive"
  exit 0
}
# Host-local dispatch kill-switch (mirrors fleet_supervisor_watchdog.py). run_supervise_loop.py
# no-ops on this sentinel, so respawning against it launches a process that exits inside the
# tick -- and the next tick sees "not alive" and does it again. Witnessed 2026-07-26 on this
# host: 25 consecutive RESPAWN ticks against a sentinel set 2026-06-21, every child dead
# immediately, invisible in the fleet report because the respawn toast is rate-limited to one
# an hour. Report it instead: the sentinel is an operator decision, not a fault to route around.
$killSwitch = Join-Path $JobDir 'DISPATCH_DISABLED'
if (Test-Path $killSwitch) {
  Note "NOOP    dispatch kill-switch present ($killSwitch) -- not respawning; delete it to re-enable dispatch"
  exit 0
}
$supEnabled = "$env:FAK_SUPERVISOR_ENABLE".Trim().ToLower() -in @('1', 'true', 'yes', 'on')
if (-not $supEnabled) {
  Note "NOOP    supervisor DOWN but FAK_SUPERVISOR_ENABLE not set -- reporting only"
  exit 0
}

# Enabled and the supervisor script exists — confirm the fleet's verdict, then respawn.
$py = Join-Path $JobDir '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }

$verdict = '?'
try {
  $j = & $py (Join-Path $JobDir 'scripts\supervise_now.py') --json 2>$null | ConvertFrom-Json
  $verdict = $j.verdict
} catch { }

$ts  = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$out = Join-Path $LogDir "supervisor-$ts.log"

# IMPORTANT: launch with JOB_SUPERVISED_WORKER cleared (that flag is for workers,
# not the supervisor) and fully detached so it outlives this watchdog tick.
$env:JOB_SUPERVISED_WORKER = $null
$p = Start-Process -FilePath $py `
  -ArgumentList @('scripts\run_supervise_loop.py', '--target', "$Target") `
  -WorkingDirectory $JobDir -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $out -RedirectStandardError "$out.err"

Note ("RESPAWN verdict_was=$verdict launched pid=$($p.Id) target=$Target log=$out")
try {
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'notify.ps1') `
    -Title 'Fleet supervisor respawned' -Message "was $verdict; relaunched pid=$($p.Id) target=$Target" `
    -Level warn -LogDir $LogDir -Key 'supervisor-respawned' -MinIntervalMinutes 60
} catch {}
exit 10
