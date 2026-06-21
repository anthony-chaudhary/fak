<#
fleet_dos_dispatch_watchdog.ps1 -- keep FLEET's OWN generic-DOS dispatch
supervisor alive, forever, independent of any Claude Code session.

Root cause this fixes: fleet's concurrency engine is `dos loop --enact --target N`
-- a long-lived supervisor that Popens one dispatch worker per free lane (via the
[supervise].worker_launch_template in dos.toml) and keeps the population at the
target. The engine works, but NOTHING keeps IT running for C:\work\fleet. The
look-alike task FleetSupervisorWatchdog watches the SIBLING repo C:\work\job's
run_supervise_loop.py, not fleet -- so fleet has run 0 supervisor-spawned workers
and concurrency has been manual-only. This watchdog is the missing PID-1 FOR
FLEET: one cheap idempotent tick that re-launches `dos loop --enact` as a
DETACHED process iff one is not already running.

Distinct from fleet_supervisor_watchdog.ps1 (which is the job-repo's PID-1).
The alive-check is keyed on the fleet `dos.exe loop --enact` supervisor and is
careful NOT to match: the job supervisor (python.exe run_supervise_loop.py), a
`dos.exe hook pretool` call, a worker's `dos-dispatch-loop`, or a transient
`dos loop --json` readiness probe.

Designed to be run on a 5-minute schedule (see register_dos_dispatch_watchdog.ps1).
Safe to run by hand any time. Never starts a second supervisor if one is alive.

Exit codes: 0 = supervisor already alive (no-op) | 10 = respawned it.
#>
[CmdletBinding()]
param(
  # Resolve the repo root from this script's own location (tools/ lives at the
  # repo root) so the watchdog works from any clone, not just C:\work\fleet.
  [string]$FleetDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [int]$Target      = 4,
  [int]$Interval    = 120,
  [string]$LogDir   = ''
)

$ErrorActionPreference = 'Stop'
$stateRoot = if ($env:FLEET_STATE_DIR) {
  $env:FLEET_STATE_DIR
} elseif ($env:LOCALAPPDATA) {
  Join-Path $env:LOCALAPPDATA 'Fleet'
} else {
  Join-Path ([System.IO.Path]::GetTempPath()) 'Fleet'
}
if (-not $LogDir) { $LogDir = Join-Path $stateRoot 'dos-dispatch-watchdog' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$log = Join-Path $LogDir 'dos-dispatch-watchdog.log'
function Note($m) {
  $line = "{0}  {1}" -f (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'), $m
  Add-Content -Path $log -Value $line
  Write-Output $line
}

# A fleet dispatch supervisor is alive iff a dos.exe process is running
# `loop ... --enact`. Two ANDed clauses (not one alternation) keep it explicit
# and robust to arg ordering: requires BOTH the `loop` subcommand and `--enact`.
# This does NOT match `dos.exe hook pretool` (no loop/--enact), the job repo's
# `python.exe run_supervise_loop.py` (wrong image), a worker's `claude.exe ...
# dos-dispatch-loop` (wrong image, no --enact), or a `dos loop --json` probe
# (no --enact).
$alive = Get-CimInstance Win32_Process -Filter "Name='dos.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'loop' -and $_.CommandLine -match '--enact' }

if ($alive) {
  Note ("ALIVE   pid(s)={0} target={1} interval={2} -- no action" -f (($alive.ProcessId) -join ','), $Target, $Interval)
  exit 0
}

# Not alive -- respawn detached so it outlives this watchdog tick.
$dos = (Get-Command dos -ErrorAction SilentlyContinue).Source
if (-not $dos) { $dos = 'dos' }

$ts  = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$out = Join-Path $LogDir "dos-supervisor-$ts.log"

$p = Start-Process -FilePath $dos `
  -ArgumentList @('loop', '--enact', '--workspace', $FleetDir, '--target', "$Target", '--interval', "$Interval") `
  -WorkingDirectory $FleetDir -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $out -RedirectStandardError "$out.err"

Note ("RESPAWN launched pid=$($p.Id) target=$Target interval=$Interval workspace=$FleetDir log=$out")
try {
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'notify.ps1') `
    -Title 'Fleet DOS dispatch supervisor respawned' -Message "relaunched pid=$($p.Id) target=$Target interval=$Interval" `
    -Level warn -LogDir $LogDir -Key 'dos-dispatch-respawned' -MinIntervalMinutes 60
} catch {}
exit 10
