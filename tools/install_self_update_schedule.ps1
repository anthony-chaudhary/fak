<#
.SYNOPSIS
  Register a durable Windows Scheduled Task that keeps the installed `fak` guard binary
  converged on the latest VERIFIED origin/main, so an always-on guard fleet never runs a
  stale binary.

.WHY
  The guard fleet launches a FIXED binary (default C:\Users\USER\bin\fak.exe) and only
  re-execs it on restart. Nothing rebuilds that file on its own, so without this task a
  restart re-execs the SAME stale binary. `fak self-update` builds from a pristine
  origin/main checkout, GATES it (build -> vet -> `version` smoke), and only then
  atomically swaps the target. A non-green tree is NEVER installed. With this task running
  every N minutes, the installed binary is always current+verified, so the next watchdog
  restart of any guard picks up the new fak. Convergence with zero in-session-restart risk.

  -Target names ONE path, but a real host runs several fak binaries. In particular
  tools/dispatch_worker.py `resolve_fak_bin` prefers the in-tree `<workspace>/tools/.bin/fak.exe`
  AHEAD of PATH when it builds every dispatched worker's `fak guard -- <backend>` argv, so while
  that file exists PATH is never consulted and converging only -Target leaves the fleet's own
  workers on a binary no updater touches. `fak self-update` therefore also converges its invoker
  and that in-tree fleet path (same build->vet->smoke gate), and exits NON-ZERO with
  `outcome=sibling-stale` if -Target lands but a sibling does not.

.SAFETY
  - self-update installs IFF the gate passes; a broken/uncompilable origin/main is skipped,
    the old binary stays in place.
  - the atomic swap (selfinstall.OSSwap) renames a mapped .exe aside, so it is safe to run
    while guards hold the old binary open.
  - building from a detached origin/main worktree (not the live shared tree) means peer
    work-in-progress is never baked into the installed binary.

.EXAMPLE
  .\install_self_update_schedule.ps1                       # register (every 15 min), default target
  .\install_self_update_schedule.ps1 -Target C:\Users\USER\bin\fak.exe -IntervalMin 10
  .\install_self_update_schedule.ps1 -RunNow               # register AND run once immediately
  .\install_self_update_schedule.ps1 -Uninstall
#>
[CmdletBinding()]
param(
  [string]$RepoRoot = 'C:\work\fak',                       # the fak checkout to build origin/main from
  [string]$Target   = 'C:\Users\USER\bin\fak.exe',         # the installed FLEET binary to converge
  [int]$IntervalMin = 15,
  [string]$TaskName = 'FakSelfUpdate',
  # Where each tick's output is appended. Without this the task ran under
  # `conhost --headless cmd.exe /c` with NO redirection, so every line self-update printed --
  # including the one that says whether the binary actually moved -- was discarded, leaving the
  # scheduler's single overwritten "Last Result" integer as the only observable. rc=0 is
  # identical for "installed", "already current", "busy" and "--check", so a fleet whose binary
  # had not advanced in nine hours was indistinguishable from a perfectly converged one.
  [string]$LogPath  = (Join-Path $env:LOCALAPPDATA 'Fleet\watchdog\self_update.log'),
  [switch]$RunNow,
  [switch]$Uninstall
)
$ErrorActionPreference = 'Continue'

if ($Uninstall) {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Host "Removed scheduled task '$TaskName'."
  return
}

# Invoke the repo-local built binary if present (it is rebuilt by `go build` / by this very
# task), else the installed target, else PATH. Self-update resolves origin/main itself; the
# explicit --target makes the FLEET binary the swap destination regardless of which fak runs.
$fakBin = @(
  (Join-Path $RepoRoot 'tools\.bin\fak.exe'),
  $Target,
  'fak'
) | Where-Object { $_ -eq 'fak' -or (Test-Path $_) } | Select-Object -First 1
if (-not $fakBin) { throw "no fak binary found (looked in $RepoRoot\tools\.bin, $Target, PATH)" }

# Register-ScheduledTask with an S4U principal + StartWhenAvailable (#3322): the
# migration off the old `schtasks /Create ... /RL LIMITED` Interactive default, which
# did not run at cold boot before logon and skipped any self-update tick missed while
# the box was off. S4U runs windowless in session 0 (no console flash, so the old
# conhost/hide-task-windows step is unnecessary) yet still AS THIS USER;
# StartWhenAvailable fires a missed tick on wake. Setting an S4U principal needs an
# ELEVATED shell; the catch below says so.
$logDir = Split-Path -Parent $LogPath
if ($logDir -and -not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }

# `/d /s /c "<whole command line>"`: /s makes cmd strip exactly the outer quote pair and take
# the rest literally, which is what lets the quoted paths AND the `>>` redirection coexist.
# The redirection is the point -- self-update prints one `outcome=<cause>` line per tick, and
# without a capture it goes to a discarded headless console.
$act = New-ScheduledTaskAction -Execute 'cmd.exe' `
  -Argument ('/d /s /c ""{0}" self-update --root "{1}" --target "{2}" >> "{3}" 2>&1"' -f $fakBin, $RepoRoot, $Target, $LogPath)
$trg = New-ScheduledTaskTrigger -Once -At (Get-Date) `
  -RepetitionInterval (New-TimeSpan -Minutes $IntervalMin) -RepetitionDuration (New-TimeSpan -Days 3650)
$prin = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -StartWhenAvailable -MultipleInstances IgnoreNew
try {
  Register-ScheduledTask -TaskName $TaskName -Action $act -Trigger $trg `
    -Principal $prin -Settings $set -Force -ErrorAction Stop | Out-Null
} catch {
  throw ("Register-ScheduledTask failed: $($_.Exception.Message). S4U registration needs an elevated shell -- re-run this installer as Administrator (right-click PowerShell -> Run as administrator).")
}
Write-Host "Registered '$TaskName' - every $IntervalMin min, current-user S4U windowless"
Write-Host "Tick output -> $LogPath  (grep for 'outcome=' to see whether each tick actually moved a binary)"

if ($RunNow) {
  Write-Host "Running '$TaskName' once now…"
  schtasks /Run /TN $TaskName | Out-Null
}
