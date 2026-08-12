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
  [switch]$Uninstall,
  # #6508: registration PINS the binary provenance it reviewed. A task registered against an
  # `+uncommitted` (or unstampable) binary freezes unreviewed code into a schedule that then
  # re-executes it every tick unattended -- which is how a stale Go-bin copy ended up
  # certifying logvault evidence. Refuse by default; this switch is the deliberate opt-out for
  # a maintainer knowingly scheduling a hand-build.
  [switch]$AllowUnreviewedBin
)
$ErrorActionPreference = 'Continue'

if ($Uninstall) {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Host "Removed scheduled task '$TaskName'."
  return
}

# Bootstrap from the installed target first. `self-update` can replace its own executable
# safely (including Windows' running-image lock), so this path is both durable and the exact
# binary whose freshness we are proving. A repo-local tools/.bin build is only a recovery
# fallback: it is intentionally ephemeral and may disappear after task registration. Pinning
# the scheduled action to that transient path made every tick fail before self-update ran while
# Task Scheduler retained an old success result. PATH remains the last-resort bootstrap.
$fakBin = @(
  $Target,
  (Join-Path $RepoRoot 'tools\.bin\fak.exe'),
  'fak'
) | Where-Object { $_ -eq 'fak' -or (Test-Path $_) } | Select-Object -First 1
if (-not $fakBin) { throw "no fak binary found (looked in $RepoRoot\tools\.bin, $Target, PATH)" }

# Pin the provenance we are about to schedule (#6508). A scheduled task is an UNATTENDED
# standing grant: whatever binary it names keeps running every tick with nobody looking, so the
# thing to review is exactly the identity of that file -- once, here, at registration. Resolve
# it to an ABSOLUTE path (a bare `fak` re-resolves through whatever PATH session 0 inherits,
# which is not a reviewable provenance) and read the build it self-reports.
$pinnedBin = $fakBin
if ($pinnedBin -eq 'fak') {
  $resolved = (Get-Command fak -ErrorAction SilentlyContinue).Source
  if ($resolved) { $pinnedBin = $resolved }
}
$pinnedBin = try { (Resolve-Path -LiteralPath $pinnedBin -ErrorAction Stop).Path } catch { $pinnedBin }
$pinnedBuild = ''
try {
  $verText = (& $pinnedBin version 2>&1 | Out-String)
  $m = [regex]::Match($verText, 'build:\s*(?<rev>[0-9a-f]{7,40})(?<dirty>\s*\+uncommitted)?')
  if ($m.Success) { $pinnedBuild = $m.Groups['rev'].Value + $(if ($m.Groups['dirty'].Success) { ' +uncommitted' } else { '' }) }
} catch { $pinnedBuild = '' }
if (-not $AllowUnreviewedBin) {
  if (-not $pinnedBuild) {
    throw ("refusing to register '$TaskName': $pinnedBin reports no VCS stamp, so the schedule cannot say which commit it will run every tick. Rebuild with `make build` (or pass -AllowUnreviewedBin to schedule it anyway).")
  }
  if ($pinnedBuild -match '\+uncommitted') {
    throw ("refusing to register '$TaskName': $pinnedBin is a working-tree build ($pinnedBuild) that no commit reviews; a scheduled task would re-execute it unattended forever. Point -Target at an installed clean binary, or pass -AllowUnreviewedBin.")
  }
}
Write-Host "Pinned binary: $pinnedBin  build=$(if ($pinnedBuild) { $pinnedBuild } else { '(unstamped)' })"

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
#
# `--pinned-bin` carries the reviewed provenance INTO every tick (#6508): the task refuses to
# run when the binary actually executing is no longer the pinned one, has lost its stamp, or has
# become a working-tree build -- skew detected BEFORE execution rather than discovered afterward
# in whatever the stale copy certified.
$act = New-ScheduledTaskAction -Execute 'cmd.exe' `
  -Argument ('/d /s /c ""{0}" self-update --root "{1}" --target "{2}" --pinned-bin "{4}" >> "{3}" 2>&1"' -f $pinnedBin, $RepoRoot, $Target, $LogPath, $pinnedBin)
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
