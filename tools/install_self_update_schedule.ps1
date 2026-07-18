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
$act = New-ScheduledTaskAction -Execute 'cmd.exe' `
  -Argument ('/c "{0}" self-update --root "{1}" --target "{2}"' -f $fakBin, $RepoRoot, $Target)
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

if ($RunNow) {
  Write-Host "Running '$TaskName' once now…"
  schtasks /Run /TN $TaskName | Out-Null
}
