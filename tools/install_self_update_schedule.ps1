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

$cmd = '"{0}" self-update --root "{1}" --target "{2}"' -f $fakBin, $RepoRoot, $Target
$tr  = 'cmd.exe /c {0}' -f $cmd

schtasks /Create /TN $TaskName /SC MINUTE /MO $IntervalMin /TR $tr /RL LIMITED /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks /Create failed ($LASTEXITCODE)" }
Write-Host "Registered '$TaskName' - every $IntervalMin min, runs: $tr"

# Re-point through conhost --headless so the task never flashes a console window, reusing the
# host-maintenance hider when it is present on this box (best-effort).
$hider = 'C:\work\host-maintenance\hide-task-windows.ps1'
if (Test-Path $hider) {
  try { & $hider -Task $TaskName | Out-Null; Write-Host "  (windowless via hide-task-windows.ps1)" }
  catch { Write-Warning "  could not hide window: $_" }
}

if ($RunNow) {
  Write-Host "Running '$TaskName' once now…"
  schtasks /Run /TN $TaskName | Out-Null
}
