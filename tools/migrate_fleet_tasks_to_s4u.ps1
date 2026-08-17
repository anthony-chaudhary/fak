<#
migrate_fleet_tasks_to_s4u.ps1 - convert Interactive-logon fleet Scheduled Tasks to S4U.

WHY (2026-07-09 incident): fleet automation tasks registered with LogonType=Interactive
("run only when user is logged on") are refused with 0x800710E0 ("operator/administrator
refused the request") on this headless / RDP-accessed box -- they stay refused even while
an RDP session is Active, and die outright when the session disconnects. Every S4U task
("run whether logged on or not", session 0, windowless, still AS THIS USER) rode through
the same window returning 0x0. On 2026-07-09 ~12:06 this silently took down 9 tasks at once,
including the fleet's safety net (resume watchdog, supervisor, resource guard, seat resume).

This migrates each Interactive fleet task's PRINCIPAL to S4U in place (UserId + RunLevel
preserved; action/trigger/settings untouched). Idempotent (already-S4U tasks are skipped)
and DRY-RUN by default -- review the plan, then re-run elevated with -Apply.

REQUIRES AN ELEVATED SHELL for -Apply: setting an S4U principal is refused non-elevated
("Access is denied"). Right-click PowerShell -> Run as administrator.

  .\migrate_fleet_tasks_to_s4u.ps1                 # DRY-RUN: list Interactive fleet tasks + plan
  .\migrate_fleet_tasks_to_s4u.ps1 -Apply          # migrate ALL Interactive fleet tasks (the safe default)
  .\migrate_fleet_tasks_to_s4u.ps1 -Apply -FailingOnly # legacy incident-only scope
  .\migrate_fleet_tasks_to_s4u.ps1 -Apply -VerifyRun   # after migrating, force-run each and report LastResult
#>
[CmdletBinding()]
param(
  [switch]$Apply,
  [switch]$FailingOnly, # legacy incident mode: migrate only currently failing/required tasks
  [switch]$VerifyRun,  # force-run each migrated task and report its LastTaskResult
  [string]$NamePattern = 'Fleet|Fak|Resume|Dispatch|Supervisor|Watchdog|Guard|Seat|Owner|Slack|Logvault|Push|Scout|Stale|Janitor|Reaper|Pane'
)
# Do NOT set ErrorActionPreference=Stop globally: a single unreadable task would abort the
# whole survey. Handle errors locally instead.
Set-StrictMode -Off

# Immediate work-discovery remediation (#4162): always include these tasks when
# they are still Interactive, even if Task Scheduler has not retained 0x800710E0.
$RequiredS4UTasks = @(
  'FleetScoutLoop',
  'FleetStaleWorkGarden'
)
function Test-Admin {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
}
function Fmt-Hr($v) { if ($null -eq $v) { '-' } else { '0x{0:X}' -f $v } }

$cands = Get-ScheduledTask |
  Where-Object { $_.TaskName -match $NamePattern -and $_.Principal.LogonType -eq 'Interactive' } |
  ForEach-Object {
    $t = $_
    $i = Get-ScheduledTaskInfo -TaskName $t.TaskName -TaskPath $t.TaskPath -ErrorAction SilentlyContinue
    $hr = Fmt-Hr $i.LastTaskResult
    [pscustomobject]@{
      Task       = $t.TaskName
      UserId     = $t.Principal.UserId
      RunLevel   = "$($t.Principal.RunLevel)"
      LastResult = $hr
      Failing    = ($hr -eq '0x800710E0')
    }
  }
$cands = @($cands)
if ($cands.Count -eq 0) { Write-Output "no Interactive-logon fleet tasks found - nothing to migrate."; return }

if ($FailingOnly) {
  Write-Warning '-FailingOnly preserves desktop-visible Interactive tasks and is intended only for legacy incident triage.'
  $targets = @($cands | Where-Object { $_.Failing -or $RequiredS4UTasks -contains $_.Task })
} else {
  $targets = $cands
}

Write-Output "=== Interactive-logon fleet tasks ==="
foreach ($c in ($cands | Sort-Object Task)) {
  Write-Output ("  {0,-26} {1,-8} last={2,-12} failing={3}" -f $c.Task, $c.RunLevel, $c.LastResult, $c.Failing)
}
Write-Output ("candidates: {0} Interactive | selected to migrate: {1} ({2})" -f $cands.Count, @($targets).Count, $(if ($FailingOnly) { 'currently failing 0x800710E0 / required compatibility set' } else { 'all Interactive' }))

if (-not $Apply) {
  Write-Output ""
  Write-Output "DRY-RUN - nothing changed. Re-run from an ELEVATED shell with -Apply to migrate the selected tasks to S4U."
  Write-Output "  (all Interactive tasks are selected by default; add -FailingOnly only for legacy incident triage.)"
  return
}

if (-not (Test-Admin)) {
  Write-Output "REFUSED: -Apply needs an ELEVATED shell (setting an S4U principal is 'Access is denied' non-elevated)."
  Write-Output "Right-click PowerShell -> Run as administrator, then re-run with -Apply."
  exit 1
}

Write-Output ""
Write-Output "=== migrating $(@($targets).Count) task(s) Interactive -> S4U ==="
foreach ($c in $targets) {
  $verify = ''
  try {
    $p = New-ScheduledTaskPrincipal -UserId $c.UserId -LogonType S4U -RunLevel $c.RunLevel
    Set-ScheduledTask -TaskName $c.Task -Principal $p -ErrorAction Stop | Out-Null
    $now = "$((Get-ScheduledTask -TaskName $c.Task).Principal.LogonType)"
    $migrated = ($now -eq 'S4U')
    if ($VerifyRun -and $migrated) {
      Start-ScheduledTask -TaskName $c.Task
      $deadline = (Get-Date).AddSeconds(90)
      do { Start-Sleep -Seconds 6; $st = "$((Get-ScheduledTask -TaskName $c.Task).State)" } while ($st -eq 'Running' -and (Get-Date) -lt $deadline)
      $verify = ' verify=' + (Fmt-Hr (Get-ScheduledTaskInfo -TaskName $c.Task).LastTaskResult)
    }
    Write-Output ("  {0,-26} -> {1}{2}" -f $c.Task, $now, $verify)
  } catch {
    Write-Output ("  {0,-26} -> FAILED: {1}" -f $c.Task, $_.Exception.Message)
  }
}
Write-Output ""
if ($VerifyRun) { Write-Output "verify 0x0 = task now launches; 0x1 = launches but app exited non-zero (app-level, not the logon defect)." }
Write-Output "NOTE: the register_*.ps1 installer for each task should also be moved to S4U so a future reinstall does not"
Write-Output "reintroduce Interactive (register_resume_watchdog.ps1 and register_issue_dispatch.ps1 already are)."
