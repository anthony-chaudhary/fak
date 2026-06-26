<#
register_worktree_doctor.ps1 - install/remove/status the Scheduled Task that runs
tools/worktree_doctor.py on a cadence, so this box stays at "one worktree on the trunk"
(auto-detected, e.g. main) without anyone babysitting it. Two safety stances run together:
the converge step only ever removes the provably loss-free (non-primary, clean, no
untracked, no mid-op, fully merged); the disposable SWEEP (--sweep-disposable) reaps dead
scratch worktrees under temp/scratchpad/pr-work but ARCHIVES each dirty one's diff +
untracked files first and spares any worktree touched recently (a live session). So
unattended is safe by construction - nothing is ever lost.

  .\register_worktree_doctor.ps1                  # install: prune safe worktrees + sweep dead scratch
  .\register_worktree_doctor.ps1 -PruneBranches   # ALSO delete merged local branches (git branch -d)
  .\register_worktree_doctor.ps1 -ReportOnly      # install: report only, never remove anything
  .\register_worktree_doctor.ps1 -EveryHours 4    # repeat every N hours (0 = once daily at -At)
  .\register_worktree_doctor.ps1 -Action status
  .\register_worktree_doctor.ps1 -Action remove
  .\register_worktree_doctor.ps1 -At 02:00 -AllowBranch fak-v0.1,my-release

ASCII-only on purpose: this box has only Windows PowerShell 5.1, which misreads a
BOM-less non-ASCII .ps1 as Windows-1252 and breaks the parse.
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [switch]$ReportOnly,                       # install in report-only mode (no removals at all)
  [switch]$PruneBranches,                    # ALSO delete merged local branches (opt-in; default off)
  [string]$At = '03:30',                     # daily run time (HH:mm, 24h)
  [int]$EveryHours = 4,                      # also repeat every N hours within the day (0 = daily only)
  [string[]]$AllowBranch = @('fak-v0.1'),    # long-lived worktree branches to RETAIN, never prune
  [string]$TaskName = 'FleetWorktreeDoctor'
)
$ErrorActionPreference = 'Stop'

# Resolve the repo from THIS script's location (tools/ -> repo root). No hardcoded path.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo      = (Resolve-Path (Join-Path $ScriptDir '..')).Path
$Doctor    = Join-Path $ScriptDir 'worktree_doctor.py'
$LogDir    = Join-Path $env:LOCALAPPDATA 'Fleet\watchdog'
$Log       = Join-Path $LogDir 'worktree_doctor.log'

function Get-Python {
  foreach ($c in 'python','python3','py') {
    $g = Get-Command $c -ErrorAction SilentlyContinue
    if ($g) { return $g.Source }
  }
  throw "no python on PATH (need it to run worktree_doctor.py)"
}

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  $a = ($t.Actions | Select-Object -First 1).Arguments
  $parts = @()
  if ($a -match '--prune\b') { $parts += 'prune-worktrees' }
  if ($a -match '--sweep-disposable') { $parts += 'sweep-scratch' }
  if ($a -match '--prune-branches') { $parts += 'prune-branches' }
  $mode = if ($parts.Count) { ($parts -join '+') } else { 'REPORT-ONLY' }
  Write-Output "State=$($t.State) mode=$mode LastRun=$($i.LastRunTime) LastResult=$($i.LastTaskResult) NextRun=$($i.NextRunTime)"
  Write-Output "log: $Log"
  return
}
if ($Action -eq 'remove') {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  Write-Output "removed $TaskName"
  return
}

# ---- install ----------------------------------------------------------------
$Python = Get-Python
New-Item -ItemType Directory -Force $LogDir | Out-Null

# Build the doctor args. --fetch makes the merged-check accurate against the auto-detected
# trunk ref. Default removals: safe worktree prune + the archived disposable-scratch sweep.
# Branch deletion (git branch -d) is opt-in via -PruneBranches so a fresh install never
# purges merged local branches by surprise.
$dargs = @("`"$Doctor`"", '--repo', "`"$Repo`"", '--fetch')
if (-not $ReportOnly) { $dargs += @('--prune', '--sweep-disposable') }
if ($PruneBranches -and -not $ReportOnly) { $dargs += '--prune-branches' }
foreach ($b in $AllowBranch) { if ($b) { $dargs += @('--allow-branch', $b) } }
$dargStr = $dargs -join ' '

# Run via powershell so we can append ALL streams to the rolling log with a stamp.
# Force UTF-8 end-to-end: `[Console]::OutputEncoding` makes powershell decode the
# python's UTF-8 stdout correctly, `-X utf8` makes python EMIT UTF-8 (not the OEM
# code page), and `Out-File -Encoding UTF8` writes the log as UTF-8. The old
# `*>>` redirect wrote UTF-16LE and Add-Content wrote ANSI, so the log came out
# garbled (double-spaced / unreadable) when tailed with ordinary tools.
$inner = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; `"===== `$((Get-Date -Format o)) =====`" | Out-File -FilePath '$Log' -Append -Encoding UTF8; & '$Python' -X utf8 $dargStr 2>&1 | Out-File -FilePath '$Log' -Append -Encoding UTF8"
$psArg = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command `"$inner`""

# NB: $action would alias the $Action PARAM (PowerShell vars are case-insensitive),
# so these are deliberately named $task*.
$taskAction   = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $psArg -WorkingDirectory $Repo
$taskTrigger  = New-ScheduledTaskTrigger -Daily -At $At
# Scratch worktrees accrue across a busy multi-session day, not just overnight. Repeating
# every N hours keeps the checkout swept through the day; the doctor's freshness guard
# (--fresh-minutes) makes frequent runs safe (a live session's worktree is never reaped).
if ($EveryHours -gt 0) {
  $rep = New-ScheduledTaskTrigger -Once -At $At `
           -RepetitionInterval (New-TimeSpan -Hours $EveryHours) `
           -RepetitionDuration (New-TimeSpan -Days 1)
  $taskTrigger.Repetition = $rep.Repetition
}
# StartWhenAvailable: a laptop asleep at $At still gets a catch-up run when it wakes.
$taskSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -MultipleInstances IgnoreNew
$taskDesc     = "Keep this checkout at one-worktree-on-master, safely (worktree_doctor.py). Retains: $($AllowBranch -join ',')."
# S4U (non-interactive, session 0), NOT the Register-ScheduledTask default (Interactive):
# a console powershell.exe launched in the interactive session FLASHES a window on every
# daily trigger -- one of the "random popup windows". -WindowStyle Hidden does NOT suppress
# it (the flash is the session-1 console host spawning before the flag applies). S4U runs
# the doctor windowless in session 0 yet still AS THIS USER, so its git fetch/prune still
# work. Same pattern as register_runaway_reaper.ps1 / register_issue_dispatch.ps1.
$taskPrincipal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $taskTrigger -Settings $taskSettings `
  -Principal $taskPrincipal -Description $taskDesc -Force | Out-Null

$mode = if ($ReportOnly) {
  'REPORT-ONLY (no removals)'
} elseif ($PruneBranches) {
  'PRUNE worktrees + SWEEP scratch (archived) + git branch -d merged'
} else {
  'PRUNE worktrees + SWEEP scratch (archived)'
}
$cadence = if ($EveryHours -gt 0) { "daily at $At, repeating every $EveryHours h" } else { "daily at $At" }
Write-Output "installed $TaskName - $cadence, $mode"
Write-Output "repo:    $Repo"
Write-Output "retains: $($AllowBranch -join ', ')  (never pruned / no false alarm)"
Write-Output "log:     $Log"
Write-Output "run now: Start-ScheduledTask -TaskName $TaskName    |    status: .\tools\register_worktree_doctor.ps1 -Action status"
