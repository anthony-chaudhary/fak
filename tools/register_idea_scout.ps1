<#
register_idea_scout.ps1 -- install/remove the OS Scheduled Task that runs the
daily idea-scout (tools/idea_scout.py): search arXiv + GitHub for ideas RELATED
to fak, and file the genuinely-new, on-topic hits as GitHub issues -- deduped
against the existing backlog + a persistent seen-cache, and hard-capped per run.

This is the FEEDER for the issue-dispatch loop (docs/dispatch-loop.md), which
RESOLVES the backlog. Unlike register_issue_dispatch.ps1 (a 10-minute SPAWN tick
bounded by a worker cap), this task fires ONCE A DAY and creates at most
-MaxIssues issues; it spawns no worker, so there is no DoS surface to bound --
the only side effect is `gh issue create`, gated behind -Live.

SAFE BY DEFAULT: installed WITHOUT -Live, the run is DRY-RUN -- it only PRINTS
the issues it would file and writes nothing (not even the seen-cache). Add -Live
to actually create issues (the explicit opt-in to the side effect), mirroring the
dispatch tools' dry-run-first contract.

  .\register_idea_scout.ps1                         # install, DRY-RUN daily (files nothing)
  .\register_idea_scout.ps1 -Live -MaxIssues 3      # install, LIVE (files <=3 issues/day)
  .\register_idea_scout.ps1 -Action status
  .\register_idea_scout.ps1 -Action remove
  .\register_idea_scout.ps1 -At 09:00 -Live         # pick the daily fire time
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName  = 'FleetIdeaScout',
  [string]$Workspace = $(Split-Path -Parent $PSScriptRoot),
  # Daily fire time (local), HH:mm. arXiv/GitHub move on a daily-ish cadence, so
  # once a day is plenty -- a morning slot lands fresh issues for the day's triage.
  [string]$At        = '09:00',
  # Hard cap on issues filed per daily run -- the anti-storm bound. The tool also
  # enforces this; passing it here keeps the registered command self-documenting.
  [int]$MaxIssues    = 3,
  # Optional JSON config overriding the baked-in topics/thresholds
  # (see tools/idea_scout_topics.example.json).
  [string]$Config    = '',
  [switch]$Live
)
$ErrorActionPreference = 'Stop'

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  Write-Output "State=$($t.State)  LastRun=$($i.LastRunTime)  LastResult=$($i.LastTaskResult)  NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'remove') {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Output "removed $TaskName"
  return
}

# install -- resolve python and the daily scout; pick dry-run vs live.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "python not found on PATH" }
$tick = Join-Path $Workspace 'tools\idea_scout.py'
if (-not (Test-Path $tick)) { throw "idea_scout.py not found at $tick" }

$liveFlag = if ($Live) { ' --live' } else { '' }
# Register python.exe DIRECTLY (Execute = program, Argument = its args), NOT via a
# `powershell.exe -Command "..."` wrapper -- the wrapper is a SILENT-NO-OP trap:
# a standard python lives under "C:\Program Files\Python3xx\python.exe" (a SPACE in
# the path), and the nested quotes needed to protect it do not survive the
# PowerShell -> schtasks /TR handoff, so the stored command truncates at
# "C:\Program", exits 0, and the tick never runs while LastResult reads 0. Splitting
# Execute from Argument sidesteps all nested quoting and makes python's own exit code
# the task's LastTaskResult. (Same lesson as register_issue_dispatch.ps1.)
$pyArgs = "`"$tick`" --workspace `"$Workspace`" --max-issues $MaxIssues$liveFlag --json"
if ($Config) { $pyArgs += " --config `"$Config`"" }

$taskAction = New-ScheduledTaskAction -Execute $py -Argument $pyArgs -WorkingDirectory $Workspace
$trigger    = New-ScheduledTaskTrigger -Daily -At $At
# S4U (non-interactive, session 0), NOT Interactive: this tick executes python.exe
# DIRECTLY, and a console exe launched in the interactive session flashes a console
# window on EVERY trigger — the "random popup windows". S4U runs the tick windowless
# yet still AS THIS USER (same profile/config/oauth), so the headless tick is unaffected.
$principal  = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$settings   = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
                -Principal $principal -Settings $settings -Force | Out-Null

$runMode = if ($Live) { "LIVE (files <=$MaxIssues issues/day)" } else { "DRY-RUN (logs the plan, files nothing)" }
Write-Output "installed $TaskName -- daily at $At, current-user interactive, $runMode"
Write-Output "run it once now (dry-run):  python tools\idea_scout.py"
if (-not $Live) {
  Write-Output "to go live later:  .\tools\register_idea_scout.ps1 -Live -MaxIssues $MaxIssues -At $At"
}
