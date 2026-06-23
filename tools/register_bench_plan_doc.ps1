<#
register_bench_plan_doc.ps1 -- install/remove the OS Scheduled Task that keeps the
committed hardware bench-plan doc fresh (docs/bench-plan.md).

The benchmark catalog (experiments/benchmark/catalog.json) is a passive registry of what
each bench-node (macbook / datacenter-A100 / cloud-L4 / RTX-laptop) HAS run. This task renders the
one surface that says what to run NEXT per machine: empty (machine x workload-kind)
coverage holes, first-ever measurements on new hardware, and the recorded baselines due
for a regression re-measure. It runs tools/bench_plan_tick.ps1 -- which stamps a fresh
UTC --now and calls tools/bench_plan.py --md -- every 12 hours by default (a benchmark
cadence is slow; minute-scale would only churn the doc).

PURE READ-ONLY FOLD: the tick WRITES only the working-tree doc and git-commits NOTHING.
The repo is a shared multi-session tree where commits are by explicit path only --
automating git here would steal a sibling session's in-flight files. An operator commits
docs/bench-plan.md by path when ready; the task just keeps the working copy current.

NO LIVE ARM: the planner has no execute mode and never runs a benchmark, and this box is
the agent-host -- "live" execution is a human/remote action on the bench-node later.
Rendering the plan IS the only effect, which is the strongest honesty guarantee.

  .\register_bench_plan_doc.ps1 -Workspace C:\work\fak            # install (every 12h)
  .\register_bench_plan_doc.ps1 -Workspace C:\work\fak -EveryMinutes 360
  .\register_bench_plan_doc.ps1 -Action status
  .\register_bench_plan_doc.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName    = 'FleetBenchPlanDoc',
  [string]$Workspace   = $(Split-Path -Parent $PSScriptRoot),
  [string]$DocPath     = 'docs\bench-plan.md',
  [int]$EveryMinutes   = 720
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

# install -- run the tick wrapper via powershell.exe -File. We deliberately do NOT embed
# python directly (as the sibling register_* scripts do): the planner needs a FRESH --now
# each tick, so the stamp must be computed at RUN time inside the wrapper, not frozen into
# the stored task argument at install time. The wrapper path is under the repo (no space),
# and the only space-bearing path (python under "C:\Program Files") is resolved + called
# INSIDE the wrapper via the call operator -- so the schtasks /TR quoting trap that bit the
# sibling scripts (a path truncating at "C:\Program") never applies here.
$ps = (Get-Command powershell -ErrorAction SilentlyContinue).Source
if (-not $ps) { $ps = (Get-Command pwsh -ErrorAction SilentlyContinue).Source }
if (-not $ps) { throw "powershell not found on PATH" }
$tick = Join-Path $Workspace 'tools\bench_plan_tick.ps1'
if (-not (Test-Path $tick)) { throw "bench_plan_tick.ps1 not found at $tick" }

$psArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$tick`" -Workspace `"$Workspace`" -DocPath `"$DocPath`""
$taskAction = New-ScheduledTaskAction -Execute $ps -Argument $psArgs -WorkingDirectory $Workspace
$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
               -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null

$everyHrs = [Math]::Round($EveryMinutes / 60.0, 1)
Write-Output "installed $TaskName -- every $EveryMinutes min (~${everyHrs}h), renders $DocPath (read-only fold; commits nothing)"
Write-Output "read the doc:   $Workspace\$DocPath"
Write-Output "commit it by path when ready:   git commit -s -- $DocPath"
