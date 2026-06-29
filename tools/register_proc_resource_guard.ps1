<#
register_proc_resource_guard.ps1 - install/remove the Scheduled Task that runs the
cross-platform process-resource guard (tools/proc_resource_guard.py) on a standing
interval, so the host has a durable watch for the runaway classes the level
watchdogs miss: a process whose thread/handle/working-set count has gone
pathological, an orphaned ephemeral helper (a dos_mcp.server outliving its session),
and -- the reason this companion to the find/grep reaper exists -- a SINGLE-THREADED
process pinning one core to 100% (a stuck spin loop / wedged terminal), which has a
normal thread count and so trips nothing else.

Report-only by default (logs to tools/_watchdog/proc_guard.log; never kills). Pass
-Enact to flip it to reaping: it then reaps flagged NON-protected runaways + orphans,
and a CPU pin only after it has held the threshold across every sample window
(default 4 samples x 2s = 6s sustained), so a legitimate compile/test burst is never
killed. It never touches an OS-critical process (System/csrss/lsass/...) or its own
tree. See docs/perf-runaway-guard.md.

  .\register_proc_resource_guard.ps1                 # install REPORT-ONLY (safe default)
  .\register_proc_resource_guard.ps1 -Enact          # install ENACTING (reaps runaways/orphans/CPU pins)
  .\register_proc_resource_guard.ps1 -Action status
  .\register_proc_resource_guard.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [switch]$Enact,
  [int]$EveryMin = 10,
  # CPU-pin reaping bar (per-core %, sustained across $CpuSamples windows of $CpuWindow s).
  # 90%/core over 4x2s = 6s safely excludes legit bursty compute -- see the doc.
  [int]$MaxCpuPct = 90,
  [double]$CpuWindow = 2.0,
  [int]$CpuSamples = 4,
  [string]$TaskName = 'FleetProcResourceGuard',
  # Resolve the sibling guard in THIS clone, so registering from any checkout
  # schedules that checkout's script -- not a hardcoded operator path.
  [string]$Guard = (Join-Path $PSScriptRoot 'proc_resource_guard.py'),
  [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  $a = ($t.Actions | Select-Object -First 1).Arguments
  $modeStr = if ($a -match '--enact') { 'ENACT (reaps)' } else { 'REPORT-ONLY' }
  Write-Output "State=$($t.State) mode=$modeStr LastRun=$($i.LastRunTime) LastResult=$($i.LastTaskResult) NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'remove') {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Output "removed $TaskName"; return
}

# Resolve a Python launcher: prefer `python`, fall back to the `py` launcher.
$py = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command py.exe -ErrorAction SilentlyContinue).Source }
if (-not $py) { $py = 'python' }

# The guard scans every live process; it flags thread/handle runaways, reaps orphaned
# dos_mcp.server helpers, and (the single-threaded-core-pin witness) a process holding
# >$MaxCpuPct% of one core sustained across $CpuSamples windows. --enact only added with -Enact.
$enactArg = if ($Enact) { ' --enact' } else { '' }
$guardArgs = "`"$Guard`" --reap-orphans --max-cpu-pct $MaxCpuPct --cpu-window $CpuWindow --cpu-samples $CpuSamples$enactArg"

$taskAction = New-ScheduledTaskAction -Execute $py -Argument $guardArgs -WorkingDirectory $RepoRoot
$trigger    = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
                -RepetitionInterval (New-TimeSpan -Minutes $EveryMin) `
                -RepetitionDuration (New-TimeSpan -Days 3650)
# S4U (non-interactive, session 0), NOT the schtasks default Interactive: a console
# python.exe launched in the interactive session would flash a window on every trigger.
# S4U runs windowless in session 0 yet still AS THIS USER, so it can still enumerate
# and (with -Enact) terminate this user's runaways -- same-user terminate needs no
# elevation. -StartWhenAvailable resumes it after a reboot that missed a tick.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null
$mode = if ($Enact) { "ENACT (reaps runaways/orphans + CPU pins >$MaxCpuPct%/core sustained ${CpuSamples}x${CpuWindow}s)" } else { 'REPORT-ONLY (logs intentions only)' }
Write-Output "installed $TaskName - every $EveryMin min, $mode, S4U (windowless, restart-durable)"
Write-Output "log: tools/_watchdog/proc_guard.log (one line per scan)"
Write-Output "flip to enacting later:  .\tools\register_proc_resource_guard.ps1 -Enact"
Write-Output "remove:                  .\tools\register_proc_resource_guard.ps1 -Action remove"
