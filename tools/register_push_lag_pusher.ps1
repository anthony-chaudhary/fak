<#
register_push_lag_pusher.ps1 -- install/remove the OS Scheduled Task that runs the
push-lag BACKSTOP on a cadence: tools/auto_push_on_lag.py checks fresh_status's
git pane and, only when the oldest unpushed commit has been waiting past the
threshold (push_lag_stale), runs the repo's safe-push primitive (`fak sync push`)
to fast-forward origin/main.

Why it exists: the witnessed issue closer only closes an issue whose fix is an
ANCESTOR of origin/main, so a silently-stalled push freezes closure while workers
keep committing locally (a ~7h gap did exactly that on 2026-07-07). This task is
the backstop that drains such a stall automatically. It never force-pushes, never
fires on a merely-dirty tree, and defers all behind/diverged handling to the
safe-push primitive and the pre-push gate.

SAFE BY DEFAULT: installed WITHOUT -Live the task only REPORTS what it would push
(dry-run to its log/LastResult). Add -Live to actually push.

  .\register_push_lag_pusher.ps1 -Live                     # arm the backstop, 15-min cadence
  .\register_push_lag_pusher.ps1 -Live -EveryMinutes 10    # tighter cadence
  .\register_push_lag_pusher.ps1 -Live -PushLagMins 30     # trip sooner than the 45-min default
  .\register_push_lag_pusher.ps1 -Action status
  .\register_push_lag_pusher.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName  = 'FleetPushLagPusher',
  [string]$Workspace = $(Split-Path -Parent $PSScriptRoot),
  [int]$EveryMinutes = 15,               # matches FleetResolveProgress: closure resumes within one closer cycle
  [int]$PushLagMins  = 45,               # trip past this many minutes of push lag (fresh_status default)
  [switch]$Live                          # without -Live the tick is dry-run (reports, pushes nothing)
)
$ErrorActionPreference = 'Stop'

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  $a = ($t.Actions | Select-Object -First 1).Arguments
  $modeStr = if ($a -match '--live') { 'LIVE' } else { 'DRY-RUN' }
  Write-Output "State=$($t.State) mode=$modeStr LastRun=$($i.LastRunTime) LastResult=$($i.LastTaskResult) NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'remove') {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Output "removed $TaskName"
  return
}

# install -- resolve python and the push-lag backstop tick.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "python not found on PATH" }
# pythonw.exe (console-less sibling) for the non-elevated fallback so a current-user
# Interactive task runs windowless -- no popup per tick.
$pyw = Join-Path (Split-Path -Parent $py) 'pythonw.exe'
$tick = Join-Path $Workspace 'tools\auto_push_on_lag.py'
if (-not (Test-Path $tick)) { throw "auto_push_on_lag.py not found at $tick" }

# Build the child args. --json so python's exit code (0 = healthy no-op or push
# landed, 1 = a needed push failed / fak unavailable) becomes LastTaskResult and
# the operator sees a stalled/blocked push.
$childArgs = @("`"$tick`"", '--workspace', "`"$Workspace`"", '--push-lag-mins', "$PushLagMins", '--json')
if ($Live) { $childArgs += '--live' }
$pyArgs = ($childArgs -join ' ')

$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
               -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

# Register python DIRECTLY via the ScheduledTasks cmdlets (NOT a powershell.exe -Command
# wrapper): a Program-Files python path has a SPACE, and the nested quotes protecting it
# do not survive the PowerShell -> schtasks /TR handoff (the stored -Command truncates at
# "C:\Program", the task logs LastResult=0 while python never runs). Splitting Execute from
# Argument sidesteps the quoting; WorkingDirectory anchors the relative paths.
#
# Preferred: S4U (session 0, windowless, runs AS THIS USER even when not logged in) with
# python.exe. S4U registration requires elevation, so when it is denied (a non-admin
# install) fall back to a current-user Interactive task running the console-less
# pythonw.exe -- no elevation, still never flashes a window.
$reg = $null
try {
  $taskAction = New-ScheduledTaskAction -Execute $py -Argument $pyArgs -WorkingDirectory $Workspace
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
  $reg = Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force -ErrorAction Stop
  $principalKind = 'S4U (session 0)'
} catch {
  $exe = if (Test-Path $pyw) { $pyw } else { $py }   # windowless if pythonw is present
  $headlessArgs = "--headless `"$exe`" $pyArgs"
  $taskAction = New-ScheduledTaskAction -Execute 'conhost.exe' -Argument $headlessArgs -WorkingDirectory $Workspace
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
  $reg = Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force
  $principalKind = "Interactive (non-elevated; conhost --headless $(Split-Path -Leaf $exe))"
}

$runMode = if ($Live) { 'LIVE (runs fak sync push on stall)' } else { 'DRY-RUN (reports, pushes nothing)' }
Write-Output "installed $TaskName -- every $EveryMinutes min, trip at ${PushLagMins}m lag, $principalKind, $runMode"
Write-Output "check any time:  python tools\auto_push_on_lag.py --workspace `"$Workspace`" --json"
if (-not $Live) {
  Write-Output "to go live later:  .\tools\register_push_lag_pusher.ps1 -Live"
}
