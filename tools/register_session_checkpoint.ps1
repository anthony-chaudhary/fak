<#
register_session_checkpoint.ps1 -- install/remove the OS Scheduled Task that writes a
durable, off-host SESSION WORK-STATUS CHECKPOINT to GitHub every few minutes.

WHY A SCHEDULED TASK (not just the Stop hook):
A Claude Code Stop hook fires only on a CLEAN end-of-turn. A real crash (the AMD-TDR
terminal kill on this box, an OOM, a power loss) NEVER reaches Stop -- so a Stop-only
checkpoint has zero coverage of the exact event we care about. This task is the
crash-survivor: it runs tools/session_checkpoint.py --source periodic INDEPENDENTLY of any
live session, so the most-recent periodic commit in fak-private is the durable record that
exists when the terminal dies mid-turn. The Stop hook (registered separately in
.claude/settings.local.json) adds the richer clean-end record with the transcript pointer.

The tick is a pure read-only FOLD over git + the scrub primitives; it launches NO worker,
defaults to the PRIVATE route, and is fail-soft (a push miss leaves a local commit to sync
next run). It NEVER auto-promotes to public.

  .\register_session_checkpoint.ps1 -Workspace C:\work\fak        # install (every 20 min)
  .\register_session_checkpoint.ps1 -Workspace C:\work\fak -EveryMinutes 10
  .\register_session_checkpoint.ps1 -Action status
  .\register_session_checkpoint.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName    = 'FakSessionCheckpoint',
  [string]$Workspace   = $(Split-Path -Parent $PSScriptRoot),
  [int]$EveryMinutes   = 20
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

# install -- resolve python and the checkpoint tick.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "python not found on PATH" }
$tick = Join-Path $Workspace 'tools\session_checkpoint.py'
if (-not (Test-Path $tick)) { throw "session_checkpoint.py not found at $tick" }

# Register python.exe DIRECTLY via the ScheduledTasks cmdlets, NOT a
# `powershell.exe -Command "..."` wrapper (same fix as register_dispatch_status_doc):
# a Program-Files python path has a SPACE, and the nested quotes protecting it did not
# survive the PowerShell -> schtasks /TR handoff -- the stored -Command truncated at
# "C:\Program", powershell exited 0 without launching python, and the task logged
# LastResult=0 while nothing ran. Splitting Execute from Argument sidesteps the quoting;
# WorkingDirectory anchors the relative paths, and python's exit code becomes
# LastTaskResult directly.
$pyArgs    = "`"$tick`" --source periodic --json"
$taskAction = New-ScheduledTaskAction -Execute $py -Argument $pyArgs -WorkingDirectory $Workspace
$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
               -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
# S4U (non-interactive, session 0), NOT Interactive: this tick executes python.exe
# DIRECTLY, and a console exe launched in the interactive session flashes a console
# window on EVERY trigger. S4U runs it windowless yet still AS THIS USER (same
# profile/config/oauth + the same git identity for the fak-private commit).
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null

Write-Output "installed $TaskName -- every $EveryMinutes min, writes a PRIVATE session checkpoint to fak-private (read-only fold; crash-survivor)"
Write-Output "the Stop-hook (clean-end, richer) is registered separately in .claude\settings.local.json"
Write-Output "inspect:   .\tools\register_session_checkpoint.ps1 -Action status"
