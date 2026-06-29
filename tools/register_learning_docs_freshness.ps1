<#
register_learning_docs_freshness.ps1 -- install/remove the OS Scheduled Task that
runs the durable learning-docs freshness loop.

This is the host delivery wrapper for the `learning-docs-freshness` job declared in
tools/loop-registry.json. Each tick runs tools/learning_scorecard.py --json through
`fak loop run`, so the loop ledger records the last tick while `fak loop health`
joins that tick with the registry cadence and the current learning-debt value.

The tick is read-only: it writes no docs, files no issues, and commits nothing.

  .\tools\register_learning_docs_freshness.ps1
  .\tools\register_learning_docs_freshness.ps1 -EveryMinutes 720
  .\tools\register_learning_docs_freshness.ps1 -Action status
  .\tools\register_learning_docs_freshness.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName    = 'LearningDocsFreshness',
  [string]$Workspace   = $(Split-Path -Parent $PSScriptRoot),
  [string]$LoopId      = 'learning-docs-freshness',
  [int]$EveryMinutes   = 1440,
  # Optional path to a fak binary. If unset, the installer probes ./fak.exe, PATH fak,
  # then falls back to `go run ./cmd/fak` so source-tree installs cannot silently use
  # a stale binary that lacks `fak loop health`.
  [string]$FakExe = $env:FAK_BIN
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

if ($EveryMinutes -le 0) { throw "EveryMinutes must be positive" }

. (Join-Path $PSScriptRoot 'fak_loop_task.ps1')

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "python not found on PATH" }
$tick = Join-Path $Workspace 'tools\learning_scorecard.py'
if (-not (Test-Path $tick)) { throw "learning_scorecard.py not found at $tick" }

$childArgs = @($py, $tick, '--json')
$taskAction = New-FakLoopScheduledTaskAction -Workspace $Workspace -FakExe $FakExe -LoopId $LoopId -ChildArgs $childArgs
$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
               -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null

Write-Output "installed $TaskName -- every $EveryMinutes min, loop=$LoopId, current-user S4U, read-only scorecard tick"
Write-Output "loop ledger:  .fak\loops.jsonl via fak loop run ($LoopId)"
Write-Output "inspect:      fak loop health"
