<#
register_grafana_rollup.ps1 -- install/remove the OS Scheduled Task that posts the
#grafana dashboard/debug-link ROLLUP to Slack on a cadence, so the channel stays
populated without a manual `fak grafana post`. The scheduled-cadence feeder for the
#grafana observability-link surface (GH #1299, follow-on of epic #1294).

It is the grafana twin of register_fleet_slack_status.ps1: the same safe-by-default,
loop-ledgered, S4U/Interactive register idiom, pointed at `fak grafana post --rollup`.
The child command is routed through `fak loop run` (the same wiring the dispatch-status
feeder uses), so every tick records a ledger row in .fak\loops.jsonl with the child exit
code and duration -- a silently-failing post is visible in `fak loop status`.

The card folds the committed link registry (docs/grafana/links.json) -- the long-lived
dashboards and provisioned debug dashboards -- never a fabricated URL. Channel/token
resolve the standard grafana way: --channel / -SlackChannel first, else
$FAK_GRAFANA_CHANNEL (else the public built-in #grafana default); token is
$FAK_GRAFANA_TOKEN, falling back to the scoreboard token. NO channel id or token is
baked into this script.

SAFE BY DEFAULT: installed WITHOUT -Live the tick runs `fak grafana post --rollup ...
--dry-run` (resolves the channel/token and renders the card to the log, posts nothing).
Add -Live to actually post to the channel.

  .\register_grafana_rollup.ps1                                  # install, DRY-RUN, rollup=all, every 6h
  .\register_grafana_rollup.ps1 -Live                            # post the rollup for real
  .\register_grafana_rollup.ps1 -Live -Rollup public-demo        # only the long-lived demo dashboards
  .\register_grafana_rollup.ps1 -Live -EveryMinutes 720          # explicit cadence (minutes)
  .\register_grafana_rollup.ps1 -Action status
  .\register_grafana_rollup.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName     = 'FleetGrafanaRollup',
  [string]$Workspace    = $(Split-Path -Parent $PSScriptRoot),
  [ValidateSet('all','public-demo','debug','rollup')] [string]$Rollup = 'all',
  [string]$SlackChannel = '',          # '' => resolve from $FAK_GRAFANA_CHANNEL / built-in #grafana default
  [int]$EveryMinutes    = 360,         # 6h: a long-lived link rollup, not a per-minute heartbeat
  # Optional path to a fak binary. If unset, the loop helper probes ./fak.exe, PATH fak,
  # then falls back to `go run ./cmd/fak` so a source-tree install cannot silently use a
  # stale binary that lacks `fak grafana post`.
  [string]$FakExe = $env:FAK_BIN,
  [switch]$Live                        # without -Live the tick is dry-run (resolves, renders, sends nothing)
)
$ErrorActionPreference = 'Stop'

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  $a = ($t.Actions | Select-Object -First 1).Arguments
  $modeStr = if ($a -match '--dry-run') { 'DRY-RUN' } else { 'LIVE' }
  Write-Output "State=$($t.State) mode=$modeStr LastRun=$($i.LastRunTime) LastResult=$($i.LastTaskResult) NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'remove') {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Output "removed $TaskName"
  return
}

. (Join-Path $PSScriptRoot 'fak_loop_task.ps1')

# install -- build the `fak grafana post --rollup <cat>` child command. --dry-run unless
# -Live, so a fresh install never posts until the operator opts in.
$childArgs = @('grafana', 'post', '--rollup', $Rollup)
if ($SlackChannel) { $childArgs += @('--channel', $SlackChannel) }
if (-not $Live)    { $childArgs += '--dry-run' }

# Route the child through `fak loop run` via the ScheduledTasks cmdlets (NOT a
# powershell.exe -Command wrapper): the helper splits Task Scheduler's Execute and
# Argument fields, which sidesteps the Program-Files-space quoting trap, and the loop
# ledger records the child exit code + duration so a silently-failing post shows up in
# `fak loop status` -- the same wiring the FleetDispatchStatusDoc feeder uses.
$wrapperLoop = 'grafana-rollup/task-scheduler'
$taskAction  = New-FakLoopScheduledTaskAction -Workspace $Workspace -FakExe $FakExe -LoopId $wrapperLoop -ChildArgs $childArgs

$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
               -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

# Preferred: S4U (session 0, windowless, runs AS THIS USER even when not logged in). S4U
# registration requires elevation; when denied (a non-admin install) fall back to a
# current-user Interactive task -- the rollup tick is a console exe, so a window may flash
# on each trigger, but it runs while the user is logged in, which is when an operator
# watches the channel anyway.
try {
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
  Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force -ErrorAction Stop | Out-Null
  $principalKind = 'S4U (session 0)'
} catch {
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
  Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null
  $principalKind = 'Interactive (non-elevated)'
}

$runMode = if ($Live) { 'LIVE (posts to #grafana)' } else { 'DRY-RUN (resolves channel/token, renders, sends nothing)' }
$chanStr = if ($SlackChannel) { $SlackChannel } else { '$FAK_GRAFANA_CHANNEL / built-in #grafana default' }
Write-Output "installed $TaskName -- every $EveryMinutes min, $principalKind, $runMode"
Write-Output "rollup:   --rollup $Rollup   channel: $chanStr   (token: FAK_GRAFANA_TOKEN / scoreboard token)"
Write-Output "loop ledger:  .fak\loops.jsonl via fak loop run ($wrapperLoop)"
Write-Output "preview the card any time:  fak grafana post --rollup $Rollup --dry-run"
if (-not $Live) {
  Write-Output "to go live later:  .\tools\register_grafana_rollup.ps1 -Live"
}
