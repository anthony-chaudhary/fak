<#
register_fleet_slack_status.ps1 -- install/remove the OS Scheduled Task that posts the
WHOLE fleet status to Slack on a cadence: the dispatch_status card (dispatcher +
supervisor + watchdog-installed + backlog + closure + throughput) AND the fleet_top
session/account-health snapshot, folded into one tick by tools/fleet_slack_status.py.

This is the operator's "the fleet's heartbeat lands in one channel" task. It posts to
the channel resolved by tools/slack_post: --channel / -SlackChannel first, else the
FAK_DISPATCH_CHANNEL env var (set machine-wide once), else nothing. The bot token is the
shared FAK_SCOREBOARD_TOKEN (or FAK_DISPATCH_TOKEN) from the gitignored .env.slack.local.
NO channel id or token is baked into this script.

SAFE BY DEFAULT: installed WITHOUT -Live the task posts a DRY-RUN line to its log
(resolves the channel/token, sends nothing). Add -Live to actually post to Slack.

  .\register_fleet_slack_status.ps1 -Live                              # post both cards, channel from env
  .\register_fleet_slack_status.ps1 -Live -SlackChannel C0ABC123       # post to an explicit channel
  .\register_fleet_slack_status.ps1 -Live -EveryMinutes 15 -Fast       # 15-min cadence, skip gh folds
  .\register_fleet_slack_status.ps1 -Action status
  .\register_fleet_slack_status.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName     = 'FleetSlackStatus',
  [string]$Workspace    = $(Split-Path -Parent $PSScriptRoot),
  [string]$SlackChannel = '',          # '' => resolve from $FAK_DISPATCH_CHANNEL / .env.slack.local
  [int]$EveryMinutes    = 30,
  [switch]$Fast,                        # dispatch card skips the gh-backed folds
  [switch]$Live                         # without -Live the tick is dry-run (resolves, sends nothing)
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

# install -- resolve python and the consolidated Slack-status tick.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "python not found on PATH" }
$tick = Join-Path $Workspace 'tools\fleet_slack_status.py'
if (-not (Test-Path $tick)) { throw "fleet_slack_status.py not found at $tick" }

# Build the child args. --json so python's exit code (0 every post landed, 1 a post
# failed/skipped) becomes LastTaskResult and the operator sees a misconfiguration.
$childArgs = @("`"$tick`"", '--workspace', "`"$Workspace`"", '--json')
if ($SlackChannel) { $childArgs += @('--channel', $SlackChannel) }
if ($Fast)         { $childArgs += '--fast' }
if (-not $Live)    { $childArgs += '--dry-run' }
$pyArgs = ($childArgs -join ' ')

# Register python.exe DIRECTLY via the ScheduledTasks cmdlets (NOT a powershell.exe
# -Command wrapper): a Program-Files python path has a SPACE, and the nested quotes
# protecting it do not survive the PowerShell -> schtasks /TR handoff (the stored
# -Command truncates at "C:\Program", the task logs LastResult=0 while python never
# runs). Splitting Execute from Argument sidesteps the quoting; WorkingDirectory anchors
# the relative paths; python's exit code becomes LastTaskResult directly.
$taskAction = New-ScheduledTaskAction -Execute $py -Argument $pyArgs -WorkingDirectory $Workspace
$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
               -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
# S4U (non-interactive, session 0), NOT Interactive: the tick executes python.exe
# directly, and a console exe launched in the interactive session flashes a console
# window on EVERY trigger. S4U runs it windowless yet still AS THIS USER, so the
# machine-wide FAK_DISPATCH_CHANNEL / .env.slack.local resolve exactly as in a shell.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null

$runMode = if ($Live) { 'LIVE (posts to Slack)' } else { 'DRY-RUN (resolves channel/token, sends nothing)' }
$chanStr = if ($SlackChannel) { $SlackChannel } else { '$FAK_DISPATCH_CHANNEL / .env.slack.local' }
Write-Output "installed $TaskName -- every $EveryMinutes min, current-user S4U, $runMode"
Write-Output "channel:  $chanStr   (token: FAK_SCOREBOARD_TOKEN / FAK_DISPATCH_TOKEN)"
Write-Output "check resolution any time:  python tools\fleet_slack_status.py --dry-run --fast"
if (-not $Live) {
  Write-Output "to go live later:  .\tools\register_fleet_slack_status.ps1 -Live"
}
