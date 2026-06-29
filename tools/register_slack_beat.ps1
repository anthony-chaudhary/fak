<#
register_slack_beat.ps1 -- install/remove the OS Scheduled Task that posts the Slack
LIVENESS BEAT on a cadence (#1426, epic #1425): `fak slack beat` folds the Slack-surface
health (resolution + auth + a real conversations.history read per cadence surface) and posts
ONE compact line to a status channel -- UNCONDITIONALLY on its cadence, whether or not any
feeder posted. A green beat means alive; a missing beat means this task itself died.

This is the operator's "the channel is provably alive even on a quiet day" task. It posts to
the channel `fak slack beat` resolves: -SlackChannel first, else $FAK_DISPATCH_CHANNEL, then
$FAK_SCOREBOARD_CHANNEL, then .env.slack.local. The bot token is the shared
FAK_SCOREBOARD_TOKEN (or a surface token) from the gitignored .env.slack.local. NO channel id
or token is baked into this script.

It runs the INSTALLED `fak` binary (fast, dependency-free per tick); if `fak` is not on PATH
it falls back to `go run ./cmd/fak` from the workspace.

SAFE BY DEFAULT: installed WITHOUT -Live the task runs `--dry-run` (resolves channel/token,
renders the beat, sends nothing). Add -Live to actually post.

  .\register_slack_beat.ps1 -Live                          # post the beat, channel from env
  .\register_slack_beat.ps1 -Live -SlackChannel C0ABC123   # post to an explicit channel
  .\register_slack_beat.ps1 -Live -EveryMinutes 120        # 2-hour pulse
  .\register_slack_beat.ps1 -Action status
  .\register_slack_beat.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName     = 'FleetSlackBeat',
  [string]$Workspace    = $(Split-Path -Parent $PSScriptRoot),
  [string]$SlackChannel = '',          # '' => resolve from env / .env.slack.local
  [int]$EveryMinutes    = 180,          # 3-hour pulse: bounds "is it alive?" silence to <=3h
  [string]$FakExe       = '',          # explicit fak binary path; '' => Get-Command fak, then ~/go/bin
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

# install -- resolve the fak binary (preferred) or fall back to `go run`. An explicit -FakExe
# wins (use it to pin a freshly-installed GOBIN binary when a stale `fak` shadows it on PATH).
$fak = ''
if ($FakExe) {
  if (-not (Test-Path $FakExe)) { throw "-FakExe '$FakExe' does not exist" }
  $fak = $FakExe
} else {
  $fak = (Get-Command fak -ErrorAction SilentlyContinue).Source
  if (-not $fak) {
    $gobin = Join-Path $env:USERPROFILE 'go\bin\fak.exe'
    if (Test-Path $gobin) { $fak = $gobin }
  }
}

# Build the beat args. --json so the verb's exit code (0 posted/dry-run, 1 a live post
# failed/skipped) becomes LastTaskResult and the operator sees a misconfiguration.
$beatArgs = @('slack', 'beat', '--json')
if ($SlackChannel) { $beatArgs += @('--channel', $SlackChannel) }
if (-not $Live)    { $beatArgs += '--dry-run' }

if ($fak) {
  $exe     = $fak
  $exeArgs = ($beatArgs -join ' ')
  $runWith = "fak ($fak)"
} else {
  # No installed binary: run `go run ./cmd/fak slack beat ...` from the workspace. Slower per
  # tick but dependency-free for a fresh checkout.
  $go = (Get-Command go -ErrorAction SilentlyContinue).Source
  if (-not $go) { throw "neither 'fak' nor 'go' found on PATH -- run 'go install ./cmd/fak' first" }
  $exe     = $go
  $exeArgs = (@('run', './cmd/fak') + $beatArgs) -join ' '
  $runWith = "go run ./cmd/fak ($go)"
}

$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
               -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

# Register the binary DIRECTLY via the ScheduledTasks cmdlets (NOT a powershell.exe -Command
# wrapper): an executable path may contain a SPACE, and the nested quotes protecting it do not
# survive the PowerShell -> schtasks /TR handoff. Splitting Execute from Argument sidesteps the
# quoting; WorkingDirectory anchors the relative paths (and the .env.slack.local walk-up).
#
# Preferred: S4U (session 0, windowless, runs AS THIS USER even when not logged in). S4U
# registration requires elevation, so when it is denied (a non-admin install) fall back to a
# current-user Interactive task via conhost --headless so no console window flashes per tick.
$reg = $null
try {
  $taskAction = New-ScheduledTaskAction -Execute $exe -Argument $exeArgs -WorkingDirectory $Workspace
  $principal  = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
  $reg = Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force -ErrorAction Stop
  $principalKind = 'S4U (session 0)'
} catch {
  $headlessArgs = "--headless `"$exe`" $exeArgs"
  $taskAction = New-ScheduledTaskAction -Execute 'conhost.exe' -Argument $headlessArgs -WorkingDirectory $Workspace
  $principal  = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
  $reg = Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force
  $principalKind = "Interactive (non-elevated; conhost --headless)"
}

$runMode = if ($Live) { 'LIVE (posts to Slack)' } else { 'DRY-RUN (resolves channel/token, sends nothing)' }
$chanStr = if ($SlackChannel) { $SlackChannel } else { '$FAK_DISPATCH_CHANNEL / $FAK_SCOREBOARD_CHANNEL / .env.slack.local' }
Write-Output "installed $TaskName -- every $EveryMinutes min, $principalKind, $runMode"
Write-Output "runs:     $runWith"
Write-Output "channel:  $chanStr   (token: FAK_SCOREBOARD_TOKEN)"
Write-Output "check resolution any time:  fak slack beat --dry-run"
if (-not $Live) {
  Write-Output "to go live later:  .\tools\register_slack_beat.ps1 -Live"
}
