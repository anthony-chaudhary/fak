<#
register_loops_inventory.ps1 -- install/remove the OS Scheduled Task that keeps the
committed recurring-loops inventory fresh (docs/loops-inventory.md) and, opt-in, posts a
compact rollup to Slack.

The fleet's recurring work is declared across two surfaces that never meet: the OS
Scheduled Tasks installed by the sibling tools/register_*.ps1 scripts, and the cron
GitHub Actions under .github/workflows/. There is no single place to see the whole
recurring surface -- which loops exist, how often each fires, and whether each reports to
Slack, files GitHub issues, commits a repo doc, or only writes operator-local telemetry.
tools/loops_inventory.py DISCOVERS the loops from those declaration files (so the list is
always as current as the tree) and folds them into a committed markdown doc, an opt-in
Slack card, and a trend ledger. This task runs that fold on a cadence -- every 12h by
default (the inventory changes only when a register_*.ps1 or a workflow is added/removed,
so minute-scale would just churn the doc).

PURE READ-ONLY FOLD: the tick parses declaration files, WRITES only the working-tree doc
(docs\loops-inventory.md) and its own gitignored trend ledger (.fak\nightrun), and
git-commits NOTHING. The repo is a shared multi-session tree where commits are by explicit
path only -- automating git here would steal a sibling session's in-flight files. An
operator commits docs\loops-inventory.md by path when ready; the task keeps the working
copy current.

SAFE BY DEFAULT: installed WITHOUT -Live the task renders the doc + trend and posts a
DRY-RUN Slack line to its log (resolves the channel/token, sends nothing). Add -Live to
actually post the rollup to Slack. The channel is resolved by tools/slack_post
(--channel / -SlackChannel first, else $FAK_DISPATCH_CHANNEL, else nothing); no channel
id or token is baked into this script.

  .\register_loops_inventory.ps1 -Workspace C:\work\fak                 # install (every 12h, doc + trend)
  .\register_loops_inventory.ps1 -Live                                 # also post the rollup to Slack
  .\register_loops_inventory.ps1 -Live -SlackChannel C0ABC123
  .\register_loops_inventory.ps1 -EveryMinutes 360                     # 6h cadence
  .\register_loops_inventory.ps1 -Action status
  .\register_loops_inventory.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName     = 'FleetLoopsInventory',
  [string]$Workspace    = $(Split-Path -Parent $PSScriptRoot),
  [string]$DocPath      = 'docs\loops-inventory.md',
  [string]$SlackChannel = '',          # '' => resolve from $FAK_DISPATCH_CHANNEL / .env.slack.local
  [int]$EveryMinutes    = 720,
  [switch]$Live                         # without -Live the Slack post is dry-run (resolves, sends nothing)
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

# install -- resolve python and the inventory tick.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "python not found on PATH" }
# pythonw.exe (the console-less interpreter, sibling of python.exe) for the non-elevated
# fallback path so a current-user Interactive task runs windowless -- no popup per tick.
$pyw = Join-Path (Split-Path -Parent $py) 'pythonw.exe'
$tick = Join-Path $Workspace 'tools\loops_inventory.py'
if (-not (Test-Path $tick)) { throw "loops_inventory.py not found at $tick" }

# Build the child args. --md renders the committed doc; --ledger appends the trend row;
# --slack posts the rollup (dry-run unless -Live). --json so python's exit code becomes
# LastTaskResult and the operator sees a misconfiguration.
$childArgs = @("`"$tick`"", '--workspace', "`"$Workspace`"", '--md', "`"$DocPath`"", '--ledger', '--slack', '--json')
if ($SlackChannel) { $childArgs += @('--channel', $SlackChannel) }
if (-not $Live)    { $childArgs += '--dry-run' }
$pyArgs = ($childArgs -join ' ')

$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
               -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
               -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

# Register python DIRECTLY via the ScheduledTasks cmdlets (NOT a powershell.exe -Command
# wrapper): a Program-Files python path has a SPACE, and the nested quotes protecting it do
# not survive the PowerShell -> schtasks /TR handoff (the stored -Command truncates at
# "C:\Program", the task logs LastResult=0 while python never runs). Splitting Execute from
# Argument sidesteps the quoting; WorkingDirectory anchors the relative paths.
#
# Preferred: S4U (session 0, windowless, runs AS THIS USER even when not logged in) with
# python.exe. S4U registration requires elevation, so when it is denied (a non-admin
# install) fall back to a current-user Interactive task running the console-less
# pythonw.exe -- it needs no elevation and still never flashes a window.
try {
  $taskAction = New-ScheduledTaskAction -Execute $py -Argument $pyArgs -WorkingDirectory $Workspace
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
  Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force -ErrorAction Stop | Out-Null
  $principalKind = 'S4U (session 0)'
} catch {
  $exe = if (Test-Path $pyw) { $pyw } else { $py }   # windowless if pythonw is present
  $headlessArgs = "--headless `"$exe`" $pyArgs"
  $taskAction = New-ScheduledTaskAction -Execute 'conhost.exe' -Argument $headlessArgs -WorkingDirectory $Workspace
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
  Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
               -Principal $principal -Settings $settings -Force | Out-Null
  $principalKind = "Interactive (non-elevated; conhost --headless $(Split-Path -Leaf $exe))"
}

$everyHrs = [Math]::Round($EveryMinutes / 60.0, 1)
$runMode = if ($Live) { 'LIVE (posts rollup to Slack)' } else { 'DRY-RUN Slack (resolves channel/token, sends nothing)' }
$chanStr = if ($SlackChannel) { $SlackChannel } else { '$FAK_DISPATCH_CHANNEL / .env.slack.local' }
Write-Output "installed $TaskName -- every $EveryMinutes min (~${everyHrs}h), $principalKind, $runMode"
Write-Output "renders $DocPath + trend ledger (read-only fold; commits nothing)"
Write-Output "channel:  $chanStr   (token: FAK_SCOREBOARD_TOKEN / FAK_DISPATCH_TOKEN)"
Write-Output "check any time:  python tools\loops_inventory.py --slack --dry-run"
Write-Output "commit the doc by path when ready:  git commit -s -- $DocPath"
if (-not $Live) {
  Write-Output "to post to Slack later:  .\tools\register_loops_inventory.ps1 -Live"
}
