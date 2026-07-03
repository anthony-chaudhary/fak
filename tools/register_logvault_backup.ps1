<#
register_logvault_backup.ps1 -- install/remove the OS Scheduled Tasks that run the
logvault backup cadence (#2450, part of the logvault epic #2447): `fak logvault capture`
DAILY (to beat the 7-day GC in tools/stale_work_watchdog.py and bound the loss window on
truncate-replaced snapshot files like vcache-turns.jsonl) and `fak logvault verify -sample N`
WEEKLY (a bitrot watch that fails loudly -- exit 1 -- on any chain/mirror mismatch).

Both ticks run through `fak loop run --loop <id> --source task-scheduler -- fak logvault ...`
(the #765 cron-emit convention: fak owns overlap-lock via the loop ledger + guard containment;
the OS scheduler only owns wall-clock firing) so every vault run is witnessed in the loop
ledger -- the vault ends up capturing its own scheduler's ledger too.

SAFE BY DEFAULT: installed WITHOUT -Live neither task passes -notify-slack, so ticks only
write to the local vault directory -- no external Slack send. Add -Live to also enqueue the
capture/verify digest through the slack outbox (channel resolved by `fak logvault`: an
explicit -SlackChannel, else $FAK_DISPATCH_CHANNEL, then $FAK_SCOREBOARD_CHANNEL).

  .\register_logvault_backup.ps1                              # install both, local-only (dry re: Slack)
  .\register_logvault_backup.ps1 -Live                        # install both, with Slack digests
  .\register_logvault_backup.ps1 -Live -SlackChannel C0ABC123 # explicit channel
  .\register_logvault_backup.ps1 -Which verify -VerifySample 0 # full re-hash, weekly task only
  .\register_logvault_backup.ps1 -Action status
  .\register_logvault_backup.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [ValidateSet('capture','verify','both')]    [string]$Which  = 'both',
  [string]$CaptureTaskName = 'FakLogvaultCapture',
  [string]$VerifyTaskName  = 'FakLogvaultVerify',
  [string]$Workspace       = $(Split-Path -Parent $PSScriptRoot),
  [string]$VaultDir        = '',          # '' => fak logvault default ($FAK_LOG_VAULT, else <repo-parent>/fak-log-vault)
  [string]$SlackChannel    = '',          # '' => resolve from env / .env.slack.local
  [int]$CaptureIntervalHours = 24,        # daily: must stay well under the 7-day GC floor
  [int]$VerifyIntervalDays   = 7,         # weekly bitrot watch
  [int]$VerifySample         = 250,       # mirrors re-hashed per verify tick (0 = all)
  [string]$FakExe          = '',          # explicit fak binary path; '' => Get-Command fak, then ~/go/bin
  [switch]$Live                            # without -Live neither tick passes -notify-slack
)
$ErrorActionPreference = 'Stop'

$taskNames = @()
if ($Which -eq 'capture' -or $Which -eq 'both') { $taskNames += $CaptureTaskName }
if ($Which -eq 'verify'  -or $Which -eq 'both') { $taskNames += $VerifyTaskName }

if ($Action -eq 'status') {
  foreach ($name in $taskNames) {
    $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if (-not $t) { Write-Output "NOT INSTALLED ($name)"; continue }
    $i = Get-ScheduledTaskInfo -TaskName $name
    $a = ($t.Actions | Select-Object -First 1).Arguments
    $modeStr = if ($a -match '-notify-slack') { 'LIVE (slack)' } else { 'LOCAL-ONLY' }
    Write-Output "$name  State=$($t.State) mode=$modeStr LastRun=$($i.LastRunTime) LastResult=$($i.LastTaskResult) NextRun=$($i.NextRunTime)"
  }
  return
}
if ($Action -eq 'remove') {
  foreach ($name in $taskNames) {
    schtasks /Delete /TN $name /F 2>$null | Out-Null
    Write-Output "removed $name"
  }
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
if (-not $fak) { throw "'fak' not found on PATH or in ~/go/bin -- run 'go install ./cmd/fak' first" }

# Build the wrapped tick argv per the #765 cron-emit convention: `fak loop run --loop <id>
# --source task-scheduler -- fak logvault <verb> ...`. The tick's own `fak` token is resolved
# on PATH by the guard child at fire time (same as `fak cron emit`'s default --fak-bin fak), so
# it does not need to be the absolute -Execute path.
function New-LogvaultTickArgs([string]$loopId, [string[]]$verbArgs) {
  $args = @('loop', 'run', '--loop', $loopId, '--source', 'task-scheduler', '--', 'fak') + $verbArgs
  if ($VaultDir)     { $args += @('-vault', $VaultDir) }
  if ($Live)         { $args += '-notify-slack' }
  if ($Live -and $SlackChannel) { $args += @('-slack-channel', $SlackChannel) }
  return $args
}

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
              -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)

function Install-LogvaultTask([string]$name, [string]$loopId, [string[]]$verbArgs, [TimeSpan]$interval) {
  $tickArgs = New-LogvaultTickArgs $loopId $verbArgs
  $exeArgs  = ($tickArgs -join ' ')
  $trigger  = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
                -RepetitionInterval $interval -RepetitionDuration (New-TimeSpan -Days 3650)

  # Preferred: S4U (session 0, windowless, runs AS THIS USER even when not logged in). S4U
  # registration requires elevation, so when it is denied (a non-admin install) fall back to a
  # current-user Interactive task via conhost --headless so no console window flashes per tick.
  try {
    $taskAction = New-ScheduledTaskAction -Execute $fak -Argument $exeArgs -WorkingDirectory $Workspace
    $principal  = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $taskAction -Trigger $trigger `
                 -Principal $principal -Settings $settings -Force -ErrorAction Stop | Out-Null
    $principalKind = 'S4U (session 0)'
  } catch {
    $headlessArgs = "--headless `"$fak`" $exeArgs"
    $taskAction = New-ScheduledTaskAction -Execute 'conhost.exe' -Argument $headlessArgs -WorkingDirectory $Workspace
    $principal  = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $taskAction -Trigger $trigger `
                 -Principal $principal -Settings $settings -Force | Out-Null
    $principalKind = "Interactive (non-elevated; conhost --headless)"
  }
  return $principalKind
}

$capturePrincipal = $null
$verifyPrincipal  = $null
if ($Which -eq 'capture' -or $Which -eq 'both') {
  $capturePrincipal = Install-LogvaultTask $CaptureTaskName 'logvault-capture' @('logvault', 'capture') `
                         (New-TimeSpan -Hours $CaptureIntervalHours)
}
if ($Which -eq 'verify' -or $Which -eq 'both') {
  $verifyPrincipal = Install-LogvaultTask $VerifyTaskName 'logvault-verify' @('logvault', 'verify', '-sample', "$VerifySample") `
                        (New-TimeSpan -Days $VerifyIntervalDays)
}

$runMode = if ($Live) { 'LIVE (also enqueues a slack digest)' } else { 'LOCAL-ONLY (no -notify-slack; add -Live to arm it)' }
Write-Output "mode: $runMode"
if ($capturePrincipal) {
  Write-Output "installed $CaptureTaskName -- every $CaptureIntervalHours h, $capturePrincipal"
}
if ($verifyPrincipal) {
  Write-Output "installed $VerifyTaskName -- every $VerifyIntervalDays d, sample=$VerifySample, $verifyPrincipal"
}
Write-Output "runs:     $fak (loop run --loop logvault-capture|logvault-verify --source task-scheduler -- fak logvault ...)"
Write-Output "vault:    $(if ($VaultDir) { $VaultDir } else { '$FAK_LOG_VAULT, else <repo-parent>/fak-log-vault' })"
Write-Output "check status any time:  .\tools\register_logvault_backup.ps1 -Action status"
if (-not $Live) {
  Write-Output "to arm slack digests later:  .\tools\register_logvault_backup.ps1 -Live"
}
