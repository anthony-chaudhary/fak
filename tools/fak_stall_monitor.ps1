<#
.SYNOPSIS
  Durable background self-monitor for the low-usage machine stalls (issue #3153).
  Wraps `fak stallscan --watch` so recurrence is caught, logged, and — optionally
  — auto-mitigated, without a human watching.

.DESCRIPTION
  The box intermittently locks up while CPU/RAM/disk read low; the cause is
  kernel-path CHURN (soft-fault + spawn storms) that no usage meter shows. This
  monitor samples the churn signals on an interval, appends every fingerprint to
  a rolling JSONL, prints a line on each elevated/stall verdict, and (with
  -AutoMitigate) fires a bounded, rate-limited remediation when stalls persist.

  Wire it to run in the background by default via a Windows Scheduled Task (see
  -Install), the same way the fleet resume watchdog is installed. It is cheap by
  construction: `fak stallscan` does one counter batch + one process snapshot per
  interval (default 20s).

.PARAMETER Interval
  Sample interval (default 20s). Do not go below ~5s — enumeration has a cost.

.PARAMETER Log
  Rolling JSONL path. Default: $env:LOCALAPPDATA\Fleet\stallscan.jsonl (matches
  `fak stallscan`'s own default so one file is the shared source of truth).

.PARAMETER AutoMitigate
  On a run of consecutive stalls, run tools/host_stall_mitigations.ps1 -Apply to
  tame the non-fak daemon floor. Rate-limited to once per -MitigateCooldownMin.

.PARAMETER MitigateCooldownMin
  Minimum minutes between auto-mitigations (default 60).

.PARAMETER Install
  Register a Scheduled Task 'FakStallMonitor' that runs this at logon and keeps
  it alive. Prints the UNDO command. Requires elevation.

.EXAMPLE
  pwsh -File tools/fak_stall_monitor.ps1                 # watch in this console
.EXAMPLE
  pwsh -File tools/fak_stall_monitor.ps1 -AutoMitigate   # watch + auto-tame floor
.EXAMPLE
  pwsh -File tools/fak_stall_monitor.ps1 -Install        # run in background by default
#>
[CmdletBinding()]
param(
  [int]$Interval = 20,
  [string]$Log = "$env:LOCALAPPDATA\Fleet\stallscan.jsonl",
  [switch]$AutoMitigate,
  [int]$MitigateCooldownMin = 60,
  [switch]$Install,
  [int]$StallRunToMitigate = 3
)

$ErrorActionPreference = 'Continue'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent $scriptDir

function Resolve-Fak {
  $c = Get-Command fak -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  foreach ($p in @("$repoRoot\fak.exe", "$env:USERPROFILE\go\bin\fak.exe")) {
    if (Test-Path $p) { return $p }
  }
  return 'fak'
}

if ($Install) {
  $ErrorActionPreference = 'Stop'
  $fak = Resolve-Fak
  $fakCmd = Get-Command $fak -ErrorAction SilentlyContinue
  if ($fakCmd -and -not $fakCmd.Source) { $fakCmd = $null }
  if (-not $fakCmd -or -not (Test-Path -LiteralPath $fakCmd.Source -PathType Leaf)) {
    throw 'fak executable not found; install/build fak and put it on PATH before -Install'
  }
  $fak = $fakCmd.Source
  $fakDir = Split-Path -Parent $fak
  if (-not ($env:PATH -split [IO.Path]::PathSeparator | Where-Object { $_.TrimEnd('\') -ieq $fakDir.TrimEnd('\') })) {
    throw "fak executable directory is not on PATH: $fakDir"
  }
  $pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
  if (-not $pwsh) { $pwsh = 'powershell.exe' }
  $self = $MyInvocation.MyCommand.Path
  $args = "-NoProfile -ExecutionPolicy Bypass -File `"$self`" -Interval $Interval"
  if ($AutoMitigate) { $args += ' -AutoMitigate' }
  # Route the pwsh child through `conhost.exe --headless` so the logon task never
  # flashes a console window on the desktop, and pin the task to an off-desktop
  # S4U principal (session 0) as a second guard — both are the windowless recipes
  # the popup gate (internal/windowgate) accepts.
  $conhost = "$env:SystemRoot\System32\conhost.exe"
  $action  = New-ScheduledTaskAction -Execute $conhost -Argument "--headless `"$pwsh`" $args"
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
  # AtStartup, not AtLogOn: the control plane is born independently of RDP/WT and
  # survives TermService tearing an interactive logon session down.
  $trigger = New-ScheduledTaskTrigger -AtStartup
  $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
  try {
    Register-ScheduledTask -TaskName 'FakStallMonitor' -Action $action -Principal $principal -Trigger $trigger -Settings $settings -Force -ErrorAction Stop | Out-Null
  } catch {
    throw "FakStallMonitor registration failed (run an elevated PowerShell): $($_.Exception.Message)"
  }

  # Session-0 cannot activate an AppX WT window in a user desktop. Keep the
  # long-lived watchdog S4U and register only this on-demand UI adapter as
  # InteractiveToken. The broker drains a typed durable spool and does no sensing,
  # policy, retry, or ownership work.
  $brokerSpool = Join-Path (Split-Path -Parent $Log) 'relaunch'
  $brokerArgs = "host-relaunch-broker --dir `"$brokerSpool`""
  $brokerAction = New-ScheduledTaskAction -Execute $fak -Argument $brokerArgs
  $brokerPrincipal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
  $brokerTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
  $brokerSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
  try {
    Register-ScheduledTask -TaskName 'FakHostRelaunchBroker' -Action $brokerAction -Principal $brokerPrincipal -Trigger $brokerTrigger -Settings $brokerSettings -Force -ErrorAction Stop | Out-Null
  } catch {
    throw "FakHostRelaunchBroker registration failed (run an elevated PowerShell): $($_.Exception.Message)"
  }
  $watchdog = Get-ScheduledTask -TaskName 'FakStallMonitor' -ErrorAction Stop
  $broker = Get-ScheduledTask -TaskName 'FakHostRelaunchBroker' -ErrorAction Stop
  if ($watchdog.Principal.LogonType -ne 'S4U') { throw "FakStallMonitor principal is $($watchdog.Principal.LogonType), want S4U" }
  if ($broker.Principal.LogonType -ne 'InteractiveToken') { throw "FakHostRelaunchBroker principal is $($broker.Principal.LogonType), want InteractiveToken" }
  if ($broker.Actions.Execute -ne $fak -or $broker.Actions.Arguments -notmatch '^host-relaunch-broker(?:\s|$)') { throw 'FakHostRelaunchBroker action read-back mismatch' }
  Write-Host "[stall-mon] installed FakStallMonitor (AtStartup/S4U) + on-demand FakHostRelaunchBroker (InteractiveToken adapter)."
  Write-Host "[stall-mon] UNDO: Unregister-ScheduledTask -TaskName 'FakStallMonitor' -Confirm:`$false; Unregister-ScheduledTask -TaskName 'FakHostRelaunchBroker' -Confirm:`$false"
  return
}

$fak = Resolve-Fak

$hostCrashLog = Join-Path (Split-Path -Parent $Log) 'host-crashes.jsonl'

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) -ErrorAction SilentlyContinue | Out-Null
Write-Host "[stall-mon] fak=$fak interval=${Interval}s log=$Log hostCrashLog=$hostCrashLog autoMitigate=$AutoMitigate"

$stallRun = 0
$lastMitigate = [DateTime]::MinValue

while ($true) {
  # Poll Event 1000 in the same always-on task. --once avoids orphan watcher
  # processes across task restarts; the durable signal ledger makes overlap safe.
  & $fak host-crash --once --since 5m --log $hostCrashLog --resurrect 2>&1 | ForEach-Object {
    Write-Host "[host-crash] $_"
  }
  # One snapshot via the shipped classifier; JSON so we can read the verdict.
  $raw = & $fak stallscan --json 2>$null
  $rc = $LASTEXITCODE
  $level = 'unknown'; $cause = ''
  try {
    $obj = $raw | ConvertFrom-Json -ErrorAction Stop
    $level = $obj.verdict.level; $cause = $obj.verdict.cause
    # Mirror into the shared rolling log (fak stallscan --watch would do this too;
    # here we drive the snapshot ourselves so we can react).
    #
    # ONE LINE PER RECORD IS LOAD-BEARING. `fak stallscan --json` pretty-prints, and
    # PowerShell captures a native command's stdout as an ARRAY OF LINES -- so the
    # obvious `$raw -replace "\r?\n",' '` runs ELEMENT-WISE over lines that each
    # already contain no newline (a no-op), and Add-Content then writes every element
    # on its own line. That silently turns this "JSONL" ledger into pretty-printed
    # multi-line JSON, and the admission-path reader (dispatchPreflightChurn ->
    # dispatchLastLine) parses only the LAST line -- a bare `}` -- fails, and abstains.
    # The churn gate then reads as "calm host" forever instead of "not measured".
    # Join FIRST, then collapse: JSON is whitespace-insensitive, so the joined record
    # stays valid on a single line.
    Add-Content -Path $Log -Value (((@($raw) -join ' ') -replace "`r?`n", ' ')) -ErrorAction SilentlyContinue
  } catch {
    # rc==3 from snapshot mode also signals a stall even if JSON parse hiccups.
    if ($rc -eq 3) { $level = 'stall' }
  }

  if ($level -eq 'stall') {
    $stallRun++
    Write-Host ("{0}  STALL (run {1})  cause={2}" -f (Get-Date -Format HH:mm:ss), $stallRun, $cause)
  } elseif ($level -eq 'elevated') {
    Write-Host ("{0}  elevated  cause={1}" -f (Get-Date -Format HH:mm:ss), $cause)
    $stallRun = 0
  } else {
    $stallRun = 0
  }

  if ($AutoMitigate -and $stallRun -ge $StallRunToMitigate) {
    $mins = ([DateTime]::Now - $lastMitigate).TotalMinutes
    if ($mins -ge $MitigateCooldownMin) {
      Write-Host "[stall-mon] persistent stalls -> running host_stall_mitigations.ps1 -Apply"
      & pwsh -NoProfile -ExecutionPolicy Bypass -File "$scriptDir\host_stall_mitigations.ps1" -Apply
      $lastMitigate = [DateTime]::Now
      $stallRun = 0
    } else {
      Write-Host ("[stall-mon] would mitigate but in cooldown ({0:N0}/{1} min)" -f $mins, $MitigateCooldownMin)
    }
  }

  Start-Sleep -Seconds $Interval
}
