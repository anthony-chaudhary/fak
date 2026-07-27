<#
Shared helpers for Windows Scheduled Task installers that route a child command
through `fak loop run`.

The helpers keep Task Scheduler's Execute and Argument fields split. That avoids
PowerShell -Command quoting traps while still letting fak own the loop ledger row
around the child process.
#>

function Quote-FakLoopTaskArg {
  param([string]$Value)
  if ($Value -notmatch '[\s"]') { return $Value }
  return '"' + ($Value -replace '"', '\"') + '"'
}

function Join-FakLoopTaskArgs {
  param([string[]]$ArgumentList)
  return (($ArgumentList | ForEach-Object { Quote-FakLoopTaskArg $_ }) -join ' ')
}

function Test-FakLoopAction {
  param(
    [string]$Execute,
    [string[]]$PrefixArgs = @()
  )
  $probeLedger = Join-Path ([System.IO.Path]::GetTempPath()) ("fak-loop-probe-$([guid]::NewGuid().ToString('n')).jsonl")
  try {
    & $Execute @($PrefixArgs + @('loop','status','--ledger',$probeLedger,'--json')) *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

function Resolve-FakLoopAction {
  param(
    [string]$Workspace,
    [string]$FakExe = $env:FAK_BIN
  )
  $candidates = @()
  if ($FakExe) { $candidates += $FakExe }
  $repoExe = Join-Path $Workspace 'fak.exe'
  if (Test-Path $repoExe) { $candidates += $repoExe }
  $pathFak = Get-Command fak -ErrorAction SilentlyContinue
  if ($pathFak) { $candidates += $pathFak.Source }

  foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (Test-FakLoopAction -Execute $candidate) {
      return [pscustomobject]@{ Execute = $candidate; PrefixArgs = @() }
    }
  }

  $go = Get-Command go -ErrorAction SilentlyContinue
  if ($go -and (Test-FakLoopAction -Execute $go.Source -PrefixArgs @('run','./cmd/fak'))) {
    return [pscustomobject]@{ Execute = $go.Source; PrefixArgs = @('run','./cmd/fak') }
  }

  throw "no usable fak loop command found; set -FakExe, put fak on PATH, or install Go"
}

function Resolve-FakLoopPowerShellHost {
  <#
  Resolve a PowerShell host path that SURVIVES a PowerShell upgrade.

  A Scheduled Task action freezes whatever path it is handed at INSTALL time, but the
  Microsoft Store build of pwsh lives under a version-stamped directory:
  ...\WindowsApps\Microsoft.PowerShell_<VERSION>_x64__8wekyb3d8bbwe\pwsh.exe. Every pwsh
  update renames that directory, so a task still holding the old absolute path dies with
  exit 1 on every single fire -- silently, with no diagnostic beyond the exit code, and
  with the loop ledger recording only `command = pwsh.exe` so the row cannot even be
  reproduced. That is exactly how scout-loop/task-scheduler went dark: pinned to a
  7.6.3.0 that no longer existed while 7.6.4.0 was installed.

  `Get-Command pwsh` returns that version-stamped path, so it is precisely the wrong
  thing to freeze into a task. Prefer the paths that are stable across upgrades: a
  per-machine MSI install, then the App Execution Alias under %LOCALAPPDATA% that a
  Store install keeps pointed at the current version. If only the version-stamped path
  exists we still return it -- refusing to install is worse than installing -- but we
  warn, because that task WILL break on the next pwsh update.
  #>
  param()
  foreach ($candidate in @(
    (Join-Path $env:ProgramFiles 'PowerShell\7\pwsh.exe'),
    (Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\pwsh.exe')
  )) {
    if ($candidate -and (Test-Path $candidate)) { return $candidate }
  }

  $resolved = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
  if ($resolved) {
    if ($resolved -match 'WindowsApps\\Microsoft\.PowerShell_[\d\.]+') {
      Write-Warning ("the only pwsh found is the version-stamped Store path '$resolved'; " +
                     "this task will stop firing the next time PowerShell updates. Enable the " +
                     "pwsh App Execution Alias (Settings > Apps > Advanced app settings > App " +
                     "execution aliases) and re-run this installer.")
    }
    return $resolved
  }

  $windowsPowerShell = (Get-Command powershell -ErrorAction SilentlyContinue).Source
  if ($windowsPowerShell) { return $windowsPowerShell }
  throw "no PowerShell host (pwsh/powershell) found on PATH"
}

function New-FakLoopScheduledTaskAction {
  param(
    [string]$Workspace,
    [string]$LoopId,
    [string[]]$ChildArgs,
    [string]$FakExe = $env:FAK_BIN,
    [string]$Ledger = '',
    [string]$Source = 'task-scheduler',
    [string]$Principal = $env:USERNAME
  )
  $fak = Resolve-FakLoopAction -Workspace $Workspace -FakExe $FakExe
  if (-not $Ledger) { $Ledger = Join-Path $Workspace '.fak\loops.jsonl' }
  $loopArgs = @()
  $loopArgs += [string[]]$fak.PrefixArgs
  $loopArgs += @(
    'loop','run',
    '--ledger', $Ledger,
    '--loop', $LoopId,
    '--source', $Source,
    '--principal', $Principal,
    '--'
  )
  $loopArgs += $ChildArgs
  return New-ScheduledTaskAction -Execute $fak.Execute -Argument (Join-FakLoopTaskArgs $loopArgs) -WorkingDirectory $Workspace
}
