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
