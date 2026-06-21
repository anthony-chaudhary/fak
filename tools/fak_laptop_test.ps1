<#
fak_laptop_test.ps1 - PowerShell entry point for Anthony's laptop CPU/NVIDIA lanes.

Examples:
  .\tools\fak_laptop_test.ps1 accept
  .\tools\fak_laptop_test.ps1 accept --cpu-only
  .\tools\fak_laptop_test.ps1 check --require-nvidia
  .\tools\fak_laptop_test.ps1 cpu --smoke
  .\tools\fak_laptop_test.ps1 nvidia --setup
  .\tools\fak_laptop_test.ps1 verify
  .\tools\fak_laptop_test.ps1 status

This wrapper only locates Python and delegates to tools\fak_laptop_test.py. The
Python runner owns lane selection and WSL/CUDA command construction.
#>
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Rest)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $ScriptDir 'fak_laptop_test.py'
if (-not (Test-Path $Runner)) {
  throw "runner not found: $Runner"
}

function Get-WorkingPython([string] $Name, [string[]] $PrefixArgs) {
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if (-not $cmd) {
    return $null
  }
  try {
    & $cmd.Source @PrefixArgs --version *> $null
  } catch {
    return $null
  }
  if ($LASTEXITCODE -ne 0) {
    return $null
  }
  return @{
    Exe = $cmd.Source
    PrefixArgs = $PrefixArgs
  }
}

$Python = $null
foreach ($candidate in @(
  @{ Name = 'py'; PrefixArgs = @('-3') },
  @{ Name = 'python'; PrefixArgs = @() },
  @{ Name = 'python3'; PrefixArgs = @() }
)) {
  $Python = Get-WorkingPython $candidate.Name $candidate.PrefixArgs
  if ($Python) {
    break
  }
}
if (-not $Python) {
  throw "no Python on PATH (tried python, python3, py)"
}

& $Python.Exe @($Python.PrefixArgs) $Runner @Rest
exit $LASTEXITCODE
