<#
Build and/or Taildrop the latest prebuilt benchmark-node payload.

Examples:
  .\tools\send_prebuilt_node_payload.ps1 -Build
  .\tools\send_prebuilt_node_payload.ps1 -Archive C:\path\payload.tgz -Target worker-a
#>
[CmdletBinding()]
param(
  [string]$FleetDir = 'C:\work\fleet',
  [string]$Target = 'benchmark-driver',
  [string]$Archive = '',
  [switch]$Build,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

Push-Location $FleetDir
try {
  if ($Build) {
    & bash tools/build_prebuilt_node_payload.sh
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }

  if (-not $Archive) {
    $root = Join-Path $FleetDir 'tools\_registry\prebuilt-node'
    $latest = Get-ChildItem $root -Recurse -Filter 'fak-node-payload-*.tgz' |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1
    if (-not $latest) { throw "no prebuilt payload archive found under $root" }
    $Archive = $latest.FullName
  }

  if (-not (Test-Path $Archive)) { throw "archive not found: $Archive" }
  $hash = Get-FileHash -Algorithm SHA256 $Archive
  Write-Host "payload: $Archive"
  Write-Host "sha256:  $($hash.Hash)"
  Write-Host "target:  $Target"

  if ($DryRun) { exit 0 }

  $ts = (Get-Command tailscale -ErrorAction SilentlyContinue).Source
  if (-not $ts) { $ts = 'C:\Program Files\Tailscale\tailscale.exe' }
  if (-not (Test-Path $ts)) { throw "tailscale CLI not found" }

  & $ts file cp $Archive "${Target}:"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
  Pop-Location
}
