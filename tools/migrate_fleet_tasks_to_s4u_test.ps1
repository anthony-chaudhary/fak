$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $PSScriptRoot 'migrate_fleet_tasks_to_s4u.ps1'
$text = Get-Content -LiteralPath $scriptPath -Raw
$required = @(
  '[switch]$FailingOnly',
  'if ($FailingOnly)',
  'all Interactive tasks are selected by default'
)
foreach ($needle in $required) {
  if (-not $text.Contains($needle)) {
    throw "migration script missing broad-default contract: $needle"
  }
}
if ($text.Contains('if (-not $All)')) {
  throw 'migration still narrows its default to failed tasks; desktop-visible healthy tasks would survive'
}
Write-Output 'PASS: Interactive Scheduled Tasks are migrated by default'
