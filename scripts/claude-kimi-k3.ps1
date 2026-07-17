<#
.SYNOPSIS
  Claude Code harness for Moonshot Kimi K3 through fak.

.DESCRIPTION
  Sets the Moonshot OpenAI-compatible upstream knobs for scripts/dogfood-claude.ps1.
  Required by default: MOONSHOT_API_KEY.

.EXAMPLE
  .\scripts\claude-kimi-k3.ps1 --probe "Reply with exactly: pong"
#>
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dogfood = Join-Path $ScriptDir 'dogfood-claude.ps1'

if (-not $env:FAK_DOGFOOD_BACKEND) { $env:FAK_DOGFOOD_BACKEND = 'openai' }
if (-not $env:FAK_DOGFOOD_BASE_URL) {
  $env:FAK_DOGFOOD_BASE_URL = if ($env:FAK_KIMI_K3_BASE_URL) { $env:FAK_KIMI_K3_BASE_URL } else { 'https://api.moonshot.ai/v1' }
}
if (-not $env:FAK_DOGFOOD_MODEL) {
  $env:FAK_DOGFOOD_MODEL = if ($env:FAK_KIMI_K3_MODEL) { $env:FAK_KIMI_K3_MODEL } else { 'kimi-k3' }
}
if (-not $env:FAK_DOGFOOD_API_KEY_ENV) {
  $env:FAK_DOGFOOD_API_KEY_ENV = if ($env:FAK_KIMI_K3_API_KEY_ENV) { $env:FAK_KIMI_K3_API_KEY_ENV } else { 'MOONSHOT_API_KEY' }
}
if (-not $env:FAK_DOGFOOD_ACCOUNT) {
  $env:FAK_DOGFOOD_ACCOUNT = if ($env:FAK_KIMI_K3_ACCOUNT) { $env:FAK_KIMI_K3_ACCOUNT } else { 'faklocal' }
}

& powershell -NoProfile -ExecutionPolicy Bypass -File $Dogfood @args
exit $LASTEXITCODE
