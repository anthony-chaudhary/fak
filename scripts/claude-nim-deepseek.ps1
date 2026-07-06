<#
.SYNOPSIS
  Claude Code harness for NVIDIA NIM DeepSeek V4 Pro through fak.

.DESCRIPTION
  Sets the OpenAI-compatible upstream knobs for scripts/dogfood-claude.ps1, then
  delegates to that launcher. The live path is:

    Claude Code -> fak serve (/v1/messages) -> NVIDIA NIM /v1 -> DeepSeek V4 Pro

  Required by default: NVIDIA_API_KEY.

.EXAMPLE
  .\scripts\claude-nim-deepseek.ps1 --probe "Reply with exactly: pong"
#>
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dogfood = Join-Path $ScriptDir 'dogfood-claude.ps1'

if (-not $env:FAK_DOGFOOD_BACKEND) {
  $env:FAK_DOGFOOD_BACKEND = 'openai'
}
if (-not $env:FAK_DOGFOOD_BASE_URL) {
  $env:FAK_DOGFOOD_BASE_URL = if ($env:FAK_NIM_DEEPSEEK_BASE_URL) { $env:FAK_NIM_DEEPSEEK_BASE_URL } else { 'https://integrate.api.nvidia.com/v1' }
}
if (-not $env:FAK_DOGFOOD_MODEL) {
  $env:FAK_DOGFOOD_MODEL = if ($env:FAK_NIM_DEEPSEEK_MODEL) { $env:FAK_NIM_DEEPSEEK_MODEL } else { 'deepseek-ai/deepseek-v4-pro' }
}
if (-not $env:FAK_DOGFOOD_API_KEY_ENV) {
  $env:FAK_DOGFOOD_API_KEY_ENV = if ($env:FAK_NIM_DEEPSEEK_API_KEY_ENV) { $env:FAK_NIM_DEEPSEEK_API_KEY_ENV } else { 'NVIDIA_API_KEY' }
}
if (-not $env:FAK_DOGFOOD_ACCOUNT) {
  $env:FAK_DOGFOOD_ACCOUNT = if ($env:FAK_NIM_DEEPSEEK_ACCOUNT) { $env:FAK_NIM_DEEPSEEK_ACCOUNT } else { 'faklocal' }
}

& powershell -NoProfile -ExecutionPolicy Bypass -File $Dogfood @args
exit $LASTEXITCODE
