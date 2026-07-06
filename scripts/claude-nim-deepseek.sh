#!/usr/bin/env bash
# Claude Code harness for NVIDIA NIM DeepSeek V4 Pro through fak:
# Claude Code -> fak serve (/v1/messages) -> NVIDIA NIM /v1 -> DeepSeek V4 Pro.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FAK_DOGFOOD_BACKEND="${FAK_DOGFOOD_BACKEND:-openai}"
export FAK_DOGFOOD_BASE_URL="${FAK_DOGFOOD_BASE_URL:-${FAK_NIM_DEEPSEEK_BASE_URL:-https://integrate.api.nvidia.com/v1}}"
export FAK_DOGFOOD_MODEL="${FAK_DOGFOOD_MODEL:-${FAK_NIM_DEEPSEEK_MODEL:-deepseek-ai/deepseek-v4-pro}}"
export FAK_DOGFOOD_API_KEY_ENV="${FAK_DOGFOOD_API_KEY_ENV:-${FAK_NIM_DEEPSEEK_API_KEY_ENV:-NVIDIA_API_KEY}}"
export FAK_DOGFOOD_ACCOUNT="${FAK_DOGFOOD_ACCOUNT:-${FAK_NIM_DEEPSEEK_ACCOUNT:-faklocal}}"

exec "$SCRIPT_DIR/dogfood-claude.sh" "$@"
