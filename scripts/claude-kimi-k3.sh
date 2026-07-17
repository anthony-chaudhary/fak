#!/usr/bin/env bash
# Claude Code harness for Moonshot Kimi K3 through fak:
# Claude Code -> fak serve (/v1/messages) -> Moonshot /v1 -> Kimi K3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FAK_DOGFOOD_BACKEND="${FAK_DOGFOOD_BACKEND:-openai}"
export FAK_DOGFOOD_BASE_URL="${FAK_DOGFOOD_BASE_URL:-${FAK_KIMI_K3_BASE_URL:-https://api.moonshot.ai/v1}}"
export FAK_DOGFOOD_MODEL="${FAK_DOGFOOD_MODEL:-${FAK_KIMI_K3_MODEL:-kimi-k3}}"
export FAK_DOGFOOD_API_KEY_ENV="${FAK_DOGFOOD_API_KEY_ENV:-${FAK_KIMI_K3_API_KEY_ENV:-MOONSHOT_API_KEY}}"
export FAK_DOGFOOD_ACCOUNT="${FAK_DOGFOOD_ACCOUNT:-${FAK_KIMI_K3_ACCOUNT:-faklocal}}"

exec "$SCRIPT_DIR/dogfood-claude.sh" "$@"
