# OpenAI hosted live proof prerequisites

- generated: `2026-06-25T15:39:22.721376Z`
- status: **`PARTIAL`**
- hosted_openai_ready: `True`
- platform_api_ready: `False`
- codex_login_ready: `True`
- agents_sdk_ready: `False`

## Evidence

- OPENAI_API_KEY_set: `False`
- OPENAI_BASE_URL_set: `False`
- Codex auth_mode: `chatgpt`
- Codex auth_json_present: `True`
- Codex CLI present: `True`
- Codex access_token_present: `True`
- Codex refresh_token_present: `True`
- Codex access_token_exp_iso: `2026-06-29T19:53:19Z`
- Codex access_token_expired: `False`
- openai package: `2.41.0`
- openai-agents distribution: `None`
- agents distribution: `None`
- agents module file: `C:\work\job\agents\__init__.py`
- agents.tracing installed: `False`

## Blockers

- openai-agents distribution is not installed
- importable agents module is not an installed OpenAI Agents SDK distribution

## Privacy

This audit records only presence booleans, package versions, module paths, sanitized Codex auth state, and token-expiry metadata. It never writes API key values, Codex token values, raw account identifiers, or request payloads.
