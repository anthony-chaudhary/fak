# OpenAI hosted live pilot

- generated: `2026-06-25T15:40:08.998556Z`
- status: **`PASS`**
- model: `gpt-5.5`
- auth_mode: `codex-login`
- auth_source: `codex_login`
- hosted_openai_ready: `True`
- platform_api_ready: `False`
- codex_login_ready: `True`
- agents_sdk_ready: `False`

## Guard

- status: `PASS`
- dangerous tool: `git_push` -> `DENY` / `POLICY_BLOCK`
- useful tool: `git_status` -> `ALLOW`

## Hosted OpenAI

- status: `PASS`
- auth_source: `codex_login`
- response_id_present: `None`
- codex_exec_exit_code: `0`
- contains_expected_marker: `True`
- output_text_sha256: `73705eac30b4a6796a90adca93824a8987db9e942e783307ed70f60e0fb4403a`
- json_event_count: `5`

## Blockers

- openai-agents distribution is not installed
- importable agents module is not an installed OpenAI Agents SDK distribution

## Privacy

This artifact records verdict metadata, hosted-output hashes, and Codex exec event counts only. It never writes API key values, Codex token values, raw hosted OpenAI response text, or request payloads.
