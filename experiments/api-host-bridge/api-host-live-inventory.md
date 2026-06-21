# API-Host Live Inventory

> Evidence inventory for live API-host bridge runs and typed external-host blockers.

## Summary

- Live frontier successes: 2
- Local OpenAI-compatible successes: 1
- Billing-required hosts: 1
- Auth-required hosts: 1
- Incomplete or unclassified proofs: 0
- Live inventory gate: yes

## Proofs

| proof | status | evidence |
|---|---|---|
| `gemini_openai_compatible_turntax` | LIVE_CONFIRMED | `fak/experiments/agent-live/turntax-injection-live.json` |
| `gemini_live_safety_floor` | LIVE_CONFIRMED | `5 rows` |
| `glama_gateway_billing_state` | BILLING_REQUIRED | `https://gateway.glama.ai/v1` |
| `pollinations_tool_call_auth_state` | AUTH_REQUIRED | `https://gen.pollinations.ai/v1` |
| `local_openai_compatible_shims` | LOCAL_OPENAI_COMPAT_CONFIRMED | `4 rows` |
