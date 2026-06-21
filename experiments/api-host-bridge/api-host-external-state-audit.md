# API-Host External State Audit

> Credential, billing, readiness, and retry state for API-host roster targets.

## Summary

- Roster targets: 13
- Env present: 1
- Env missing: 11
- No auth declared: 1
- Live confirmed: 1
- Ready for live run: 0
- Blocked auth: 1
- Blocked billing: 1
- Unprobed templates: 0
- Artifact errors: 0
- External-state audit gate: yes

| target | credential | external state | next evidence |
|---|---|---|---|
| `gemini_openai_compatible` | ENV_MISSING | LIVE_CONFIRMED | Committed evidence already confirms a live bridge run for this host/base URL. |
| `glama_gateway` | ENV_PRESENT | BLOCKED_BILLING | Attach billing/payment method for the account behind GLAMA_API_KEY. |
| `pollinations_no_key` | NO_AUTH_DECLARED | BLOCKED_AUTH | Configure a valid bearer token or access path via POLLINATIONS_API_KEY. |
| `openai_api` | ENV_MISSING | NEEDS_CREDENTIAL | Set OPENAI_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `xai_api` | ENV_MISSING | NEEDS_CREDENTIAL | Set XAI_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `openrouter_gateway` | ENV_MISSING | NEEDS_CREDENTIAL | Set OPENROUTER_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `groq_openai` | ENV_MISSING | NEEDS_CREDENTIAL | Set GROQ_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `together_openai` | ENV_MISSING | NEEDS_CREDENTIAL | Set TOGETHER_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `mistral_openai` | ENV_MISSING | NEEDS_CREDENTIAL | Set MISTRAL_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `deepseek_openai` | ENV_MISSING | NEEDS_CREDENTIAL | Set DEEPSEEK_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `fireworks_openai` | ENV_MISSING | NEEDS_CREDENTIAL | Set FIREWORKS_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `perplexity_openai` | ENV_MISSING | NEEDS_CREDENTIAL | Set PERPLEXITY_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `cerebras_openai` | ENV_MISSING | NEEDS_CREDENTIAL | Set CEREBRAS_API_KEY, then rerun readiness, acceptance, and live smoke. |
