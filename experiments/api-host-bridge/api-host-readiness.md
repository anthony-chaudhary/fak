# API-Host Readiness Probe

> Current-state `/models` probe for compatible API hosts.

## Summary

- Targets: 13
- Models confirmed: 1
- Auth env missing: 11
- Auth required: 0
- Access denied: 1
- Billing required: 0
- Transient transport: 0
- Invalid targets: 0
- Readiness gate: yes

| target | status | HTTP | models |
|---|---|---:|---:|
| `gemini_openai_compatible` | AUTH_ENV_MISSING |  | 0 |
| `glama_gateway` | MODELS_CONFIRMED | 200 | 25 |
| `pollinations_no_key` | ACCESS_DENIED | 403 | 0 |
| `openai_api` | AUTH_ENV_MISSING |  | 0 |
| `xai_api` | AUTH_ENV_MISSING |  | 0 |
| `openrouter_gateway` | AUTH_ENV_MISSING |  | 0 |
| `groq_openai` | AUTH_ENV_MISSING |  | 0 |
| `together_openai` | AUTH_ENV_MISSING |  | 0 |
| `mistral_openai` | AUTH_ENV_MISSING |  | 0 |
| `deepseek_openai` | AUTH_ENV_MISSING |  | 0 |
| `fireworks_openai` | AUTH_ENV_MISSING |  | 0 |
| `perplexity_openai` | AUTH_ENV_MISSING |  | 0 |
| `cerebras_openai` | AUTH_ENV_MISSING |  | 0 |
