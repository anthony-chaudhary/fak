# API-Host Acceptance Probe

> Candidate-host gate for the scoped FAK/DOS API-host bridge contract.

## Summary

- Targets: 13
- Ready for live bridge run: 0
- Live bridge confirmed: 0
- Wire supported but unprobed: 0
- Typed external blockers: 13
- Unsupported wire: 0
- Invalid targets: 0
- Sweep artifact errors: 0
- Acceptance gate: yes

| target | provider | class | status | readiness |
|---|---|---|---|---|
| `gemini_openai_compatible` | `openai-compatible` | `openai_compatible_upstream` | NEEDS_AUTH_ENV | AUTH_ENV_MISSING |
| `glama_gateway` | `openai-compatible` | `openai_compatible_upstream` | BILLING_REQUIRED | MODELS_CONFIRMED |
| `pollinations_no_key` | `openai-compatible` | `openai_compatible_upstream` | AUTH_REQUIRED | ACCESS_DENIED |
| `openai_api` | `openai-compatible` | `openai_compatible_upstream` | NEEDS_AUTH_ENV | AUTH_ENV_MISSING |
| `xai_api` | `xai` | `openai_compatible_upstream` | NEEDS_AUTH_ENV | AUTH_ENV_MISSING |
| `openrouter_gateway` | `openai-compatible` | `openai_compatible_upstream` | NEEDS_AUTH_ENV | AUTH_ENV_MISSING |
| `groq_openai` | `openai-compatible` | `openai_compatible_upstream` | NEEDS_AUTH_ENV | AUTH_ENV_MISSING |
| `together_openai` | `openai-compatible` | `openai_compatible_upstream` | NEEDS_AUTH_ENV | AUTH_ENV_MISSING |
| `mistral_openai` | `openai-compatible` | `openai_compatible_upstream` | NEEDS_AUTH_ENV | AUTH_ENV_MISSING |
| `deepseek_openai` | `openai-compatible` | `openai_compatible_upstream` | NEEDS_AUTH_ENV | AUTH_ENV_MISSING |
| `fireworks_openai` | `openai-compatible` | `openai_compatible_upstream` | NEEDS_AUTH_ENV | AUTH_ENV_MISSING |
| `perplexity_openai` | `openai-compatible` | `openai_compatible_upstream` | NEEDS_AUTH_ENV | AUTH_ENV_MISSING |
| `cerebras_openai` | `openai-compatible` | `openai_compatible_upstream` | NEEDS_AUTH_ENV | AUTH_ENV_MISSING |
