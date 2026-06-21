# API-Host Qualification

> Per-target qualification against the proven API-host bridge contract.

## Summary

- In-contract targets: 13/13
- Live confirmed: 1
- Ready for live smoke: 0
- External blocked: 2
- Needs credential: 10
- Needs probe: 0
- Out of contract: 0
- Invalid targets: 0
- Unclassified: 0
- Qualification gate: yes

| target | rule | qualification | evidence | next evidence |
|---|---|---|---|---|
| `gemini_openai_compatible` | `openai_compatible_wire` | IN_CONTRACT_LIVE_CONFIRMED | LIVE_CONFIRMED | Committed evidence already confirms a live bridge run for this host/base URL. |
| `glama_gateway` | `openai_compatible_wire` | IN_CONTRACT_EXTERNAL_BLOCKER | NEEDS_OPERATOR_STATE | Attach billing/payment method for the account behind GLAMA_API_KEY. |
| `pollinations_no_key` | `openai_compatible_wire` | IN_CONTRACT_EXTERNAL_BLOCKER | NEEDS_OPERATOR_STATE | Configure a valid bearer token or access path via POLLINATIONS_API_KEY. |
| `openai_api` | `openai_compatible_wire` | IN_CONTRACT_NEEDS_CREDENTIAL | NEEDS_OPERATOR_STATE | Set OPENAI_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `xai_api` | `openai_compatible_wire` | IN_CONTRACT_NEEDS_CREDENTIAL | NEEDS_OPERATOR_STATE | Set XAI_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `openrouter_gateway` | `openai_compatible_wire` | IN_CONTRACT_NEEDS_CREDENTIAL | NEEDS_OPERATOR_STATE | Set OPENROUTER_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `groq_openai` | `openai_compatible_wire` | IN_CONTRACT_NEEDS_CREDENTIAL | NEEDS_OPERATOR_STATE | Set GROQ_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `together_openai` | `openai_compatible_wire` | IN_CONTRACT_NEEDS_CREDENTIAL | NEEDS_OPERATOR_STATE | Set TOGETHER_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `mistral_openai` | `openai_compatible_wire` | IN_CONTRACT_NEEDS_CREDENTIAL | NEEDS_OPERATOR_STATE | Set MISTRAL_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `deepseek_openai` | `openai_compatible_wire` | IN_CONTRACT_NEEDS_CREDENTIAL | NEEDS_OPERATOR_STATE | Set DEEPSEEK_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `fireworks_openai` | `openai_compatible_wire` | IN_CONTRACT_NEEDS_CREDENTIAL | NEEDS_OPERATOR_STATE | Set FIREWORKS_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `perplexity_openai` | `openai_compatible_wire` | IN_CONTRACT_NEEDS_CREDENTIAL | NEEDS_OPERATOR_STATE | Set PERPLEXITY_API_KEY, then rerun readiness, acceptance, and live smoke. |
| `cerebras_openai` | `openai_compatible_wire` | IN_CONTRACT_NEEDS_CREDENTIAL | NEEDS_OPERATOR_STATE | Set CEREBRAS_API_KEY, then rerun readiness, acceptance, and live smoke. |
