# API-Host Live Smoke Runner

> Execution ledger for ready API-host live-smoke queue rows.

## Summary

- Execute ready rows: no
- Targets: 13
- Already complete: 1
- Ready to execute: 0
- Ready not executed: 0
- Executed: 0
- Passed: 0
- Failed: 0
- Skipped external state: 2
- Waiting for credential: 10
- Ready for probe: 0
- Runner gate: yes

| target | queue state | runner status | command count |
|---|---|---|---:|
| `gemini_openai_compatible` | COMPLETE | ALREADY_COMPLETE | 0 |
| `glama_gateway` | BLOCKED_EXTERNAL_STATE | SKIPPED_EXTERNAL_STATE | 1 |
| `pollinations_no_key` | BLOCKED_EXTERNAL_STATE | SKIPPED_EXTERNAL_STATE | 3 |
| `openai_api` | WAITING_FOR_CREDENTIAL | SKIPPED_WAITING_FOR_CREDENTIAL | 3 |
| `xai_api` | WAITING_FOR_CREDENTIAL | SKIPPED_WAITING_FOR_CREDENTIAL | 3 |
| `openrouter_gateway` | WAITING_FOR_CREDENTIAL | SKIPPED_WAITING_FOR_CREDENTIAL | 3 |
| `groq_openai` | WAITING_FOR_CREDENTIAL | SKIPPED_WAITING_FOR_CREDENTIAL | 3 |
| `together_openai` | WAITING_FOR_CREDENTIAL | SKIPPED_WAITING_FOR_CREDENTIAL | 3 |
| `mistral_openai` | WAITING_FOR_CREDENTIAL | SKIPPED_WAITING_FOR_CREDENTIAL | 3 |
| `deepseek_openai` | WAITING_FOR_CREDENTIAL | SKIPPED_WAITING_FOR_CREDENTIAL | 3 |
| `fireworks_openai` | WAITING_FOR_CREDENTIAL | SKIPPED_WAITING_FOR_CREDENTIAL | 3 |
| `perplexity_openai` | WAITING_FOR_CREDENTIAL | SKIPPED_WAITING_FOR_CREDENTIAL | 3 |
| `cerebras_openai` | WAITING_FOR_CREDENTIAL | SKIPPED_WAITING_FOR_CREDENTIAL | 3 |
