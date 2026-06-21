# API-Host Roster

> No-spend target templates for compatible API-host bridge candidates.

## Summary

- Targets: 13
- Supported templates: 13
- OpenAI-compatible templates: 13
- Invalid targets: 0
- Unsupported wire: 0
- Duplicate names: 0
- Roster gate: yes

| target | provider | class | env | status |
|---|---|---|---|---|
| `gemini_openai_compatible` | `openai-compatible` | `openai_compatible_upstream` | `GEMINI_API_KEY` | SUPPORTED_TEMPLATE |
| `glama_gateway` | `openai-compatible` | `openai_compatible_upstream` | `GLAMA_API_KEY` | SUPPORTED_TEMPLATE |
| `pollinations_no_key` | `openai-compatible` | `openai_compatible_upstream` | `` | SUPPORTED_TEMPLATE |
| `openai_api` | `openai-compatible` | `openai_compatible_upstream` | `OPENAI_API_KEY` | SUPPORTED_TEMPLATE |
| `xai_api` | `xai` | `openai_compatible_upstream` | `XAI_API_KEY` | SUPPORTED_TEMPLATE |
| `openrouter_gateway` | `openai-compatible` | `openai_compatible_upstream` | `OPENROUTER_API_KEY` | SUPPORTED_TEMPLATE |
| `groq_openai` | `openai-compatible` | `openai_compatible_upstream` | `GROQ_API_KEY` | SUPPORTED_TEMPLATE |
| `together_openai` | `openai-compatible` | `openai_compatible_upstream` | `TOGETHER_API_KEY` | SUPPORTED_TEMPLATE |
| `mistral_openai` | `openai-compatible` | `openai_compatible_upstream` | `MISTRAL_API_KEY` | SUPPORTED_TEMPLATE |
| `deepseek_openai` | `openai-compatible` | `openai_compatible_upstream` | `DEEPSEEK_API_KEY` | SUPPORTED_TEMPLATE |
| `fireworks_openai` | `openai-compatible` | `openai_compatible_upstream` | `FIREWORKS_API_KEY` | SUPPORTED_TEMPLATE |
| `perplexity_openai` | `openai-compatible` | `openai_compatible_upstream` | `PERPLEXITY_API_KEY` | SUPPORTED_TEMPLATE |
| `cerebras_openai` | `openai-compatible` | `openai_compatible_upstream` | `CEREBRAS_API_KEY` | SUPPORTED_TEMPLATE |

## Bulk Commands

```bash
python tools/api_host_readiness_probe.py --target 'gemini_openai_compatible|https://generativelanguage.googleapis.com/v1beta/openai|GEMINI_API_KEY|gemini-2.5-flash' --target 'glama_gateway|https://gateway.glama.ai/v1|GLAMA_API_KEY|openai/gpt-4.1-nano-2025-04-14' --target 'pollinations_no_key|https://gen.pollinations.ai/v1||openai-fast' --target 'openai_api|https://api.openai.com/v1|OPENAI_API_KEY|gpt-4.1-mini' --target 'xai_api|https://api.x.ai/v1|XAI_API_KEY|grok-3-mini' --target 'openrouter_gateway|https://openrouter.ai/api/v1|OPENROUTER_API_KEY|openai/gpt-4.1-mini' --target 'groq_openai|https://api.groq.com/openai/v1|GROQ_API_KEY|llama-3.3-70b-versatile' --target 'together_openai|https://api.together.xyz/v1|TOGETHER_API_KEY|meta-llama/Llama-3.3-70B-Instruct-Turbo' --target 'mistral_openai|https://api.mistral.ai/v1|MISTRAL_API_KEY|mistral-small-latest' --target 'deepseek_openai|https://api.deepseek.com|DEEPSEEK_API_KEY|deepseek-chat' --target 'fireworks_openai|https://api.fireworks.ai/inference/v1|FIREWORKS_API_KEY|accounts/fireworks/models/llama-v3p1-8b-instruct' --target 'perplexity_openai|https://api.perplexity.ai|PERPLEXITY_API_KEY|sonar' --target 'cerebras_openai|https://api.cerebras.ai/v1|CEREBRAS_API_KEY|llama3.1-8b' --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md
```

```bash
python tools/api_host_acceptance_probe.py --target 'gemini_openai_compatible|openai-compatible|https://generativelanguage.googleapis.com/v1beta/openai|GEMINI_API_KEY|gemini-2.5-flash' --target 'glama_gateway|openai-compatible|https://gateway.glama.ai/v1|GLAMA_API_KEY|openai/gpt-4.1-nano-2025-04-14' --target 'pollinations_no_key|openai-compatible|https://gen.pollinations.ai/v1||openai-fast' --target 'openai_api|openai-compatible|https://api.openai.com/v1|OPENAI_API_KEY|gpt-4.1-mini' --target 'xai_api|xai|https://api.x.ai/v1|XAI_API_KEY|grok-3-mini' --target 'openrouter_gateway|openai-compatible|https://openrouter.ai/api/v1|OPENROUTER_API_KEY|openai/gpt-4.1-mini' --target 'groq_openai|openai-compatible|https://api.groq.com/openai/v1|GROQ_API_KEY|llama-3.3-70b-versatile' --target 'together_openai|openai-compatible|https://api.together.xyz/v1|TOGETHER_API_KEY|meta-llama/Llama-3.3-70B-Instruct-Turbo' --target 'mistral_openai|openai-compatible|https://api.mistral.ai/v1|MISTRAL_API_KEY|mistral-small-latest' --target 'deepseek_openai|openai-compatible|https://api.deepseek.com|DEEPSEEK_API_KEY|deepseek-chat' --target 'fireworks_openai|openai-compatible|https://api.fireworks.ai/inference/v1|FIREWORKS_API_KEY|accounts/fireworks/models/llama-v3p1-8b-instruct' --target 'perplexity_openai|openai-compatible|https://api.perplexity.ai|PERPLEXITY_API_KEY|sonar' --target 'cerebras_openai|openai-compatible|https://api.cerebras.ai/v1|CEREBRAS_API_KEY|llama3.1-8b' --out fak/experiments/api-host-bridge/api-host-acceptance.json --markdown fak/experiments/api-host-bridge/api-host-acceptance.md
```
