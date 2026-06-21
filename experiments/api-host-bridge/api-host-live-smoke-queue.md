# API-Host Live Smoke Queue

> Credential-conditioned execution queue for remaining API-host live-smoke evidence.

## Summary

- Targets: 13
- Complete: 1
- Ready to execute: 0
- Blocked external state: 2
- Waiting for credential: 10
- Ready for probe: 0
- Unqualified: 0
- Unclassified: 0
- Command gaps: 0
- Queue gate: yes

| target | queue state | prerequisite | command count |
|---|---|---|---:|
| `gemini_openai_compatible` | COMPLETE | none | 0 |
| `glama_gateway` | BLOCKED_EXTERNAL_STATE | Attach billing/payment method for the account behind GLAMA_API_KEY. | 1 |
| `pollinations_no_key` | BLOCKED_EXTERNAL_STATE | Configure a valid bearer token or access path via POLLINATIONS_API_KEY. | 3 |
| `openai_api` | WAITING_FOR_CREDENTIAL | Set OPENAI_API_KEY for this host. | 3 |
| `xai_api` | WAITING_FOR_CREDENTIAL | Set XAI_API_KEY for this host. | 3 |
| `openrouter_gateway` | WAITING_FOR_CREDENTIAL | Set OPENROUTER_API_KEY for this host. | 3 |
| `groq_openai` | WAITING_FOR_CREDENTIAL | Set GROQ_API_KEY for this host. | 3 |
| `together_openai` | WAITING_FOR_CREDENTIAL | Set TOGETHER_API_KEY for this host. | 3 |
| `mistral_openai` | WAITING_FOR_CREDENTIAL | Set MISTRAL_API_KEY for this host. | 3 |
| `deepseek_openai` | WAITING_FOR_CREDENTIAL | Set DEEPSEEK_API_KEY for this host. | 3 |
| `fireworks_openai` | WAITING_FOR_CREDENTIAL | Set FIREWORKS_API_KEY for this host. | 3 |
| `perplexity_openai` | WAITING_FOR_CREDENTIAL | Set PERPLEXITY_API_KEY for this host. | 3 |
| `cerebras_openai` | WAITING_FOR_CREDENTIAL | Set CEREBRAS_API_KEY for this host. | 3 |

## glama_gateway

```powershell
pwsh tools/run_transcript_adapter_sweep.ps1 -OutDir fak/experiments/agent-live/transcript-adapter-sweep-glama-gateway-retry -ApiBaseUrl 'https://gateway.glama.ai/v1' -ApiKeyEnv GLAMA_API_KEY -ApiModels 'openai/gpt-4.1-nano-2025-04-14' -SkipOffline -SkipLocalShim -SkipMicrobench -MaxTurns 12 -Trials 1
```


## pollinations_no_key

```powershell
python tools/api_host_readiness_probe.py --target 'pollinations_no_key|https://gen.pollinations.ai/v1|POLLINATIONS_API_KEY|openai-fast' --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md
```

```powershell
python tools/api_host_acceptance_probe.py --target 'pollinations_no_key|openai-compatible|https://gen.pollinations.ai/v1|POLLINATIONS_API_KEY|openai-fast' --out fak/experiments/api-host-bridge/api-host-acceptance.json --markdown fak/experiments/api-host-bridge/api-host-acceptance.md
```

```powershell
pwsh tools/run_transcript_adapter_sweep.ps1 -OutDir fak/experiments/agent-live/transcript-adapter-sweep-pollinations-no-key-retry -ApiBaseUrl 'https://gen.pollinations.ai/v1' -ApiKeyEnv POLLINATIONS_API_KEY -ApiModels 'openai-fast' -SkipOffline -SkipLocalShim -SkipMicrobench -MaxTurns 12 -Trials 1
```


## openai_api

```powershell
python tools/api_host_readiness_probe.py --target 'openai_api|https://api.openai.com/v1|OPENAI_API_KEY|gpt-4.1-mini' --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md
```

```powershell
python tools/api_host_acceptance_probe.py --target 'openai_api|openai-compatible|https://api.openai.com/v1|OPENAI_API_KEY|gpt-4.1-mini' --out fak/experiments/api-host-bridge/api-host-acceptance.json --markdown fak/experiments/api-host-bridge/api-host-acceptance.md
```

```powershell
pwsh tools/run_transcript_adapter_sweep.ps1 -OutDir fak/experiments/agent-live/transcript-adapter-sweep-openai-api-retry -ApiBaseUrl 'https://api.openai.com/v1' -ApiKeyEnv OPENAI_API_KEY -ApiModels 'gpt-4.1-mini' -SkipOffline -SkipLocalShim -SkipMicrobench -MaxTurns 12 -Trials 1
```


## xai_api

```powershell
python tools/api_host_readiness_probe.py --target 'xai_api|https://api.x.ai/v1|XAI_API_KEY|grok-3-mini' --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md
```

```powershell
python tools/api_host_acceptance_probe.py --target 'xai_api|xai|https://api.x.ai/v1|XAI_API_KEY|grok-3-mini' --out fak/experiments/api-host-bridge/api-host-acceptance.json --markdown fak/experiments/api-host-bridge/api-host-acceptance.md
```

```powershell
pwsh tools/run_transcript_adapter_sweep.ps1 -OutDir fak/experiments/agent-live/transcript-adapter-sweep-xai-api-retry -ApiBaseUrl 'https://api.x.ai/v1' -ApiKeyEnv XAI_API_KEY -ApiModels 'grok-3-mini' -SkipOffline -SkipLocalShim -SkipMicrobench -MaxTurns 12 -Trials 1
```


## openrouter_gateway

```powershell
python tools/api_host_readiness_probe.py --target 'openrouter_gateway|https://openrouter.ai/api/v1|OPENROUTER_API_KEY|openai/gpt-4.1-mini' --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md
```

```powershell
python tools/api_host_acceptance_probe.py --target 'openrouter_gateway|openai-compatible|https://openrouter.ai/api/v1|OPENROUTER_API_KEY|openai/gpt-4.1-mini' --out fak/experiments/api-host-bridge/api-host-acceptance.json --markdown fak/experiments/api-host-bridge/api-host-acceptance.md
```

```powershell
pwsh tools/run_transcript_adapter_sweep.ps1 -OutDir fak/experiments/agent-live/transcript-adapter-sweep-openrouter-gateway-retry -ApiBaseUrl 'https://openrouter.ai/api/v1' -ApiKeyEnv OPENROUTER_API_KEY -ApiModels 'openai/gpt-4.1-mini' -SkipOffline -SkipLocalShim -SkipMicrobench -MaxTurns 12 -Trials 1
```


## groq_openai

```powershell
python tools/api_host_readiness_probe.py --target 'groq_openai|https://api.groq.com/openai/v1|GROQ_API_KEY|llama-3.3-70b-versatile' --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md
```

```powershell
python tools/api_host_acceptance_probe.py --target 'groq_openai|openai-compatible|https://api.groq.com/openai/v1|GROQ_API_KEY|llama-3.3-70b-versatile' --out fak/experiments/api-host-bridge/api-host-acceptance.json --markdown fak/experiments/api-host-bridge/api-host-acceptance.md
```

```powershell
pwsh tools/run_transcript_adapter_sweep.ps1 -OutDir fak/experiments/agent-live/transcript-adapter-sweep-groq-openai-retry -ApiBaseUrl 'https://api.groq.com/openai/v1' -ApiKeyEnv GROQ_API_KEY -ApiModels 'llama-3.3-70b-versatile' -SkipOffline -SkipLocalShim -SkipMicrobench -MaxTurns 12 -Trials 1
```


## together_openai

```powershell
python tools/api_host_readiness_probe.py --target 'together_openai|https://api.together.xyz/v1|TOGETHER_API_KEY|meta-llama/Llama-3.3-70B-Instruct-Turbo' --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md
```

```powershell
python tools/api_host_acceptance_probe.py --target 'together_openai|openai-compatible|https://api.together.xyz/v1|TOGETHER_API_KEY|meta-llama/Llama-3.3-70B-Instruct-Turbo' --out fak/experiments/api-host-bridge/api-host-acceptance.json --markdown fak/experiments/api-host-bridge/api-host-acceptance.md
```

```powershell
pwsh tools/run_transcript_adapter_sweep.ps1 -OutDir fak/experiments/agent-live/transcript-adapter-sweep-together-openai-retry -ApiBaseUrl 'https://api.together.xyz/v1' -ApiKeyEnv TOGETHER_API_KEY -ApiModels 'meta-llama/Llama-3.3-70B-Instruct-Turbo' -SkipOffline -SkipLocalShim -SkipMicrobench -MaxTurns 12 -Trials 1
```


## mistral_openai

```powershell
python tools/api_host_readiness_probe.py --target 'mistral_openai|https://api.mistral.ai/v1|MISTRAL_API_KEY|mistral-small-latest' --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md
```

```powershell
python tools/api_host_acceptance_probe.py --target 'mistral_openai|openai-compatible|https://api.mistral.ai/v1|MISTRAL_API_KEY|mistral-small-latest' --out fak/experiments/api-host-bridge/api-host-acceptance.json --markdown fak/experiments/api-host-bridge/api-host-acceptance.md
```

```powershell
pwsh tools/run_transcript_adapter_sweep.ps1 -OutDir fak/experiments/agent-live/transcript-adapter-sweep-mistral-openai-retry -ApiBaseUrl 'https://api.mistral.ai/v1' -ApiKeyEnv MISTRAL_API_KEY -ApiModels 'mistral-small-latest' -SkipOffline -SkipLocalShim -SkipMicrobench -MaxTurns 12 -Trials 1
```


## deepseek_openai

```powershell
python tools/api_host_readiness_probe.py --target 'deepseek_openai|https://api.deepseek.com|DEEPSEEK_API_KEY|deepseek-chat' --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md
```

```powershell
python tools/api_host_acceptance_probe.py --target 'deepseek_openai|openai-compatible|https://api.deepseek.com|DEEPSEEK_API_KEY|deepseek-chat' --out fak/experiments/api-host-bridge/api-host-acceptance.json --markdown fak/experiments/api-host-bridge/api-host-acceptance.md
```

```powershell
pwsh tools/run_transcript_adapter_sweep.ps1 -OutDir fak/experiments/agent-live/transcript-adapter-sweep-deepseek-openai-retry -ApiBaseUrl 'https://api.deepseek.com' -ApiKeyEnv DEEPSEEK_API_KEY -ApiModels 'deepseek-chat' -SkipOffline -SkipLocalShim -SkipMicrobench -MaxTurns 12 -Trials 1
```


## fireworks_openai

```powershell
python tools/api_host_readiness_probe.py --target 'fireworks_openai|https://api.fireworks.ai/inference/v1|FIREWORKS_API_KEY|accounts/fireworks/models/llama-v3p1-8b-instruct' --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md
```

```powershell
python tools/api_host_acceptance_probe.py --target 'fireworks_openai|openai-compatible|https://api.fireworks.ai/inference/v1|FIREWORKS_API_KEY|accounts/fireworks/models/llama-v3p1-8b-instruct' --out fak/experiments/api-host-bridge/api-host-acceptance.json --markdown fak/experiments/api-host-bridge/api-host-acceptance.md
```

```powershell
pwsh tools/run_transcript_adapter_sweep.ps1 -OutDir fak/experiments/agent-live/transcript-adapter-sweep-fireworks-openai-retry -ApiBaseUrl 'https://api.fireworks.ai/inference/v1' -ApiKeyEnv FIREWORKS_API_KEY -ApiModels 'accounts/fireworks/models/llama-v3p1-8b-instruct' -SkipOffline -SkipLocalShim -SkipMicrobench -MaxTurns 12 -Trials 1
```


## perplexity_openai

```powershell
python tools/api_host_readiness_probe.py --target 'perplexity_openai|https://api.perplexity.ai|PERPLEXITY_API_KEY|sonar' --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md
```

```powershell
python tools/api_host_acceptance_probe.py --target 'perplexity_openai|openai-compatible|https://api.perplexity.ai|PERPLEXITY_API_KEY|sonar' --out fak/experiments/api-host-bridge/api-host-acceptance.json --markdown fak/experiments/api-host-bridge/api-host-acceptance.md
```

```powershell
pwsh tools/run_transcript_adapter_sweep.ps1 -OutDir fak/experiments/agent-live/transcript-adapter-sweep-perplexity-openai-retry -ApiBaseUrl 'https://api.perplexity.ai' -ApiKeyEnv PERPLEXITY_API_KEY -ApiModels 'sonar' -SkipOffline -SkipLocalShim -SkipMicrobench -MaxTurns 12 -Trials 1
```


## cerebras_openai

```powershell
python tools/api_host_readiness_probe.py --target 'cerebras_openai|https://api.cerebras.ai/v1|CEREBRAS_API_KEY|llama3.1-8b' --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md
```

```powershell
python tools/api_host_acceptance_probe.py --target 'cerebras_openai|openai-compatible|https://api.cerebras.ai/v1|CEREBRAS_API_KEY|llama3.1-8b' --out fak/experiments/api-host-bridge/api-host-acceptance.json --markdown fak/experiments/api-host-bridge/api-host-acceptance.md
```

```powershell
pwsh tools/run_transcript_adapter_sweep.ps1 -OutDir fak/experiments/agent-live/transcript-adapter-sweep-cerebras-openai-retry -ApiBaseUrl 'https://api.cerebras.ai/v1' -ApiKeyEnv CEREBRAS_API_KEY -ApiModels 'llama3.1-8b' -SkipOffline -SkipLocalShim -SkipMicrobench -MaxTurns 12 -Trials 1
```
