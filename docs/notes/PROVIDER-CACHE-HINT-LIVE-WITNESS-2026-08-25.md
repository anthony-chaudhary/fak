# Provider cache-hint live witness — issue #7309

Observed 2026-08-25. This artifact records the required supported-provider and
negative runs without storing credentials or full prompts.

## Supported-provider attempt

Route: OpenAI `POST /v1/chat/completions`; requested a repeated stable prefix and
`prompt_cache_key` through the exact provider field emitted by the new adapter
contract. The sanctioned environment's `OPENAI_API_KEY` resolved to the repository
placeholder credential, and OpenAI returned HTTP 401 `invalid_api_key` before model
execution. Consequently no provider usage object existed and a nonzero cached-token
claim cannot honestly be made from this worktree. The attempted run did not modify
repository state or substitute a local model. Re-run command shape (credential omitted):

```text
POST https://api.openai.com/v1/chat/completions
model=gpt-4.1-mini, repeated stable prefix, prompt_cache_key=fak-issue-7309-live-witness
result=401 invalid_api_key; provider cache writes/reads/latency/billed-input unavailable
```

## Negative captured run

The table-driven `TestCacheHintNegotiationCompatibility` witness requests OpenAI
24-hour retention on `gpt-4.1` as advisory and receives:

```json
{"status":"downgraded","provider":"openai-responses","model":"gpt-4.1","reason":"model/API does not support 24h prompt-cache retention"}
```

The strict form returns `rejected`; a 24-hour request under a memory-only privacy
ceiling also returns `rejected`. Neither case silently drops the request.
