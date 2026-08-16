---
title: "Managed Claude Tool Search compatibility witness — 2026-08-13"
description: "Documentation for Managed Claude Tool Search compatibility witness — 2026-08-13, including the captured behavior, operating context, and reproducible fak evidence."
---

# Managed Claude Tool Search compatibility witness — 2026-08-13

Issue: #6653  
Unblocks: #6520 paired execution  
Verdict: **PASS**

## Failure

Claude Code 2.1.229 advertised the retired Tool Search protocol pair:

```text
anthropic-beta: tool-search-2025-09-17
type: tool_search_tool_20250917
name: tool_search_tool
```

Anthropic rejected the beta first, then rejected the stale descriptor when only the header was removed. A managed request therefore failed before the user's task could run.

## Repair

The served Anthropic boundary now treats the header and descriptor as one compatibility unit:

- remove only `tool-search-2025-09-17` while preserving other negotiated betas;
- migrate the stale descriptor to Anthropic's current regex Tool Search contract:
  `tool_search_tool_regex_20251119` / `tool_search_tool_regex`;
- use that current contract for fak-injected cold-tool deferral too;
- preserve current descriptors and custom tools unchanged.

Current protocol source checked on 2026-08-13:
`https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool`.

## Live managed request

Built from a clean archive of committed tip plus this issue's paths, outside the shared checkout:

```powershell
fak manage --quiet --probe --lease mode=off -- claude `
  -p 'Reply with exactly READY' --output-format json --max-turns 1 `
  --tools '' --model sonnet --setting-sources ''
```

Observed result:

```json
{
  "is_error": false,
  "num_turns": 1,
  "terminal_reason": "completed",
  "result": "READY",
  "modelUsage": {
    "claude-sonnet-5": {
      "inputTokens": 2,
      "outputTokens": 679,
      "cacheReadInputTokens": 9026,
      "cacheCreationInputTokens": 822,
      "costUSD": 0.0159813,
      "provider": "firstParty"
    }
  }
}
```

The larger output token count includes managed harness setup; completion and provider cost are provider-reported, not modeled.

## Paired end-to-end receipt

The same clean binary then ran identical task text through the remote shared-kernel microagent and the managed Claude baseline:

```json
{
  "schema": "fak-micro-paired/1",
  "execution_verdict": "PASS",
  "value_verdict": "NOT_YET",
  "microagent": {
    "model": "qwen2.5:14b",
    "correct": true,
    "answer": "READY",
    "input_tokens": 33,
    "output_tokens": 2,
    "wall_ms": 1077,
    "cost_usd": null,
    "cost_status": "provider-unsupported"
  },
  "managed_baseline": {
    "model": "claude-sonnet-5",
    "correct": true,
    "answer": "READY",
    "input_tokens": 184,
    "output_tokens": 4,
    "cache_read_tokens": 9026,
    "wall_ms": 5622,
    "cost_usd": 0.0057963,
    "cost_status": "provider-reported"
  }
}
```

The paired runner uses `--probe`, an empty settings-source set, and disables only fak's session-start affordance prompt. That isolates task execution while retaining the real `fak manage` gateway/admission path. It does not bypass the kernel.

`value_verdict` correctly remains `NOT_YET`: the remote OpenAI-compatible gateway did not report dollars, so no quality-per-dollar winner is claimed.

## Validation

```text
go test ./internal/gateway -run 'Test(PrepareServedAnthropicRequestMigratesRetiredToolSearchContract|MigrateRetiredToolSearchPreservesCurrentAndCustomTools|DeferColdTools|ToolDefer)' -count=1
ok github.com/anthony-chaudhary/fak/internal/gateway

go test ./internal/gateway -short -count=1
go vet ./internal/gateway
go build -o <temporary-path> ./cmd/fak
```
