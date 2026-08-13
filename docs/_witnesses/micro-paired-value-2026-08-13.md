# Microagent paired-value spine witness — 2026-08-13

Issue: #6520  
Compatibility blocker: #6653  
Verdict: **NOT_YET**

## Value frame

- **For:** operators deciding whether a shared-kernel microagent is a net-true alternative to a tuned managed CLI.
- **Problem:** `fak micro` previously could not accept a real task or return answer/model/provider usage, so a paired quality/cost run was impossible.
- **Today:** `fak micro --task` drives a real gateway turn, and `fak micro paired` emits a machine-readable receipt for an identical one-step task.
- **Better because:** correctness, provider tokens, wall time, mechanism, kernel control, and nullable provider cost are recorded per arm instead of inferred.
- **Witness:** the live real-kernel receipt below plus focused tests for honest unsupported-cost folding.

## Reproduction

Build outside the shared checkout, then target the sanctioned remote GPU-server gateway through an operator-provided endpoint:

```powershell
fak micro --engine gateway --gateway $env:FAK_GATEWAY --model qwen2.5:14b `
  --task 'Reply with exactly READY' --agents 1 --turns 1 --json

fak micro paired --gateway $env:FAK_GATEWAY --model qwen2.5:14b `
  --task 'Reply with exactly READY' --expect READY --cli-model sonnet `
  --complexity one-step --json
```

## Captured live real-kernel arm

```json
{
  "mode": "run",
  "engine": "gateway",
  "agents": 1,
  "done": 1,
  "failed": 0,
  "results": [{
    "id": "micro-000",
    "steps": 1,
    "done": true,
    "answer": "READY",
    "model": "qwen2.5:14b",
    "prompt_tokens": 33,
    "completion_tokens": 2,
    "total_tokens": 35
  }]
}
```

The tuned direct Claude CLI control also completed `READY` and returned provider-reported usage and cost (15,951 input tokens, 5 output tokens, USD 0.07988). It is not substituted for the guarded arm.

## Captured paired receipt

The microagent arm completed correctly in 1,115 ms with 35 provider tokens. The guarded managed arm failed before usage evidence, so the receipt returned:

```json
{
  "schema": "fak-micro-paired/1",
  "execution_verdict": "FAIL",
  "value_verdict": "NOT_YET",
  "microagent": {
    "completed": true,
    "correct": true,
    "answer": "READY",
    "input_tokens": 33,
    "output_tokens": 2,
    "wall_ms": 1115,
    "cost_usd": null,
    "cost_status": "provider-unsupported",
    "managed": true
  },
  "guarded_cli": {
    "completed": false,
    "correct": false,
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": null,
    "managed": true
  }
}
```

Current Claude Code advertises the retired `tool-search-2025-09-17` beta and `tool_search_tool_20250917` descriptor through the guarded Anthropic passthrough; upstream rejects the request with HTTP 400. #6653 owns that focused protocol migration.

## Honesty boundary

- Missing gateway cost is serialized as `null` with `provider-unsupported`, never zero.
- Missing CLI cost is serialized as `null` with `provider-unreported`, never zero.
- No quality-per-dollar winner is claimed.
- #6520 remains open: the pinned corpus and retry/context/verify/mode ablations still belong to the issue after #6653 restores the baseline.

## Validation

Against a clean archive of committed tip plus only the four `cmd/fak` paths:

```text
go test ./cmd/fak -run 'Test(Micro|FoldPaired|ClaudeResult|FilteredPaired)' -count=1
ok github.com/anthony-chaudhary/fak/cmd/fak

go vet ./cmd/fak
go build -o <temporary-path> ./cmd/fak
```
