# Issue #8311 — Q5_K_M platform qualification attempt

**Status: INVALID — serving API contract failed before campaign qualification.**

This directory preserves the failed 2026-08-21 attempt as diagnostic evidence. It does not rank Q5_K_M against Q4_K_M, recommend retaining either quant, or provide usable quality, latency, throughput, cache, memory-efficiency, or promotion evidence.

## Why analysis stops here

Both attempted arms produced zero successful API completions:

- Text, JSON-schema, and cache responses could not be decoded because `usage` contained nested objects where the frozen OpenAI-compatible contract requires integer token counters.
- Tool and coding requests failed closed with HTTP 502 after an unrecognized tool-call format.
- Long-context requests reached the 10-minute client deadline.

The checked-in campaign contract now treats the first text request as an API canary. An undecodable response or non-positive `usage.completion_tokens` stops the run before the other 17 requests and emits no campaign report. `qwen38quant.Validate` also rejects an imported or hand-assembled campaign with no successful completion, even if it says `HOLD`.

## Platform outcome

| Arm | Qualification | Next action |
|---|---|---|
| Q5_K_M | Invalid API contract | Fix serving response decoding/tool format, then rerun Q5_K_M independently. |
| Q4_K_M | Invalid API contract | Fix serving response decoding/tool format, then rerun Q4_K_M independently if this arm still needs qualification. |

Issue #8311 asks whether fak supports Q5_K_M end to end. The platform-support question remains unanswered. Cross-quant comparison is a later consumer and is admissible only after each compared arm independently qualifies on the same frozen corpus.

## Retained diagnostics

The raw files retain model identity, launch/topology, residency observations, request failures, restart readiness, and cleanup evidence:

- `q5-report.json` and `q5-archive.json`
- `q4-report.json` and `q4-archive.json`

`summary.json` records only the invalid qualification status and remediation. The earlier derived latency/throughput/cache comparison was removed because failed API calls do not establish comparable model behavior. Artifact size and residency observations may help diagnose deployment capacity, but they do not rescue campaign validity.

## Rerun gate

Use the frozen corpus at `docs/benchmarks/qwen38-quant/corpus.json`. A rerun must pass the API canary, complete every required workload at least three times, and pass `qwen38quant.Validate` before any downstream report performs cross-arm analysis.