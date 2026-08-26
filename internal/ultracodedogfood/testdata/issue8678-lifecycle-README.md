# Issue #8678: lifecycle-bound scoped-prefix replay

This bounded, single-runtime dogfood campaign replays one accepted task across cold, warm, natural TTL expiry, explicit runtime clear, and harness context compaction. The committed source ledger is `testdata/issue8678-lifecycle-session.json`; `EvaluateLifecycleSession` is the integrated, fail-closed evaluator.

## Replay

The campaign ran on the sanctioned `fak-realmodel` node with `qwen2.5:0.5b` through Ollama 0.30.10 / llama.cpp. It used deterministic generation (`temperature=0`, seed 8678), an exact `ACCEPTED` outcome, and one scoped prompt. A full-history counterfactual evaluated 2,256 input tokens; every scoped replay evaluated 635, so fak-owned omission remained 1,621 tokens through every boundary.

1. Run the full-history counterfactual and scoped cold/warm requests through `/api/generate`.
2. Set the warm request's `keep_alive` to `2s`, wait 5 seconds, and require `/api/ps` to show no loaded model before the TTL replay.
3. Rewarm, then send `/api/generate` with `keep_alive: 0`; require `done_reason: unload` and `/api/ps` `models: []` before the clear replay.
4. Compact by dropping the frozen history prefix while retaining the exact scoped task; record before/after SHA-256 digests and dropped byte count.
5. Join each request to the runtime's `cached n_tokens` log. Record zero at cold/TTL/clear and a positive count after each immediate recovery. If this join or any boundary receipt is unavailable, encode the evidence as missing; the evaluator emits `ABSTAIN` and marks that cell `UNKNOWN`.
6. Replay locally: `go test ./internal/ultracodebench -run LifecycleSession -count=1`.

Provider cache contribution is not inferred from latency. The committed ledger reports only runtime-log-backed cache values and records evaluation duration as descriptive telemetry.

## Generation evidence

- **Promotion evidence:** promote from `gen/next` toward `gen/now` after a second bounded campaign reproduces reset/recovery with exact per-request cache-token logs.
- **Demotion/retirement evidence:** demote or retire if authoritative provider cache telemetry cannot be joined to every replay, or equal accepted outcomes stop holding.
- **Invalidating assumption:** the runtime cache-token log rows are attributable to this isolated bounded campaign rather than concurrent requests.

The result is intentionally limited to one model, runtime, tokenizer, task, node, and campaign. It does not establish the cross-runtime matrix tracked by #8650.
