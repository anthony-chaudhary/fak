# Issue #9777 performance-RSI loop-turn witness

This witness captures one real `fak loop run` turn automatically invoking the
internal performance-RSI score. The command in [`loop-turn.txt`](loop-turn.txt)
asks only for a dispatch turn; it does not invoke the scorecard CLI or shell out
to it. `runLoopRun` calls `internal/perfrsiscore` once after the child and dispatch
result complete, then emits the versioned `fak-performance-rsi-loop-turn/1`
receipt on stderr.

`FAK_PERFORMANCE_RSI_INPUT` points the automatic scorer at the independently
produced evidence document. If it is unset or unreadable, the same seam emits an
`unavailable` receipt with `SCORE_INPUT_UNAVAILABLE`; the child exit code, stdout
report, and loop event sequence remain unchanged.

## Reproduce

```bash
: > /tmp/issue-9777-loop-turn.jsonl
FAK_PERFORMANCE_RSI_INPUT=docs/_witnesses/issue-9768-performance-rsi-dogfood/input.json \
  go run ./cmd/fak loop run \
  --ledger /tmp/issue-9777-loop-turn.jsonl \
  --loop dispatch/issues \
  --run issue-9777-loop-turn \
  --source witness \
  --no-guard -- sh -c true
```

`loop-turn.txt` labels stderr and stdout separately so cross-stream terminal
buffering cannot make their relative display order look like a second seam.

The captured receipt reports `status=scored`, snapshot
`issue-9768-live-repository-dogfood-2026-08-28`, health `D 62.6/100`, debt `9`,
and dominant bottleneck `cycle_time`; the dispatch still exits `0`.
