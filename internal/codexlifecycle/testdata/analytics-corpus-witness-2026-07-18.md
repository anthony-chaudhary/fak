# #4767 corpus witness — native Codex critical-path + typed tool-outcome analytics

Captured 2026-07-18 by `TestAnalyticsCorpus_LiveStore` over the local native Codex
rollout store (`~/.codex/sessions`, cwd-scoped to this repository checkout), i.e.
the same store and scoping method as the issue's 2026-07-14 evidence pass.
Everything below is scrubbed: opaque rollout/turn ids, closed class/reason tokens,
tool names, and hashed signatures — no raw commands, no result bodies.

## Scale (issue's 2026-07-14 pass → this 2026-07-18 pass)

| metric        | 2026-07-14 (issue) | 2026-07-18 (this witness) |
|---------------|--------------------|---------------------------|
| sessions      | 2,126              | 2,179                     |
| completed     | 4,994              | 4,988 (5,188 tasks total) |
| tool calls    | 225,934            | 277,482                   |

The witnessed operating envelope (`tasks >= 4994`) holds: 5,188 reconciled tasks.

## Recorded task duration (seconds, completed tasks, n=4,988)

| p50   | p90     | p95     | p99     | max      |
|-------|---------|---------|---------|----------|
| 282.8 | 1,570.8 | 2,565.4 | 5,899.8 | 37,699.2 |

(4.7-minute median / 26.2-minute p90 / 42.8-minute p95 / 1.6-hour p99 /
10.5-hour max — the issue's 5.4-min / 23.0-min / 34.7-min / 84.6-min / 8.90-hour
distribution, shifted by four more days of long fleet runs in the same store.)

## Time to first token (seconds, n=5,081)

| p50  | p90  | p95  | p99  | max     |
|------|------|------|------|---------|
| 10.8 | 18.7 | 23.6 | 44.4 | 1,605.0 |

## Typed tool outcomes (closed vocabulary, total = 277,482 calls)

| class             | count   | reason breakdown                                  |
|-------------------|---------|---------------------------------------------------|
| ok                | 253,436 | exit_0 227,094 · no_envelope 23,926 · structured_no_exit 2,416 |
| failure           | 17,232  | exit_1 17,115 · exit_2 37 · exit_-1 31 · others   |
| expected_negative | 5,375   | grep_no_match 4,422 · merge_head_probe 953        |
| timeout           | 1,358   | harness deadline kills                            |
| interrupted       | 69      | open call at aborted/superseded/dead task boundary |
| missing_result    | 12      | later output in same task proves the gap          |
| control_exit      | 0       | no segment-start `wait` non-zero exits in the current store |

`failure_calls = 18,590` (failure + timeout). The 5,375 expected negatives are
VISIBLE but excluded — the naive parser's "18,408 error-shaped results" collapses
once probes with documented negative exit semantics are typed separately.

## Top critical-path outliers

The longest tasks decompose into typed contributors (tool / model / wait / idle),
e.g. the 10.5-hour completed task: tool:shell_command 7.9h, model 1.9h, wait 1.4h.
The two longest rows are `superseded` mid-session gaps — non-success boundaries
the #4785 fold synthesizes, never fabricated completions.

## Behavioral detectors (ported from #2365)

timeout_kills=1,358 · sleep_polls=2,265 · stall_gaps=107

## Findings feed (dos unstick / issue candidates)

| reason                     | count  |
|----------------------------|--------|
| repeated_failure:exit_1    | 17,115 |
| repeated_failure:timeout   | 1,358  |
| repeated_failure:exit_2    | 37     |
| repeated_failure:exit_-1   | 31     |
| repeated_idle_gap          | 107    |
| foreground_polling         | 2,265  |

## Manual precision adjudication (bounded sample)

48 non-ok outcomes sampled deterministically (every 40th of 1,942) via the
local-only `CODEX_AUDIT_SAMPLE=1` seam and adjudicated by hand against their
command heads:

| class sampled     | rows | adjudicated correct |
|-------------------|------|---------------------|
| timeout           | 6    | 6                   |
| expected_negative | 10   | 10                  |
| failure           | 32   | 28                  |

Overall sample precision 44/48 ≈ 0.92; failure-class precision 28/32 ≈ 0.88.
The four adjudicated misses are probe-shaped negatives not yet in the registry
(a `--check`-style staleness probe, a `--exit-status` CI-watch propagation, a
file-existence probe, a process-liveness probe) — candidates for future
registered rows, kept OUT of the registry now because their exit semantics are
tool-specific rather than documented-universal.
