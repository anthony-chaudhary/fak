---
title: "Micro-context S8 large-input operator spine"
description: "Fixture-backed 1,000-issue partition, prefilter, semantic-map, cache-invalidation, bounded-fold, and oracle witness."
status: observed
last_reviewed: 2026-08-09
---

# S8: 1,000-record large-input operator spine

## Verdict

**Observed locally:** the runnable large-input spine processes an immutable 1,000-record
GitHub-issue-shaped fixture through deterministic prefiltering, one-record semantic
micro-contexts, typed outcomes, content-addressed reuse, and a bounded hierarchical fold.
The committed artifact is
[`s8-local-large-input-1000-pass-2026-08-09.json`](../../experiments/microcontext/s8-local-large-input-1000-pass-2026-08-09.json).

This is a **fixture-backed correctness and concurrency witness**, not model throughput,
provider-cache, or production-quality evidence. The semantic worker is intentionally declared
`fixture-backed semantic worker` in the artifact.

## Reproduce

```bash
go run ./cmd/microcontextdemo \
  -large-input-selfcheck -contexts 1000 -workers 16 \
  -large-input-output /tmp/large-input.json

go run ./cmd/microcontextdemo -verify-large-input /tmp/large-input.json
```

## Captured result

- 1,000 source records accounted exactly once.
- 100 deterministic documentation-label exclusions before semantic work.
- 900 semantic micro-context calls on the baseline pass through 16 bounded workers.
- Outcomes remain separate: 25 kept, 973 excluded, one abstention, and one fixture error.
- An unchanged second pass makes zero semantic calls and reuses all 900 pure-stage records.
- Mutating one previously-negative issue makes exactly one semantic call and adds cited issue
  `10996`; 899 unchanged semantic records remain cache hits.
- Every fold node receives at most 32 children; 1,000 leaves reduce through two bounded levels.
- Baseline and mutated cited IDs exactly match the independently-authored fixture oracle.

The command keeps all 1,000 typed facts in memory for tests, while the committed artifact keeps
aggregate accounting, hashes, fold bounds, and final citations so the proof remains inspectable
without becoming another large-input payload.

## What remains unproven

- Real-model semantic quality and provider costs.
- Adaptive selection among deterministic filters, semantic filters, wider neighborhoods, and
  enrichment tools (#6030).
- Read-only tool execution (#6031), generalized fold laws (#6032), tuned-baseline comparison
  (#6033), and effectful stages (#6034).
