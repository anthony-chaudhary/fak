---
title: "Micro-context S8d tuned-baseline falsification spine"
description: "Fixture-modeled decision-boundary harness comparing five large-input strategies at one strict quality contract."
status: simulated
last_reviewed: 2026-08-09
---

# S8d: tuned-baseline falsification spine

## Verdict

**Simulated fixture result, not a net-true performance claim:** the benchmark harness can identify
both win and loss regimes while refusing to call a cheaper, lower-quality pipeline a winner. Across
six modeled corpus mixes, tuned SQL/search wins the fully structured case, long context wins one
dense mixed case, and micro-context wins four sparse/adaptive or very dense cases. These boundaries
are hypotheses for live measurement, not endpoint facts.

Captured artifact:
[`s8d-local-falsification-1000-pass-2026-08-09.json`](../../experiments/microcontext/s8d-local-falsification-1000-pass-2026-08-09.json).

## Reproduce

```bash
go run ./cmd/microcontextdemo \
  -falsification-selfcheck \
  -falsification-output /tmp/falsification.json

go run ./cmd/microcontextdemo \
  -verify-falsification /tmp/falsification.json
```

## Shared contract

All five strategies receive the same 1,000-record fixture and independent deterministic grader:

1. tuned exact SQL/search;
2. retrieval plus reranking;
3. one long-context call;
4. coarse chunk map-reduce;
5. adaptive micro-context selector/tool/fold.

Eligibility requires zero false positives, zero false negatives, zero abstentions, and all emitted
citations resolving. Only eligible pipelines compete on modeled work plus scheduler overhead.

## Modeled boundary

| Ambiguity | Relations | Fixture winner | Interpretation |
|---:|---:|---|---|
| 0% | 0% | tuned SQL/search | deterministic structure dominates |
| 1% | 0% | micro-context | sparse residual semantics amortize dispatch |
| 5% | 1% | micro-context | selective relation work remains sparse |
| 20% | 5% | micro-context | adaptive work remains below long-context fixed cost |
| 40% | 20% | long context | micro scheduling/context multiplication exceeds one-call model |
| 70% | 40% | micro-context | fixture long-context capacity abstentions disqualify it |

The final row is especially provisional: its result depends on a modeled long-context capacity
failure and must not be generalized without live evidence.

## Steelman interpretations

- **Database/search advocate:** correct for structured predicates and exhaustive aggregates. The
  harness makes this strategy win rather than charging it an LLM call it does not need.
- **Retrieval advocate:** retrieval minimizes context while preserving conventional infrastructure;
  it may dominate when embedding/index cost is amortized and relation recall is tuned. The fixture's
  relation misses are not evidence against a production reranker.
- **Long-context advocate:** one call avoids scheduler and summary-loss complexity, often improving
  chronology and contradiction reasoning. Prefix caching/provider batching can narrow its cost gap.
- **Chunk map-reduce advocate:** coarse chunks reduce per-item overhead and may preserve local
  discourse better; overlap and a stronger reducer can improve quality at additional cost.
- **Micro-context advocate:** sparse adaptive routing, exact cache keys, cancellation, and path-local
  invalidation should win when only a small residual needs semantic/tool work and repeated updates
  reuse most facts.

## What remains before #6033 can close

- leakage-controlled non-fixture corpus and independent grader;
- tuned retrieval/index, long-context, and chunk prompts;
- live API results separated from controlled-kernel results;
- provider-native batching/prefix caching where available;
- actual tokens, dollars, TTFT/tail, retries, scheduler CPU/allocation, and tool latency;
- sensitivity/error bars and a `fak claim-check` bundle against the strongest tuned alternative.

Therefore this spine is tracked by #6100 and intentionally leaves parent #6033 open.
