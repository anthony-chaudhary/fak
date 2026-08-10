---
title: "Micro-context S8b read-only tool enrichment"
description: "Fixture-backed bounded fan-out from adaptive micro-windows into allowlisted read tools."
status: observed
last_reviewed: 2026-08-09
---

# S8b: bounded read-only tool enrichment

## Verdict

**Observed in the fixture:** a micro-window can request external evidence without acquiring
open-ended tool authority. The controller converts typed `fetch-comments(issue_id)` requests
into bounded reads, deduplicates identical keys, releases model slots before tool waits, enforces
global and per-resource quotas, caches typed observations across restart, and folds only sources
that an independent read-back verified.

Captured artifact:
[`s8b-local-tool-enrichment-1000-pass-2026-08-09.json`](../../experiments/microcontext/s8b-local-tool-enrichment-1000-pass-2026-08-09.json).

This is a local fixture/stub witness, not GitHub API throughput, token, dollar, or latency evidence.

## Reproduce

```bash
go run ./cmd/microcontextdemo \
  -tool-enrichment-selfcheck -workers 16 \
  -tool-enrichment-output /tmp/tool-enrichment.json

go run ./cmd/microcontextdemo \
  -verify-tool-enrichment /tmp/tool-enrichment.json
```

## Captured contract

- 1,000 source records produce 28 selector windows and 33 logical read requests.
- Four duplicate reads reuse existing observations; 26 unique requests consume quota.
- Twenty-five reads become independently read-back observations, one times out after one retry,
  two unopened reads are cancelled before dispatch, and one excess unique read is denied by quota.
- Physical invocation count is 27: 26 first attempts plus one bounded retry.
- A serialized observation checkpoint is loaded into a fresh coordinator; all 26 completed keys
  hit cache after restart and cause zero new dispatches.
- Peak tool concurrency is 16 while `model_slots_during_tool_wait` is zero.
- One large result partitions to fan-out four at depth one, below caps of four and two.
- Every final fold citation names the fixture URI and exact SHA-256 read-back hash.
- An undeclared write such as `close-issue` is denied before dispatch.

## Why this is stronger than “let each agent call tools”

The model proposes a typed request; it does not own dispatch. This preserves four separate states
that a prose-only agent loop tends to blur: `observed`, `timeout`, `cancelled`, and `not_run`.
Cancellation only prevents unopened work—it does not pretend to reverse a dispatched read—and a
timeout never silently becomes negative evidence.

## Steelman perspectives

### Why this pattern may generalize

The expensive input is replaced by a sparse evidence graph. Most records remain model-free; only
uncertain records acquire evidence, identical reads collapse to one physical call, and oversized
tool results recursively re-enter the same bounded partition/fold machinery. The same shape fits
issues/comments, database rows plus foreign-key lookups, log events plus traces, or documents plus
retrieval, provided reads have stable semantic keys and provenance.

### Why a conventional query or deterministic pipeline may still win

If the desired predicate is expressible in SQL, `jq`, GraphQL selection, an index, or server-side
search, run that first: it is cheaper, more reproducible, and easier to capacity-plan. Micro-windows
are justified for residual semantic ambiguity and adaptive evidence acquisition, not as a fashionable
replacement for ordinary query planning.

### Why “one tiny agent per record” can be worse

Fine granularity adds scheduler overhead, correlated model error, cache-key complexity, tail latency,
and dependency load. A selector that is nearly as expensive as the skipped filter is a loss. Quotas
can also bias results by starving late records, and stale cached observations can preserve a wrong
answer efficiently. The tuned-baseline program (#6033) must falsify the economic case rather than
assuming sparse fan-out wins.

### Security and reliability view

Read-only does not mean harmless: reads can exfiltrate secrets, amplify traffic, or expose prompt
injection in fetched content. Capability allowlists, egress/resource bounds, untrusted-data marking,
meaning-complete cache keys, and independent read-back remain mandatory. Effectful tools are a
separate contract (#6034) because cancellation and caching cannot erase an already-landed write.

## Boundary

This spine composes the authority shape already represented by `internal/microagent.ToolExec`, but
keeps fixture-specific batch dedupe and quota policy in the demo. Production extraction belongs only
after the tuned baseline and real adapter identify a stable shared abstraction.
