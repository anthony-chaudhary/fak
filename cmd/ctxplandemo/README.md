# Context Planner Demo

`ctxplandemo` is the runnable witness for the context planner: a fresh turn is treated as an O(1) materialized view over a lossless history store, with elided spans kept recoverable by reference.

## Prerequisites

Requires Go only. It does not need a model, tokenizer file, GPU, network, or persistent store.

## Quick Start

From the repo root:

```bash
go run ./cmd/ctxplandemo -selfcheck
```

The run completes in a few seconds and returns exit code 0 only if the budget, poison exclusion, faithfulness, candidate-index, and scaling invariants hold. Output is deterministic because the history store, forecast, and cost model are synthetic and fixed.

## What You See

```text
== Planned O(1) view (the fresh turn's history) ==
  rendered 3 span(s) through the trust gate
== Faithful (planned view) vs lossy (compaction) - same residency, opposite recoverability ==
== Scaling law: resident tokens & exact-recall across the turn horizon ==

OK - O(1) view planned under budget, poison kept out, faithfulness proven, scaling curve bent.
```

## What This Does Not Claim

This demo does not claim the live HTTP/model loop is using this planner on every turn. It proves the planner's structural properties over a synthetic in-memory store, including recoverability and poison-never-resident behavior.

## Related Docs

- [O(1) turn context planner note](../../docs/notes/O1-TURN-CONTEXT-PLANNER-2026-06-23.md)
- [Engineering is building loops](../../docs/explainers/engineering-is-building-loops.md)
