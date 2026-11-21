---
title: "Super workstream cost model: many workstreams are cheap, navigation must stay linear"
description: "The Big-O contract for super workstreams: O(1) context in transcript length, O(1) allocation per decision step, Theta(Q) scan only at a seed boundary, Theta(W*Q) to sweep W streams — with the measured cost and the regression gates that pin it."
---

# Super workstream cost model

> Companion to [`super-workstreams.md`](super-workstreams.md), which defines the
> primitive. This page states and measures what it costs to *have many* and to
> *navigate* them.

A workstream is meant to be cheap to *have* and proportional to *navigate*. Two
different quantities are involved, and conflating them is how this surface drifts
into accidental quadratic behavior.

| Quantity | Bound | Why |
|---|---|---|
| Session context carried across items | **O(1) in turns/transcript**, Θ(completed) receipts | `StreamCarryoverSeed` replaces the transcript, not the queue |
| Resident state per stream | **Θ(Q)** | the queue of `Q` work items |
| One `DecideStep`, no seed boundary | **O(1) time and allocation** | defaults are applied to the limits only; no queue copy, no queue-sized prealloc |
| One contended step (`FindNextDisjointItem`) | Θ(Q·live leases) | forward scan for a disjoint queue item |
| One seed-emitting step (`ADVANCE_QUEUE`, `STREAM_COMPLETE`, `RESET_CONTEXT`) | Θ(Q) scan time, Θ(completed) space | the scan is inherent; the seed carries completed receipts only |
| Draining a `Q`-item stream | Θ(Q²) scan time when every boundary emits a seed; Θ(Σcompleted) summaries | next lever: a last-K receipt cap or an incremental cursor |
| Navigating `W` streams (status/plan sweep) | Θ(W·Q) time, O(max Q) peak memory when swept one at a time | W independent spec reads; nothing aggregates all queues |

Measured on `internal/superstream` (AMD Ryzen 9 9950X, go1.26.11 windows/amd64,
`-benchtime=200x`, `BenchmarkDecideStepQueueScaling`): a decision step allocates
**632–634 B/op and 21 allocs/op flat from Q=1 to Q=10,000** — the same bytes at
every queue length, ~0.7–1.5 µs/op. Before 2026-11-21 the same probe read
**369 B per queue item** (3.69 MB/op at Q=10,000) because `DecideStep` normalized
the queue and then `EvaluateContextSafety` and `BuildCarryoverSeed` normalized it
again, while `ApplyStep` normalized it for no result. A first fix collapsed that
to one copy (184 B/item); the residual copy was then removed outright, because
the decision helpers read only the limit fields and base pins — the queue they
walk is the caller's `state.Queue`.

Two deterministic gates pin this (allocation bytes, no wall clock):

- `TestDecideStepCostIsQueueIndependent` — Q=200 vs Q=2,000 must allocate within
  8 KiB of each other. On the copy shape it fails at
  `Q=200 41588 B/op, Q=2000 373767 B/op`.
- `TestCarryoverSeedAllocatesCompletedOnly` — with one completed item, the seed
  must not size itself to the queue. Before the fix it allocated
  `len(state.Queue)` summaries and failed at `Q=200 62320 B/op, Q=2000 567395 B/op`.

Consequences worth designing against:

- **`Q` is the remaining superlinear term.** No queue-length cap exists today
  (`MaxTurns*` bound turns, not items) and every seed boundary rescans the queue.
  Keep queues bounded, or make the completed-receipt sweep incremental.
- **The seed is not O(1) in `Q`.** `StreamCarryoverSeed.CompletedItems` grows with
  completed work; "O(1) context" means independent of transcript length, not of
  queue length. For long streams, carry the last `K` receipts plus a count.
- **`W` streams have no index.** Each `fak superstream … --spec <path>` reads one
  file and computes one stream; there is no registry to enumerate or rank streams,
  so aggregate navigation costs one invocation per known spec. If an aggregate
  surface is added, retain per-stream summaries (O(W)) and never hold the
  concatenation of all queues (O(ΣQ)).
- **Lane navigation follows the same shape.** A served snapshot answers a fleet
  census in O(1) file reads, and a fresh census is linear in lanes; the durable
  snapshot is what keeps the default path off the linear term.

Re-measure with:

```bash
go test ./internal/superstream -run '^$' -bench 'BenchmarkDecideStepQueueScaling' -benchtime=200x -count=3
go test ./internal/superstream -run 'TestDecideStepCostIsQueueIndependent|TestCarryoverSeedAllocatesCompletedOnly' -count=1
```
