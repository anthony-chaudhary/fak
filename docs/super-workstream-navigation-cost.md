---
title: "Super workstream cost model: many workstreams are cheap, navigation must stay linear"
description: "The Big-O contract for super workstreams: O(1) context in transcript length, Theta(Q) per decision step, Theta(Q^2) per drain with bounded Q, Theta(W*Q) to sweep W streams — with the measured per-step cost and the defects that violate it."
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
| Session context carried across items | **O(1) in turns/transcript**, O(pins + completed) in receipts | `StreamCarryoverSeed` replaces the transcript, not the queue |
| Resident state per stream | **O(Q)** | the queue of `Q` work items |
| One `DecideStep` | **Θ(Q)** time and bytes | exactly one normalized copy of the queue |
| Draining a `Q`-item stream | **Θ(Q²)** total | `Q` steps × `Θ(Q)` each, small constant |
| Navigating `W` streams (status/plan sweep) | **Θ(W·Q)** time, **O(max Q)** peak memory when swept one at a time | W independent spec reads; nothing aggregates all queues |

Measured on `internal/superstream` (AMD Ryzen 9 9950X, go1.26.11 windows/amd64,
`-benchtime=200x`, `BenchmarkDecideStepQueueScaling`): a step costs **184 B per
queue item** at `Q=10,000` (1.85 MB/step), i.e. exactly one `WorkItem` copy.
Before 2026-11-21 the same probe read **369 B/item** (3.69 MB/step) because
`DecideStep` normalized the queue and then `EvaluateContextSafety` and
`BuildCarryoverSeed` normalized it again; `ApplyStep` normalized a third time for
no result. Those repeated copies are gone; the remaining single copy is the price
of a pure function over a value-typed spec. `TestDecideStepCopiesQueueOnce` pins
the one-copy bound in allocation bytes (deterministic, no wall clock) so the extra
passes cannot return unnoticed: it fails at **371 B/item > 276 B/item** on the old
shape and passes at 184 B/item now.

Consequences worth designing against:

- **`Q` is the only superlinear term.** No queue-length cap exists today
  (`MaxTurns*` bound turns, not items), so a pathological spec pays `Q²` to drain.
  Keep queues bounded, or pass an already-normalized spec through an executor loop
  that steps without re-normalizing.
- **The seed is not O(1) in `Q`.** `StreamCarryoverSeed.CompletedItems` grows with
  completed work; "O(1) context" in the main page means independent of transcript
  length, not of queue length. For long streams, carry the last `K` receipts plus
  a count.
- **`W` streams have no index.** Each `fak superstream … --spec <path>` reads one
  file and computes one stream; there is no registry to enumerate or rank streams,
  so aggregate navigation costs one invocation per known spec. If an aggregate
  surface is added, retain per-stream summaries (O(W)) and never hold the
  concatenation of all queues (O(ΣQ)).
- **The lane fleet is the worked example of the same shape.** `fak-flow wip`
  navigates 718 live worktrees at **0.101 s served** (one durable snapshot read,
  0 git spawns), **8–13 s warm** (~231 git spawns), **63.4 s cold** — linear in
  lanes, O(1) on the default path (measured 2026-11-20).

Re-measure with:

```bash
go test ./internal/superstream -run '^$' -bench 'BenchmarkDecideStepQueueScaling' -benchtime=200x -count=3
go test ./internal/superstream -run TestDecideStepCopiesQueueOnce -count=1
```
