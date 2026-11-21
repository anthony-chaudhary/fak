---
title: "Super workstream cost model: many workstreams are cheap, navigation must stay linear"
description: "The Big-O contract for super workstreams: O(1) context in transcript length, Theta(Q) for a one-shot decision but O(1) per step through a Stepper, Theta(Q) allocation for a whole drain, a MaxItems queue cap, and Theta(W*Q) to sweep W streams — with the measured cost and the gates that pin it."
---

# Super workstream cost model

> Companion to [`super-workstreams.md`](super-workstreams.md), which defines the
> primitive. This page states and measures what it costs to *have many* and to
> *navigate* them.

A workstream is meant to be cheap to *have* and proportional to *navigate*. Two
different quantities are involved, and conflating them is how this surface drifts
into accidental quadratic behavior.

| Quantity | Bound | Basis |
|---|---|---|
| Context carried across items | **O(1) in turns/transcript**; O(32) receipts | `StreamCarryoverSeed` replaces the transcript; `CompletedItems` is windowed to `DefaultCarryoverReceiptWindow` (32) plus `CompletedItemsTotal` |
| Resident state per stream | **Θ(Q)** | the queue |
| One `DecideStep` call | **Θ(Q)** time and bytes | one normalized queue copy — correct for a one-shot decision |
| Any drain loop | use **`Stepper.Step`**, **O(1) in Q** | `NewStepper` normalizes once; the copy is amortized across the whole drain |
| Draining a Q-item stream via `Stepper` | **Θ(Q) allocation total** | Q steps × O(1); measured flat at 656–662 B/op for Q=1..10,000 |
| Queue length | **bounded by `MaxItems`** (default 1000) | over-cap specs fail validation and halt navigation: split the stream or raise `max_items` explicitly |
| One contended step (`FindNextDisjointItem`) | Θ(Q·live leases) | forward scan for a disjoint item |
| Navigating W streams (plan/status sweep) | Θ(W·Q) time; **O(max Q)** peak memory when swept one at a time | W independent spec reads; nothing aggregates all queues |

Measured on `internal/superstream` (AMD Ryzen 9 9950X, go1.26.11 windows/amd64,
`-benchtime=100x`):

| Probe | Q=1 | Q=100 | Q=10,000 |
|---|---|---|---|
| `BenchmarkDecideStepQueueScaling` (one-shot call) | 853 B/op | 19,093 B/op | **1,845,639 B/op** |
| `BenchmarkStepperStepQueueScaling` (drain step) | 662 B/op | 661 B/op | **656 B/op** |

Before 2026-11-21 a `DecideStep` normalized the queue and then had
`EvaluateContextSafety` and `BuildCarryoverSeed` normalize it two more times
(3.69 MB/op at Q=10,000). Collapsing those passes halved it to one copy
(1.85 MB/op); amortizing that copy across a drain through `Stepper` removed the
per-step term entirely (0.66 KB/op, flat). The lesson is the contract: **`DecideStep`
is a one-shot primitive; a loop must hold a `Stepper`.**

Gates that pin it (allocation bytes, no wall clock):

- `TestDecideStepCopiesQueueOnce` — a single call copies the queue at most once.
- `TestStepperStepDoesNotCopyQueue` — `Step` allocation does not scale with Q.
- `TestStepperDrainAllocationLinear` — a full drain allocates Θ(Q), not Θ(Q²);
  fails on the per-call-copy shape.
- `TestStreamSpecValidateCap` / `TestDecideStepRefusesOversizedQueue` — the
  `MaxItems` cap refuses an unbounded queue instead of paying for it.

Consequences worth designing against:

- **Many bounded streams are the intended shape.** `MaxItems` is the line between
  "many workstreams" (navigable, linear) and "one enormous queue" (a Θ(Q²) sweep).
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
go test ./internal/superstream -run '^$' -bench 'Benchmark(DecideStep|StepperStep)QueueScaling' -benchtime=100x -count=1
go test ./internal/superstream -count=1
```
