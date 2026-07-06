---
title: "Context control surfaces — the middle, the pins, and the audit gap"
description: "A map of every lever an operator or the model has over fak's planned context view (the four-area 'middle'), what pins actually guarantee, and the one long-term-correct gap: pin outcomes are only reconciled at materialize time, so the dry-run preview can lie about the pinned region. Names real signatures across internal/ctxplan and cmd/fak."
---

# Context control surfaces — the middle, the pins, and the audit gap

This note maps how much control there is over the planned context view, what
"the middle" and "pin points" mean concretely, and where the one durable gap
is. It is a design note, not a change — every claim below is grounded in a
named signature so the next rung can be built without re-discovering the terrain.

## What "the middle" is

Two distinct things share the word.

1. **The planned view itself** — `ctxplan`'s four-area profile
   (`internal/ctxplan/layout.go`, `type Layout`): **base / current / recent /
   deep**, at descending precision (`PrecisionExact → PrecisionPlanned →
   PrecisionPointer`). This is "the middle" in the architectural sense: the
   thing that sits between a linear window that blows up and lossy compaction
   that launders facts away. `DefaultLayout()` (`layout.go:77`) seeds exact base
   pins, one exact current entry, a recent tail of `DefaultRecencyWindow`, and a
   bounded deep probe.

2. **The temporal middle of history** — the span range the experimental
   **`Primacy`** weight (`forecast.go:68`) is designed to *de-emphasize*.
   `Recency` alone is monotone in step, so the four-term benefit cannot express
   "the oldest spans (system framing, original constraints) are ALSO
   load-bearing." `Primacy` is the opposite-end twin: older spans slightly
   favored, producing a U-shaped "remove-the-middle" prior. It ships **OFF**
   (default `0`) and is explicitly gated: "Gate any claim on a multi-turn fault
   measure (fak horizon-recovery), never on faith."

## What "pin points" are

`Forecast.Pins` (`forecast.go:38`) — cell IDs forced resident regardless of
score, charged against the budget **first**, landing in the **base** area at
`PrecisionExact` (`layout.go:198-213`). Two hard invariants make a pin a
guarantee rather than a loophole:

- **A pin cannot launder poison.** A pin naming a sealed/tombstoned cell is
  *refused*, not forced in (`plan.go:186`, and at page-in `fault.go:38`
  `FaultRefused`). Proven in `primacy_test.go` `TestPrimacyNeverLiftsSealed` and
  `materialize_test.go:66` `TestMaterializeWitnessReconcilesRefused`.
- **No goal ⇒ no change.** With no goal set the incremental pin set is
  byte-identical to pre-edge behavior (`ctxplan_session.go:277` `pins()`,
  `ctxplan_goalpin_test.go`).

Pins are the **imperative** control (name a span, force it). Their declared dual
is **`Forecast.Releases`** (`forecast.go:39`) — the model's `free()`: a released,
un-pinned cell is elided up front as `ElideReleased` (`plan.go:153`), cold and
recoverable, so a wrong release costs one demand-page, never a lost fact.

## The full control-surface inventory

| Lever | Where | Who sets it | Reversible |
|---|---|---|---|
| `Forecast.Pins` — force resident | `forecast.go:38` | forecast author / operator | yes (drop the id) |
| `Forecast.Releases` — `free()` dual | `forecast.go:39` | **the model** (declared) | yes, one demand-page |
| Goal-as-pin-root (first `RoleGoal`) | `ctxplan_session.go:206` | session | set a new goal |
| `Weights` incl. experimental `Primacy` | `forecast.go:56` | operator / learner | yes |
| `Layout` four areas (`--base/--recent/--deep`) | `layout.go:61` | operator | yes |
| `ProbeOptions` (candidate bound) | `index.go:157` | operator | yes |
| `Budget` + pace/throughput scaling | `ctxplan_session.go:120-158` | session pace signal | idempotent, restorable |
| Dry-run **preview** | `cmd/fak/debug_ctxplan.go` (`context-plan-preview`) | operator | read-only |

The split: pins are imperative; everything else is declarative/statistical —
weights, layout, budget shape *what wins the knapsack*, they don't name spans.

## The gap — pin outcomes are reconciled at materialize time, not plan time

The refused/rendered reconciliation vocabulary already exists — but only on the
**render** path. `MaterializeLayout` produces a `Witness` where
`Rendered + Refused == Selected` (`faithful.go:135` `Reconcile`,
`materialize.go:163`), and a sealed pin lands in `Refused` with a gate reason.

The **preview** path (`PreviewLayout` → `PreviewOf`, `debug_ctxplan.go:48`)
never materializes (by design — `debug_ctxplan.go:24`, "previewing can never
page sealed bytes in"). But that means the preview only knows pins that
*resolved to a candidate*: in `ProbeLayout` a pin id absent from `ix.byID` is
**silently dropped** (`layout.go:207`, the `if i, ok := ix.byID[id]; ok` guard)
*before the Plan exists*, and a pin naming a **sealed** span survives into the
preview's pinned region even though the served view will refuse it.

So the dry-run surface an operator inspects **can disagree with the served
view about the pinned region**, in two silent ways:

- a pin that **names no span** — dropped, invisible;
- a pin that **will be refused as poison** — shown as honored.

That is the one long-term-correct gap: the preview should tell the truth about
pins *before* any bytes move, reusing the existing `Refusal`/reconciliation
vocabulary rather than inventing a parallel one.

## Recommended sequencing (build order, each rung load-bearing for the next)

1. **Lift pin outcome to plan time — a `PinOutcome` on the Plan.**
   In the layout probe, classify each `Forecast.Pins` id as
   `honored` / `skipped-no-span` / `refused-sealed` (the classifier already
   exists for materialize — reuse `store` seal state at plan time, which is a
   metadata read, not a page-in, so the no-materialize invariant holds). Surface
   it in `PreviewLayout` and in `context-plan-preview` (text/md/json). Pure
   render + one metadata read; **no change to what wins the knapsack**. This is
   the highest-value, lowest-risk rung and it makes the sealed-pin invariant
   *visible* instead of provable-only-in-tests.

2. **Operator pin/unpin channel.** Today pins are author-set and releases are
   model-set; there is no runtime operator override. A `fak debug` verb that
   pins/unpins a specific span for a session closes the symmetry. Depends on (1)
   so the operator can *see* whether their pin resolved. Touches the live loop —
   more surface area, do it second.

3. **Graduate or kill `Primacy`.** The harness already exists —
   `internal/horizonrecovery` + `fak horizon-recovery` + `cmd/ctxplanbench`.
   What is missing is a Primacy-on-vs-off comparison *through* that harness: run
   `ctxplanbench` twice (Primacy 0 vs 0.2), feed both reports to
   `horizon-recovery`, and compare the fault-rate fence. `horizon-recovery`
   structurally refuses to print a bare `r`, so the comparison stays honest.
   Only then does `Primacy` earn a non-zero default — or a deletion.

Sequenced this way, (1) is a self-contained honesty fix that makes (2) usable
and is independent of (3)'s research question.
