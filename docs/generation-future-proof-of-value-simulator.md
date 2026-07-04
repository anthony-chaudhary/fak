---
title: "Generation Future Proof-of-Value Simulator"
description: "A decision-model spec for a cross-stream simulator that compares option cost, projected payoff, and uncertainty across generation streams, so a long-horizon bet can be valued as an option band instead of a pretend-shipped point estimate."
---

# Generation Future Proof-of-Value Simulator

**Issue:** #1672.
**Parent:** [#1625](https://github.com/anthony-chaudhary/fak/issues/1625).
**Stream:** `gen/future`.
**Status:** simulator spec / decision model for cross-stream option valuation.

This page specifies a **proof-of-value simulator** for long-horizon bets. It lets a
future agent estimate what a `gen/future` bet is worth **without pretending it has
shipped**, and compare option cost, projected payoff, and uncertainty **across the
four generation streams** — so capacity is allocated to horizons deliberately rather
than to whichever issue is loudest. A future agent can act on this page alone;
reading the whole generation epic
([#1625](https://github.com/anthony-chaudhary/fak/issues/1625)) is not required.

## Why A Simulator, Not A Spreadsheet

The problem this issue names: *future work needs a way to estimate value without
pretending it has shipped.* A single projected-payoff number does exactly the thing
the stream forbids — it launders a research option into a fact. The honest object is
a **distribution**: a band of net outcomes, plus the probability the exercise
condition fires inside the horizon. The simulator's job is to produce that band and
compare it across streams, never to emit one fabricated point estimate.

This composes with, and does not duplicate, the per-issue valuation:

- The [real-options model](generation-future-real-options-model.md) (#1666) values
  **one** `gen/future` issue via six variables (`V`, `X`, `C`, `σ`, `T`, `S`) and
  folds them to a disposition verb (exercise / hold / retire / hedge).
- **This simulator lifts that per-option valuation to a portfolio across streams.**
  It runs many bets through a scenario sweep and reports, per stream, the aggregate
  option cost, the projected-payoff band, and the aggregate uncertainty — the
  cross-stream comparison the per-issue model cannot give on its own.
- The [sunset and kill criteria](generation-future-sunset-criteria.md) (#1670)
  consume the simulator's `retire` signals through their closed trigger vocabulary
  (`CARRY_EXHAUSTED`, `STRIKE_UNREACHABLE`, …).

## Inputs

Each bet is one row. The per-bet variables are exactly the six the real-options
model already defines, estimated **ordinally** (better/worse, cheaper/dearer), never
cardinally — these are not tradeable assets and no honest cardinal price exists.
Two fields are added for the cross-stream fold:

| Field | Meaning | Source |
|---|---|---|
| `stream` | `now` / `next` / `second-next` / `future` | the issue's generation label |
| `V` | underlying value if exercised with current evidence | the issue's `## Why` |
| `X` | exercise cost to promote and build | leaf tier + [capacity model](generation-agent-capacity-model.md) |
| `C` | carry cost per recheck cycle | named on the issue (the real-options model requires it) |
| `σ` | uncertainty — how far `V` could move before `T` | count of open / invalidating assumptions |
| `T` | time to decision (recheck cadence or market window) | the issue's cadence |
| `S` | strike evidence that makes exercise rational | the issue's promotion criteria |
| `p_S` | ordinal probability the strike fires within `T` (low / med / high) | the recheck trail |

A bet with no named `C` or `S` is **not** admissible as a zero — it is refused the
same way the real-options model refuses an un-priced carry, because a missing input
silently understates cost and overstates payoff.

## Mechanics

The simulator is a **scenario sweep**, not a cardinal Monte-Carlo. Dollar-precise
draws would manufacture exactly the false confidence the stream bans, so each bet is
evaluated in three ordinal scenarios and the spread between them **is** the reported
uncertainty:

1. **Widen by uncertainty.** For each bet, take a payoff bucket in each of
   `{pessimistic, base, optimistic}`. `σ` sets how far pessimistic and optimistic
   sit from base: a high-`σ` bet has a wide band, a low-`σ` bet a narrow one.
2. **Net against cost.** In every scenario, subtract exercise cost `X` and the carry
   `C` accumulated over horizon `T` (`C_accumulated = C × rechecks_until_T`) from the
   scenario payoff. The net can be negative — a bet that costs more to carry and build
   than it can return in its own base case is doing real, reportable harm.
3. **Weight by strike reachability.** Scale the payoff side by `p_S`: an optimistic
   payoff that requires a strike almost certain never to fire is discounted toward its
   carry-only (retire) outcome.
4. **Fold per stream.** Aggregate the per-bet net bands within each generation stream.

The load-bearing rule is step 2's honesty: **the output is a band and a verb, never a
single number.** If a caller wants one figure, the correct figure is the *floor* of
the base-case net band, because the stream's promise is "do not overstate a future
bet."

## Outputs

For each generation stream, and for the portfolio as a whole:

- **Option cost** — total `X + C_accumulated` the stream is committing / carrying.
- **Projected-payoff band** — `[pessimistic, base, optimistic]` net-of-cost, strike-weighted.
- **Aggregate uncertainty** — the width of the band (a wide band on a stream that is
  already consuming capacity is the signal to slow intake there).
- **Positive-net count** — how many bets in the stream clear a positive base-case net.
- **Retire candidates** — bets whose `C_accumulated ≥` their optimistic payoff, or
  whose `p_S` is `low` with a horizon nearly spent. These feed the
  [sunset trigger](generation-future-sunset-criteria.md) vocabulary directly.

The decision readout is a **comparison, not a ranking-by-priority**: "stream *future*
is carrying more option cost than its base-case band returns" is an allocation
finding, and it is orthogonal to how urgent any single bet is.

## Worked Example

A synthetic four-bet portfolio (ordinal buckets; L/M/H):

| Bet | stream | V | X | C_acc | σ | p_S | Base-case net (strike-weighted) |
|---|---|---|---|---|---|---|---|
| addressable-KV product tier | future | H | H | M | H | L | **negative** (wide band, strike unlikely) |
| hosted multi-tenant gateway | future | H | H | M | H | M | thin-positive optimistic, negative floor |
| backend conformance badge | next | M | M | L | M | H | positive (near strike, low carry) |
| standards-body seat | future | M | L | H | H | L | **negative** (carry `H` swamps a `L`-cost, low-`p_S` option) |

Readout: the `future` stream is holding three bets but only one has a non-negative
base case, and its total carry exceeds its base-case payoff band — an
**over-carried** stream. The simulator flags *standards-body seat* as a retire
candidate (`C_acc ≥` payoff, `p_S = L`) and marks *addressable-KV product tier* as
hold-and-recheck (wide band, strike could still open). The `next` bet clears on its
own and is a promote candidate. No number was invented; every cell is an ordinal
bucket and the output is a set of dispositions plus one over-carry finding.

## Generation Stays Orthogonal

The simulator answers an allocation question and touches none of the three axes the
generation contract keeps separate:

- **Orthogonal to priority.** A `future` bet with a strong band is not automatically
  urgent, and a weak band is not automatically deprioritized work — it is a *carry*
  decision. Payoff band and priority are separate axes that multiply, not substitute.
  `gen/future` remains a horizon label, not a value judgment (the issue's non-goal).
- **Orthogonal to shared trunk.** A promote signal still lands through `main` by
  explicit path with the normal witness and DCO rules. The simulator never authorizes
  a per-generation feature branch or a worktree escape (the issue's first non-goal),
  and it produces no branch of its own — this spec lands on trunk like any other doc.
- **Orthogonal to runtime feature gates.** A bet the simulator says is worth
  exercising may still ship inert behind a default-off gate; the gate decides
  exposure, the simulator decided only whether the option cost was worth paying.

## Promotion And Retirement Evidence

**Promotion evidence** (this simulator earns a move toward `now`/`next` — e.g. a
runnable `fak generation simulate` verb) is a captured cross-stream readout that a
human acted on: a dated fold showing one stream over-carried, an operator retiring or
reweighting at least one bet as a result, and the decision reproduced from the same
inputs. Promotion of an individual *bet* the simulator evaluates is the real-options
model's exercise readout — new `V`/`X`/`C`/`σ` plus a fired strike `S` — landed
through `main` with a witnessed commit.

**Demotion / retirement evidence** for the simulator itself is equally concrete:

- The bands it emits never change a real allocation decision across two recheck
  cycles — it is decoration, and decoration retires (`CARRY_EXHAUSTED`).
- The lane-aggregate `debt_score` in [`generation.md`](generation.md) already
  captures the over-carry signal well enough that the per-stream band adds nothing —
  fall back to the coarser measure and retire this one (`SUPERSEDED`).
- The ordinal inputs prove un-rankable in practice, so the fold is fabricated
  precision (`ASSUMPTION_FIRED`; see below).

Retirement is named with a trigger token, a witness, and a disposition — never by a
silent label change — exactly as the [sunset criteria](generation-future-sunset-criteria.md)
require.

## Invalidating Assumptions

At least one assumption, stated so it can fire:

- **Ordinal payoff buckets are comparable across streams.** The simulator assumes a
  `future` "base-case H" and a `next` "base-case H" are on the same rough scale enough
  to fold into a per-stream band. If a `now` deliverable's value and a `future`
  option's value are genuinely incommensurable, the cross-stream comparison is
  meaningless and the simulator must retire in favor of per-stream, within-stream-only
  reads. **This is the load-bearing assumption.**
- **`σ` and `p_S` are estimable cheaply.** If most bets cannot honestly name an
  uncertainty width or a strike probability, the band collapses to `[V, V, V]` and the
  simulator adds nothing over listing `V`.
- **Over-carry is actionable.** The simulator assumes that surfacing an over-carried
  stream causes an operator to actually retire or reweight. If nobody ever acts on the
  finding, the fold is a report tax and is itself carry (`CARRY_EXHAUSTED`).

If these fail, replace the simulator with the stronger measured surface — do not keep
an ordinal band as an operator-facing fact once its inputs have stopped being honest.

## Future Implementation Hooks

Each a separate issue, so this spec is not silently deferred into code:

- A `fak generation simulate` verb (pure fold in `internal/<name>/`, thin shell in
  `cmd/fak/`) that reads the per-bet variables off open `gen/future` issues and emits
  the per-stream band + retire candidates as JSON.
- A fixture that walks the worked example above and asserts the dispositions
  (over-carry finding, one retire candidate, one promote candidate) match the rule.
- A carry-budget field in the milestone report's `gen/future` lane, fed by the
  simulator's per-stream option-cost total, reported next to the existing `debt_score`.

Use [`generation.md`](generation.md) for the contract,
[`generation-future-real-options-model.md`](generation-future-real-options-model.md)
for the per-bet valuation this simulator aggregates, and
[`generation-future-sunset-criteria.md`](generation-future-sunset-criteria.md) for the
retirement machinery its `retire` signals feed.
