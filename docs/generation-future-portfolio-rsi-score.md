---
title: "Generation Portfolio RSI Score"
description: "A decision-model spec for a portfolio-level RSI score that tracks whether the whole generation portfolio is improving over time — generation balance, promotion velocity, retired assumptions, and avoided contention — as a frontier and trend, never a bounded completion percentage."
---

# Generation Portfolio RSI Score

**Issue:** #1677.
**Parent:** [#1625](https://github.com/anthony-chaudhary/fak/issues/1625).
**Stream:** `gen/future`.
**Status:** decision-model / scorecard spec for portfolio-level recursive self-improvement.

This page specifies a **portfolio-level RSI score** for the generation program. RSI
here is *recursive self-improvement*: the object measured is not any single issue but
the **whole four-stream portfolio**, and the question is "is the portfolio getting
healthier over time?" — not "is issue #X done?" A future agent can act on this page
alone; reading the whole generation epic
([#1625](https://github.com/anthony-chaudhary/fak/issues/1625)) is not required.

## Why A Portfolio Score, Not Another Per-Issue One

The problem this issue names: *the whole portfolio should improve over time, not only
individual issues.* fak already values a **single** `gen/future` bet
([real-options model](generation-future-real-options-model.md), #1666) and compares
option value **across** streams at one instant
([proof-of-value simulator](generation-future-proof-of-value-simulator.md), #1672).
Neither answers the recursive question: *between last week and this week, did the
portfolio move in a healthier direction?* That is a **trend over the whole set**, and
it is what an RSI score is for.

This composes with, and does not duplicate, the existing surfaces:

- The [real-options model](generation-future-real-options-model.md) (#1666) values one
  issue and emits a disposition verb (exercise / hold / retire / hedge).
- The [proof-of-value simulator](generation-future-proof-of-value-simulator.md) (#1672)
  compares option cost and payoff bands **across** streams at one moment.
- The `debt_score` in [`generation.md`](generation.md) §Debt Metric is a cheap
  lane-aggregate *pressure* signal (higher = worse) read once.
- **This RSI score is the portfolio-over-time fold that sits above all of them.** It
  reads the same cheap inputs those surfaces use, but reports a **direction between two
  dated folds** so a human can see the portfolio climbing or slipping, not just its
  instantaneous debt.

## It Is A Frontier And A Trend, Never A Percentage

The generation program is an **ongoing optimization program**, not a discrete
deliverable — there is always a better balance and a faster honest promotion cadence,
and it is never "done." The project already forbids putting such a program on a
completion bar (see [`AGENTS.md`](../AGENTS.md) §Planning: two kinds of work, and
[`generation.md`](generation.md) §"Generation is independent of completion
percentage"). So the RSI score is reported the way a program is: **a frontier value
plus a trend**, never a bounded 0–100 grade. A single absolute RSI number is not a
verdict; it is only meaningful as a **Δ against its own prior fold**.

The score is also a **scorecard side-channel, not a gate**. It mirrors
`rsiloop.Measurement.Score` (see [`rsi-loop.md`](rsi-loop.md) §"Structured scores for
multi-axis controls"): "A scorecard can explain a metric; it cannot make a REVERT into
a KEEP." The portfolio RSI **explains** whether the portfolio is improving so an
operator can act on intake and rebalancing. It never blocks a commit, never reorders
priority, and never sets a runtime gate. It is evidence, not authority.

## The Four Axes

The composite RSI is a **vector**, folded to a scalar frontier only for a headline. The
vector is always reported alongside the scalar, because a rise on one axis and a fall
on another tell different health stories that a single number hides. Every input is a
count or an ordinal bucket read from surfaces the fleet already keeps — no cardinal
dollar figure is invented anywhere.

### 1. Generation balance (`B`) — is capacity spread deliberately?

Is live evidence spread across the four streams inside each stream's capacity
envelope ([capacity model](generation-agent-capacity-model.md))?

- **Inputs:** count of live-evidence issues per stream (open + witnessed-in-flight),
  against that stream's capacity envelope.
- **Direction:** a within-envelope spread scores well. A **starved** stream (zero live
  evidence — the horizon has no seed corn) *and* a **glutted** stream (over its carry
  budget — a graveyard) each subtract. Balance rewards the deliberate middle: a
  portfolio that is 100% `now` is mortgaging every later horizon and scores badly; a
  `future`-heavy graveyard also scores badly.
- `B` is a **distribution** term, not a raw count. More issues is not more balance.

### 2. Promotion velocity (`P`) — is research becoming product?

The net rate of evidence-backed movement toward `now` per recheck window.

- **Inputs:** witnessed `promote` events (a stream/milestone move toward `now` carrying
  an evidence comment or a commit generation sidecar), **minus** `demote` events caused
  by a regressed or failed witness.
- **Direction:** a portfolio that never promotes anything is **inert** — its RSI is
  flat no matter how busy it looks. Rising net promotion is the core "improves over
  time" signal the issue's `## Why` names.
- **Guard:** only *witnessed* promotions count. A bare label flip with no evidence is
  not a promotion, or the axis just rewards label-churn (the [Hidden demotion /
  Branch-by-label anti-patterns](generation.md) apply here unchanged).

### 3. Retired assumptions (`R`) — is the portfolio self-correcting?

The rate of invalidating assumptions that **fired** and produced a named
`retire`/`demote` with a closed sunset-trigger token
([sunset criteria](generation-future-sunset-criteria.md): `CARRY_EXHAUSTED`,
`ASSUMPTION_FIRED`, `STRIKE_UNREACHABLE`, `SUPERSEDED`, `ORPHANED`, `STALE_RECHECK`,
`HORIZON_LAUNDERED`).

- **Inputs:** retirement-evidence rows carrying a trigger token plus a witness.
- **Direction — the counter-intuitive, load-bearing one:** retirement is a **positive**
  contribution. A portfolio that kills dead bets is self-correcting; one that never
  retires is hoarding carry and letting `gen/future` rot into a graveyard. `R` rewards
  **honest death**, which is precisely what stops the future lane from becoming the
  undead-bet pile the real-options model warns about.
- **Guard (self-gaming trap):** a retirement counts only with a *fired named
  assumption* and a witness. You cannot inflate `R` by filing junk issues and retiring
  them, because the fire requires a real assumption line that was really invalidated
  against live repo/issue/market evidence.

### 4. Avoided contention (`A`) — is the shared-trunk partition working?

The collisions the generation partition and lane discipline prevented per window.

- **Inputs:** `dos arbitrate` GO decisions on disjoint, generation-adjacent lanes;
  `COLLISION_RISK` refusals that correctly held two workers off one tree; and the
  absence of trunk merge conflicts attributable to two streams editing the same path.
- **Direction:** rising avoided-contention means multiple generations are advancing on
  one `main` without stepping on each other — the measurable payoff of "generation is
  orthogonal to shared trunk."
- **Honesty fence (see the load-bearing assumption below):** a GO is credited to the
  *partition* only when the lanes were generation-adjacent. Generic lane leasing that
  would have happened with no generation labels at all is **not** credited, or the axis
  inflates itself with contention it never actually avoided.

### Composite

```text
RSI_portfolio(fold) = ( B, P, R, A )              # the always-reported vector
                    → scalar frontier             # a headline only
                    → Δ vs previous dated fold     # the actual verdict
```

The scalar is a headline; the **Δ against the prior fold is the verdict**. RSI **rises**
when the portfolio self-improves — balanced spread, net promotion, honest retirement,
contention held off — and **falls** when a stream starves or gluts, promotion stalls,
dead bets accumulate unretired, or two streams begin colliding on the trunk. One fold
is a snapshot; the loop is the trend.

## Worked Example

Two dated folds one recheck window apart. All inputs are counts or ordinal buckets;
no number is invented.

| Axis | Fold 1 (baseline) | Fold 2 (one window later) | Δ |
|---|---|---|---|
| `B` balance | `next` starved (0 live), `future` at budget | `next` seeded (2 live), all streams in-envelope | **up** |
| `P` velocity | 0 promotes, 0 demotes | 2 witnessed promotes, 0 regress-demotes | **up** |
| `R` retired | 0 | 1 `future` bet retired (`CARRY_EXHAUSTED` + witness) | **up** |
| `A` contention | 1 cross-stream trunk conflict | 3 arbitrate-GO on disjoint gen lanes, 0 conflicts | **up** |

Readout: **RSI rose** — the portfolio seeded a starved horizon, converted two research
bets toward `now` with evidence, honestly retired one dead future bet, and ran three
generations concurrently on `main` with no collision. That is self-improvement, and it
is visible only as a *trend*, not from either fold alone.

Now the degenerate fold that should read as a **fall**: `P = 0` (nothing promoted),
`R = 0` (nothing retired despite three future bets past their recheck cadence), `B`
glutted on `future` (over carry budget), `A` shows a fresh cross-stream conflict. RSI
falls. The operator response the score is built to trigger: **slow `future` intake,
fund the starved stream, and run the overdue rechecks** — an allocation move, not a
priority change and not a code gate. No bet is made more urgent; the *portfolio's*
health trend is what moved.

## Generation Stays Orthogonal

The RSI score answers a portfolio-health question and touches none of the three axes the
generation contract keeps separate:

- **Orthogonal to priority.** RSI measures self-improvement *over time*, not urgency. A
  rising RSI makes no bet more urgent, and a low RSI deprioritizes no stream — it is a
  *health trend* that priority multiplies against, not a priority score. `gen/future`
  stays a horizon label, not a value judgment (the issue's non-goal).
- **Orthogonal to shared trunk.** The score is computed from trunk evidence (issue
  labels, commit sidecars, the arbitration journal) and produces **no branch**. A
  rebalance it prompts still lands through `main` by explicit path with the normal
  witness and DCO rules. The `A` axis specifically *measures* the shared-trunk partition
  working; it never authorizes a per-generation branch or a worktree escape (the issue's
  first non-goal), and this spec lands on trunk like any other doc.
- **Orthogonal to runtime feature gates.** RSI is planning observability; it never gates
  code exposure. A promotion the `P` axis rewards may still ship inert behind a
  default-off gate — the gate decides exposure, the score only observed that the
  portfolio moved.

## Promotion And Retirement Evidence

**Promotion evidence** (this spec earns a move toward `now`/`next` — e.g. a runnable
`fak generation rsi` verb) is a captured portfolio fold that a human acted on: **two
dated folds** showing the trend, an operator rebalancing intake because of it (slowing a
glutted stream, seeding a starved one, running overdue rechecks), and the decision
reproduced from the same inputs. Promotion of an individual *bet* the score observes is
still the real-options model's exercise readout, landed through `main` with a witnessed
commit — the RSI score never promotes a bet, it only reports that promotions happened.

**Demotion / retirement evidence** for the score itself is equally concrete, and is
named with a closed trigger token, a witness, and a disposition — never a silent label
change — exactly as the [sunset criteria](generation-future-sunset-criteria.md) require:

- The RSI trend never changes an intake or rebalance decision across two recheck cycles
  — it is decoration, and decoration retires (`CARRY_EXHAUSTED`).
- The `debt_score` in [`generation.md`](generation.md) already captures portfolio
  health well enough that the four-axis trend adds nothing over reading it twice — fall
  back to the coarser signal and retire this one (`SUPERSEDED`).
- The four axes prove un-witnessable cheaply, so the fold is fabricated precision
  (`ASSUMPTION_FIRED`; see below).

## Invalidating Assumptions

At least one assumption, stated so it can fire:

- **Avoided contention is attributable to the generation partition.** The `A` axis
  assumes a chunk of the collisions the fleet avoids are avoided *because* generation
  labels partitioned the work — not merely because ordinary path-lane leasing
  (`dos arbitrate` on file trees) would have separated the same workers regardless of
  their stream labels. If contention avoidance is fully explained by lane discipline
  that owes nothing to generations, `A` is crediting the partition for work it did not
  do, and the axis must be dropped (or re-derived as a strict generation-attributable
  residual over the lane baseline). **This is the load-bearing assumption.**
- **The four axes are independently cheap to witness.** The score assumes balance,
  promotion, retirement, and contention are each readable per window from labels, commit
  sidecars, retirement rows, and the arbitration journal. If promotion velocity needs
  project-field history that is not cheap to read, or retirement rows are not actually
  emitted, the vector collapses and the score adds nothing over `debt_score` — retire it
  in favor of the coarser measure rather than defend it with fabricated precision.
- **A portfolio trend is more informative than the per-issue signals.** The score
  assumes an operator will act on the *portfolio* Δ (rebalance intake) and not only on
  per-issue dispositions. If nobody ever acts on the trend, the fold is a report tax and
  is itself carry (`CARRY_EXHAUSTED`).

If these fail, replace the score with the stronger surface — do not keep a four-axis
trend as an operator-facing fact once its inputs have stopped being honest.

## Future Implementation Hooks

Each a separate issue, so this spec is not silently deferred into code (and so the
generation program keeps its "spine first, then fan out" default — see
[`spine-first-defaults.md`](spine-first-defaults.md)):

- A `fak generation rsi` verb (pure fold in `internal/<name>/`, thin shell in
  `cmd/fak/`) that reads the four axes off open generation issues + the commit-sidecar
  and arbitration journals, and emits the vector, the scalar frontier, and the Δ vs the
  last recorded fold as JSON — appending one row to a versioned JSONL ledger so the
  trend is durable (the additive-only ledger discipline in
  [`generation-abi-compatibility-policy.md`](generation-abi-compatibility-policy.md)).
- A carry/trend field in the milestone report's generation section, fed by the RSI Δ,
  reported next to the existing `debt_score` so the operator sees pressure *and*
  direction side by side.
- A fixture that walks the worked example above (the rising fold and the falling fold)
  and asserts the Δ direction on each axis matches the rule — the deterministic,
  wall-clock-free witness an RSI score requires (see [`rsi-loop.md`](rsi-loop.md) §"The
  metric is a legal witness").
- A discoverability entry in [`llms.txt`](../llms.txt) §"Start here" for this page,
  landed with a clean `llms-full.txt` regeneration on a quiet tree (the generator inlines
  each linked doc's working-tree body, so regenerating on a dirty shared tree would
  sweep peers' in-flight doc edits into the aggregate — do it from a pristine snapshot).

Use [`generation.md`](generation.md) for the contract,
[`generation-future-real-options-model.md`](generation-future-real-options-model.md) for
the per-bet valuation, and
[`generation-future-proof-of-value-simulator.md`](generation-future-proof-of-value-simulator.md)
for the cross-stream option-band fold this portfolio trend sits above.
