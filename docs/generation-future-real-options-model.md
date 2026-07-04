---
title: "Generation Future Real-Options Model"
description: "Values each gen/future issue as a real option with carry cost, exercise trigger, and a kill condition, so the future lane pays its carry deliberately instead of accumulating parked bets."
---

# Generation Future Real-Options Model

**Issue:** #1666.
**Stream:** `gen/future`.
**Status:** research memo / scoring model for future-generation issues.

This page gives a future agent the valuation model needed to decide whether a
`gen/future` issue should be carried, exercised, or retired, without rereading
the generation epic ([#1625](https://github.com/anthony-chaudhary/fak/issues/1625)).

## Core Rule

A `gen/future` issue is an **option**, not a backlog entry. The fleet holds the
right, but not the obligation, to invest in the direction later. Keeping that
right open costs something; exercising it requires evidence. This model makes
both explicit so the future lane pays its carry deliberately instead of
accumulating permanently-parked bets (the anti-pattern
[`docs/generation.md`](generation.md) already names).

The model is a **disposition model, not a priority model.** Priority still
answers how valuable or urgent an item is. Shared trunk still requires one
`main`, explicit-path commits, and witnessed closes. Runtime feature gates
still decide whether shipped code is visible. Real-options valuation answers a
different question: given what we know now, is it still worth paying carry to
keep this future option open, or has the evidence moved enough to exercise,
demote, or retire it?

Generation stays orthogonal under this model in three ways:

- **Orthogonal to priority.** A high-priority `gen/future` issue (a decision
  window that is genuinely opening) can receive capacity; a low-value future
  option can be retired regardless of how "strategic" it sounds. Option value
  and priority are separate axes that multiply, not substitute.
- **Orthogonal to shared trunk.** Exercising an option still lands through
  `main` by explicit path with the normal witness and DCO rules. The model
  never authorizes a branch, a worktree escape, or stale trunk hygiene.
- **Orthogonal to runtime feature gates.** An exercised future option may ship
  inert behind a default-off gate; the gate decides exposure, the option model
  decided whether the investment was worth making at all.

Do not solve carry pressure with generation branches, hidden worktrees, or
ungated runtime exposure.

## Option Variables

Each `gen/future` issue is described by six variables. They are estimated
**ordinally** (better/worse, cheaper/dearer), not cardinally: these are not
tradeable assets and no honest cardinal price exists.

| Symbol | Name | What it measures | Source |
|---|---|---|---|
| `V` | underlying value | the decision, product, or market value if exercised with current evidence | the issue's `## Why`; sharpened at each recheck |
| `X` | exercise cost | cost to promote and build: engineering, gate, dogfood, review capacity | leaf tier + [`docs/generation-agent-capacity-model.md`](generation-agent-capacity-model.md) |
| `C` | carry cost | cost to keep the option open per recheck cycle: attention, recheck overhead, report tax, context pollution | named explicitly below |
| `σ` | uncertainty | how much `V` could move before the decision window closes | count of open assumptions / invalidating-assumption lines |
| `T` | time to decision | horizon until the option must be exercised, promoted, or retired | the recheck cadence / market window |
| `S` | strike evidence | the witness condition that makes exercise rational | the issue's promotion criteria |

### Carry cost in detail

`C` is the variable `#1666` explicitly asks to be valued. It is the price of
*not* deciding. It compounds each recheck cycle an option stays open:

```text
C_accumulated(issue) = sum of per-cycle carry across rechecks since intake
```

Carry is denominated in the scarce resources the capacity model already names:

- **Attention/context** — every open future issue occupies a slot in the
  milestone report, the dispatch views, and every operator/agent scan of the
  backlog. More open future issues = more scan tax = thinner focus for the
  items that matter.
- **Recheck overhead** — a future option that names a recheck cadence must be
  re-witnessed on that cadence or it goes stale. Each re-witness is a turn of
  human or agent work.
- **Assumption drift** — the longer an option is carried, the more its stated
  `V` and `S` drift from the live repo/market, raising the chance the next
  recheck is a surprise rather than a confirmation.
- **Model tax** — future issues that survive in the lane without a named
  exercise trigger bloat the lane and erode the signal value of the
  `gen/future` label itself (priority laundering becomes easier to hide).

A future issue with no named carry cost is not free; it has simply externalized
its carry onto the lane. The model refuses that: every carried option states
its `C`.

## Disposition Rules

The six variables fold into one of four dispositions. This is the scoring
surface — it produces a *verb*, not a number.

| Disposition | Promotion verb | Fires when |
|---|---|---|
| **exercise** | `promote` | `S` has fired, `V − X − C_accumulated` is positive, and `σ` has fallen enough to commit. Move toward `now`/`next`. |
| **hold** | `park` | `C` is affordable, `σ` is high, `T` is open, `S` not yet fired. Keep open; recheck on the named cadence. |
| **retire** | `retire` | `C_accumulated ≥ V`, or `S` is structurally unreachable (an invalidating assumption fired), or a nearer stream superseded it. |
| **hedge** | `demote` | `σ` fell but `V` fell with it; keep as context but move to a cheaper/deeper carry tier. |

The economically load-bearing rule is **retire on accumulated carry**. An
option that never exercises and never dies is not a free option; it is a
leak in the carry budget. The retirement disposition is what stops
`gen/future` from becoming a graveyard of undead bets.

### Carry budget

The `gen/future` lane has a finite carry budget. The sum of per-issue `C`
across the lane must stay under it, or the lowest value-per-carry options
retire first. Concretely:

```text
if  Σ C(issue) over open gen/future issues  >  carry_budget:
        retire the issues with the lowest  V / C_accumulated
        until the lane is back under budget
```

`carry_budget` is set by operator judgement against the attention/recheck
capacity the lane actually has; it is not a fixed number. The discipline is
that the budget *exists* and that retirement is the release valve, not that
the number is precise.

### Exercise trigger contract

Every `gen/future` issue should name three things for the model to apply. These
are the same three things the issue template already asks for; the model just
makes them first-class rather than incidental:

1. **Strike witness `S`** — the concrete evidence (test, command, benchmark,
   market signal, standards shift) that would make exercise rational.
2. **Recheck cadence** — when carry `C` is next paid and the variables are
   re-estimated. A future issue with no recheck date is permanently parked,
   which is itself a retirement signal.
3. **Kill condition** — which named invalidating assumption, if it fires,
   retires the option immediately rather than waiting for the next recheck.

## Worked Example

A hypothetical `gen/future` issue: *"Research whether fak should expose a
hosted multi-tenant gateway as a product surface."*

- `V` — high if it changes the go-to-market; the decision it could influence is
  "do we build a hosted tier."
- `X` — large: a hosted surface means auth, tenancy, billing, SLOs, and the
  L3 share-scope machinery the kernel already gates (`L3_CROSS_TENANT_SCOPE_DENIED`).
- `C` — moderate per quarter: one recheck of the market + one witness that the
  multi-tenant isolation proofs still hold.
- `σ` — high: the hosting market, the model-serving economics, and the
  isolation guarantees could all move.
- `T` — open: no external deadline forcing a decision.
- `S` — strike: a repeatable customer ask AND a passed `ScopeTenant`/`ScopeFleet`
  isolation proof AND a unit-economics memo showing positive contribution.

Disposition today: **hold.** Carry is affordable, uncertainty is high, the
strike has not fired. Recheck quarterly. Retire if the isolation proof cannot
be strengthened within two recheck cycles (kill condition), or if `C_accumulated`
exceeds the expected decision value before the strike fires.

This is the shape a `gen/future` memo should have: variables named, a
disposition, a recheck date, and a kill condition.

## Relationship To Existing Surfaces

This model composes with, and does not duplicate, the existing generation
surfaces:

- **`debt_score` ([`docs/generation.md`](generation.md) §Debt Metric)** is the
  *lane-aggregate* signal; this model is the *per-issue* valuation. The lane's
  `unpromoted_bets` term is the coarse signal that carry is accumulating; the
  per-item model explains *why* a given bet is still worth carrying. Falling
  back to `debt_score` alone is the documented retirement path if per-item `C`
  proves unmeasurable (see Invalidating Assumptions).
- **Capacity model ([`docs/generation-agent-capacity-model.md`](generation-agent-capacity-model.md))**
  sizes the fleet resources each horizon consumes; `gen/future` "receives
  small, expiring research/decision capacity." Carry `C` is denominated in
  exactly that capacity, and the carry budget is bounded by it.
- **Promotion verbs ([`docs/generation.md`](generation.md) §Promotion Verbs)**
  — `promote` / `demote` / `retire` / `park` map one-to-one onto
  exercise / hedge / retire / hold. The option model adds the *valuation* that
  justifies the verb; it does not introduce a new verb set.
- **Sunset and kill criteria
  ([`docs/generation-future-sunset-criteria.md`](generation-future-sunset-criteria.md))**
  is the operational companion to this model. It defines the closed
  sunset-trigger vocabulary that *fires* the `retire` disposition defined here
  (`CARRY_EXHAUSTED`, `STRIKE_UNREACHABLE`, `ASSUMPTION_FIRED`, …), the default
  recheck cadence, and the retirement-evidence contract. This model says
  *whether* to retire; the sunset criteria say *when, why, and with what
  witness*.

## Promotion And Retirement Evidence

Promotion (exercise) evidence for this model is a re-estimation readout:

- A recheck names the new `V`, `X`, `C_accumulated`, and `σ` and shows the
  strike witness `S` has fired.
- The promoted issue lands through `main` with a witnessed commit, a feature
  gate if runtime exposure is involved, and a generation sidecar.
- A held option's recheck produces a dated note confirming carry was paid and
  the disposition is unchanged (hold ≠ silence).

Demotion or retirement evidence is equally concrete:

- `C_accumulated` has reached or exceeded `V` and no near-term strike is
  plausible.
- A named invalidating assumption fired against live repo, issue, benchmark,
  or market evidence.
- A nearer stream shipped a design that supersedes the option.
- The strike witness `S` is structurally unreachable (e.g. a runtime gate or
  compatibility policy proved the surface cannot be safely exposed).
- The issue has no recheck date and no owner — permanent parking, which the
  model treats as a retirement signal, not a valid disposition.

Do not promote, demote, or retire by changing labels alone. Name the variable
that moved and the witness that moved it.

## Invalidating Assumptions

This model depends on these assumptions:

- **Carry `C` is estimable ordinally.** The model assumes a future issue's
  carry can be ranked cheaply enough (higher/lower than the next issue) to
  drive the carry-budget retirement rule. If carry proves genuinely
  unmeasurable per-item, the model degrades to the lane-aggregate
  `debt_score` and *should* be retired in favor of it rather than defended
  with fabricated precision. This is the single most important assumption.
- **Strike `S` can be stated up front.** If most `gen/future` issues cannot or
  will not name a concrete strike witness, the exercise-trigger contract is
  vacuous and the model collapses into "hold everything forever."
- **A carry budget is enforceable.** If operators never actually retire
  low-value-per-carry options, the budget is decorative and the lane still
  becomes a graveyard.
- **Ordinal valuation resists laundering.** The model assumes an operator can
  tell a high-uncertainty/high-value option from priority laundering. The
  existing anti-patterns (priority laundering, current-work laundering) apply
  here unchanged.

If these assumptions fail, replace the model with the stronger measured
surface. Do not keep ordinal option math as an operator-facing fact once its
inputs have stopped being honest.

## Future Implementation Hooks

Reasonable next slices, each a separate issue:

- Add a `gen/future` issue template section for `S` (strike), recheck cadence,
  and kill condition, so the exercise-trigger contract is captured at intake.
- Add a `fak future audit` (or a `gen/future` view) that lists open future
  issues with their last recheck date and flags any with no recheck in the
  cadence window as retirement candidates.
- Add a carry-budget field to the milestone report's `gen/future` lane,
  reported alongside the existing `debt_score`.
- Add a fixture that walks three synthetic future issues through
  exercise / hold / retire and asserts the disposition matches the rule.

Use [`docs/generation.md`](generation.md) for the contract,
[`docs/generation-agent-capacity-model.md`](generation-agent-capacity-model.md)
for the capacity envelope, and
[`docs/generation-loop-scheduling.md`](generation-loop-scheduling.md) for
admission and conflict arbitration.
