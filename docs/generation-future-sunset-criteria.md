---
title: "Generation Future Sunset And Kill Criteria"
description: "The operational kill machinery for gen/future: a closed sunset-trigger vocabulary, a default review cadence, and the retirement-evidence contract that turns the real-options 'retire' disposition into an executable, auditable verb."
---

# Generation Future Sunset And Kill Criteria

**Issue:** #1670.
**Stream:** `gen/future`.
**Status:** research memo / kill-criteria contract for future-generation issues.

This page gives a future agent the sunset and kill machinery needed to decide
*when* and *how* a `gen/future` issue is retired, without rereading the
generation epic ([#1625](https://github.com/anthony-chaudhary/fak/issues/1625))
or the valuation model
([`docs/generation-future-real-options-model.md`](generation-future-real-options-model.md)).

It is the operational companion to the real-options model. That model defines
the *disposition* `retire` and the variable that justifies it
(`C_accumulated ≥ V`, strike unreachable, assumption fired, superseded). This
memo defines the closed trigger vocabulary that fires that disposition, the
default recheck cadence that surfaces the triggers on time, and the evidence a
closing artifact must carry so a retirement is auditable rather than a silent
label change.

## Core Rule

A future bet is retired by a **named trigger from a closed vocabulary**, on a
**stated cadence**, with a **witnessed evidence trail** — never by label
movement alone, and never by silence.

Generation stays orthogonal to the three other controls under this rule:

- **Orthogonal to priority.** A high-priority future issue that hits a sunset
  trigger is still retired; a low-priority one that keeps paying its way is
  still carried. Priority decides urgency; sunset criteria decide whether the
  option is still alive.
- **Orthogonal to shared trunk.** Retirement lands through `main` like any other
  change: the closing comment, issue close, or doc edit is by explicit path,
  signed, and witnessed. A sunset trigger never authorizes a branch, force-push,
  or history rewrite to "clean up" the retired bet.
- **Orthogonal to runtime feature gates.** Retiring a future *option* (the
  research/decision) is a different act from removing shipped *code*. Code that
  landed behind a gate is retired through the gate's own removal witness, not
  through this criteria. The two retirements can coincide, but one does not
  substitute for the other.

Do not solve sunset pressure with generation branches, hidden worktree deletes,
or unwitnessed issue closes.

## Sunset Trigger Vocabulary

Every retirement of a `gen/future` issue cites exactly one trigger from this
closed set. The set is small on purpose: a retirement that cannot name one of
these is a hidden demotion, which the anti-pattern list
([`docs/generation.md`](generation.md) §Anti-Patterns) already forbids.

| Trigger | Fires when | Witness required |
|---|---|---|
| `CARRY_EXHAUSTED` | Accumulated carry `C_accumulated` has reached or exceeded the underlying value `V`, and no near-term strike is plausible. | A recheck note naming the new `V`, `C_accumulated`, and why no strike is expected before the next recheck. |
| `STRIKE_UNREACHABLE` | The strike witness `S` is structurally unreachable — a runtime gate, compatibility policy, or proved invariant shows the surface cannot be safely exposed. | The gate, policy, or proof that closed the path, with a file/test reference. |
| `ASSUMPTION_FIRED` | A named invalidating assumption (kill criterion) recorded on the issue fired against live repo, issue, benchmark, or market evidence. | The assumption text, the evidence that fired it, and a link to the commit/issue/PR that surfaced it. |
| `SUPERSEDED` | A nearer stream shipped a design, doc, or feature that covers the option's decision. | The superseding artifact (commit, issue, doc) and one line on why it subsumes the future bet. |
| `ORPHANED` | The issue has no owner, no named recheck cadence, and no consumer — permanent parking, which the model treats as a retirement signal, not a valid disposition. | The intake note confirming no owner and no recheck date was ever recorded. |
| `STALE_RECHECK` | The issue missed its named recheck for two consecutive cycles (the default cadence window × 2). | The recheck history showing the two missed windows. |
| `HORIZON_LAUNDERED` | The issue is detected as priority laundering or current-work laundering (the `gen/future` anti-patterns). | The detection note — which anti-pattern, and the evidence the issue was mis-classified. |

The first three (`CARRY_EXHAUSTED`, `STRIKE_UNREACHABLE`, `ASSUMPTION_FIRED`)
are the *valuation* triggers: they come straight from the real-options model's
retire disposition. `SUPERSEDED` is a *portfolio* trigger: the bet died
because something better shipped. `ORPHANED`, `STALE_RECHECK`, and
`HORIZON_LAUNDERED` are *hygiene* triggers: the bet died because the lane
stopped being honest about it.

A trigger outside this vocabulary is `UNCLASSIFIED` and must not be used to
retire an issue. Declare a new trigger in this doc (with summary, witness, and
the anti-pattern it closes) before relying on it — mirroring the closed
refusal-vocabulary discipline the kernel uses for guard tokens.

## Review Cadence

A sunset trigger only fires if someone looks. The cadence is what makes the
look happen.

| Cadence class | Default window | Applies to | Recheck produces |
|---|---|---|---|
| Standard | 90 days (quarterly) | `gen/future` issues with no external deadline. | A dated recheck note: new `V`/`X`/`C_accumulated`/`σ`, disposition (hold/exercise/retire), and the next recheck date. |
| Decision-windowed | The named external deadline, or 30 days — whichever is sooner. | Issues whose `T` is bound to a market, standards, or vendor signal. | A recheck aligned to the window; retirement if the window closes without a strike. |
| Held-fast | 180 days (twice yearly) | Issues explicitly demoted to a cheaper carry tier (`hedge` disposition). | A lighter recheck confirming carry is still affordable and the strike is still conceivable. |

The discipline is the *recheck event*, not the precision of the number:

- A recheck that reaffirms **hold** must still produce a dated note. A held
  option with no recheck note for one full cadence window is a `STALE_RECHECK`
  candidate; for two windows it is a retirement candidate.
- A recheck that fires **exercise** begins promotion toward `gen/next`/`gen/now`
  per [`docs/generation.md`](generation.md) §Promotion Verbs.
- A recheck that fires **retire** cites a trigger from the vocabulary above and
  writes the retirement evidence (next section).

The cadence is enforced by recheck, not by automation, until a `fak future
audit` (or equivalent view) exists — see Future Implementation Hooks. Until
then, an operator or agent performs the recheck on the cadence and records the
note. A lane with no rechecks for a full cycle is itself a hygiene signal: the
carry budget is decorative, and the whole lane is a `HORIZON_LAUNDERED`
candidate.

## Retirement Evidence Contract

A retirement is not complete until the closing artifact carries this evidence.
The artifact is usually the issue's closing comment, but may be a commit
sidecar, a doc edit, or a project-field change.

1. **Trigger** — the single token from the sunset vocabulary that fired (e.g.
   `ASSUMPTION_FIRED`). Not prose; the token.
2. **Witness** — the variable, assumption, or artifact that moved. For
   `CARRY_EXHAUSTED`, the `V` and `C_accumulated` values; for
   `ASSUMPTION_FIRED`, the assumption text and the evidence that fired it; for
   `SUPERSEDED`, the superseding commit/issue/doc.
3. **Disposition of the option** — `retired` (closed, the decision is final) or
   `parked` (re-openable if the trigger reverses; e.g. an assumption that could
   un-fire). Parking is allowed only with a new recheck date; parking without a
   date is the `ORPHANED` trigger in disguise.
4. **Orthogonality note** — one line confirming priority, shared-trunk rules,
   and runtime gates are unaffected by this retirement (e.g. "shipped code
   behind the `X` gate is unaffected; only the research option is closed").

A retirement that omits any of these four is treated as a hidden demotion and
re-opened. Do not retire by closing the issue with no comment, by removing the
label, or by editing the milestone in silence.

## Worked Example

The hosted-multi-tenant-gateway example from the real-options model, carried
forward to its sunset:

- The option was held with a quarterly recheck and a kill condition: *"retire
  if the isolation proof cannot be strengthened within two recheck cycles."*
- At the second recheck, the `ScopeTenant`/`ScopeFleet` isolation proof has not
  advanced and the `L3_CROSS_TENANT_SCOPE_DENIED` gate still blocks the
  multi-tenant read path. The strike witness `S` (passed isolation proof AND
  unit-economics memo) is structurally unreachable on this horizon.
- **Trigger:** `STRIKE_UNREACHABLE` (with `ASSUMPTION_FIRED` as the secondary —
  the kill condition fired).
- **Witness:** the gate reference, the two missed recheck notes, and the
  unchanged isolation-proof status.
- **Disposition:** `parked`, not `retired` — if a future kernel change
  strengthens tenant isolation, the option re-opens. New recheck date set to the
  next standard cadence.
- **Orthogonality note:** no shipped code is removed; the `gen/now` L3
  machinery is unaffected; the gate stays as-is.

This is the shape a retirement memo should have: a trigger token, a witness, a
disposition with a recheck date if parked, and the orthogonality line.

## Relationship To Existing Surfaces

This criteria composes with, and does not duplicate, the existing generation
surfaces:

- **Real-options model
  ([`docs/generation-future-real-options-model.md`](generation-future-real-options-model.md))**
  provides the *valuation* (`V`/`X`/`C`/`σ`/`T`/`S`) and the *disposition*
  (`retire`). This memo provides the *trigger vocabulary* and *cadence* that
  make that disposition fire on a schedule with a witness. The two are paired:
  the valuation says *whether* to retire; this criteria says *when, why, and
  with what evidence*.
- **Debt metric ([`docs/generation.md`](generation.md) §Debt Metric)** is the
  lane-aggregate retirement signal. `unpromoted_bets` rising is the coarse view
  that the carry budget is under pressure; the per-issue trigger vocabulary
  explains *which* bet should retire first. If per-issue triggers prove
  unenforceable, fall back to the debt metric (see Invalidating Assumptions).
- **Capacity model
  ([`docs/generation-agent-capacity-model.md`](generation-agent-capacity-model.md))**
  sizes the recheck capacity the cadence consumes. A cadence the lane has no
  capacity to honor is itself a hygiene signal.
- **Promotion verbs ([`docs/generation.md`](generation.md) §Promotion Verbs)**
  — `retire` here is the same verb; this memo restricts *when* it may be
  emitted to the closed trigger set.

## Promotion And Retirement Evidence

Promotion (exercise) evidence for *this criteria itself* — i.e. evidence the
kill machinery is working — is a recheck trail:

- A recheck produced a dated note on the cadence and named a disposition.
- A retirement cited a token from the vocabulary and carried the four-piece
  evidence contract.
- The lane's open `gen/future` count tracks the carry budget rather than
  growing without bound.

Demotion or retirement evidence for *this criteria* is equally concrete:

- Rechecks stop happening on the cadence (the look is decorative).
- Retirements cite `UNCLASSIFIED` triggers or omit the evidence contract.
- The trigger vocabulary drifts (operators retire by ad-hoc prose).
- The real-options valuation model itself is retired (per its own invalidating
  assumption about `C` being estimable), which removes the inputs this criteria
  fires on.
- The invalidating assumptions below fail.

Do not promote, demote, or retire this criteria by changing labels alone. Name
the recheck or retirement that moved and the witness that moved it.

## Invalidating Assumptions

This criteria depends on these assumptions, stated so a later agent can check
them cheaply:

1. **The trigger vocabulary stays closed.** The criteria assumes retirements
   cite one of the seven tokens and that new triggers are declared in this doc
   before use. If ad-hoc prose retirements become common, the vocabulary has
   failed and the criteria degrades to "retire whenever" — at which point the
   debt metric is the honest surface. **This is the assumption most likely to
   fail**, because it depends on operator discipline rather than an enforced
   gate.
2. **The cadence is honored.** The criteria assumes someone performs the
   recheck on the stated window. If rechecks are skipped silently, the
   `STALE_RECHECK` trigger never fires and the lane still becomes a graveyard —
   the criteria exists but does not bind. A `fak future audit` that flags
   overdue rechecks is the promotion path for this assumption.
3. **The real-options inputs are estimable.** `CARRY_EXHAUSTED` and
   `STRIKE_UNREACHABLE` inherit the real-options model's assumption that `V`,
   `C`, and `S` can be stated ordinally. If that model is retired (its own
   invalidating assumption), the two valuation triggers lose their inputs and
   only the hygiene triggers (`ORPHANED`, `STALE_RECHECK`,
   `HORIZON_LAUNDERED`) remain — a weaker but still useful residue.
4. **Retirement is distinguishable from parking.** The criteria assumes an
   operator can tell a final close from a re-openable hold. If every retirement
   is silently parked (or every park silently closed), the evidence contract's
   disposition field becomes noise.

If these assumptions fail, replace the criteria with the stronger measured
surface (the debt metric, or a `fak future audit` view that reads recheck dates
from git/issue history). Do not keep an unenforced trigger vocabulary as an
operator-facing fact once operators have stopped citing it.

## Future Implementation Hooks

Reasonable next slices, each a separate issue:

- Add a `gen/future` issue-template section for the kill condition (the named
  invalidating assumption that fires `ASSUMPTION_FIRED`) and the recheck date,
  so the exercise-trigger contract from the real-options model and the cadence
  here are both captured at intake.
- Add a `fak future audit` (or a `gen/future` view) that lists open future
  issues with their last recheck date, flags any past one cadence window as a
  `STALE_RECHECK` candidate, and flags any past two windows as a retirement
  candidate. This is the promotion path for the "cadence is honored"
  assumption.
- Add a carry-budget + sunset-trigger summary to the milestone report's
  `gen/future` lane, reported alongside the existing `debt_score`.
- Add a fixture that walks three synthetic future issues through
  `CARRY_EXHAUSTED`, `STRIKE_UNREACHABLE`, and `STALE_RECHECK` and asserts the
  retirement evidence contract is satisfied.

Use [`docs/generation.md`](generation.md) for the contract,
[`docs/generation-future-real-options-model.md`](generation-future-real-options-model.md)
for the valuation that feeds these triggers, and
[`docs/generation-agent-capacity-model.md`](generation-agent-capacity-model.md)
for the recheck capacity the cadence consumes.
