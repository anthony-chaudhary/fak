---
title: "A surfaced choice is a triage, not a page: decenter the human from the decision"
description: "The decision-side dual of the no-babysitting doctrine. Almost every choice a fleet surfaces to a human is fake: it is really one obvious action, an evaluation for a fresh context window, a ticket-sized piece of work, or the small irreducible remainder that genuinely needs a person. internal/choicetriage is the closed vocabulary that folds a surfaced choice into that taxonomy, defaulting AWAY from paging anyone."
date: 2026-07-07
---

# A surfaced choice is a triage, not a page

Status: concept note + doctrine + one shipped leaf (`internal/choicetriage`) and
its first consumer (`internal/operatorbrief`). This note is a rung under the
[no-babysitting umbrella](CONCEPT-NO-BABYSITTING-2026-07-01.md): that note bounds
how a *genuine* human page behaves (rare, cheap, evidence-carrying, bounded
silence); this note asks the prior question — **was the page genuine at all?**
It is the decision-side dual of [`internal/waiting`](../../internal/waiting):
`waiting` bounds how long a real human hold may sit; `choicetriage` decides
whether the hold was ever a human's to make.

## The operator's ask

> Look at how we surface choices. Most of the time we stop and ask a human to
> pick one of two or three options. But that framing centers the human by
> default. Decenter it. Almost every "choice" is really: one obvious option an
> agent should just take; or an evaluation that wants a fresh context window at
> the best model to work it end-to-end; or a piece of work too big for one step
> that should become a GH ticket. The genuine "a human must decide this" case is
> the small remainder, not the default.

## The thesis

A narrating fleet **manufactures decisions**. It reaches a fork, prints two or
three options, and waits. The act of printing options makes the fork *look* like
a human's to resolve — but the framing is the bug, not the fork. Fold any
surfaced choice honestly and it lands in exactly one of four places:

| Disposition     | What it really is                                   | Who moves it                                  |
| --------------- | --------------------------------------------------- | --------------------------------------------- |
| `TAKE_OBVIOUS`  | one sane move; the "choice" is a to-do              | an agent, now, best-effort                     |
| `FRESH_CONTEXT` | knowable but not obvious                             | a fresh context window at the top model tier   |
| `FILE_TICKET`   | real work, too large for one context                | the fleet, via a new/existing GH ticket        |
| `HUMAN_RESIDUAL`| a genuine policy / auth / release / priority / trust call | a person — the only case that waits      |

The load-bearing inversion: **the default is `FRESH_CONTEXT`, never
`HUMAN_RESIDUAL`.** A human decision must be *earned* by a real signal that
names authority a person holds (approve/release a build, grant an auth, set a
priority, make a policy/legal/budget/trust call). A choice that names none of
those is, by construction, not a human decision — so it routes to a clean
evaluation, an obvious action, or a ticket. The human residual shrinks toward
zero as the other three absorb what was only ever pretending to need a person.

## Why this is the dual of no-babysitting

Babysitting is *polling* — a human running a busy-wait loop over a system with no
interrupt controller. The no-babysitting note converts every poll into a rare,
typed interrupt. But an interrupt is only cheap if it is *warranted*. A fleet
that pages a human for every fork has a working interrupt controller wired to a
smoke alarm that trips on toast. `choicetriage` is the filter on the interrupt
line: it keeps the "seven watches" honest by refusing to raise a human-decision
interrupt for something that was an obvious action or an evaluation. Green means
witnessed, not quiet; and *quiet* here means the human is not paged for a
decision the fleet could make itself.

## What shipped

- **`internal/choicetriage`** (tier 1, pure, stdlib-only, imports nothing
  internal): the closed `Disposition` vocabulary above plus `Triage(Signal)
  Verdict`, a deterministic fold — state in, disposition out, no I/O and no
  clock — so any layer (the operator brief, a dispatch loop, a stop hook) can
  route a surfaced choice through the same taxonomy. The authority test is a
  small, conservative, declared token list (mirroring
  `waiting.BlockedReasonHints`); it is deliberately hard to trip, because the
  whole point is to make `HUMAN_RESIDUAL` rare.
- **`internal/operatorbrief`** now folds every `Choice` through `Triage` and
  carries `Disposition` / `Resolve` / `NeedsHuman` as the load-bearing fields.
  A missing-report page whose fix is a runnable command (`generate
  \`fak cadence --json\`…`) becomes `TAKE_OBVIOUS` — regenerate it, page no one;
  a release go/no-go stays `HUMAN_RESIDUAL`. The legacy `Default`/`Options`
  "pick one" fields are kept for back-compat and marked for retirement.

## Rungs (what this note does NOT yet ship)

- **R1 — more consumers.** The dispatch stop-hook and notify path still phrase
  some forks as "pick one" without folding through `choicetriage`. Route them.
- **R2 — retire `Default`/`Options`.** They encode the old human-centered
  framing; `Disposition`/`Resolve` supersede them. Deprecate then remove.
- **R3 — the deeper rename.** `worktype` classifies *work*; `choicetriage`
  classifies *choices*. There is a latent shared idea — a "program" of work the
  fleet owns end-to-end vs. the residual a human owns — worth unifying so the
  two taxonomies do not drift. Tracked as [#3137](https://github.com/anthony-chaudhary/fak/issues/3137), not done here.
- **R4 — a residual counter.** Count `HUMAN_RESIDUAL` dispositions per witnessed
  unit of work, the decision-side analogue of the babysitting touch counter, and
  ratchet it down. A rising residual rate is a regression in the doctrine.

## The falsifiable property

> Over a soak window, the fraction of surfaced choices that reach a human is
> bounded and trending down, and every choice that *does* reach a human names an
> authority token a person actually holds. A human paged for an obvious action,
> an evaluation, or ticket-sized work is a bug against this note.
