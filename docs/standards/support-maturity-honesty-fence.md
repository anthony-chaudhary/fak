---
title: "The support-maturity honesty fence — how a support rung stays witnessed, not wished"
description: "The provenance-honesty standard for fak's support-maturity ladder (M0 none → M7 beyond-SOTA, epic #1243). It is the net-true-value lens applied to *support* claims instead of perf claims, and it holds three rules that keep the ladder from becoming a wish-list: (1) no self-reported promotion — a cell's rung rises ONLY on a witness its own author did not write (the shipgate non-forgeable keep-bit); (2) a rung can DROP — the scorecard re-derives every rung from the live coverage grid each run, so a regressed or stale witness demotes the cell with no latch; (3) every rung carries a WITNESSED / OBSERVED / MODELED label, the same closed vocabulary net-true-value and observer-effect use, and only a WITNESSED rung counts as attained. The asymmetry is the point: promotion is hard (needs a non-author witness), demotion is free (any witness loss drops it). The witness is the M6-author-only fixture: a cell claiming M6 on author-only evidence is MODELED, so it is held at its witnessed rung."
---

# The support-maturity honesty fence

"Supported" is one fuzzy word collapsing a whole ladder — none / loads / runs /
correct / optimized / SOTA-parity / beyond-SOTA. The support-maturity epic (#1243)
turns that fuzz into a graded, ordered ladder (`M0 none → M1 fenced → M2 loads →
M3 runs → M4 correct → M5 optimized → M6 SOTA-parity → M7 beyond-SOTA`) so a
10,000-step optimization loop is never pointed at something actually at "loads." A
ladder is only as honest as the rule that moves a cell up it. This page is that rule.

The commitment is one sentence: **a support rung is an envelope with a witness, not
a promise — a cell sits at the highest rung a non-author witness confirms, and not
one rung higher.** This is the [net-true-value](net-true-value.md) rubric (which
grades a *gain*) and the [observer-effect](observer-effect.md) contract (which grades
a *cost number*), turned on a third kind of claim: how mature is this support? Same
provenance vocabulary, same "no number a participant can move by narrating it,"
pointed at the ladder.

This is not a new gate. It is the *name* for a discipline the repo already runs in
pieces — the `internal/shipgate` non-forgeable keep-bit, the `internal/covmatrix`
`OracleInCI` / `StaleUnwitnessed` witness state, the `internal/supportmaturityscore`
live re-derivation, and the provenance labels the
[conflation scorecard](../CONFLATION-SCORECARD.md) already enforces. The table at the
end binds each fence rule to the stick that mechanizes it.

## The three rules

### Rule 1 — no self-reported promotion

A cell's rung **rises only on a witness its own author did not write.** This is the
one rule no prior auto-improver enforced, and `internal/shipgate` already implements
it for the RSI loop: a candidate is `KEEP`-bound only when *a witness the candidate's
author did not write* confirms a strict metric gain — otherwise `REVERT`. The
keep/revert decision is a typed verdict derived from the measurement, never from the
candidate's own claim (the "non-forgeable keep-bit").

Applied to the ladder: a cell advances to M4 (correct) only on a CI-runnable oracle
(`covmatrix.OracleInCI`), to M5 (optimized) only on a committed bench a third party
can re-run, to M6 (SOTA-parity) only on the baseline-letter discipline of
[`BENCHMARK-AUTHORITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md). An author's own "this is M6"
is not a promotion event; it is a hypothesis awaiting a non-author witness. The author
proposes the rung; the witness disposes of it.

### Rule 2 — a rung can drop

Promotion is hard; demotion is free. `supportmaturityscore.Build()` recomputes every
rung from `covmatrix.Grid()` **on each run** — it never latches a prior `SUPPORTED`.
So when a witness regresses (a test goes red) or goes stale (`covmatrix.StaleUnwitnessed`:
"SUPPORTED but no CI oracle — accelerated path runs, numeric claim unwitnessed in CI"),
the next scorecard run reads the lower rung with no human in the loop and no override.

This asymmetry is the load-bearing design: a ladder where rungs only ever rise is a
wish-list with timestamps. A ladder where any witness loss drops the cell — while a
rise demands a fresh non-author witness — is a measurement. A correct-but-slow cell
that loses its bench is honestly back at M4, not still flying an M5 flag.

### Rule 3 — every rung carries a provenance label

Every rung carries one of the same closed labels [net-true-value](net-true-value.md)
Q4 and [observer-effect](observer-effect.md) require, now read as *what kind of
evidence stands under this rung*:

| Label | The rung rests on | Can it raise a rung? |
|---|---|---|
| **WITNESSED** | a fact fak authored and a third party can re-derive — a CI oracle, a committed bench, a passing test | **Yes** — the only label that attains a rung |
| **OBSERVED** | a value relayed from an external party fak does not control — an upstream "it's supported", a third-party benchmark | No — it can *describe* a cell, but it is their witness, not fak's |
| **MODELED** | a deterministic projection — a roofline estimate, an author's expectation, "should be M5 once the fast-path lands" | No — it names a *target* rung, never an *attained* one |

The rule that follows: **a cell sits at its highest WITNESSED rung.** An OBSERVED or
MODELED claim above that is recorded as a target, never as the cell's position. A
MODELED "M6" sitting over a WITNESSED M4 is a cell at M4 with an M6 target — not a
cell at M6.

## The witness — the M6-author-only fixture

The fence is provable, not just stated. The binding invariant the code half
(`internal/supportmaturity`, the C1 ladder + C2 witness binding) must satisfy:

> A fixture cell that claims **M6** with **author-only evidence** (no non-author
> witness — its support label is therefore MODELED, an author projection) is **held at
> its witnessed rung** — the highest rung an actual non-author witness confirms (e.g.
> M4 from a CI oracle). The provenance label is asserted on **every** cell, never
> absent.

Two assertions, one per failure mode: the held-rung assertion proves Rule 1 (the
author-only claim did not promote the cell); the label-on-every-cell assertion proves
Rule 3 (no rung is unlabeled, so no claim can smuggle itself in as "just supported").
Rule 2 is witnessed by `internal/supportmaturityscore`'s existing test that the debt
count re-derives from the live grid each run.

## How this is encountered by default

The fence is reachable today from [`INDEX.md`](https://github.com/anthony-chaudhary/fak/blob/main/INDEX.md), listed directly beside
its two sibling standards ([net-true-value](net-true-value.md) and
[observer-effect](observer-effect.md)), and is pinned as epic #1243's honesty contract —
Definition-of-Done item 8, "the honesty fence holds (no self-reported promotion;
labels)." Two further placements are the named follow-on; this page does not yet claim
them as present, because the fence forbids reporting an un-landed placement as attained:

- **Agents** — a pointer in [`AGENTS.md`](https://github.com/anthony-chaudhary/fak/blob/main/AGENTS.md) beside the "every claim
  carries a tag" rule and the [net-true-value](net-true-value.md) /
  [observer-effect](observer-effect.md) lenses, so the support-side check sits where an
  agent already reads before reporting "this family is now optimized / at parity."
- **Humans** — an [`llms.txt`](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt) doc-map entry beside the two sibling
  standards. Deferred because adding one forces a regen of the generated `llms-full.txt`
  companion (the `gen_llms_full.py --check` drift gate), a separate generated-file step
  taken off the shared trunk.

## Rule → stick

| Fence rule | Stick that mechanizes it | State |
|---|---|---|
| 1 · No self-reported promotion | `internal/shipgate` non-forgeable keep-bit (non-author witness ⇒ KEEP); `covmatrix.OracleInCI`; `BENCHMARK-AUTHORITY` baseline letters | enforced for the RSI loop; binding into the ladder is the C2 follow-on (#1245) |
| 2 · A rung can drop | `supportmaturityscore.Build()` re-derives from `covmatrix.Grid()` each run (no latch); `covmatrix.StaleUnwitnessed` demotes a stale oracle | enforced |
| 3 · Provenance label on every rung | the WITNESSED / OBSERVED / MODELED vocabulary; the [conflation scorecard](../CONFLATION-SCORECARD.md) enforces WITNESSED-vs-OBSERVED separation | vocabulary enforced; per-rung label is the C1 follow-on (#1244) |

## Honest fences

- This page is a **standard plus a lens over existing sticks**, not the ladder enum
  itself. The closed M0–M7 type is C1 (#1244); the executable per-rung witness binding
  + shipgate-gated promotion + drop-on-regression is C2 (#1245). The doc states the
  rule those land; the `internal/supportmaturity` package does not yet exist, so the
  M6-author-only fixture above is the **specified, not yet executable** witness — a
  normative test the C1/C2 code must pass, named MODELED here rather than claimed as a
  green Go test it is not. That is the fence working on itself: the rule we have is
  witnessed by the shipped sticks it binds; the rung-level fixture we do not yet have
  is named as the follow-on.
- The fence governs *whether* a rung is honestly held, not *what to do* at that rung —
  the rung→dev-regime router (R0 explore … R3 production) is Plane B of #1243 (C7,
  #1250) and is out of scope here.
- A WITNESSED rung is an envelope at a stated scope, not a guarantee of zero defects:
  M4 "correct" means a CI oracle passes on the cells it covers, not that every input is
  proven — the same scope-stated reading [net-true-value](net-true-value.md) Q3 asks.
