---
title: "conceptbench findings — what we learned + which fak affordance to adapt (2026-07-24)"
description: "The first conceptbench analysis (epic #2721, issue #2741): the commit_stamp frontier-vs-small contrast the dos referee has graded so far, an honest result-claim gate (replay-only ⇒ no measured gap claimed), and the concrete fak affordance adaptation filed as #5380."
---

# conceptbench findings — the adaptation loop's first turn (2026-07-24)

> Owns issue **#2741** (analysis) under conceptbench epic **#2721**. The benchmark's
> payoff is not a leaderboard — it is the **adaptation loop**: read where a model falls
> off a fak concept, change fak so weaker models stay productive, then re-measure. This
> doc turns the rows the dos referee has actually graded into **one concrete fak change**
> (filed as **[#5380](https://github.com/anthony-chaudhary/fak/issues/5380)**), and is
> scrupulous about what is and is not yet claimable.

## Result-claim gate (read this first)

**No measured frontier-vs-small gap is claimed in this doc.** As of 2026-07-24 every
graded conceptbench row is a **`replay`** row: the harness replays a *recorded*
transcript's produced commit into a scratch repo and grades it with a real
`dos commit-audit` call. The grade is real; the *provenance is replay*, so the report's
computed honesty gate correctly pins `result_claim_allowed: false`
([`experiments/agent-live/conceptbench/spine-replay.json`](../../experiments/agent-live/conceptbench/spine-replay.json),
`result_claim_reason`: "a live frontier-vs-small gateway comparison has not run … mirrors
#868"). Per this issue's DoD, **replay rows are excluded from every claim below**; they
are used only to *localize a failure mode*, which is the adaptation loop's actual input.

The measured half of #2741 — a real (non-replay) frontier-vs-small gap — is **gated on a
live gateway run** (see [Promotion evidence](#promotion-evidence-what-moves-this-toward-now)).

## The rows the referee has graded (concept: `commit_stamp`)

One concept has graded rows so far — `commit_stamp` (Conventional-Commits subject +
`(fak <leaf>)` trailer, path-scoped diff, on `main`). Both rows are `source: "replay"`,
`witness_source: dos_commit_audit`. Figures link the report artifact that carries them.

| Arm (tier) | Model | `source` | `verdict` | `witness` | `stamp_kind` | `pass` |
|---|---|---|---|---|---|---|
| frontier | `claude-opus-4-8` | **replay** | `OK` | `diff-witnessed` | `trailer/gateway` | ✅ |
| small | `claude-3-5-haiku` | **replay** | `CLAIM_UNWITNESSED` | `subject-only` | `none` | ❌ |

Source rows (verbatim `witness_source` + commit SHA):
[`spine-replay.json` rows](../../experiments/agent-live/conceptbench/spine-replay.json)
— frontier `7d19695` ("code-effect claim witnessed by a touched source file |
stamp=trailer/gateway"), small `f2a2200` ("claims tests but the diff touches no test file
| stamp=none").

**This is a replay-illustrated contrast, not a measured gap.** It says nothing yet about
how either model behaves *spontaneously* — the transcripts were authored to be a
deterministic PASS and a deterministic FAIL. What it *does* give the adaptation loop is a
concrete, referee-graded picture of the small-model failure mode to design against.

## Where is the gap widest — and is it the model or a fak affordance?

The user's hypothesis for the epic is that **verdict/injection handling** (concept #4) is
the widest frontier-vs-small gap. That concept does **not yet have graded rows**, so the
hypothesis stays **open** (named as promotion evidence below). On the one concept graded
so far, the small-model arm fails `commit_stamp` **two ways at once**, and both point at a
**fak-affordance discoverability gap**, not (only) raw capability:

1. **Unwitnessed claim.** It emitted `"fix(gateway): resolve the ready race, all tests
   pass"` over an **empty diff** → `CLAIM_UNWITNESSED` / subject-only. The "report `not
   yet` with evidence; never claim a witness you did not produce" rule is conveyed only as
   implicit doctrine, not echoed in-band at the point of the task.
2. **Missing ship stamp.** `stamp=none` — the `(fak <leaf>)` trailer grammar was not
   discovered. The task frame states it once, terse
   (*"…`(fak gateway)` ship trailer, on main"*); a weaker model does not reliably parse
   the exact trailer template from that clause.

The frontier arm cleared both (`stamp=trailer/gateway`, `diff-witnessed`). Because the
same terse injection produced a discoverable stamp for the frontier arm and not the small
arm, the **cheapest testable hypothesis is affordance, not capability**: echo the exact
grammar and the witness rule in-band for the weaker tier and re-measure before concluding
the small model simply cannot do it.

## The adaptation (named + filed): #5380

**Filed:** [#5380 — echo the `(fak <leaf>)` stamp + witness rule as a tier-gated
affordance hint for weaker models](https://github.com/anthony-chaudhary/fak/issues/5380)
(`benchmark, model, class:dev, gen/next`; own DoD + witness + contract test).

It adds a **model-tier-aware, in-band affordance hint** to the conceptbench commit-stamp
task frame (and the shared injection seam it reuses) that, for a weaker arm, **echoes the
concept's exact `(fak <leaf>)` trailer template** and the **"witness, don't claim"** rule.
Gated behind a tier/model flag (dogfood before default). Its promotion evidence is a re-run
row where the small arm's `stamp_kind` flips off `none` and/or its `verdict` moves off
`CLAIM_UNWITNESSED`, with the frontier arm unregressed.

Two sibling adaptations were considered and deliberately **left to their own issues** (out
of #5380's scope): a `lane_models` frontier-pin for a concept the small model keeps failing
(`cmd/fak/dispatch_model_policy.go`), and a new `ModelSwitchableReason` class (36185e28) if
a failure proves model-switchable. Neither is justified yet — both need the *measured*
gap the result-claim gate is still holding closed.

## Generation frame (gen/next)

### Promotion evidence — what moves this toward `now`
- A **live (non-replay) gateway run** of the two-arm spine
  ([`cmd/conceptbench/testdata/spine/spine-live.json`](../../cmd/conceptbench/testdata/spine/spine-live.json),
  wired by #5311) producing a `fak.conceptbench.v1` report with
  `result_claim_allowed: true` (≥1 live referee-witnessed headline row, no unwitnessed
  rows). This is the missing witness for a *measured* `commit_stamp` gap.
- Graded rows for the **headline concept #4** (verdict/disposition + call repair) — the
  epic's central hypothesis — for both a frontier and a small model.
- A #5380 re-run row showing the small arm's `stamp_kind`/`verdict` moved after the hint.

### Demotion / retirement evidence
- If a live run shows the small arm passing `commit_stamp` **without** the hint, #5380 is
  retired as unnecessary (the failure was a replay-transcript artifact, not a real gap).
- If the hint fires but the small arm's row does **not** move, demote the "affordance, not
  capability" diagnosis and escalate to the `lane_models` frontier-pin instead.

### Invalidating assumption
This analysis assumes the small arm's `commit_stamp` failure is **representative** of how
`claude-3-5-haiku` handles the concept. It may not be: both spine arms are **recorded
transcripts authored to a fixed outcome**, and even the live arm currently sources its
*diff* from the fixture recipe (only the subject is model-produced —
[`cmd/conceptbench/spine.go`](../../cmd/conceptbench/spine.go) `driveAndGradeLiveArm`). A
live full-forward-pass could show the small model discovering the stamp on its own,
invalidating the affordance diagnosis. The result-claim gate exists precisely so this
assumption cannot harden into a claim before the live witness lands.

## Honesty fences

- Every graded row cited here is `source: "replay"` and is **excluded from any measured
  claim** — used only to localize a failure mode.
- No leaderboard, no per-model ranking, no "gap = N%" figure is asserted; the honesty gate
  is `result_claim_allowed: false` and this doc does not override it.
- The `verdict/injection` hypothesis (concept #4) is **open**, not confirmed — it has no
  graded rows yet.
- Scope of this issue (#2741) is *analysis + one filed adaptation*; **implementing** the
  adaptation is #5380's job, and **producing the live rows** is the named promotion
  evidence, tracked, not silently deferred.
