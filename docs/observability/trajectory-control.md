---
title: "fak trajectory control (trajctl)"
description: "The live forward-progress control plane over declared objectives: witnessed score curves, closed steering signals, and a regime gate that defaults to doing nothing."
---

# Trajectory control — score anything, steer by curve

**Trajectory control (`trajctl`) is the live, forward-progress control plane over
declared objectives.** Anything you want to progress gets a named objective and a
score; every conversation about progress becomes one of two moves — *improve the
score* or *improve the scoring method*. Steering reads curves, never points.

It exists because fak's other trajectory machinery answers different questions.
This page is the operating description of the shipped surface
(`internal/trajctl` + `internal/trajctlhook`); the durable concept statement and
SOTA anchors live in the
[design note](../notes/TRAJECTORY-CONTROL-SCORE-ANYTHING-2026-07-03.md) (epic
#2533), and the first real end-to-end run is the
[dogfood witness report](../notes/TRAJCTL-DOGFOOD-2026-07-07.md) (#2541).

## Not the trajectory data plane (and not four other things)

The word "trajectory" already names the *retrospective* corpus toolkit on the
[trajectory observability page](trajectory.md). The line, drawn once:

| name | question it answers | tense |
|---|---|---|
| **trajctl** (this page) | is this live run still moving toward its declared objective? | live |
| **`fak traj`** (`internal/trajectory` + `internal/trajhook`) | which *past* turns were notable / redundant / expensive? | retrospective |
| **scorecard family** | what is the measured debt of a *repo surface*? | repo, not runs |
| **score-signal** | which CI scorecard regressions should auto-file issues? | feeder |
| **`fak signal steer`** | how does an operator inject free-text into a running session? | one actuator rung trajctl *uses* |
| **loopdrive GOAL.md witness** | is the goal witnessed-done, yes or no? | the binary endpoint trajctl stretches into a curve |

trajhook's `Score` is *notability of a past turn*; trajctl's score is *progress
of a live objective*. Same word, different tense, different ledger.

## The model (`internal/trajctl`)

An **Objective** is a declared goal with an id, an optional parent (hierarchy:
epic > session goal > sub-agent assignment > detour), attached scorer methods, a
budget, and a status (`active | paused | met | abandoned`). A **ScoreRow** is one
scored observation of one objective: a normalized value, the method and version
that produced it, a **witness rung**, and an evidence pointer. The ledger is
append-only JSONL (`fak-trajctl/1`), same discipline as the guard decision
journals: objectives and scores survive process death, and evidence pointers are
re-verified at read time (`audit.go` demotes a row whose evidence no longer
resolves).

### Witness rungs — scores are witnessed, not self-reported

Every ScoreRow carries its evidence strength, same doctrine as `dos verify`:

- **W3** — deterministic evidence: a witnessed commit, a green suite, a benchmark
  harness. The shipped `CommitProgressScorer` is W3: fraction of the declared
  plan's phases with a real landing commit, resolved through `git cat-file` —
  zero model calls.
- **W2** — transcript-derived heuristic: the shipped stall scorer folds
  activity/progress divergence from sessionaudit-shaped signals.
- **W1** — a structured, pinned-schema judge verdict.
- **W0** — the session's own self-report. Recorded, never load-bearing: a stale
  or dangling W3 row is *demoted* to W0 by the audit, not silently kept.

## The curve and the closed signal vocabulary

`curve.go` folds the time-ordered ScoreRows per (objective, method) into a curve
(`fak-trajctl-curve/1`) and derives one signal from a closed vocabulary:

- **HEALTHY** — witnessed progress rising or steady. HEALTHY is *nothing*: this
  is the **regime gate**. Intervening in a high-scoring session is harm, not
  help (arXiv:2602.03338 — mid-trajectory intervention degrades performance in
  high-success regimes), so every steering rung above "annotate" sits behind
  this gate, and the controller's default action is to do nothing.
- **STALL** — a flat progress curve while activity stays high.
- **DRIFT** — a declining witnessed-progress curve; alignment is decaying.
- **DETOUR_OVERRUN** — a **detour objective** (a child objective opened for a
  side-quest, with its own scorers and budget) has run past its declared budget
  while its parent sits paused. A detour is not noise to suppress — it is an
  objective too, and trajectory control means the detour *returns*.

The steering ladder above the gate (`observe → annotate → nudge → warn →
suspend → escalate`) is the epic's follow-on wave; the re-anchor nudge (#2540)
is not landed yet.

## The shipped surface

- `internal/trajctl` — Objective + ScoreRow model, witness rungs, the JSONL
  ledger, the curve fold, the read-time evidence audit, and the first two
  scorers (W3 commit progress, W2 stall).
- `internal/trajctlhook` — `RunTurnEnd`, the turn-end sampling seam a harness
  stop-hook drives so the curve gets a point every turn.
- `fak trajctl declare | close | list | curve` — the objective-lifecycle CLI is
  authored in `cmd/fak/trajctl.go` but its `main.go` wiring is parked behind an
  in-flight lane (#2765); `fak focus-score` already reads the same ledger.

The dogfood run scored seven plan phases against their real landing commits: a
monotonically rising 0.14→0.86 W3 curve and a verified-abstention regime-gate
decision (`HEALTHY → withhold`). Known blind spots are filed, not hidden: no
live sampler producer / phase→commit binding source yet (#3129), and the commit
scorer proves SHA existence, not semantic match (#2566/#2568).

## Anti-gaming fences

Every trajctl metric gets an intent-vs-literal honesty row; any score that rises
with session length by construction is banned as a steering input (the
cache-hit vanity-metric lesson); every scorer declares its termination
condition; and scores gate harness actions only — they never feed a model
reward loop directly.

## Read next

- [Trajectory observability primitives](trajectory.md) — the retrospective data
  plane this control plane is *not*.
- [The trajectory-control family in the concept glossary](../fak/concept-glossary.md)
  — the one-line disambiguations, scorecard-anchored.
