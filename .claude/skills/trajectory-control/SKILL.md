---
name: trajectory-control
description: "The operator on-ramp to trajectory control (`trajctl`) — fak's live, forward-progress control plane over a DECLARED objective. Teaches the one primitive that carries the family (anything you want to progress gets a named, witnessed score, and every move is either improve-the-score or improve-the-scorer), then the four operator loops: declare an objective + plan + budget, read the CURVE (never a point) for the closed signal vocabulary (HEALTHY / STALL / DRIFT / DETOUR_OVERRUN), apply the when-to-nudge doctrine (the REGIME GATE. Use when this named workflow matches the task."
---

# trajectory-control — declare an objective, steer by curve, nudge only by regime

> **What this does.** A long-horizon session drifts: the goal from turn 1 is
> diluted by turn 200, the context fills with operational detail, and the run
> starts optimizing for the most recent error message instead of the objective.
> A sub-agent adopts a nearby, more tractable goal and reports success on the
> wrong thing. An error opens an unbounded repair side-quest. This skill is the
> operator on-ramp to `trajctl`, fak's control plane for exactly this: give the
> thing you want to progress a **named objective with a witnessed score**, read
> its **curve** (not a snapshot), and intervene only when the curve — and the
> regime — say you should. The default action is nothing.

The one primitive carries the whole family: **anything you want to progress gets
a named score**, and every conversation about progress becomes one of two moves —
*improve the score* or *improve the scoring method*. A goal without a score is a
wish; a score without a curve is a snapshot. **Steering reads curves, never
points.** The doctrine, witness rungs, steering ladder, and SOTA anchors live in
`docs/notes/TRAJECTORY-CONTROL-SCORE-ANYTHING-2026-07-03.md` (epic **#2533**);
this skill is the runnable operator path over the shipped spine.

---

## Positioning — reach for the right trajectory skill

Three skills share the word "trajectory"; they do not overlap.

- **trajectory-control (this skill)** steers a **live** objective forward. It
  reads a progress curve against a goal you *declared* and decides whether to
  nudge the running session. Present tense, forward-looking.
- **trajectory-audit** is the *retrospective* token/cost sweep of past Claude
  Code transcripts — machine-wide I:O ratio, cache reuse, heaviest sessions,
  stuck/churn detectors. It scores *cost of past runs*, read-only, and never
  touches a live objective.
- **trajectory-garden** gardens a *recorded corpus* (`fak traj`) — near-duplicate
  queries, cost outliers, refused traces — and proposes prunes. It scores
  *notability of past turns*, not progress of a live goal.

Rule of thumb: **audit** asks "where did the tokens go?", **garden** asks "which
past turns are cruft?", **control** asks "is this run still going where I asked?"

`trajctl` is also **not** the scorecard family (those fold a repo surface into a
`*_debt` integer), **not** `score-signal` (the CI-regression issue feeder), and
**not** `loopdrive`'s binary GOAL.md witness — it adds the continuous curve
*between* 0 and done, importing GOAL.md's Objective/Plan/Budget as its objective.

---

## Shipped surface vs. not-yet-shipped (be honest about the rung)

`trajctl` is an epic mid-build. **What ships today is the data-plane leaf**
`internal/trajctl`: the Objective + ScoreRow model, the four witness rungs, the
closed status/witness vocabularies, and the append-only JSONL ledger
(`Append` / `ReadLedgerFile` / `Fold`, schema `fak-trajctl/1`). It is a pure
data-plane leaf by design — it does not score, steer, spawn, or shell out.

**Not yet shipped** (later children of #2533): the `fak trajctl declare` /
`curve` verbs, the scorer registry + built-in scorers, the stop-hook that writes
a curve point per turn, and the regime-gated re-anchor nudge. Where a step below
would use one of those verbs, it is marked **[NOT YET SHIPPED]** and described
conceptually. **Do not invent output for those steps** — run only the witnessed
command.

The deterministic, runnable witness of the shipped spine today:

```bash
go test ./internal/trajctl/     # model + witness rungs + JSONL ledger round-trip, all green
```

That test appends an objective and one score row per witness rung (W3/W2/W1/W0),
folds the ledger, and proves the fold keeps the latest objective plus the full
score history — the ground the operator loops below stand on.

---

## The four operator loops

### 1. Declare an objective

Name the thing you want to progress, its plan, and its budget. An objective is
the durable record `{id, parent_id, statement, plan[], scorers[], budget,
status}`; the status vocabulary is closed: `active | paused | met | abandoned`.
Parent/child ids give you the hierarchy — epic > session goal > sub-agent
assignment > detour — so a side-quest is a *child* objective, not lost context.

- **[NOT YET SHIPPED]** `fak trajctl declare` will take a GOAL.md spec (or a
  `/goal` condition / dispatch issue) and append the objective row. Today you can
  construct the same record through the `internal/trajctl` API
  (`trajctl.ObjectiveRecord` → `trajctl.Append`), which is what the green test
  above exercises.
- Declare the **budget** up front (turns / tokens before escalation). An
  objective with no budget cannot detect a `DETOUR_OVERRUN`.

### 2. Read the curve, not the point

A single score is a snapshot and lies. Steering reads the **time-ordered
ScoreRows per (objective, method)** and folds them into a closed signal
vocabulary:

| signal | what the curve shows | operator read |
|---|---|---|
| `HEALTHY` | rising or plateaued-at-done progress | do nothing — the default |
| `STALL` | flat curve × high activity | the run is busy but not progressing |
| `DRIFT` | declining alignment with the objective | it is optimizing the wrong thing |
| `DETOUR_OVERRUN` | a detour past budget while the parent is paused | the side-quest ate the session |

Every score row carries a **witness rung**, same doctrine as `dos verify`:
**W3** deterministic evidence (witnessed commit, green suite, benchmark) ·
**W2** transcript/session heuristic · **W1** judge/rubric verdict · **W0**
self-report (recorded, never gates alone). Trust the curve to the height of its
rung — a W0 self-report is not a reason to act.

- **[NOT YET SHIPPED]** `fak trajctl curve <objective-id>` will render the fold +
  the signal. Today the fold primitive (`trajctl.Fold` → `State.ScoresFor`) is
  shipped and tested; the CLI rendering and the STALL/DRIFT classifier are later
  leaves.

### 3. When to nudge — the regime gate

**The default action is nothing.** Intervening in a high-scoring session is harm,
not help: mid-trajectory intervention consistently *degrades* performance in
high-success regimes and helps only in low-success ones
([arXiv:2602.03338](https://arxiv.org/abs/2602.03338)). So every rung above
"annotate" sits behind the **regime gate**: *recent curve healthy ⇒ do nothing.*

The steering ladder, weakest sufficient rung wins:

```
observe → annotate (ledger only)
        → nudge   (re-anchor: re-inject objective + curve through the session steer channel)
        → warn    (advisory guard rung / exit summary)
        → suspend (structured refusal reason, e.g. TRAJECTORY_STALL)
        → escalate (operator / Slack)
```

Re-anchor content follows **checkpoint-and-re-read**: serialize objective + curve
+ plan state and read it back fresh, rather than trusting context continuity
(the most robust goal-drift mitigation in the literature —
[2505.02709](https://arxiv.org/abs/2505.02709),
[2603.03258](https://arxiv.org/abs/2603.03258)). A nudge's outcome is itself
ledgered and scored (did the curve recover after the nudge?), so the intervention
policy is calibrated from evidence, not vibes.

- **[NOT YET SHIPPED]** the regime-gated re-anchor nudge over the `fak signal
  steer` channel is the first steering leaf of the epic. Until it lands, apply
  the gate **manually**: read the curve, and if it is `HEALTHY`, keep your hands
  off the running session.

### 4. Budget the detour — a side-quest is a child objective

A detour is not noise to suppress — it is a **child objective with its own score
and a budget**, and the point of trajectory control is that the detour
*returns*. An error burst (or a correct-but-unrelated infra fix) opens a child
objective with a turn/token budget; `DETOUR_OVERRUN` fires the return-to-main
nudge; the detour's own score records whether the repair actually landed, and the
parent's paused time stays visible in the curve. Do not kill a recoverable
detour — score the recovery (the [DataPRM](https://arxiv.org/abs/2604.24198)
corollary: scorers that punish recoverable errors prune self-correcting runs).

### The meta loop — improve the score or improve the scorer

Scoring methods are themselves objectives. A scorer's **calibration against
witnessed outcomes** is scored like anything else — same shape as the scorecard
family's intent-vs-literal honesty stick, pointed at the scorers. Anti-gaming
fences are load-bearing: every metric gets an `intent_literal_scorecard` row; any
score that rises with session length by construction is **banned** as a steering
input; every scorer declares its termination condition (an unbounded scoring loop
is a named enemy, not an accepted risk); and scores gate **harness actions only**,
never a model reward loop.

---

## The runnable demo (honest scope)

The one deterministic, model-free, network-free witness that runs end-to-end
today is the data-plane test — it proves the objective/score/witness-rung model
and the JSONL ledger that every loop above stands on:

```bash
go test ./internal/trajctl/     # → ok  github.com/anthony-chaudhary/fak/internal/trajctl
```

The full **declare → curve → nudge** walkthrough against a live demo session
depends on the `fak trajctl declare` / `curve` verbs and the stop-hook scorer,
which are later children of epic #2533 and **not yet shipped**. When they land,
this section becomes a `go run ./cmd/fak trajctl …` walkthrough with a real curve;
until then, do not present those verbs as runnable. See
`docs/run-the-demos.md` for the demo entry.

---

## What to report

A short operator note, not a wall:

- the objective (statement + budget) and whether it is `active | paused | met`,
- the **curve signal** (`HEALTHY / STALL / DRIFT / DETOUR_OVERRUN`) and the
  **witness rung** of the scores behind it — never quote a bare W0 as a reason,
- the regime-gate verdict: if `HEALTHY`, the recommendation is **do nothing**,
- for any detour, its budget, whether it overran, and whether it returned,
- honestly, which step ran (the `go test` witness) vs. which is **not yet
  shipped** and was described conceptually.

Stop there. A healthy curve is a reason to keep your hands off, not to steer.
