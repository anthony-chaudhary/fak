---
name: steer-prs
description: The operator loop over the steer-prs overlay (`fak steer prs`) — fak's read-only view that folds the pending dev->release trunk delta into PR-sized units per (fak <leaf>) ship-stamp and renders them WORST-ATTENTION-FIRST (RESIDUAL -> UNVERIFIABLE -> CLEARED). Teaches the four-step loop (run the view and read worst-first; apply the REGIME GATE — a CLEARED unit with a healthy curve is a reason to do nothing; pick the WEAKEST SUFFICIENT RUNG on the observe -> comment -> ack -> redirect ->. Use when this named workflow matches the task.
---

# steer-prs — read the overlay worst-first, gate on regime, steer with the weakest rung

> **What this does.** A shared trunk that many agents commit to accretes work
> faster than a human can read `git log`. `fak steer prs` folds the pending
> dev->release delta into operator-legible, PR-sized units per `(fak <leaf>)`
> ship-stamp and bands each by where attention is owed. This skill is the
> operator **loop** over that view: read worst-first, keep your hands off what
> the kernel already witnessed, and when something is owed, act with the
> weakest rung that suffices. **The default action is nothing** — intervening
> in a healthy intent is a measurable harm, not diligence
> (the `/trajectory-control` regime gate, [arXiv:2602.03338](https://arxiv.org/abs/2602.03338)).

The overlay is epic **#5015** (operator steerability PRs): a read-mostly PR
overlay on a PR-free trunk, leaf `internal/steerpr`. It is **READ-ONLY and
gates NOTHING** — no rung below ever blocks a commit or a merge; the whole
thesis is observability without a merge gate.

---

## Positioning — reach for the right surface

- **steer-prs (this skill)** reads the **forming** units on the trunk,
  continuously, worst-attention-first, and asks "does anything owe a look?"
- **`fak release prplan`** is the release-time twin: the same range and the
  same `(fak <leaf>)` fold, but ordered biggest-first as a **promotion plan**.
- **`/trajectory-control`** steers a **live declared objective** by its
  witnessed progress curve. steer-prs borrows its regime gate; the curve
  doctrine (signals, witness rungs W3..W0) lives there.
- **`dos_review` / `dos commit-audit`** are the witness **oracle** under this
  view: per-commit verdicts come from one `dos commit-audit <base>..<head>
  --json` call mapped through the same keep-bit the dispatch sweep uses, so
  the view and the sweep can never disagree about whether a commit is
  witnessed.

The track's dispatchable backlog is the `steer-prs` issue view in
[`.github/issue-views.json`](../../../.github/issue-views.json).

---

## The operator loop

### 1. Run the view; read worst-attention-first

```bash
fak steer prs                 # human view, worst-first: RESIDUAL -> UNVERIFIABLE -> CLEARED
fak steer prs --json          # machine view (schema fak.steerpr.v1)
fak steer prs --base HEAD~20 --head HEAD   # bound the range explicitly
```

The band vocabulary is closed, and it is a **view** over the kernel's witness
oracle, never a second oracle:

| band | meaning | operator read |
|---|---|---|
| `RESIDUAL` | a commit in the unit made a claim the kernel could **not** witness | the only place a human look buys something — read these first |
| `UNVERIFIABLE` | not yet graded, or no checkable claim | lower priority than an unwitnessed claim; never fabricated as CLEARED |
| `CLEARED` | every claim in the unit is diff-witnessed | ~0 attention owed for "did it do what it said" |

A unit's band folds to its **worst member**. Commits with no `(fak <leaf>)`
stamp are listed separately as unstamped — visible, but carrying no band.

`--check` exits 1 when any forming unit is RESIDUAL. It **reports** — for CI
or a cadence loop — and must never sit in a commit or promotion path.

### 2. Apply the regime gate — CLEARED + healthy means hands off

**The first move is the gate, not the ladder.** A CLEARED band with no
adverse curve signal is a reason to **do nothing**: the kernel confirmed each
claim, and intervening in a high-success run degrades it (the
`/trajectory-control` regime doctrine). Over-steering is the named harm this
skill exists to prevent — a view that reads as "here are five things to do"
causes it.

Two orthogonal axes, never folded into each other:

- the **Band** answers "was each claim confirmed" (this view);
- the **Curve** answers "is the objective behind it progressing" — a CLEARED
  unit whose bound objective is `DRIFT`ing still deserves attention, and a
  RESIDUAL band is never excused by a healthy curve.

**[NOT YET WIRED]** The curve carrier on an overlay unit shipped in the leaf
(#5038: `steerpr.Curve`, mirroring trajctl's `HEALTHY / STALL / DRIFT /
DETOUR_OVERRUN` and rungs W3..W0, with W0 self-reports rendered but never
actionable), but `fak steer prs` does not yet bind live trajctl objectives to
units, so today's output carries no curve line. Until it does, read the curve
side from `/trajectory-control` directly; do not invent a curve for a unit.

### 3. Pick the weakest sufficient rung

Only when something is owed (a RESIDUAL unit, or a drifting objective behind
a cleared one), climb from the bottom — and stop at the first rung that
suffices:

```
observe  → comment → ack → redirect → pause
(weakest)                              (strongest — actively harmful if misapplied)
```

Be honest about which rungs ship. **Today only `observe` is runnable**:

- **observe** — `fak steer prs` (shipped, #5021 wave). Rendering a unit and
  deciding it needs nothing IS the loop working, not a wasted pass.
- **comment** — `fak steer comment` **[NOT YET SHIPPED — #5029]** will
  annotate a unit onto its bound issue. Until it lands, the manual rung is
  the unit's `Closes #N` issue itself:

  ```bash
  gh issue comment <issue> --body "steer-prs: <what the overlay showed and what is owed>"
  ```

- **ack** — `fak steer ack` **[NOT YET SHIPPED — #5028]** will let a human
  record "I looked at this RESIDUAL". Even when it ships, **an ack never
  lowers the residual count** (see the anti-gaming laws below).
- **redirect** — `fak steer redirect` **[NOT YET SHIPPED — #5030]** will
  re-aim an intent without touching the merge.
- **pause** — `fak steer pause` **[NOT YET SHIPPED — #5031]** will hold an
  intent via `BLOCKED_BY_HUMAN` backpressure. This is the rung the regime
  gate exists to fence: pausing a healthy intent is actively harmful.

**Do not invent output for an unshipped verb.** Run only what is marked
shipped; describe the rest conceptually, the way this section does.

### 4. Confirm the effect on the next tick

A steer without a confirmation is a guess. Re-run the view and compare:

```bash
fak steer prs --json    # re-read residual_count / the unit's band next tick
```

The residual pile falls only when the underlying work gets **witnessed** — a
follow-up commit whose diff proves the claim — never because a rung was
climbed. If the number did not move, the steer did not land; that is signal,
not a rendering problem.

---

## The anti-gaming laws (named, load-bearing)

Restated here because this is where an operator will hit them:

1. **An ack is not a witness.** `diff-witnessed` is a non-forgeable machine
   bit (the diff proves the claim); `acked` means "a human looked". The
   residual count is deliberately independent of any ack: **the pile falls
   when work gets WITNESSED, not when it gets acked.** A chronically high
   pile is a true fact about the fleet's witness discipline, not a number to
   ack away.
2. **A band reconciles pessimistically.** An operator affordance can only
   ever make a commit read *worse* than its witness verdict allows, never
   better — a CLEARED band written onto an unwitnessed commit still floors at
   RESIDUAL. Flagging something the machine cleared is allowed; laundering is
   structurally impossible.
3. **Ungraded never reads CLEARED.** If `dos` is unavailable, commits stay
   ungraded → UNVERIFIABLE — the honest read, never a fabricated CLEARED.
4. **The overlay gates nothing.** `--check` reports; it never blocks a
   commit, a merge, or a promotion. A rung that gated would turn a view into
   a second oracle.
5. **A W0 curve is rendered, never acted on.** Trust a curve to the height of
   its witness rung; a bare self-report is not a reason to steer.

---

## The deterministic witness (runs green today)

```bash
go test ./internal/steerpr/...   # → ok  github.com/anthony-chaudhary/fak/internal/steerpr
```

That suite proves the ground this loop stands on: the `(fak <leaf>)` unit
fold, worst-member banding and the deterministic worst-first order, the
pessimistic band/verdict reconcile, the curve carrier and its W0 fence, and
the structural anti-gaming property — of the whole verdict space, exactly one
value (the machine's witness bit) opens the CLEARED band, so an ack that ever
reaches the band reds the suite on the commit that wires it.

A live walkthrough is one bounded read against the real repo:

```bash
fak steer prs --base HEAD~10 --head HEAD
```

---

## What to report

A short operator note, not a wall:

- the range read and the headline fold (N commits, M units, R RESIDUAL),
- for each RESIDUAL unit: the leaf, the unwitnessed claim, and the weakest
  rung you chose (or that you chose none),
- the regime-gate verdict for everything CLEARED: **do nothing** — say so
  explicitly,
- the next-tick confirmation: did `residual_count` move, and why,
- honestly, which rungs you ran (observe, a manual `gh issue comment`) vs.
  which are **[NOT YET SHIPPED]** and were only described.

Stop there. A cleared, healthy unit is a reason to keep your hands off — the
loop succeeded every time it correctly did nothing.
