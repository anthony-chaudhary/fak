---
title: "Operator-steerability PRs: observability, not a merge gate"
description: "How `fak steer prs` bundles continuous-merge trunk commits into operator-legible PR units, and why that overlay must never become a merge gate."
---

# Operator-steerability PRs — observability without a merge gate

**The doctrine note for epic [#5015](https://github.com/anthony-chaudhary/fak/issues/5015).**
Leaf: [`internal/steerpr`](../internal/steerpr/steerpr.go) · CLI:
[`fak steer prs`](../cmd/fak/steer_prs.go) (schema `fak.steerpr.v1`) · operator loop:
the [`/steer-prs`](../.claude/skills/steer-prs/SKILL.md) skill.

This page exists to preserve the one idea in the epic that is easy to state and easy to
forget. A future reader who finds a banded view with a `--check` that exits 1 will
reasonably conclude the obvious next step is to wire it into CI as a merge gate. **That
conclusion is wrong, and this note is the reason why.**

## The separation: a PR is two functions, and fak wants exactly one of them

A conventional pull request bundles two orthogonal functions:

1. **Merge gate** — nothing lands until a human (or a bot) approves.
2. **Observability / steering** — the work arrives as a coherent, reviewable unit a
   human can react to.

fak's trunk takes work PR-free — one issue, one commit, one leaf, stamped
`(fak <leaf>)` — because continuous merge is what makes a many-agent fleet fast. But
dropping the PR dropped **both** functions, and only the first was the intended
sacrifice. What remained was the raw commit firehose: no unit of operator *attention*.

The operator-steerability-PR overlay rebuilds the second function **without rebuilding
the first**. It folds landed trunk commits into PR-sized units ("Operator Steerability
PRs", OSPs) and bands each by where attention is owed — and it is a **read-mostly
overlay that gates nothing**. No rung of it blocks, delays, or approves a commit, a
merge, or a promotion.

**The standing rule:** `fak steer prs --check` (exit 1 on any RESIDUAL unit) *reports*,
for a cadence loop or a dashboard. It must never sit in a commit or promotion path.
Wiring it into a merge gate re-introduces exactly the serialization continuous merge
exists to avoid — the fence without this reason is a speed bump someone will eventually
remove, so this paragraph is the reason.

## The OSP unit — six facets

An OSP unit ([`steerpr.Unit`](../internal/steerpr/steerpr.go)) carries six facets,
each deliberately thin:

- **Binding** — the unit is keyed by the `(fak <leaf>)` ship-stamp; each member commit
  parses its issue bindings apart as `Resolves` (a `#N` in the subject — closure-grade)
  vs `Mentions` (a `#N` only in the body — a safe mention). Unstamped commits are
  listed separately: visible, but carrying no band.
- **Grouping basis** — leaf is the default and the fallback, but the fleet dispatches by
  **wave**, not by leaf ([`grouping.go`](../internal/steerpr/grouping.go), #5040). Pass
  `fak steer prs --cohort PLAN.json` (a `fak issue cohort --json` plan) and the commits
  whose subject-bound `#N` belongs to a planned wave fold into **one unit per wave**,
  keyed `wave:<n>` — the wave is what actually got spawned together, so it is the unit an
  operator can stop or redirect. Everything else keeps folding by leaf. Because two bases
  coexist, every unit states which one it used (`grouped_by: wave|leaf`, and a wave unit
  lists the `leaves` it spans): a unit whose basis you have to guess at is worse than one
  basis, not better. `fak release prplan` stays leaf-grouped — a promotion PR is a lane
  artifact — and a wave unit bands by the same worst-member rule, so regrouping can never
  clear a band.
- **Membership** — the fold is deterministic over git history. There is no plan file to
  go stale: every stamped commit in the range is already a line item in the unit of the
  lane that owns it, so re-running the fold on the same range always yields the same
  units.
- **Band** — where attention is owed: `RESIDUAL` (a member claimed something the
  machine could not confirm) → `UNVERIFIABLE` (no checkable claim, or not yet graded) →
  `CLEARED` (every member diff-witnessed). A unit's band is its **worst member's** —
  the fold is pessimistic by design, so a CLEARED unit means *every* member was
  witnessed, not most.
- **Curve** — the bound trajctl objective's progress signal (`HEALTHY / STALL / DRIFT /
  DETOUR_OVERRUN`, witness rungs W3..W0), carried as a pure mirror value
  ([`curve.go`](../internal/steerpr/curve.go), #5038) so the stdlib-only leaf never
  imports `internal/trajctl`. Orthogonal to the band and never folded into it: Band
  answers "was each claim confirmed", Curve answers "is the objective progressing" — a
  CLEARED unit whose objective is DRIFTing still deserves attention, and a RESIDUAL
  band is never excused by a healthy curve. **[NOT YET WIRED]** `fak steer prs` does
  not yet bind live trajctl objectives onto units, so today's output carries no curve
  line; read the curve side from `/trajectory-control` directly.
- **Affordances** — the operator rung ladder, weakest-first: observe → comment → ack →
  redirect → pause. Today only **observe** (`fak steer prs`) is runnable; `fak steer
  comment` **[NOT YET SHIPPED — #5029]**, `ack` **[#5028]**, `redirect` **[#5030]**,
  and `pause` **[#5031]** are named children of the epic. Whatever ships, an affordance
  can only ever make a commit read *worse* than its witness verdict allows, never
  better (see the anti-gaming rule below).

## Why the band is the HUMAN_RESIDUAL doctrine pointed at landed commits

The band vocabulary is the
[choice-triage doctrine](notes/CONCEPT-CHOICE-IS-A-TRIAGE-2026-07-07.md) — *escalate
only what an oracle could not resolve* — aimed at landed commits instead of harness
prompts. `CLEARED` is the `TAKE_OBVIOUS` analogue: the machine already confirmed each
claim, so human attention buys ~nothing there. `RESIDUAL` is the only place a human
look purchases something the machine could not: a claim the diff did not witness.

The doctrine carries through to paging:
[`internal/operatorbrief`](../internal/operatorbrief/osp.go) folds the overlay into the
operator brief, and a RESIDUAL unit reaches the *human* bucket (the pager) **only**
when `choicetriage` judges it a genuine authority decision — every other RESIDUAL is a
watch, CLEARED is background, and an unreadable overlay folds as `UNMEASURED`, never as
a clean zero. "An oracle could not confirm" is not automatically "a human must decide
now"; the residual is where attention *starts*, not a pager feed.

## The read-mostly fence and the anti-gaming rule

The overlay's verdicts are **supplied, never derived**: per-commit verdicts come from
one `dos commit-audit <base>..<head> --json` call mapped through the same keep-bit the
dispatch sweep uses (`dispatchtick.CommitWitnessed`), so the view and the sweep can
never disagree about whether a commit is witnessed. The band is a *view* over the
kernel's one witness oracle, never a second oracle. If `dos` is unavailable, commits
stay ungraded → `UNVERIFIABLE` — the honest read, never a fabricated `CLEARED`.

The anti-gaming rule, in one line: **an ack is not a witness.** `diff-witnessed` is a
non-forgeable machine bit (the diff proves the claim); `acked` means "a human looked".
The residual pile falls when work gets **witnessed** — a follow-up commit whose diff
proves the claim — never when it gets acked. Structurally, of the whole verdict space
exactly one value (the machine's witness bit) opens the CLEARED band, and a supplied
band reconciles **pessimistically** against its verdict — an operator may flag
something the machine cleared (pessimism is always safe), but a CLEARED band written
onto an unwitnessed commit still floors at RESIDUAL. The fence is enforced by
[`internal/steerpr/antigaming_test.go`](../internal/steerpr/antigaming_test.go): an ack
that ever reaches the band reds the suite on the commit that wires it.

## Relationship map

| Surface | Relation to the overlay |
|---|---|
| [`fak release prplan`](../cmd/fak/release_prplan.go) | **Same fold, different scope.** The release-time twin: the same `(fak <leaf>)` unit fold via `internal/steerpr`, ordered biggest-first as a promotion plan; `steer prs` reads the *forming* units continuously, worst-attention-first. |
| `dos review` / `dos commit-audit` | **Same oracle, different projection.** The per-commit witness verdicts under both this overlay and `dos_review`'s bands come from `dos commit-audit`; the overlay re-projects them per `(fak <leaf>)` unit instead of per commit. |
| [`internal/operatorbrief`](../internal/operatorbrief/osp.go) | **Folds in.** The overlay is one more optional brief source beside heaviness and fleet, bucketed through choice-triage so it cannot spam the pager. |
| `trajctl` / [`/trajectory-control`](../.claude/skills/trajectory-control/SKILL.md) | **Curve supplier, mirrored not imported.** The overlay consumes trajctl's folded curve signal as a pure mirror; the regime gate ("a CLEARED unit with a healthy curve is a reason to do nothing") is trajctl doctrine the operator loop borrows. |

## The deterministic witness

```bash
go test ./internal/steerpr/...   # the fold, the bands, the pessimistic reconcile, the anti-gaming fence
fak steer prs --base HEAD~10 --head HEAD   # one bounded live read; READ-ONLY, gates nothing
```
