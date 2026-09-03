---
title: "fak super loops: the operator-intent meta-loop that walks a set of loops"
description: "A super loop is keyed on an operator intent (improve quality), not a task: it walks its member loops/scorecards/gardens to read their status first, selects the worst-first member to enter, and exits on the aggregate clearing — the layer above a normal loop. Five properties separate them, and fak superloop makes the distinction executable."
---

# Super loops (`fak superloop`)

> An operator says **"improve quality"**. That is not one task — it is a standing
> intent that spans a dozen loops: the code-quality scorecard, the slop scorecard,
> the disambiguation scorecard, the gardening bundle, and more. A **super loop** is
> the thing that takes that intent, **walks those loops first to read their status**,
> and tells you **worst-first what to enter** — before it does any work.

The fleet already runs many loops. The [issue-dispatch loop](dispatch-loop.md)
resolves one issue per tick; the [RSI loop](rsi-loop.md) keeps-or-reverts one
candidate; the garden tick reaps one class of stale work; a scorecard run reports one
debt number; `fak loop drive` settles one `GOAL.md` witness. Each is a **normal
loop**: keyed on a *task* and a cadence, its tick *does* one concrete thing, and it
is a **leaf** in the work graph — it acts on the codebase or the world directly.
When a super loop spans generation-labeled work, use the
[generation super-loop budget contract](generation-super-loop-budgets.md) to keep
time, token, worker, and review capacity explicit without turning generation into
priority, a branch, or a runtime gate.

A **super loop** sits one altitude up. It is keyed on an operator **intent**, and its
tick is a **traversal over other loops**, not a task.

## The differentiation — five properties

`fak superloop explain <name>` prints this table for a registered intent, side by side
with a normal loop. The classification is executable (`internal/superloop.Classify`):
a loop is a super loop **iff all five hold**.

| Property | Super loop | Normal loop | What it means |
|---|:--:|:--:|---|
| `has_members` | yes | no | it walks ≥1 member loop; a normal loop has none |
| `walks_first` | yes | no | its tick **reads** each member's status before acting (orient-over-loops) |
| `selects_worst_first` | yes | no | it **selects** which member to enter, worst-first; a normal loop just runs its body |
| `exits_on_aggregate` | yes | no | it exits when the **fold** clears (aggregate debt ≤ floor), not on a single task's witness |
| `interior_node` | yes | no | it **mutates nothing at its own altitude** — only its members mutate the world |

The load-bearing line is the last one. A normal loop is a **leaf**: its unit of work
is a task, and it changes the repo. A super loop is an **interior node**: its unit of
work is *another loop*, and its only effects are *reading* members and *driving* them.
That is why a super loop is safe to run as a read-first orientation pass — by
construction it has nothing to commit.

## What the walk does — the four moves

```
fak superloop walk improve-quality
```

1. **WALK** — read each member's status. Cheaply and honestly: a scorecard member's
   debt comes from the pinned control-pane baseline (`tools/scorecard_baseline.json`,
   the last *measured, committed* value); a loop member's live/stale/dark state comes
   from the cross-ledger loop-health fold (`internal/loopfleet`). A member whose
   status cannot be read is surfaced as **unmeasured**, never silently treated as
   clean.
2. **SELECT** — fold the members worst-first. Dark or unmeasured leaves rank first
   (a gone-dark loop or an unknown status is the most urgent thing to enter), then by
   debt descending. The result is a **worklist**, not a pass/fail.
3. **DESCEND** — a member that is itself a container (the **garden** bundle,
   another **super loop**, or a domain-specific command surface) is surfaced as a
   *descend pointer*: its status is only knowable by walking it in turn. This is the
   recursion — *loops that themselves have many loops* — and it is the named
   follow-on the walk hands you.
4. **FOLD** — the intent is **satisfied** only when the aggregate debt is at-or-below
   its floor **and** every member was measured **and** none is dark. An unread or dark
   member can never let the intent read as done.

A real walk on this repo — the seven-surface sweep. Every action is directly
runnable (each member carries an **enter hint**: the owning skill, or the scorecard
script where no skill exists yet), and the two clean surfaces (doc-appeal,
agent-readiness at debt 0) are correctly absent from the worklist:

```
superloop walk: sweep-surfaces — ACTION (superloop_debt)
  aggregate debt 998 (floor 0)  members 7  walked 7  unmeasured 0  dark 0

  worst-first — enter these in order:
  #  MEMBER                     DEBT  ACTION
  1  scorecard slop             746   enter `/slop-score` to retire slop debt
  2  scorecard disambiguation   153   enter `/disambiguation-score` to retire disambiguation debt
  3  scorecard tooling_quality  68    enter `python tools/tooling_quality_scorecard.py --json` to retire tooling_quality debt
  4  scorecard code             28    enter `/quality-score` to retire code debt
  5  scorecard learning         3     enter `python tools/learning_scorecard.py --json` to retire learning debt

  → worst-first: scorecard "slop" — enter `/slop-score` to retire slop debt
```

And one altitude up, `improve-quality` DESCENDS the sweep inline — the sub-walk's
folded debt arrives as one measured row, so nothing is counted twice:

```
superloop walk: improve-quality — ACTION (superloop_debt)
  aggregate debt 1009 (floor 0)  members 6  walked 5  unmeasured 0  dark 0
  rollup (2 intents): 11 leaf member(s)  walked 11  unmeasured 0  dark 0  containers 1

  worst-first — enter these in order:
  #  MEMBER                    DEBT  ACTION
  1  superloop sweep-surfaces  998   descend: `fak superloop walk sweep-surfaces`
  2  scorecard intent_literal  7     enter the intent_literal scorecard's reduce loop (its skill)
  3  scorecard ui_quality      3     enter the ui_quality scorecard's reduce loop (its skill)
  4  scorecard claim_repro     1     enter `/claim-repro-score` to retire claim_repro debt
  5  garden garden             →     run `fak garden` then `fak garden tick`

  → worst-first: superloop "sweep-surfaces" — descend: `fak superloop walk sweep-surfaces`
```

### Knowing the denominator — direct vs roll-up

A super loop distinguishes two denominators:

- **Direct candidate denominator (`members`)** — the members evaluated directly at this intent's altitude (e.g. 6 in `improve-quality`: 5 evaluated plus 1 container pointer; or the expanded candidate count in `tend-fleet`). By conservation: `Walked + Unmeasured + Containers == Members`.
- **Roll-up leaf denominator (`rollup.leaf_members`)** — the true population of terminal leaves (scorecards, loops, utilization pools, trajectory curves) evaluated across the entire descend tree. When sub-superloops are shared (fan-in > 1, such as `drain-issues`), distinct leaves are counted once, preserving the once-only invariant without double-counting. An unmeasured or dark leaf anywhere in the subtree is tracked in `rollup.unmeasured` and `rollup.dark` rather than being masked by the parent container.

## What the walk reads on a loop member — three dimensions, not just liveness

A loop member is scored on the **product** of three independent dimensions, because
they compound: a loop can be *up* and still be doing nothing useful, and it can be
doing useful work at its own grain while the work it emits rots.

| dimension | doubles the product when | closed reason |
| --- | --- | --- |
| liveness | `stale` — slipping past its cadence (DARK is worse still: the `Dark` flag, tier 0) | — |
| progress (#4956) | **SPINNING** — ticking on cadence with zero advanced verified progress | `RELAY_NO_PROGRESS` |
| follow-on (#4957) | **ORPHANED** — it emitted work nobody advances | `RELAY_ORPHANED_FOLLOWON` |

`FleetDebt` is that product minus one, so a clean live leaf folds to debt 0 and the
ordinary worst-first sort ranks the fleet with no rival walker.

### ORPHANED-FOLLOWON — emitted work nobody advances

A tick emitted downstream work — a durable `relay.ArtifactIssue` pointer (`#1234`) or
the issue an `a2achan.WorkerStatus` names — and **nobody advanced or closed it within
a cadence window**. This is progress at the loop's own grain, zero progress at the
fleet grain, and it is *distinct from SPINNING*: a member can be `advancing` and
`orphaned` at the same time. Each axis drives its own action; the walk emits the
`superloop_orphaned` finding and ranks the member into the debt band (tier 1) ahead of
a clean live leaf, where before the follow-on verdict it read clean.

The verdict is one of four values, and only one of them is debt:

| verdict | meaning | weighs |
| --- | --- | --- |
| `""` | axis **unread** — gate off, or the member emitted nothing | nothing |
| `advancing` | every resolved emission advanced within the window, or closed | nothing |
| `orphaned` | ≥1 emitted issue is OPEN with no advance in the window | **debt** |
| `unknown` | an emission was unreadable — **fail closed** | nothing |

**It fails closed, deliberately.** Any single unresolvable emission collapses the
*whole* member read to `unknown` — even alongside a positively-witnessed orphan — and
`unknown` is surfaced on the status but gates nothing: no debt, no worklist item,
`Satisfied` stays true. An orphan is never fabricated from an absence, the same
asymmetry `relay.ReadVerifiedProgress` keeps for a missing ledger. Gate off is **not**
"clean" — it is the axis unread. The verdict is assembled only from durable
issue/artifact state; there is no field where a member asserts it.

**It only witnesses.** The walk surfaces the orphan and hands you a redirect action
naming the closed token; it never re-files or re-dispatches the emitted work — that
goes through the member's own front door (#4958).

### Turning it on — the dogfood path

The live join costs one `gh issue view` per emitted ref, so the default walk stays
offline and fast. The witness reads the axis only when the gate is set:

```
FAK_SUPERLOOP_FOLLOWON=1 fak superloop walk tend-fleet
```

- **Window** — the loop's own declared cadence, widened to a **24h floor**, so a
  fast-ticking loop cannot fabricate an orphan from a window narrower than a work day.
  The fail-closed direction here is to *under*-report orphans, never to slander live
  work with a window that was too tight.
- **Which refs** — today only the `dispatch` loop kind binds emitted refs, from its
  latest durable tick row (the issues that tick witnessed as worked yet still open).
  Every other loop leaves the axis unread rather than inventing an emission. Only the
  *refs* come from the ledger; whether each one advanced is re-read from live issue
  state, so no upbeat self-report can make the verdict read clean.

**Promotion evidence** (what would move this from `gen/next` to on-by-default): a
gated run where the orphan count tracks a real backlog — every flagged orphan is one
an operator agrees is genuinely unowned, and no orphan is raised against work someone
was actively carrying.

**Demotion/retirement evidence**: if an armed run produces orphans the operator
consistently judges wrong, or the dispatch tick ledger stops being the emission
record, retire the axis to unread — do not loosen the window to hide the noise.

**Invalidating assumption**: the witness treats *any* `updatedAt` movement as "someone
advanced it". A bot relabel, a stale-bot nudge, or a drive-by comment all read as
carried — so the witness under-reports orphans, and a repo with chatty automation
could read clean while nothing real moves. If that breaks, the advance test has to
move off `updatedAt` onto a durable progress witness. A second, narrower assumption:
only the *latest* dispatch tick row is read, so an emission from an earlier tick that
is still open falls out of the axis. A third, *witnessed* on the first armed run below:
an operator-**gated** emission is indistinguishable from an unowned one.

### First armed run — 2026-08-04

The witness was armed against this host's live dispatch ledger and joined against live
GitHub issue state. It fired, and it fired correctly at the mechanism grain:

| | gate off | `FAK_SUPERLOOP_FOLLOWON=1` |
| --- | --- | --- |
| the `dispatch` member | clean, axis unread | `follow_on: orphaned` / `RELAY_ORPHANED_FOLLOWON` |
| its debt | 0 | 1 |
| walk total debt | 2 | 3, `orphaned: 1` |
| worklist rank 1 | the `drain-throughput` container | **the `dispatch` loop**, chase/redirect |

The latest tick emitted seven refs — #5497, #5495, #5493, #5254, #5170, #5106, #4835 —
and every one resolved OPEN with its last `updatedAt` between 2026-07-17 and
2026-07-30: 5 to 18 days outside the 24h window. Every ref was positively read, so
nothing was fabricated from an absence, and a member that read clean before the verdict
existed now ranks first as debt — the exact "emitted work nobody advances" the witness
was built to stop hiding.

**What this does not yet prove.** The run shows the mechanism fires, binds its closed
token, and ranks; it does *not* show the count tracks **unowned** work. At least one
flagged ref (#4835) is operator-*gated* — it waits on a hardware witness nobody can
produce on demand — so it is parked, not orphaned. The witness separates "not recently
touched" from "carried", but not **blocked** from **orphaned**, and a backlog of
legitimately-parked work reads as orphan debt. The promotion bar above therefore stays
open. The smallest next step toward it is a park/blocked signal the join can subtract
before it judges, so a gated emission drops out of the count instead of inflating it.

## What the walk reserves — the budget

Each intent also declares a **generation budget**: a planned reservation across the
four dimensions the [budget contract](generation-super-loop-budgets.md) names —
**time**, **tokens**, **workers**, **review**. The walk folds it into one row per
dimension and **divides it down** across the worklist members it just built, so the
budget *reserves attention* without touching the worst-first order (priority stays
debt-ordered; the budget only says how much capacity each member may draw):

```
  budget gen/next — declared reservation, divided across 3 worklist member(s):
  DIMENSION  DECLARED      PER-MEMBER
  time       20 minutes    6
  tokens     120000 tokens 40000
  workers    1 workers     0
  review     —             HELD
```

Two rules make this honest. Division is **floored** — the per-member shares never sum
past the declared cap (a scarce integer cap like `workers 1` over three members
floors to a `0` share the members must timeshare, which is *budgeted*, not held). And
a dimension with **no declared cap** renders as **HELD** — the contract's "no row =
hold for later-horizon work" case, surfaced for the operator rather than silently
treated as an unlimited grant. `walk --json` carries the same rows plus each worklist
member's `allocation`. These are *declared* caps (the top of the cascade), not
measured consumption; binding a member's share to its drive env and comparing planned
against witnessed spend are the [remaining hooks](generation-super-loop-budgets.md#implementation-hooks).

## How a super loop relates to what already exists

A super loop **generalizes the garden bundle** (`internal/gardenbundle`). The garden
is a *fixed* bundle of members folded into one OK/RED **gate**. A super loop is an
**intent-named**, **worst-first-selecting**, **recursively-nestable** bundle whose
members are themselves loops/gardens/scorecards, and whose output is a **worklist**
(what to enter next) rather than only a pass/fail. The status reads it folds are the
same ones the fleet already computes — `loopfleet` for loop health, `scorecardpane`
for scorecard debt — so a super loop adds a *view and a selection*, never a new
oracle.

## The registry

Super loops are **data** (`internal/superloop`). Each binds an operator intent to an
ordered member set; every scorecard member references a real control-pane card key
(a no-drift test enforces it), and a member may carry an **enter hint** — the
concrete skill or command that retires its debt — so the worklist action column is
runnable as printed. The registered set:

- **`sweep-surfaces`** — the seven quality surfaces, swept worst-first: **code**
  (`/quality-score`), **doc-appeal** (`/appeal-score`), **agent-readiness**
  (`/agent-readiness`), **code-slop** (`/slop-score`), **concept-disambiguation**
  (`/disambiguation-score`), **learning** (`tools/learning_scorecard.py`), and
  **tooling-quality** (`tools/tooling_quality_scorecard.py`).
- **`improve-quality`** — descends `sweep-surfaces`, then the remaining
  quality-bearing scorecards (conflation, intent-literal, ui-quality, claim-repro)
  + the gardening bundle. Nesting the sweep (instead of duplicating its members)
  keeps each surface's debt counted exactly **once** at the root — a once-only test
  pins that no scorecard key is walked by two intents.
- **`improve-loops`** — the loop-index scorecard + the dogfood scorecard + the
  goal-scoped issue-dispatch intent + the live loop ledgers (cadence, dojo) + the
  gardening bundle.
- **`drain-issues`** — the aggregate issue-dispatch intent. It keeps the legacy
  dispatch progress row visible, then descends into the goal-specific issue-drain
  intents.
- **`drain-throughput`** — the throughput issue-drain intent. It walks
  `issue-resolve-dispatch/claude/throughput` and enters
  `fak dispatch auto --goal throughput` when the loop is stale or dark.
- **`drain-high-priority`** — the high-priority issue-drain intent. It walks
  `issue-resolve-dispatch/claude/high-priority` and enters
  `fak dispatch auto --goal high-priority` when that loop is stale or dark.
- **`manage-benchmarks`** — the benchmark-DX scorecard + the `nightrun` collection
  loop + a descend pointer into `fak bench-loop status`, the benchmark-specific
  control surface.
- **`tend-scoreboards`** — the reporting family (`internal/scoreboard` names it:
  scoreboard, blockers, bench, cachevalue, capacity, node-usage, backlog, dojo,
  product, releases, steering — all folded onto one CI/CD report channel). Its
  **measurable** members are the four outward-facing scorecards whose numbers get
  posted — **product**, **release-readiness**, **steerability**, **milestone** — each
  a real control-pane card key walked by no other intent, so the once-only fold still
  counts each once. Alongside them rides one **`KindLoop`** member — the
  operator-steerability overlay's maintenance loop (`steerpr-overlay`, whose ticks
  append `docs/nightrun/steerpr-overlay.jsonl`), entered with `fak steer prs`. It
  carries **liveness, not a card**, so it adds no scorecard ref and double-counts
  nothing at the root; an overlay that stops recomputing the residual pile now surfaces
  here instead of going unnoticed, and a host with no foldable overlay ledger reads
  **UNMEASURED** rather than a clean 0. The feeds that carry no scorecard (blockers,
  cachevalue, capacity, node-usage, backlog) are a delivery-liveness question — *are the
  channels actually receiving the posts?* — surfaced as a `fak slack beat` descend
  pointer that is shown for entry but never weighed, so an unread pulse can't red a
  clean walk. Answers "is every scoreboard number healthy, are the steering numbers
  still being recomputed, and are the feeds delivering?".
- **`run-the-night`** — the overnight productivity meta-loop. It walks the three
  dimensions that must move together or a night wastes itself: **issue-drain**
  (descends `drain-issues`), **account-limit utilization**, and **lab/node resource
  utilization** (Mac, A100s, dgx). The two utilization members are a new
  **`KindUtilization`** member type: unlike a scorecard (a committed baseline) or a
  loop (a ledger fold), their status is read **live** by the shell at walk time, and
  their debt is **unused capacity** — offerable-but-idle account seats
  (`rotationHeadroom`), and up-but-idle boxes (the `fak lab status` fleet fold).
  Worst-first, the walk enters whichever dimension is most underused; a night that
  leaves seats or boxes idle reds the fold until they are put to work. The intent also
  carries a **declared `IssueTarget`** (the operator's ~200-issue overnight headline) —
  a stated policy like `Floor`/`Budget`, **surfaced** by the walk (`issue target: 200`)
  so the number is explicit and testable rather than buried prose. Binding it to a
  **live count of issues progressed** (and reddening the night until it is met) is the
  named follow-on — the same declared-vs-measured posture the budget rows keep.
- **`tend`** — the root: every other registered intent, reachable directly as a
  member or by descent (a no-escape test pins the reachability). It descends
  `run-the-night`, so the root walk answers "is the night actually producing?"
  alongside the quality/loop/benchmark intents.

For benchmark work, use the generic intent to orient across loop health and debt:

```
fak superloop walk manage-benchmarks
```

Then descend into the benchmark surface when the worklist points there:

```
fak bench-loop status              # registry + run catalog + ledger + local next + authority gap
fak bench-loop next                # the single next benchmark-loop action
fak bench-loop walk                # map the benchmark surfaces to enter
fak bench-loop run --apply --loop  # delegate to the local nightrun collection loop
```

```
fak superloop list                  the named super loops + their members
fak superloop explain <name>        the five-property differentiation, super vs normal
fak superloop walk <name> [--json]  walk the members' status, fold the worst-first plan
fak superloop drive <name> [--lane] walk, ENTER the one worst-first member through the
                                    same admission gate any spawn passes, then re-fold
fak superloop drive <name> --batch N  ENTER up to N worst-first members whose regions
                                    are mutually disjoint (and disjoint from live
                                    leases), through the SAME gate; N<=0 = every member
```

## The drive rung

`walk` is the orientation half — *understand the status of everything under this
intent, and what to do next* — and it **mutates nothing**. `drive` (`fak superloop
drive <intent>`, #2224) is the acting half: it walks, **selects the single worst-first
member** (`internal/superloop.Drive` — one member per invocation), and **enters** it
through the **same admission gate any spawn passes** — region admission over the live
lease fabric (`COLLISION_RISK` on lease overlap), armed by `--lane`/`--tree`, reusing
the loop-drive region hold. The super loop gets **no private spawn path** (the
band-ladder discipline: B6 reaches the world only through B4's gates): a refused gate
**surfaces the token and enters nothing**, never bypasses it. The admission decision is
recorded on the loop ledger with the standing witness vocabulary
(`admitted`/`refused`), and the drive **re-walks and folds** — the aggregate re-fold is
the exit check, so a driven-but-unwitnessed member can never satisfy the intent.

**Batch drive (`--batch N`).** `Drive` returns one member; `Walk` already produces a
*fully-ranked* worklist, so a single invocation leaves the rest of the ranked work on
the table. `--batch N` (`internal/superloop.DriveBatch`) offers the **top-N worst-first
members** and admits each through the **same gate**, under a **member-scoped lease
identity** so N holds coexist. Because each admitted lease stays held while the next is
gated, the gate itself enforces the batch's **mutual disjointness** — a later member
whose region overlaps one already admitted this batch refuses `COLLISION_RISK`, exactly
as it refuses overlap with a live peer lease — so **throughput scales with the available
non-colliding work** instead of one member per invocation. With no `--lane`/`--tree`,
each member is scoped to its own region (distinct members run concurrently; a peer
entering the *same* member is refused); an explicit operator region fences the whole
batch to one tree (members serialize on it, by the operator's choice). Refused members
**surface their token and are skipped** (no private spawn path); the drive **re-folds
once** over the whole intent as the exit check. `--batch 1` (the default) is the
historical single-member drive, unchanged.

Honest fence: the drive keeps the **interior-node** property — it mutates nothing at
its own altitude. The single action it takes is **surfacing the member's own front
door** (a loop member via `fak loop drive` / the dispatch tick; a scorecard member via
its enter hint). Live execution of the member's child *behind* that front door — under
a real lease, landing the member's own `witnessed_done` — is the named follow-on;
per-member budget allocation arrives via #2222. *Descending* into a container member's
own walk (the recursion case) is already live in `walk`.
