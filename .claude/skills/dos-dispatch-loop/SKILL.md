---
name: dos-dispatch-loop
description: "Run recurring `dos-dispatch` cycles, switching to `dos-replan` when the backlog drains and stopping on the kernel's loop verdict. Use for unattended dispatch->replan->dispatch work across disjoint lanes."
---

# dos-dispatch-loop — the generic dispatch⇄replan cadence

> **The unattended plan-and-ship loop.** It runs `/dos-dispatch` repeatedly and
> falls to `/dos-replan` when a lane drains, stopping on a typed, kernel-decided
> condition (not a prose guess). The stop/continue logic is the kernel's
> `loop_decide.decide` — the loop carries counters, the kernel decides. Several
> loops on disjoint lanes run in parallel, each holding its own lane lease.

The stop conditions are the kernel's, in one place:

1. **iteration cap** — reached `max_iterations` (default 10).
2. **drained-twice** — a DRAIN after a *productive* `/dos-replan` that itself
   followed a DRAIN (the lane is genuinely exhausted).
3. **consecutive-unclear** — the dispatch subprocess is failing systematically.
4. **rate-limited** — a usage window is exhausted (don't burn launches).
5. **launch-failed** — a subprocess never started.
6. **pick-held-invariant** — the next unit is held ONLY by a reason a re-dispatch
   cannot change (draft-class / operator-gated / soak-open / dependency-unmet);
   re-dispatching it would re-block identically, so honest-STOP + surface the hold.
7. **pick-cooldown** — the next unit was attempted-and-didn't-move inside its
   cooldown window AND nothing fresher is offerable; re-dispatching it would
   re-storm a known drain (the ~5%-shipping re-pick loop the bare loop hit).
8. **not-ratcheting** — the OUTER RATCHET (docs/351): the loop has run too many
   iterations in a row with no *witnessed net gain* — the reconcile-VERIFIED
   ship-count (Step 3) is not rising, even though each iteration reported SHIPPED.
   The loop is running but not improving (the "spinning, narrating progress while
   net-shipping nothing real" failure that worsens the longer the loop runs). The
   verdict is the kernel's `improve` ESCALATE, read VERBATIM — not the loop's
   self-report. Hand the judgment back to a human.

Conditions 6–7 are the docs/207 anti-churn rungs (the loop stops re-picking work
it cannot move); condition 8 is the docs/351 ratchet (the loop stops when it is
moving but not *improving* — RSI made first-class here, the same `improve` keep-gate
the `/dos-self-improve` loop uses, now gating each dispatch iteration's net gain).

## Inputs

- `--lane <name>` (optional) — focus the whole loop on one lane (fixed for the
  run; a bare loop auto-picks a free lane at Step 0).
- `--gate hard|soft|drive` (default `hard`) — the verdict policy. `hard` routes a
  non-LIVE verdict through `/dos-replan`; `soft`/`drive` stop on a true DRAIN;
  `drive` self-heals a STALE-STAMP inline.
- `--max-iterations <N>` (default 10).

## Step 0 — Pre-flight: take the lane, read the taxonomy

```bash
dos doctor --workspace . --json
dos arbitrate --workspace . --lane <LANE> --kind cluster --leases '<SIBLING_LEASES>'
```

The arbiter ADMITs a free lane (or auto-picks one); a REFUSE means a sibling loop
already holds an overlapping lane — pick a free one from `free_clusters` or exit.
Initialise the loop counters (iteration=1, the breakers at 0).

## Step 1 — Pick-selection: skip held + cooled units (the anti-churn gate)

Before a `dispatch` iteration offers a unit, screen the candidate unit set so the
loop never re-storms work it cannot move (the docs/207 §6 throughline). For each
candidate, in order:

```bash
dos pickable <UNIT> --state '<host-gathered state>'   # OFFERABLE=0; HELD=per-reason code
dos cooldown <UNIT>                                    # CLEAR=0; RECENTLY_ATTEMPTED=3
```

- A `pickable` exit of **0** (OFFERABLE) **and** a `cooldown` exit of **0** (CLEAR)
  → this unit is dispatchable; offer it and proceed to the iteration.
- A `pickable` HELD by a re-dispatch-CURABLE reason (IN_FLIGHT / SOFT_CLAIMED /
  STALE_CLAIM / UNPARSEABLE) or a `cooldown` of **3** (RECENTLY_ATTEMPTED) → **skip
  this unit, try the next candidate** (the skip-to-next is pick-selection's job).
- A `pickable` HELD by a re-dispatch-INVARIANT reason (DRAFT_CLASS=10 /
  OPERATOR_GATED=11 / SOAK_OPEN=12 / DEPENDENCY_UNMET=13) → carry that verdict into
  Step 2; the kernel will honest-STOP on it (don't re-dispatch a unit a re-dispatch
  cannot un-gate).

When EVERY remaining candidate is skipped (all cooled / curably-held), carry the
last `cooldown` RECENTLY_ATTEMPTED (or the invariant `pickable` hold) into Step 2
as the loop's pre-dispatch evidence — the kernel turns "nothing fresh is offerable"
into the `pick-cooldown` / `pick-held-invariant` honest-STOP. **Do not re-dispatch
the cooled/held unit yourself** — that is the re-pick storm this gate prevents.

## Step 1b — Run one iteration

For a `dispatch` iteration, invoke `/dos-dispatch --lane <LANE>` (it snapshots,
gates, and ships). For a `replan` iteration, invoke `/dos-replan`. Capture the
iteration's outcome:

- a **dispatch** iteration that reached the gate carries a typed verdict — get it
  from `dos gate` over the packet's dispositions sidecar (LIVE/DRAIN/STALE-STAMP/
  BLOCKED/RACE).
- a **replan** iteration carries a productivity signal (did it refill/garden?).

## Step 2 — Decide continue / replan / stop (the kernel decides)

This is the load-bearing step: **the decision is a kernel mechanism, not prose.**
Feed the iteration outcome + the carried counters to the loop decider. In code a
host calls `dos.loop_decide.decide(state, outcome)`; the screenplay's job is to
construct the typed `IterationOutcome` and read the returned `LoopDecision`:

- `action: "continue"` → run the next iteration in `next_mode` (`dispatch` or
  `replan`); if `reconcile` is set (a soft/drive STALE-STAMP), run an inline
  stamp-reconcile pass first. Carry `next_state` forward (the updated counters).
- `action: "stop"` → the loop ends; report `stop_reason` (one of the five above)
  and `surface` (whether it needs operator attention).
- `action: "retry-same-iter"` → a transient overload; sleep `backoff_seconds`
  and re-run the SAME iteration.

The drained-twice rule is the kernel's: a DRAIN counts toward an early stop ONLY
after a *productive* `/dos-replan`. A STALE-STAMP or BLOCKED gate routes to
`/dos-replan` (under `hard`) but never arms a false drained-twice stop — that is
the structural fix the typed gate exists for.

The pre-dispatch evidence from Step 1 rides into the decision too: the kernel reads
the carried `Pickability` (→ `pick-held-invariant` stop) and `Cooldown` (→
`pick-cooldown` stop). So the loop's continue/stop is driven END-TO-END by kernel
rungs — the honest-STOP that used to be a per-run human override is now a kernel
rule, not prose the loop re-applies each iteration.

### Step 2b — Gather the outer-ratchet verdict (docs/351, the RSI rung)

One more piece of in-flight evidence rides into `decide()`: the **outer ratchet**,
the verdict on whether this iteration produced a *witnessed net gain* or the loop is
spinning while narrating progress. The work-metric is the cross-run KEEP the loop
already computes in **Step 3** — the count of picks `dos reconcile` confirms VERIFIED
(exit 0) against git ancestry, NOT the SHIPPED self-report. Carry a cumulative
baseline of VERIFIED picks across iterations and ask the kernel:

```bash
# work = the carried cumulative VERIFIED count + this iteration's VERIFIED picks;
# baseline_work = the carried cumulative count BEFORE this iteration.
# suite/truth come from the gate being LIVE and reconcile not flagging QUIET_INCOMPLETE.
dos improve --workspace . \
  $( [ "$GATE_LIVE" = 1 ] && echo --suite-passed ) \
  $( [ "$RECONCILE_CLEAN" = 1 ] && echo --truth-clean ) \
  --work "$VERIFIED_TOTAL" --baseline-work "$VERIFIED_BASELINE" \
  --consecutive-reverts "$RATCHET_REVERTS" --max-reverts 3 --json
```

Branch on the verdict (the exit code): `0` KEEP — net gain witnessed, raise the
baseline (`VERIFIED_BASELINE=$VERIFIED_TOTAL`), reset `RATCHET_REVERTS=0`; `3` REVERT
— no net gain this iteration, bump `RATCHET_REVERTS`; `4` ESCALATE — the breaker is
open. Pass the verdict as `loop_decide.decide`'s `ratchet` evidence. When it is
ESCALATE the kernel STOPs the loop with **not-ratcheting** (condition 8): N iterations
of "SHIPPED" that all reconcile QUIET_INCOMPLETE accrue reverts and stop the loop
instead of burning the cap — the exact failure the bare loop hit (motion without
measured progress). This is the same `improve` keep-gate the `/dos-self-improve` loop
uses, now gating each dispatch iteration's net gain; the metric is the non-forgeable
reconcile-VERIFIED count (git ancestry), never the loop's word.

## Step 3 — Reconcile each claimed pick (the cross-run KEEP)

A `dispatch` iteration that SHIPPED claims picks done — but a claim is a
self-report. Before the archive drops a claimed pick from the residual, reconcile
its claim against ground truth so a quietly-incomplete pick re-enters the pickable
set next iteration, flagged:

```bash
dos reconcile <UNIT> --claimed-done --plan <PLAN> --phase <PHASE>   # oracle from git
```

Branch on the exit code (the verdict IS the code):

- `0` **VERIFIED** → the oracle confirms it shipped; it leaves the residual.
- `3` **QUIET_INCOMPLETE** → CLAIMED done but the oracle says NOT_SHIPPED — KEEP it
  in the residual, flagged; it re-enters the pickable set next iteration so the
  host routes it (a verifier pass / `/dos-replan` / a finding). **Do NOT believe
  the claim** — only ground truth removes work (the FQ-336 touch-counts-as-ship
  false-DRAIN is exactly what this catches).
- `4` **HONEST_OPEN** → not claimed, not shipped; honest open work, stays in the
  residual.

This is the cross-run KEEP wired at the boundary that runs the write (the
`CLAUDE.md` "wire the contract into the step that runs the write" rule).

## Step 4 — Archive + release (and leave the lane's tree clean)

When the loop stops, write a loop record under `paths.runs` (the run dir from
`dos doctor --json`: the per-iteration verdicts, the reconcile flags, the stop
reason) and release the lane lease. Commit with a generic subject read from config
— no hardcoded prefix.

**Leave the tree clean for the lane you held — an unattended loop must not strand
its own writes.** A loop that ships work but leaves it uncommitted is the bug the
oracle catches next session: `dos verify` answers from git ancestry, so an
uncommitted change is a phase the kernel reports `NOT_SHIPPED` (the "a commit IS the
ship-stamp" contract). So at close-out, commit your lane's writes — driven by the
same `dos` verbs the loop already uses, with generic git:

- **Commit your lane's writes by explicit pathspec** — stage exactly the files
  under the lane region you leased (the `tree` globs `dos arbitrate` handed back),
  then commit naming those paths: `git add <lane paths>`; `git commit -m "<subject>"
  -- <lane paths>`. **Never a bare `git add -A`** — when sibling loops hold disjoint
  lanes on the same tree, a blanket add sweeps another loop's in-flight edits into
  your commit. The lane lease you held names exactly which paths are yours; commit
  only those.
- **Confirm the tree is clean for your lane before you exit.** `git status
  --porcelain -- <lane paths>` over the region you leased should come back empty once
  you have committed. If it does not, you stranded durable work — `log` it and commit
  it (or, if a path turns out to belong to a *still-live* sibling lease — check
  `dos arbitrate`/the lane journal — leave it for that loop). Either way the loop
  must not exit leaving its own lane dirty off a self-reported "done".
- **Scratch is not stranded work.** Short-lived probe output (a host's scratch
  convention — temp dirs, `*.err`, leading-underscore probes) is deletable noise, not
  a phase to ship; `rm` it or leave it gitignored, don't commit it into the lane.

This close-out is what keeps an unattended fleet's tree **clean and well-organized
across runs**: each loop commits its own lane and confirms it left nothing behind, so
the trunk never accumulates anonymous WIP from a loop that stopped mid-write.

> **DOS-repo note (not part of the generic skill):** when this loop runs *in the DOS
> kernel repo itself*, that repo ships an advisory `scripts/git_hygiene.py --strict`
> that mechanizes the "is my lane clean?" check above (exit 1 on stranded durable
> work, lease-aware). It is a DOS-repo convenience, not a `dos` verb — a foreign
> workspace uses the generic `git status` form above. Both express the same
> discipline.

## What this skill deliberately does NOT do (no silent gap, `CLAUDE.md` heavy tier)

- **No soft-claim lease core.** It coordinates loops by *lane* lease
  (`dos arbitrate`), not the per-pick soft-claim machinery that stays host-side.
  `log` this when a sibling lane is busy rather than waiting on a soft-claim.
- **No value-greedy focus scheduler.** It picks by lane order + the gate verdict,
  not a host's per-iteration focus ranking. That scheduler is the heavy tier.
- **No rate-limit predictive monitor / resume manifest.** It stops on a
  RATE_LIMITED outcome and reports it; it does not pre-empt or auto-resume. A
  host adds that as a driver concern.

It `log`s each of these the first time it would have used them, so the capability
gap is named, never silent.

## Worked example (live transcript)

> **The cadence loop, by hand.** Take a lane, keep the WAL beat alive, ask the
> kernel if the run is moving. Every line is a real `dos` verb; the heartbeat is
> the `HEARTBEAT` op the trajectory-audit cross-signal reads.

Step 0 — take a lane. You ask for `src`; a live lease made it contended, so the
arbiter auto-picks a free cluster lane instead (it never double-books a region):

```bash
$ dos arbitrate --workspace . --lane src
{"auto_picked":true,"free_clusters":[],"lane":"benchmark","lane_kind":"cluster","outcome":"acquire","pick_count":null,"reason":"auto-picked free cluster lane benchmark (requested src was refused: lane src would edit the orchestrator's own running code … (SELF_MODIFY) …).","tree":["benchmark/**"]}
```

→ exit `0` (`acquire`); the redirect IS the admission kernel refusing an
inadmissible region, with the real reason named in the parenthetical (a free,
admissible lane you name is granted directly).

Keep the beat alive across iterations — `acquire` once, `heartbeat` each pass,
`release` at the end. The `HEARTBEAT` op is what makes `SPINNING` reachable from
real evidence (a beat is not an event):

```bash
$ dos lease-lane acquire   --workspace . --lane benchmark    # writes ACQUIRE to the WAL
$ dos lease-lane heartbeat --workspace . --lane benchmark    # writes HEARTBEAT each iteration
$ dos lease-lane release   --workspace . --lane benchmark    # writes RELEASE when the loop stops
```

Ask the kernel if the run is moving (the temporal verdict, from git delta — never
the loop's "making progress" self-report):

```bash
$ dos liveness --workspace . --run-id R --start-sha 80d4f30
```

→ exit `0` = `ADVANCING` · `3` = `SPINNING` · `4` = `STALLED` (`2` contract-error).
Drive the loop off the exit code, not stdout prose.

Read the beat back from the WAL — an `ACQUIRE`/`HEARTBEAT`/`REFUSE`/`RELEASE`
sequence on generic lanes (ILLUSTRATIVE shape; `dos journal replay` folds it):

```bash
$ dos journal --workspace . tail 4
{"op":"ACQUIRE","lane":"benchmark","tree":["benchmark/**"]}
{"op":"HEARTBEAT","lane":"benchmark"}
{"op":"REFUSE","lane":"benchmark","reason":"lane benchmark is already held by a live loop — pick a different --lane or wait."}
{"op":"RELEASE","lane":"benchmark"}
```

→ the `REFUSE` row is the WAL recording the arbiter's "no" — a sibling loop's
overlapping lane, fossilized for `dos decisions` to surface.

## Anti-patterns

- ❌ Re-implementing the stop conditions in prose — route every outcome through
  the kernel loop decision; the counters/breakers/cap are the kernel's.
- ❌ Counting a STALE-STAMP/BLOCKED toward drained-twice — only a true DRAIN after
  a productive replan counts (the kernel enforces this; don't second-guess it).
- ❌ Running two loops on overlapping lanes — the arbiter REFUSEs the second;
  honor it, don't `--force`.
