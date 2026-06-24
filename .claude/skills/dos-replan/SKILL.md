---
name: dos-replan
description: "Refresh a plan portfolio from evidence: close shipped queue items, update cooldown state, and surface the few decisions an operator must make. Use after dispatch bursts, drained backlogs, or recurring findings."
---

# dos-replan — the generic portfolio gardening sweep

> **The planning refresh that keeps the loop honest.** Its domain-free core is
> small: a queue item whose phases now `dos verify` as shipped is *closed*; the
> operator should see only the 0-2 items that truly need a human. The heavy
> host-specific gardening passes (anchor reconciliation, soak-state drift, the
> postmortem evidence stream) are host driver hooks, scoped OUT of the generic
> baseline. Closure detection rides the truth syscall; the operator surface is
> the `dos decisions` queue.

The shape: **read the queue → close what verifies as shipped → rank what's left →
surface only what needs a human.** Closure is a kernel call (`dos verify`); the
operator inbox is the kernel's decisions projection.

## Inputs

- `--force` (optional) — run even if the no-op skip gate would fire (no new
  evidence since the last sweep).

## Step 0 — Discover the layout

```bash
dos doctor --workspace . --json
```

Read `paths` (where the queue/cooldown state live) and `stamp` (the active ship
grammar `dos verify` applies). Read the active trunk from config if you will
gate a release later (see `/dos-replan-loop`).

## Step 1 — No-op skip gate

If there is no new evidence since the last sweep (no new commits, no new findings)
and `--force` was not given, print one line and exit cheap — writing nothing.
This is the gate that keeps a recurring loop from doing 0-work sweeps; an
*unproductive* replan (this skip, or a 0/0/0 sweep) must NOT arm the loop's
drained-twice trigger (the kernel loop decision enforces that — report the
productivity honestly so `/dos-dispatch-loop` reads it right).

## Step 2 — Closure detection (the domain-free core, via the truth syscall)

For each open item in the queue, ask the kernel whether its phases have shipped:

```bash
dos verify --workspace . <PLAN> <PHASE> --json
```

If every phase of a queue item now reports `shipped: true`, the item is **closed**
— move it from the open queue to the closed section. **Never trust the plan doc's
own stamp; the truth syscall is the source.** This is the auto-close pass that
keeps the queue from carrying already-shipped work as if it were pending.

Count the closures — this is part of the productivity signal Step 5 reports.

## Step 3 — Cooldown-state tracking

Bump the cooldown timestamp in the configured state file so `/dos-next-up` knows
when this sweep last ran (the cooldown banner). Generic, no host specifics.

## Step 4 — Rank what's left

Order the remaining open items by a domain-free signal (recency, how many phases
remain). Do not impose a host's bespoke ranking — that's a driver hook.

## Step 5 — Surface only what needs a human (the operator inbox)

The ruthless filter: of everything you swept, at most **0-2 items** should reach
the operator — the ones no mechanism can resolve (a real decision-needed). The
kernel's operator inbox is the `dos decisions` queue; read it to see what is
already pending so you don't duplicate a row:

```bash
dos decisions --workspace . --json
```

The `dos decisions` projection is the generic operator inbox: a HUMAN-resolvable
row is one a mechanism (an ORACLE/JUDGE) cannot close.

**Honesty about the write path (a named open seam).** `dos decisions` is
**read-only** today — there is no generic `dos` verb to *append* a decision-needed
row (the queue's write path, `home.append_decision`, is currently reached only by
`dos arbitrate --force`'s override capture). So the generic baseline **surfaces**
the 0-2 items in its operator summary (Step 6) and reads the existing queue here;
*emitting a new decision row into the queue is a host/driver capability* (the
named open seam — see `docs/74-friction-log.md`). Do not imply `dos decisions`
writes; it lists and drills in.

## Step 5b — Concrete findings go to the issue tracker, not the inbox

A sweep surfaces more than closures: a bug in another lane, a missing test, a
doc that drifted from the code. The 0-2 operator slots are for *decisions*; a
concrete, public-subject finding with a checkable done-condition is *work*, and
its durable home is the workspace's public issue tracker (on a GitHub-hosted
repo, the `gh` CLI) — not the inbox, not a memory file, not silence. The filing
discipline is `/dos-dispatch`'s "Out-of-scope findings" section: **dedupe
first** (`gh issue list --search "<keywords>"`); give the body a checkable
**done-condition**, a lane guess, and where you found it; **leak-check the
drafted body before posting** (issue text is public output no tracked-file gate
scans — no machine-absolute paths, hostnames, or personal identifiers; if the
workspace ships a publication leak-scanner, pipe the draft through it and treat
a hit as a refusal). An issue closes only via `Fixes #N` in the body of the
commit that resolves it — never `gh issue close` off your own narration.

## Step 6 — Emit the operator summary

Print a terse summary: how many items closed (Step 2), how many remain, and the
0-2 that need a decision (with a pointer to `dos decisions`). Report the
productivity verdict honestly (PRODUCTIVE iff it closed/refilled/gardened
anything) — `/dos-dispatch-loop` reads this to decide drained-twice.

## What this skill deliberately does NOT do (no silent gap)

- **No host gardening passes.** Anchor reconciliation, soak-state drift, the
  postmortem evidence stream, gitignore drift — those are host-specific evidence
  surfaces, scoped OUT of the generic baseline (a driver hook, or out of scope).
  The generic sweep does closure detection + the operator surface, and `log`s
  what it is skipping.
- **No host inbox file.** It writes to the kernel's `dos decisions` queue — the
  generic operator-inbox surface — not a host-specific pending-decisions file.

## Worked example (live transcript)

> **The stale-stamp catch.** A queue item points at a plan whose phase the doc
> still narrates as in-flight — but the grep rung sees a commit *subject* carrying
> the phase token, so `dos verify` flips it SHIPPED. Read the **rung**, not the
> bare verdict, then reconcile the queue.

Step 0 — discover the layout (the on-ramp every sweep runs):

```bash
$ dos doctor --workspace . --json
  ... "stamp": {"style": "grep", "grammar": "generic (any/no dir prefix)"} ...
  ... "exit_codes": {"gate": {"LIVE": 0, "DRAIN": 3, "STALE-STAMP": 4, "BLOCKED": 5, "RACE": 6}} ...
```

Step 2 — closure detection. Ask the truth syscall, never the plan doc's own stamp:

```bash
$ dos verify --workspace . docs/82_liveness-oracle-plan liveness --json
{"phase":"liveness","plan":"docs/82_liveness-oracle-plan","rung":"direct","sha":"80d4f30","shipped":true,"source":"grep-subject","summary":"80d4f30 liveness: exclude the BIRTH acquire from the ADVANCING event count"}
```

Verdict carried by the **`source: grep-subject`** rung — a commit SUBJECT containing
the phase token flips this to SHIPPED even if little was built. The doc lagging the
git fact IS the stale stamp; close the item from this rung, not from the doc.

Contrast — an item still genuinely in flight returns the **`none`** rung:

```bash
$ dos verify --workspace . docs/99_runtime-validation-and-the-actuation-boundary halt --json
{"phase":"halt","plan":"docs/99_runtime-validation-and-the-actuation-boundary","shipped":false,"source":"none"}
```

Step 6 reconcile — feed the dispositions to the gate; a stale stamp returns exit 4:

```bash
$ dos gate ./dispositions.json ; echo "exit=$?"
exit=4
```

`gate` exit **4 = STALE-STAMP** — the plan doc lags the verified git fact; the
replan's reconciliation is to close the over-claimed item against the `grep-subject`
rung and drop the doc's self-narrated status. (Exit 0 = LIVE, 3 = DRAIN, 5 = BLOCKED.)

## Anti-patterns

- ❌ Closing an item from its plan-doc stamp — close from `dos verify` only.
- ❌ Surfacing more than 0-2 items — the operator surface is ruthless by design;
  everything a mechanism can resolve stays out of it.
- ❌ Reporting an unproductive sweep as productive — it would arm a false
  drained-twice stop in the loop. Report honestly.
- ❌ Parking a concrete finding in the operator inbox (or dropping it) — the
  inbox is for decisions; file the finding as an issue with a done-condition.
