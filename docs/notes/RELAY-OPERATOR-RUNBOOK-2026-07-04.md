---
title: "Relay operator runbook — start, observe, tune, park/resume"
description: "Operator-facing runbook for perpetual-session relays: start a relay, read its status, tune soft marks and caps, interpret every tombstone reason token, and resume a parked relay."
date: 2026-07-04
---

# Relay operator runbook

**Kind:** operator procedure (start / observe / tune / park-resume). **Lane:** `docs`.
**Closes:** #1910 (J8, epic #1860).

A **relay** runs one goal as an ordered sequence of bounded **legs**, each handing a
small typed **baton** to the next so peak context stays flat no matter how long the goal
runs. This is the operator surface for driving one. The doctrine and the *why* live in
[CONCEPT-PERPETUAL-SESSIONS-2026-07-01](CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md); the
baton fields live in [RELAY-BATON-SCHEMA-2026-07-01](RELAY-BATON-SCHEMA-2026-07-01.md);
the closed reason vocabulary lives in
[RELAY-REASON-VOCABULARY-2026-07-01](RELAY-REASON-VOCABULARY-2026-07-01.md). This runbook
does not re-specify any of those — it tells an operator what to *do*.

> **Readiness, honest.** The relay is in a staged rollout. What you can drive *today*:
> inspect any baton with `fak relay resume`, and read every tombstone reason against the
> action table below. What is still staging (and named by its rung issue so you can watch
> it land): the auto-rotation driver loop (#1894), the `fak relay status` view (#1900),
> the `fak relay handoff` writer verb (#1875), the `[relay]` policy table (#1888), and the
> hard-ceiling / no-progress escapes (#1898, #1893). Where a step needs a staging piece,
> this runbook says so and gives the manual operator action that holds until it ships.

## The one model to hold in your head

A relay never shrinks a pressured window. When a leg nears its ceiling it **ends cleanly
at a safe point**, externalizes every load-bearing fact to the durable store (git, the
ledger, issues, memory), writes a pointer-only baton, and a fresh leg takes over seeded
only by that baton plus a query tool. The transcript is discarded, never summarized. The
invariant you are buying:

> **Flat context.** Peak resident tokens in a relay are bounded by the per-leg ceiling,
> *independent of goal duration.* A relay that runs a week peaks no higher than one that
> runs an hour.

Three things make that safe, and all three are adjudicated *below* the agent, not by
trust: (1) rotation is two-phase — **arm** at a soft mark, **fire** only at the next safe
point; (2) an **externalize gate** fails closed if anything load-bearing lives only in the
transcript; (3) a baton carries **no `claimed` field** — its progress is a cursor the
successor re-verifies against git before trusting.

## Start a relay

1. **Pin the objective.** Write the goal as one line plus a `done_when` predicate the
   successor can evaluate against the durable store ("a pushed commit closes #NNNN and
   `dos commit-audit` passes"). The objective is carried verbatim across every leg and
   content-digested, so a successor that drifts is detected, not silently rewritten.
2. **Acquire the lane once.** A relay holds its lane across legs; the baton carries the
   `held_region` so leg N+1 re-acquires the *same* lease and does not collide with peers
   on the shared trunk. Take it with `dos arbitrate --lane <lane>` (or the lane's acquire
   verb) before leg 1.
3. **Run the leg to a safe point.** Do the work through the ordinary witnessed path —
   every result lands in git, the ledger, or an issue *as it happens*. This is what makes
   the externalize gate cheap: nothing load-bearing is ever transcript-only by the time
   you rotate.
4. **Arm, then fire.** When a rotation trigger crosses its soft mark (context %, turns,
   wall-clock, spend), the leg arms (`RELAY_ARMED`) and continues only to the next
   quiescence boundary — no in-flight tool call, green-or-parked tree, and a one-line next
   action. It fires there, not mid-action.

> **Staging:** the auto-rotation driver (#1894) and the `fak relay handoff` writer (#1875)
> are not yet wired as verbs. Until they land, a relay is driven manually: the closing leg
> writes its baton through the codec (`internal/relay.Marshal`, the same wire bytes
> `fak relay resume` reads back) and the operator launches the successor pointed at it.
> The baton schema, the gates, and the reload verifier are all shipped, so the handoff is
> real — only the one-command convenience is pending.

## Observe — read relay state

The observable unit is the **baton**: it is the whole handoff a successor leg receives.
Read it offline with the shipped verb:

```bash
# human summary — the objective, done_when, re-verifiable cursor, next action, pointers
fak relay resume --baton .fak/relay/RLY-20260701-0001.baton.json

# canonical wire bytes (byte-stable round-trip: pipe it back in and it's identical)
fak relay resume --baton leg.json --json | fak relay resume --baton - --json

# from a pipe
cat some-baton.json | fak relay resume --baton -
```

`fak relay resume` is a **pure read**: it parses the baton, gates it on the
`fak.relay.baton.v1` schema tag (a non-baton is refused), and prints. It does *not*
re-verify the cursor — that is the successor's job at reload (`relay.VerifyReload`), and a
baton is the *least* trusted signal in the system on purpose. The summary prints every
field, marking empty ones `(none)`, so you see the whole handoff, not a digest.

What you are reading, and what to check by eye:

| Field | What it tells you | What you re-check |
|---|---|---|
| `tombstone.reason` | why this leg ended | match it to the action table below |
| `objective` + `done_when` | the goal, unchanged across legs | is the goal already satisfied? |
| `progress_cursor.start_sha` | the git anchor for verified progress | is it an ancestor of `HEAD`? |
| `progress_cursor.held_region` | the lease to re-acquire | is it free to take? |
| `next_action` | the single step to resume on | is it still the right step? |
| `artifacts` / `do_not_rederive` | pointers into the durable store | do they still resolve? |

> **Staging:** the folded multi-leg view `fak relay status <relay-id>` (legs, rotation
> reasons, cost vs. a compaction baseline) is #1900, still open. Until it lands, read each
> leg's baton with `fak relay resume` and fold the view yourself — the cost benchmark that
> underpins the comparison is already shipped (#1906).

## Tune — soft marks and caps

Rotation is **two-phase on purpose**: arm at a *soft* mark well below the wall, fire at
the next safe point. SOTA consensus is to rotate around 50–70% of the window, **not 95%**
— a leg that arms at 60% still has clean headroom to reach a safe point and externalize.
The soft marks and caps are **policy data, not magic numbers in code**; they live on the
`Envelope` axes (context tokens, turns, wall-clock, spend) and, once shipped, a `[relay]`
table.

| Knob | What it controls | Lean default | Why |
|---|---|---|---|
| soft arm mark (context) | when a leg arms | ~60% of the per-leg ceiling | headroom to reach a safe point |
| hard ceiling | the fail-closed backstop | the per-leg ceiling | parks (`RELAY_PARKED_UNSAFE`) rather than blows the window |
| min progress before re-arm | anti-thrash hysteresis | one verified forward step | a leg that did nothing may not arm again |
| rotations / wall-hour | anti-thrash cap | a small N | stops a spin before `RELAY_NO_PROGRESS` |
| `max-legs` / `max-spend` | the relay envelope | operator-set | bounds even an open-ended goal |

> **Staging:** the `[relay]` policy table shape (#1888), the Envelope-axis arm triggers
> (#1890), the per-hour cap (#1892), and the hysteresis floor (#1891) are open. Until they
> land, treat these as the knobs you *will* set and the values to hold them to; the
> externalize gate and safe-point predicate that consume them are already shipped.

## Interpret every tombstone reason

Every leg ends with a typed tombstone — a closed reason token, never prose. The table
below pairs each token with the **operator action** that resolves it. The authoritative
`summary` + `fix` rows live in
[RELAY-REASON-VOCABULARY-2026-07-01](RELAY-REASON-VOCABULARY-2026-07-01.md); this is the
operator-facing reading.

| Token | What happened | Operator action |
|---|---|---|
| `RELAY_ARMED` | a soft mark crossed; the leg will rotate at the next safe point | Nothing urgent. Let the leg finish its current atomic action; confirm it externalizes and rotates rather than spinning in the armed state. |
| `RELAY_ROTATED` | the leg wrote a baton at a safe boundary and handed off cleanly | Launch the successor from the baton. **Re-verify its cursor before trusting any progress field** — the baton is a pointer, not a claim. |
| `RELAY_GOAL_DONE` | `done_when` was satisfied against the durable store; the relay ended normally | Close the relay. Do **not** launch another leg unless the durable done witness is later invalidated. |
| `RELAY_NOT_EXTERNALIZED` | rotation was refused: load-bearing state still lives only in the transcript | Commit, file, or ledger the missing state, then rerun the externalize check. If it cannot be made durable, park the relay rather than force-rotating. |
| `RELAY_PARKED_UNSAFE` | the hard ceiling arrived before any verified safe point, so the leg parked | Resume with an operator or stronger witness (below). Recover only durable state and write a clean baton before relaunching automation. |
| `RELAY_BATON_STALE` | successor re-verification found the baton cursor no longer matches git/ledger | Discard the stale baton fields and re-derive the cursor from ground truth. Write a fresh baton before continuing — never trust the stale one. |
| `RELAY_NO_PROGRESS` | the relay made no verified progress for the configured number of consecutive legs | Stop automatic rotation. Inspect the blocker and the hysteresis settings. Relaunch only after a new progress witness or a narrowed objective exists. |

Two rules cut across the whole table. First, **no `claimed` field survives a read**: a
baton's progress is a cursor the successor re-verifies (`relay.VerifyReload`), so a
`RELAY_ROTATED` tombstone is an invitation to check git, not a receipt. Second, **a parked
or stuck relay is the healthy outcome**, not a failure — the alternative (blowing the
window to keep going, or spinning fresh legs forever) is what the design refuses.

## Resume a parked relay

A parked relay (`RELAY_PARKED_UNSAFE`) stopped because continuing would have overrun
context mid-action. Resume it deliberately:

1. **Inspect the parked baton.** `fak relay resume --baton <parked.baton.json>` — read the
   tombstone note and the `next_action`. The note is display-only; do not consume it as
   progress.
2. **Re-verify the cursor from ground truth.** Check `progress_cursor.start_sha` is still
   an ancestor of `HEAD`, that `held_region` is free to re-acquire, and that every
   `artifacts` pointer resolves. If any has drifted, treat the baton as `RELAY_BATON_STALE`
   and re-derive the cursor rather than trusting the note.
3. **Re-acquire the lane.** `dos arbitrate --lane <lane>` for the baton's `held_region`.
   A parked relay does not hold its lease across the pause; take it back before writing.
4. **Recover only durable state, then relaunch.** Seed the fresh leg from the (re-verified)
   baton and a query tool — never from a transcript excerpt. Write a clean baton at the
   next safe point before handing back to automation.

The same recovery path applies to a `RELAY_BATON_STALE` resume: the only difference is
*why* the cursor mismatched (git moved under a parked baton vs. a stale handoff). In both,
the fix is re-derive-from-durable, not trust-the-note.

## What "done" looks like for a relay

A relay terminates in exactly two healthy ways, and one unhealthy one:

- **`RELAY_GOAL_DONE`** — `done_when` is satisfied against the durable store. The relay
  ends; no further leg launches. (Idempotent: a done relay stays done, #1897.)
- **`RELAY_ROTATED` at the final leg** — the last needed leg handed off and the successor
  found the goal already done on reload.
- **`RELAY_NO_PROGRESS`** — the unhealthy stop: K consecutive legs made no verified
  progress, so the relay escalated instead of thrashing. This needs an operator, not
  another automatic leg.

## Non-goals

- This runbook documents no API beyond the CLIs shipped in the epic (`fak relay resume`).
- It does not authorize any new refusal path — the reason tokens are data until a floor
  consumes them.
- It does not make transcript summaries trustworthy; relay successors always re-derive
  from durable witnesses.
