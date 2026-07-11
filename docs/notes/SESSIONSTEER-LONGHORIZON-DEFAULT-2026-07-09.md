---
title: "Sessionsteer: long-horizon persistence + managed-context, on by"
description: "Introduces sessionsteer, a pure core that turns context-value advice into a steering directive so headless workers persist long sessions, on by default."
---

# Sessionsteer: long-horizon persistence + managed-context, on by default

Date: 2026-07-09
Epic: #2198 (automatic context) · Spine issue: #3512
Sibling doctrine notes: `CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md`,
`CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md`, `AUTOCTX-R7-RELAY-DEFAULT-ADMISSION-TRIAGE-2026-07-05.md`

## The gap this closes

fak already *measures* a long session's context health: the gateway serves a context-value
report (`mcp__fak__fak_context_value`) whose `step_advice` is a closed verdict —
`any | bounded | checkpoint | rebuild | unknown` — over how much of the active window remains and
whether a context event just rewrote it. It is a sensor with no default consumer. Nothing turns
`checkpoint` into "commit now," nothing turns `rebuild` into "re-anchor from durable state," and
nothing tells a headless worker to *keep going* past the point a raw window would feel full.

Separately, the harness's own instinct is to **stop**: a worker yields the turn when the context
feels long, exactly when a managed session should press on. The managed-context machinery
(cache-preserving compaction, prefix retention) makes long horizons *cheap*, but the agent is never
told the posture changed, so it behaves as if the window were a hard wall.

`sessionsteer` is the missing consumer: a pure core that folds the context-value advice plus a few
content-free session facts into one typed directive the guard hooks act on, **on by default** for
the sessions that need it.

## What ships in the spine (this note's PR)

- `internal/sessionsteer` — a tier-1 pure core, `Steer(SteerInput) SteerDirective`, exercised by a
  golden table. Stdlib-only; the advice vocabulary is a string mirror of `gateway.StepClass` so the
  core carries no dependency on (and cannot cycle with) the gateway package.
- SessionStart hook wiring (`cmd/fak/guard_sessionstart.go`): a headless/fleet worker (`claude -p`,
  not an attended TUI) is admitted MANAGED and its first-turn injection now carries the
  **persistence + managed-context rule** on top of the existing MCP-affordance hint. Attended
  interactive sessions get the affordance alone. Default-on via `guardSessionStartManaged(command)`.
- The Stop-hook **persist** half is DESIGNED here but ships in shadow — the directive computes a
  `BLOCK_STOP` / `ALLOW_STOP` decision, but nothing enforces it yet (see fanout).

## The two decisions the core makes

### 1. Admission — MANAGED vs LEGACY (never silent)

Every session gets a structured admission reason; a session is never *silently* left unmanaged.

| DurableStore | Headless | GoalActive | → Admit | Reason |
|---|---|---|---|---|
| no | * | * | LEGACY | `LEGACY_NO_DURABLE_STORE` |
| yes | yes | yes | MANAGED | `MANAGED_HEADLESS_GOAL` |
| yes | yes | no | MANAGED | `MANAGED_HEADLESS` |
| yes | no | yes | MANAGED | `MANAGED_ATTENDED_GOAL` |
| yes | no | no | LEGACY | `LEGACY_ATTENDED_NO_GOAL` |

Rationale: persistence matters most where **no human is present to keep the session going** — a
headless fleet worker. An attended human-driven TUI with no standing goal is left alone (no heavy
posture imposed). A session with no durable store to checkpoint into cannot be managed — but it says
so, with a reason, rather than silently degrading. This mirrors R7's "admit the goal session onto
the managed posture by default; keep interactive opt-in" (`AUTOCTX-R7-...`).

### 2. Persist — BLOCK_STOP vs ALLOW_STOP (priority ladder, shadow for now)

Computed independent of admission — the core always reports the *truth* about the work state; the
Stop-hook wiring decides whether to ENFORCE a block (and, in the spine, does not yet).

1. goal active & not met → `BLOCK_STOP` / `PERSIST_GOAL_UNMET`
2. handoff artifact not written → `BLOCK_STOP` / `PERSIST_HANDOFF_REQUIRED`
3. pending checkable work → `BLOCK_STOP` / `PERSIST_WORK_REMAINS`
4. goal active & met → `ALLOW_STOP` / `STOP_CLEAN_GOAL_MET`
5. otherwise → `ALLOW_STOP` / `STOP_CLEAN_NO_WORK`

### Step-advice → steering text

`checkpoint` → land durable state now (commit, write the plan/ledger/handoff).
`rebuild` → re-anchor from durable state after the context event before any wide step.
`bounded` → single-concern step; keep new residency deliberate.
`any` / `unknown` → no directive (no evidence to steer on; unknown fails closed, never headroom).

### 3. Floor reconciliation — the persist-hook must respect the write floor

The persist decision and the capability floor are two of the levels this feature spans, and they can
**deadlock**. A `BLOCK_STOP` tells the Stop hook to keep a session from yielding until it persists —
"checkpoint/commit now." But `fak`'s guard floor denies the agent's own git-write as a write-shaped
`.git/` self-modify (`SelfModifyGlobs`). So a MANAGED headless worker told to `BLOCK_STOP` for
"commit your progress," on a floor that refuses the agent's commit, is ordered to do something it
*cannot* do — and spins forever.

This is not hypothetical: it is exactly what happened building this spine. The Stop hook demanded a
green commit; the write floor refused every agent git-write (`POLICY_BLOCK/TERMINAL`); the session
could neither commit nor cleanly stop. Naive multi-level persistence *manufactures* this deadlock.

The core reconciles it rather than letting the levels collide. `SteerInput.PersistFloorDenied` reports
that the floor denies the agent's durable-persist path and no within-floor sink exists. When set, a
would-be `BLOCK_STOP` downgrades to `ALLOW_STOP` / `STOP_PERSIST_FLOOR_DENIED`: the session is allowed
to yield and persistence becomes **operator-mediated** (the human runs the commit, or grants a scoped
capability), instead of the hook wedging against the floor. Zero value preserves the plain ladder, so
this only bites when the floor actually denies persistence.

The durable lesson for the fanout: a persist-hook must be **floor-aware**. Two clean ways to give a
MANAGED headless session a real within-floor persist path (either makes `PersistFloorDenied` false):
- **Prefer floor-passing sinks for checkpoints** — write a handoff/ledger/plan *file* (an allowed
  write, not a `.git/` self-modify). Git commit becomes an operator-mediated *finalization*, not a
  per-turn agent action. Checkpoint ≠ commit.
- **Grant one scoped, audited commit lane** — a narrow verb (e.g. a `fak` checkpoint-commit over
  agent-owned paths only) that the floor allows, keeping the general git-write deny intact.

## Why a pure core + thin shell

The whole decision is a deterministic function of a content-free snapshot, so it is exercised by a
golden table and carries no session content — the directive is safe to log. The impure parts (read
the live context-value report for a trace; know whether a goal is set; enumerate pending work) live
in the guard shell and hand the core plain data. Same layering the sibling pure cores use
(`looporphan`, `doomloop`, `resume`).

## Fanout (tracked off #3512 / #2198)

1. **Stop-hook enforcement out of shadow** — honor `BLOCK_STOP` for a MANAGED session behind a fleet
   soak + a kill-switch. This is the load-bearing, risky half; it must not wedge a session that is
   genuinely done. Gate on a real done-condition signal, not just "work remains." **Must set
   `PersistFloorDenied` from the live floor** (probe whether the agent has any within-floor persist
   path) so enforcement can never deadlock against the write floor — see "Floor reconciliation" above.
   Pair enforcement with a floor-passing checkpoint sink so the block is satisfiable, not just refused.
2. **Live dynamic directive** — a consumer that reads the running session's `step_advice` mid-flight
   (a periodic tick or a PreToolUse hook) and injects the `checkpoint`/`rebuild` steering text at the
   moment the advice flips, not only at SessionStart. The SessionStart rule is the standing posture;
   this is the just-in-time nudge.
3. **Real goal / pending-work signals** — wire `GoalActive`/`GoalMet`/`PendingWork` to actual
   sources (the `/goal` marker, the task ledger, uncommitted-checkable-work) instead of the spine's
   headless-only proxy.
4. **`fak sessionsteer` verb** — render the directive from a live trace for operator inspection and
   dogfooding, deferred out of the spine to keep the main.go dispatch surface untouched.
5. **Reason vocab into the refusal set** — surface the admission + persist reasons through the
   structured-refusal machinery so a blocked stop is an auditable "no," not prose.
