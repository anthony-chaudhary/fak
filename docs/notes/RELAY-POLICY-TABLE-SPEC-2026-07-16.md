---
title: "Relay operator policy table"
description: "Data-only field specification for the dos.toml [relay] operator policy table."
date: 2026-07-16
---

# Relay operator policy table

Status: data-only spec (issue #1888). No code reads a `[relay]` table yet, and this
file changes no `dos.toml`. It pins the table's shape — five fields, their types,
and their defaults — so the future loader builds against one contract. Every field
configures a knob that has ALREADY shipped in `internal/relay`; the table adds no
new mechanism, it only turns values the rungs take as caller-supplied scalars into
operator DATA. That seam is deliberate on the code side: the G3 evaluator's own
doc comment says its soft mark is "supplied by the caller from an Envelope field
or the `[relay]` dos.toml table, never a constant baked here"
(`internal/relay/armtriggers.go`). Reason tokens cited below are the closed relay
vocabulary in [`RELAY-REASON-VOCABULARY-2026-07-01.md`](RELAY-REASON-VOCABULARY-2026-07-01.md).

## Shape

| Field | Type | Default | Shipped knob | Reason token |
|---|---|---|---|---|
| `soft_mark` | float | `0.85` | `ArmTriggers.SoftMark` (`internal/relay/armtriggers.go`) | `RELAY_ARMED` (advisory) |
| `hard_ceiling` | float | `1.0` | `CeilingPark` park path (`internal/relay/parkunsafe.go`) | `RELAY_PARKED_UNSAFE` |
| `rotations_per_hour` | integer | `4` | `RotationCap.MaxPerHour` (`internal/relay/rotationcap.go`) | `RELAY_ROTATION_CAPPED` |
| `min_progress` | integer | `1` | `ArmHysteresis.MinSteps` (`internal/relay/hysteresis.go`); counted by `NoProgressEscape` (`internal/relay/noprogress.go`) | `RELAY_NO_PROGRESS` |
| `done_hook` | string | `""` | `done_when` check via the driver's `DoneCheck` hook / `ReasonGoalDone` (`internal/relay/driver.go`) | `RELAY_GOAL_DONE` |

Unknown fields are invalid (closed shape, matching the baton schema's discipline).
Every shipped knob keeps its fail-safe zero-value semantics: an explicit `0` (or
`""`) disables that policy rather than inventing a hidden fallback.

## `soft_mark`

Purpose: the advisory arm threshold — the fraction of any budget axis's cap at
which a leg arms for rotation at the next verified safe point.

- Type and default: float in `(0, 1]`; default `0.85`. `0` disables arming
  (the shipped `ArmTriggers` zero value is inert: an unset policy never arms).
- Shipped knob: `ArmTriggers.SoftMark` (`internal/relay/armtriggers.go`), folded
  per axis by `ArmTriggers.Cross` in the division form `(used/cap) >= SoftMark`.
- Reason token: `RELAY_ARMED` — advisory, not a refusal; the leg continues to a
  verified safe point. Must sit strictly below `hard_ceiling`.

## `hard_ceiling`

Purpose: the hard park boundary — the fraction of the leg's window at which a leg
that never reached a verified safe point parks instead of overrunning context.

- Type and default: float in `(0, 1]`; default `1.0` (park only at window
  exhaustion). Must be `>= soft_mark`; the arm phase sits below the ceiling.
- Shipped knob: the `CeilingPark` park path (`internal/relay/parkunsafe.go`):
  `Park` renders the terminal tombstone anchored at the last committed SHA
  (`Anchor`), so a park is resumable and loses no committed work.
- Reason token: `RELAY_PARKED_UNSAFE` (`ReasonParkedUnsafe`,
  `internal/relay/parkunsafe.go`) — an `OPERATOR_GATE`: resume needs a human or
  stronger witness.

## `rotations_per_hour`

Purpose: the runaway-rotation bound — the maximum accepted rotations inside the
trailing wall-clock hour before further proposals are held.

- Type and default: integer `>= 0`; default `4`. `0` disables the cap (the
  shipped zero value never holds — an unset policy never refuses).
- Shipped knob: `RotationCap.MaxPerHour` (`internal/relay/rotationcap.go`),
  folded per proposal by `RotationCap.Admit`; a held rotation consumes no slot.
- Reason token: `RELAY_ROTATION_CAPPED` (`ReasonRotationCapped`) — hold the leg
  and let the window slide before proposing again.

## `min_progress`

Purpose: the anti-thrash progress floor — the minimum LEDGER-VERIFIED progress-step
movement since the last arm before a leg may re-arm. Verified, never claimed: an
unverifiable read fails closed as no movement.

- Type and default: integer `>= 0`; default `1`. `0` disables the gate (every
  arm permitted).
- Shipped knobs: `ArmHysteresis.MinSteps` (`internal/relay/hysteresis.go`) is
  the gate this field sets directly; its terminal sibling `NoProgressEscape`
  (`internal/relay/noprogress.go`) counts the legs that show no verified forward
  movement against the same D3 verified-progress read. Whether the escape's
  `MaxEmptyLegs` (K) gets its own table field is deferred to the wiring floor.
- Reason token: `RELAY_NO_PROGRESS` (`ReasonNoProgress`) — an `OPERATOR_GATE`
  when the escape trips; the hysteresis gate itself withholds re-arm silently.

## `done_hook`

Purpose: the workspace-default done predicate — the one-line durable-store check
seeding `done_when` for relays launched without their own, so a done relay stays
done instead of minting another leg.

- Type and default: string; default `""` (unset — the launcher or incoming baton
  must carry `done_when`; the table never overrides a baton's own predicate).
- Shipped knob: the driver's done-check (`internal/relay/driver.go`): the
  `DoneCheck` hook evaluates the leg's `done_when` against the durable store
  BEFORE any work, and the wire field is `Baton.DoneWhen` (`json:"done_when"`,
  `internal/relay/baton.go`).
- Reason token: `RELAY_GOAL_DONE` (`ReasonGoalDone`, `internal/relay/driver.go`)
  — a `TRUE_DRAIN`: close the relay; do not launch another leg.

## Out of scope

The `dos.toml` wiring — the loader, validation, and any live gate consuming this
table — is deferred; this document pins the shape only.
