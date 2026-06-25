---
title: "fak note: sound restore against semantic rollback replay (2026-06-25)"
description: "A runnable fixture showing a restored fak session does not re-fire an already witnessed irreversible effect, contrasted with an ungated ACRFence-style semantic rollback control."
date: 2026-06-25
---

# Sound Restore Against Semantic Rollback Replay

> Date: 2026-06-25. Status: SHIPPED fixture.
> Witness: `go test ./internal/sessionimage -run TestSoundRestoreSkipsWitnessedEffectReplay -count=1`.

## What ACRFence Names

The ACRFence paper identifies a checkpoint-restore failure mode for LLM agents: after a
restore, the agent can synthesize a subtly different irreversible request, and an external
server may treat it as a fresh operation instead of a replay. ACRFence's proposed defense is
to track irreversible effects and enforce replay-or-fork semantics. Source:
<https://arxiv.org/abs/2603.20625>.

This note does not claim fak implements the full ACRFence protocol. The fixture isolates one
sound-restore property in fak's shipped organs: once an effect has a witness-backed
`VerifiedDone` rung, a restored session consults that rung before deciding whether to execute
the effect again.

## The Fixture

`internal/sessionimage/sound_restore_test.go` builds two arms:

| Arm | Behavior |
|---|---|
| Ungated semantic-rollback control | Fires `refund-ledger/refund-500/original`, restores, then fires `refund-ledger/refund-500/regenerated-after-restore`. The ledger count is 2, modeling the duplicate effect ACRFence warns about when the regenerated request is treated as new. |
| fak restore | Fires the original effect once, writes a receipt artifact, marks the task claimed-done, and runs `taskmgr.PathWitness` over the receipt. The witness record is `verified_done`; the session is dumped with `sessionimage.DumpDir`, loaded with `LoadDir`, and resumed with `Rehydrate`. On resume, the fixture re-reads the witness evidence and skips the regenerated effect. The ledger count stays 1. |

The image carries the witness pointer/state in `Meta.Labels` and a recall page admitted under
the witness token `taskmgr:<task>:verified_done`. The effect proof itself is not a model
self-report: `PathWitness` reads the receipt from outside the agent's claimed state. A host
can plug a stronger source, such as `dos commit-audit`, into the same `taskmgr.WitnessRecord`
shape.

## Local Fences Used

- `sessionimage.DumpDir` / `LoadDir` / `Rehydrate` preserve the drive and recall image across
  the checkpoint boundary. The test restores onto a different model and host and records the
  migration.
- `internal/session.Decide` takes `ReasonDrained="DRAINING"` at the next turn boundary. The
  fixture restores a `Draining` drive and asserts the next `Decide` finalizes it to `Stopped`
  without advancing another turn.
- `internal/agent.InKernelPlanner.EvictPoisoned` uses the `<|im_end|>` turn boundary for
  poison eviction. That is the same boundary discipline: do not splice or replay partial
  turns.
- `internal/shipgate.Evaluate` owns the keep-bit; callers can only read it through `Kept()`.
  That is the same non-self-report discipline as `VerifiedDone`: a claimed success is not the
  proof.

## Honest Scope

This is a runnable fixture, not a transactional effects store and not a full ACRFence
implementation. `sessionimage` persists the session drive/content/trajectory, while the
effect ledger and witness evidence live in the host layer. The acceptance property is narrower
and checkable: fak can restore a session image and use an independently read witness rung to
avoid re-firing an already completed irreversible effect, while an ungated semantic-rollback
control fires the regenerated effect twice.
