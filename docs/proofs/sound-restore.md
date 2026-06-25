---
title: "fak proof: sound restore does not re-fire an already-witnessed effect"
description: "A runnable fixture demonstrating fak's restore consults a persisted, non-forgeable keep-bit before re-executing an effect — the distinction ACRFence's semantic-rollback attack exploits — against an ACRFence-style control that re-executes."
---

# D · sound restore vs the ACRFence semantic-rollback attack

The deepest hole in agent durable-execution: on restore, a framework **re-executes side
effects it already performed** — a payment, a database write, an email — because its
checkpoint cannot distinguish *already-completed* from *needs-replaying*. **ACRFence**
(arXiv:2603.20625) documents this "semantic rollback attack" across LangGraph, CrewAI,
and Google ADK; Diagrid's structural statement of the same gap is blunt: *"checkpoints are
not durable execution."* A checkpoint that records *what the agent was doing* but not
*which effects are provably done* will, on restore, replay the done ones.

fak already mints the missing distinction. The **keep-bit** (`taskmgr.VerifiedDone`) is
raised onto an effect **only** by a `Witness` that read the effect back from a source the
process did not author (`internal/taskmgr/evidence.go:20-21`, the `Witness` interface at
`:66`). It is the kernel's standing rule — never trust the agent's own completion string
— applied to a side effect. This fixture shows that keep-bit surviving a real
checkpoint/restore boundary and gating re-execution, where an ACRFence-style control with
no such bit re-fires.

This is a **demonstration on fak's own organs, not a fielded transactional store** — the
honesty note below states exactly what is and is not claimed.

---

## Claim — restore consults the persisted keep-bit, and it is non-forgeable

**CLAIM.** A session performs a witnessed effect (`VerifiedDone`), checkpoints via
`sessionimage.DumpDir`, and is restored in a fresh process via `LoadDir` + `Rehydrate`.
On the restored handle, the fak arm reads the **persisted** keep-bit
(`Resumed.Witness[…].Record.VerifiedState == VerifiedDone`) and does **NOT** re-fire the
effect — the observable effect counter stays at **1**. An ACRFence-style control with no
keep-bit re-executes the regenerated effect — its counter reaches **2**. The keep-bit is
**non-forgeable across the offload boundary**: it rides a first-class, sha256-indexed part
(`witness.json`), so a tampered "already done" fails the whole image closed on `LoadDir`
rather than waving a skipped-but-not-completed effect through.

**REGIME.** D — fail-closed gate (a fixture, not a regime-D soundness *theorem*; see the
honesty note).

### The shipped organs it rides

- **The non-forgeable keep-bit** — `taskmgr.WitnessRecord.VerifiedState` is raised to
  `VerifiedDone` only by an evidence-reading `Witness` (`internal/taskmgr/evidence.go:42-50`,
  `:87-119`); the claimed `State` is never treated as proof (`:8-12`).
- **Persisted + fail-closed** — the keep-bit is written as `witness.json`
  (`internal/sessionimage/witness.go`), added to `knownParts`, and integrity-indexed +
  re-hashed on Load like every other part (`sessionimage.go:288-300`, `:341-360`). A
  forged keep-bit mismatches its recorded digest and fails the image closed
  (`TestWitnessPartGatesReplayAfterTamperFailsClosed`).
- **Boundary-not-mid-token** — the restored drive is `Draining`, and the stop is taken at
  the **next turn boundary**, never mid-decode: `ReasonDrained` (`session/decide.go:33`),
  proven independently by `TestDecideDrainingTakenAtBoundaryThenStopped`. The fixture
  re-asserts it post-restore: the resumed `Draining` session `Decide`s to `Stop` with
  `Reason==DRAINING` and `State.Run==Stopped`.
- **Quarantine rides through unchanged** — a slice the recall gate sealed in the live
  session stays sealed after pack/ship/unpack/reload (`sessionimage.go:50-57`), so a
  restore cannot be tricked into paging a poisoned effect back in.

**FIXTURE.** A session refunds a payment (`task-refund-500`), records a `VerifiedDone`
keep-bit, and checkpoints. On restore: the **fak arm** reads the persisted keep-bit and
skips the refund (counter `1`); the **ACRFence-style control** — an in-test re-executor
with no keep-bit to consult — replays the regenerated refund (counter `2`). The two
distinct refs in the control's ledger are the semantic rollback made observable.

**WITNESS.**
```
go -C fak test ./internal/sessionimage/ -count=1 -timeout 120s -run \
  'TestSoundRestoreSkipsWitnessedEffectReplay|TestWitnessPartGatesReplayAfterTamperFailsClosed' -v
```

**VERDICT.** **DEMONSTRATED** (2026-06-25). Both tests pass: the fak restore skips the
already-witnessed effect (counter 1) where the ACRFence control re-executes (counter 2),
and a forged `witness.json` fails `LoadDir` closed on the integrity check.

> **Honesty note.** This is a **fixture**, not a fielded transactional-effects engine.
> The "effect" is an in-test counter append (`effectLedger`), not a real payment; the
> ACRFence "control" is a minimal in-test re-executor, not LangGraph/CrewAI/ADK
> themselves. The claim is narrow and exact: fak **persists a non-forgeable keep-bit
> across checkpoint/restore and gates re-execution on it**, which is the precise
> distinction the semantic-rollback attack exploits the absence of. It is not a claim
> that fak provides exactly-once delivery, two-phase commit, or idempotent external I/O —
> those need an effect-side idempotency key the host supplies. What is demonstrated is the
> *restore-side* half: a checkpoint that knows which effects are provably done.

**DOS.** Shipped on the `recall` lane; bound by `dos commit-audit` at ship.
