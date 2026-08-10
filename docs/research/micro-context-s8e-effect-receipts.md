---
title: "Micro-context S8e effectful stages bound to receipts"
description: "Controlled effect batch through capability, resource, idempotency, journal, and independent read-back seams."
status: observed
last_reviewed: 2026-08-09
---

# S8e: effectful micro-windows require witnessed receipts

## Verdict

**Observed in the controlled fixture:** a selector may choose a declared effect stage, but it does
not acquire authority and its narration never enters the fold. The controller converts a selected
`set-label` action into a capability/resource/operation/idempotency-bound intent, executes only
through `internal/microagent.EffectCoordinator`, and folds only independently read-back receipts.

Captured artifact:
[`s8e-local-effect-batch-pass-2026-08-09.json`](../../experiments/microcontext/s8e-local-effect-batch-pass-2026-08-09.json).

This is a local controlled-state witness, not a claim about a remote service's transactional or
exactly-once guarantees.

## Reproduce

```bash
go run ./cmd/microcontextdemo \
  -effect-batch-selfcheck \
  -effect-batch-output /tmp/effect-batch.json

go run ./cmd/microcontextdemo \
  -verify-effect-batch /tmp/effect-batch.json
```

## Captured states

The 22 selected fixture actions yield:

- six newly confirmed writes and three independently confirmed idempotent replays;
- five capability/stage denials;
- one resource-lease conflict and one pre-write partial failure;
- one cancellation before dispatch, with zero dispatch attempts;
- one dispatched write left `unknown_pending_readback` after cancellation, then independently
  confirmed through replay/read-back without another write;
- one approval-gated not-run, one dry-run, and two breaker-open not-runs;
- eight dispatch attempts, seven physical writes, zero duplicate writes, and zero restart writes;
- nine confirmed receipts entering the fold, while the unknown remains explicitly visible but is
  excluded from confirmed effects.

## State machine

```text
selected
  -> denied | dry_run | approval_not_run | breaker_not_run
  -> cancelled_before_dispatch
  -> conflict | failed_before_write
  -> dispatched
       -> confirmed
       -> unknown_pending_readback
            -> replayed_confirmed after independent read-back
```

There is deliberately no `cancelled_after_dispatch = rolled_back` transition. Once dispatched, a
cancellation only stops waiting; it says nothing about whether the remote effect landed.

## Exactly-once qualification

The fixture proves **one physical local write per idempotency key** because the durable local store
records success and restart replay reuses that record. It does not prove universal exactly-once
remote effects. A process can crash after a remote system commits but before the local success record
is durable. Resolving that interval requires one of:

- remote idempotency keys with durable response lookup;
- transactional outbox/inbox coordination;
- operation-specific read-back that can identify the intended effect;
- or an explicit `unknown` requiring reconciliation rather than blind retry.

The demo uses the last two disciplines and never replays an effect from an ordinary content cache.

## Steelman perspectives

### Why micro-window effects are attractive

Per-record intents make capability, idempotency, resource ownership, approval, receipts, and retries
precise. Independent records can execute concurrently, duplicates collapse, failures remain local,
and confirmed effects can be folded into a transparent batch result.

### Why a conventional transaction or workflow engine may be better

Databases, queues, sagas, and workflow engines already provide durable ordering, retries, leases,
compensation, and operator consoles. An LLM micro-window should choose among declared operations,
not replace those systems. For tightly coupled multi-row invariants, one database transaction is
safer than many independently correct micro-effects.

### Why idempotency is not enough

A malformed key can deduplicate distinct intended writes; a stale key can replay obsolete output;
and a supposedly idempotent API may only scope keys to one tenant or time window. Keys must bind the
semantic operation, target, authorization context, and version, with collision/refusal telemetry.

### Why circuit breakers matter

Independent effects do not imply independent model errors. A bad selector rubric can correlate
hundreds of wrong writes. The fixture opens a breaker after three authorization denials and leaves
later actions not-run. Production breakers should also watch semantic disagreement, read-back
failure, and unusual action-rate shifts.

## Boundary

This closes the controlled effect-safety spine for #6034. Real integrations still require each
remote system's idempotency and reconciliation contract; no generic kernel can infer rollback or
exactly-once behavior from cancellation alone.
