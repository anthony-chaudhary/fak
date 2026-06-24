---
title: "Multiple-Sink Transmission for the fak KV Cache Lifecycle"
description: "How fak's ProvisionalSink seam fans one OpSpecCommit or OpSpecSquash across registered KV/context caches per (Txn, Epoch), with honest best-effort semantics."
---

# Multiple-sink transmission for the cache lifecycle (2026-06-23)

A working note on one question: is *multiple-sink transmission* — one kernel op fanning
its resolution across every registered `ProvisionalSink` — actually useful for the
lifecycle of the KV / context cache? Short answer: **yes, and it is the right shape, but
its value is latent today** (one registrant, never exercised with more than one, and the
ABI doc describes a second registrant that does not exist). This note records the
mechanism, the gap, the honest semantics, and what to wire next. It ships with an
executable witness so the conclusion is a test, not a paragraph.

## What "multiple-sink transmission" is here

`abi.ProvisionalSink` is a frozen seam (`internal/abi/types.go`):

```go
type ProvisionalSink interface {
	Promote(ctx context.Context, txn TxnID, epoch uint64) error
	Rollback(ctx context.Context, txn TxnID, epoch uint64) error
}
```

A driver that produces *retractable* effects registers one with
`abi.RegisterProvisionalSink`. The registry keeps a `[]ProvisionalSink`. The reserved
OpsSpec ops `OpSpecCommit` / `OpSpecSquash` (`internal/spec/spec.go`, `op.Invoke`) fan the
chosen resolution **across every registered sink** for one `(Txn, Spec.Epoch)`:

```go
for _, s := range abi.ProvisionalSinks() {
	if o.commit { err = s.Promote(ctx, c.Txn, c.Spec.Epoch) } else { err = s.Rollback(...) }
	...
}
```

That fan-out is the transmission. The idea: a speculative or transactional turn touches
several cache tiers at once; one kernel commit-or-squash should resolve all of them
together, instead of each cache exposing a private rollback verb the caller has to
remember to call.

## Why it is useful for the cache lifecycle

- **Atomic-intent cross-cache resolve under one `(Txn, Spec.Epoch)`.** A single
  `OpSpecSquash` retracts a misspeculated turn's KV span (`spec.Sink.Rollback` →
  bit-exact `model.KVCache.Evict`) and, once they register, any other cache's draft
  (a radix prefix node, a candidate-index span) — together, keyed by epoch.
- **One drain invariant across all sinks.** `spec.Sink.OpenCount() == 0` is the existing
  leak witness for the KV sink. The multi-sink contract generalizes it: after an epoch
  resolves, *no registered cache* holds an open provisional span. One invariant catches a
  leak in any driver.
- **A frozen cross-driver contract, not N private paths.** A new cache joins the
  lifecycle by registering, not by inventing its own evict verb a future caller must wire.
  This removes the "caller forgot to undo cache X" bug class.
- **The hardest part is already proven on the live registrant.** `spec.Sink.Rollback`'s
  bit-exact re-RoPE eviction is lossless-witnessed (`TestSpeculativeGreedyLossless`).
  Plurality — more than one sink — is the unproven axis, not exactness.

## The gap: built, dormant, doc-misrepresented

- **Exactly one registrant in the whole tree.** `spec.Sink`, registered at
  `internal/spec/spec.go` inside `Install()`, itself gated behind `polymodel.Enabled()`
  and kept off the defconfig. The registry slice and the fan loop support N; only one is
  ever registered.
- **Never exercised with more than one sink (until now).** Every prior witness registers
  exactly one sink, so the `>1`-registrant fan loop and its error branch had zero
  coverage.
- **The ABI doc names a second registrant that does not exist.** `internal/abi/types.go`
  says "the context-MMU is the canonical one" and "v0.1's MMU registers a sink"; `spec`'s
  own doc calls itself "the sibling of the context-MMU (the other ProvisionalSink)". In
  fact `internal/ctxmmu` registers only a `ResultAdmitter` and never calls
  `RegisterProvisionalSink`. (Correcting that comment is a follow-up: the `internal/abi`
  lane was peer-hot at the time of writing, so this note records the divergence rather
  than racing an edit into it.)

## The honest semantics: best-effort, not atomic

The fan-out is **not** two-phase. `op.Invoke` iterates sinks sequentially, keeps only the
*first* error, continues past a failing sink, performs **no** compensation of sinks
already resolved, and still returns the *success* `Outcome` (only `Status` flips to
`StatusError`). So a mid-fan retract failure leaves a torn cross-cache state. Today that
is harmless only *vacuously*: the single shipped sink's `Promote`/`Rollback` always return
`nil`, so the error path can never fire.

This is a real property, not a bug to hide. Two things make best-effort defensible for
now, and one names the upgrade:

- Every sink is **idempotent** on an unknown / already-resolved key, and the decode lane
  is **serial** (`polymodel.Schedule`). So a failed fan can be safely **re-driven** — the
  same `OpSpecSquash` re-invoked converges. Idempotency buys eventual atomicity by retry.
- If a caller ever needs strict all-or-nothing across heterogeneous caches, that is a
  **contract change** (a 2PC prepare/commit or a compensation pass), not just a new test.
  The witness below is exactly what such a change would have to rewrite.

## What to wire next (ranked)

| Cache tier | Priority | Why |
|---|---|---|
| `radixkv` (prefix/radix KV cache) | **high** | Retract (`EvictNode` → `KVCache.Evict`) is already bit-exact; the only gap is registration + `(txn,epoch)` node metadata. A squashed turn currently leaves phantom prefix nodes that skew hit rate in the exact speculative workload the seam serves. |
| `ctxplan.Index` (candidate index) | **high** | Append-only `Add` with no `Retract`, so a squashed turn's spans persist and poison the candidate set. Needs one `Retract(spanID)` removing from `spans`/`byID`/`posting`/`durable`, then register. Conceptually the cleanest second sink. |
| `ctxmmu` (context-MMU) | medium | The *intended* canonical sink, but its current retract is post-admission FIFO aging, not a pre-admission draft-then-promote/rollback. Becoming a real sink is a redesign, not a wiring change. |
| `vdso` (tier-2 result cache) | **no** | Category mismatch: invalidation is consistency-driven (external-witness refutation on the coherence bus), not transaction-driven. Entries carry no `(txn,epoch)`. |
| audit / decision journal | **no** | Must *never* be a sink. It is append-only, hash-chained, tamper-evident; an auditor should see decisions about calls that were later squashed. Keeping it out of the fan is a correctness property. |

`radixkv` and `ctxplan` were peer-hot at the time of writing — wire them when their trees
go quiescent, and only after the fan-out contract is itself a witness (it now is).

## The witness shipped with this note

`internal/spec/spec_test.go`:

- `TestMultiSinkTransmissionFansToEveryRegistrant` — registers the real KV `spec.Sink`
  plus a second `countingSink`, then drives `OpSpecSquash` and `OpSpecCommit` through the
  kernel op table and asserts the fan reaches **both** registrants for one
  `(Txn, Spec.Epoch)`: squash drains every sink (KV evicted + second rolled back), commit
  promotes every sink. This is the `>1`-registrant path no prior test exercised.
- `TestMultiSinkFanOutIsBestEffortNotAtomic` — pins the honest semantics with fan order
  `[kv, bad, good]`: the bad sink's `Rollback` errors after `kv` resolved and before
  `good` is reached. The op does **not** short-circuit (`good` still resolves), does
  **not** compensate (`kv` is not re-added; `bad` is left torn), and still returns
  `OutcomeSquashed` with `Status = StatusError`. A future 2PC upgrade is the change that
  would rewrite this test.

## Follow-ups

1. Correct the `internal/abi/types.go` ProvisionalSink doc comment (and `spec/doc.go`,
   `ARCHITECTURE.md`) — drop the claim that ctxmmu registers a sink, or make it true.
2. Wire `radixkv` as the first *real* second sink (node `(txn,epoch)` metadata +
   `RegisterProvisionalSink`) once the lane is quiescent.
3. Add `ctxplan.Index.Retract` + register it, so a squashed turn leaves neither stale KV
   nor a phantom index entry.
4. Decide explicitly: keep best-effort + idempotent re-drive, or upgrade the fan to 2PC /
   compensation. The witness makes either choice a visible contract.
