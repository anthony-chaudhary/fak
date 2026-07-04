---
title: "fak proof: SubmissionHandle async-addressing seam (the message-tag / multi-queue analogue)"
description: "Documents the frozen-but-inert SubmissionHandle.{Seq,Queue,Opaque} + Ext/ExtKey async addressing contract as the MPI message-tag / multi-communicator analogue — a reserved seam shaped for a future driver, with no scheduler, no tag-matcher, and no ANY_TAG/ANY_SOURCE receive behind it."
---

# A · SubmissionHandle async addressing

`internal/abi` freezes the typed identity a non-blocking `Submit` hands back and a later
`ReapAny` completes against: `SubmissionHandle{Seq uint64, Queue uint32, Opaque uint64}`
(`internal/abi/types.go:313`), plus the OPEN typed sidecar `Ext map[ExtKey]Ref` over the
registered `ExtKey uint32` (`types.go:171`, `types.go:267`). This doc names each field,
its MPI analogue, and — the load-bearing part — its **inert** status, so a future async
driver that fills the seam stays honest about what the contract does and does not promise.

It documents **existing frozen fields**. It adds no field and edits no ABI; the `abi`
package is additive-only and human-owned, and this is a reading of it, not a change to it.

> **Honesty caveat (read first).** This is an analogy to MPI message-tag / communicator
> addressing, **not** a working multi-communicator transport. `Queue` is a `uint32`
> routing slot with **no scheduler behind it**; there is no tag-matching engine, no
> `MPI_ANY_TAG` / `MPI_ANY_SOURCE` receive, and no communicator-split semantics. fak has
> **one process-global engine fold**, and these fields are **inert** — shaped like MPI
> tag/communicator addressing for a *later* driver to fill, carrying no runtime behavior
> today. The seam is routing/identity only; there is nothing to verify at runtime because
> nothing reads `Queue` to pick a queue. The doc's job is to keep that future driver from
> quietly claiming MPI semantics the field has never had.

---

## The fields

`SubmissionHandle` (`internal/abi/types.go:313`) is the typed identity of one non-blocking
`Submit`. A blocking call never mints one; an async driver returns it from `Submit` and the
kernel later completes it via `ReapAny` (`internal/kernel/kernel.go:448`).

| field    | type     | role                                                            | MPI analogue                                  | status |
|----------|----------|-----------------------------------------------------------------|-----------------------------------------------|--------|
| `Seq`    | `uint64` | request identity; `== ToolCall.SeqNo` — binds a completion to its submission | message **tag** identity (the per-message correlation handle) | **live** as an identity; it is the key `ReapAny` matches a `Completion` to |
| `Queue`  | `uint32` | which completion queue / per-engine routing slot (multi-engine / multi-queue) | **communicator id** / message-tag subset (which sub-communicator a receive draws from) | **inert** — reserved; one process-global fold, no scheduler reads it |
| `Opaque` | `uint64` | driver-private correlation token (a driver stashes its own cursor here) | driver-private metadata the transport carries opaquely | **inert** to the kernel — never interpreted above the driver that set it |

The OPEN typed sidecar `Ext map[ExtKey]Ref` (`types.go:171`) carries per-subsystem
correlation payloads — a spec id, an async cursor, label rows — keyed by `ExtKey uint32`
(`types.go:267`), an OPEN registered key with **reserved per-subsystem ranges**. Unknown
`ExtKey`s MUST be ignored (forward-compat: a new subsystem's sidecar never breaks an older
reader). `Ext` is the typed escape hatch a future async driver uses to carry routing
metadata *without* widening the frozen `SubmissionHandle` struct.

---

## Theorem A.1 — the seam is identity-and-routing only; no behavior rides on it

**THEOREM.** Of the three `SubmissionHandle` fields, only `Seq` is load-bearing today (it
is the completion-to-submission binding key). `Queue` and `Opaque` are reserved routing /
correlation slots that no scheduler, tag-matcher, or receive-side selector consumes — so
the multi-queue / multi-communicator semantics they are *shaped* for do not exist yet.

**REGIME.** Documentation of a frozen, inert contract — there is no runtime behavior to
witness because the behavior is deliberately absent. The verifiable claim is the negative
one: nothing reads `Queue` to make a scheduling decision.

**ARGUMENT.**
- `ReapAny` (`internal/kernel/kernel.go:448`) completes "one handle from a request set":
  it consumes an already-ready completion and otherwise drives the single engine fold to
  produce one. It matches a `Completion` to a submission by **`Seq`** (the `Completion`
  carries `Handle SubmissionHandle`, `types.go:307`). It does **not** branch on `Queue` to
  select among multiple queues, and it does **not** interpret `Opaque`. There is one
  global fold, not a per-`Queue` scheduler.
- There is no `MPI_ANY_TAG` / `MPI_ANY_SOURCE` analogue: a receive cannot say "give me a
  completion from any queue matching this tag pattern," because there is no tag-matcher and
  no multi-queue set to match against. `ReapAny` takes an explicit `[]SubmissionHandle`
  request set and returns one of *those* — a closed set the caller named, not a wildcard
  receive over an open communicator.
- `Queue`'s comment (`types.go:315`) — "which completion queue (multi-engine /
  multi-queue)" — names the *intended* future role. Until a driver introduces more than
  one engine fold, every handle's `Queue` is whatever the single driver stamps, and no
  code dispatches on it.

**WITNESS.** `go test ./internal/kernel -run TestReservedQueueOpaqueAreInert`
mutates both reserved fields on a submitted handle and proves `TestHandle`, `ReapAny`,
and `Reap` still complete the original submission by `Seq`. Source-audit backstop:
`rg -n "\.Queue\b|Queue:\b|Queue\s+uint32" internal/abi internal/kernel` reports only
the `SubmissionHandle.Queue` field declaration.

**VERDICT.** **PROVEN** (2026-07-01). The current kernel treats `Queue` and `Opaque`
as inert reserved slots; no scheduling, tag-matching, or receive-side selection semantics
ride on them today.

**CONSEQUENCE.** A future async/multi-engine driver MAY give `Queue` real
routing meaning and use `Opaque` / `Ext` for its own correlation — that is exactly the
seam's purpose. When it does, it inherits a *frozen* handle shape (so two independent
drivers cannot collide on the wire form) and this caveat: the moment `Queue` selects a
queue, the driver owes a written account of its matching semantics, because the bare field
guarantees none.

---

## The comm/compute-overlap arm (concurrent Reap over the seam, issue #651)

`Reap` (`internal/kernel/kernel.go`) pops a pending submission under `k.mu` and then
releases the lock **before** calling `eng.Complete` — so W host goroutines reaping
**distinct** handles run their inference *concurrently*, overlapping the slow engine
round-trip the way the MPI `Isend`/`Irecv` + progress story overlaps communication with
computation (borrowed for **shape only**). `cmd/fanbench`'s overlap arm
(`cmd/fanbench/overlap_test.go`, a go-test fixture) is the one bench that isolates that
factor: it drives the real kernel via `Submit`/`Reap` over a **deterministic offline
engine** — one whose call tally is a `sync/atomic` counter, *not* `engine.Mock`'s racy
`m.calls++`, so it is `-race`-clean — once with a single reaper (inference serialized) and
once with W concurrent reapers (inference overlapped), and reports

    overlap-efficiency = (serial adjudicate + serial infer) / measured wall

The number is **measured**, so the fixture asserts only the structural facts (the engine
was called exactly N times, results are bit-identical whichever goroutine reaped them, the
overlapped infer wall is strictly below the serialized one) and **logs** the efficiency
rather than pinning a wall-clock value — it ships no flaky number.

> **Honesty caveat.** This measures fak's **ACTUAL overlap shape** — host-driven concurrent
> `Reap` over an engine whose `Complete` runs **inline** on the reaping goroutine — **NOT**
> an MPI non-blocking progress thread, and **NOT** MPI/HPC latency-hiding. It borrows the
> `Isend`/`Irecv` + progress *story* for shape only; it **never** carries over an
> MPI/InfiniBand overlap, message-rate, or throughput number. The `overlap-efficiency`
> above is a ratio of two locally measured wall times on this box, nothing more.

---

## What this is NOT

- **Not** a multi-communicator transport. One process-global engine fold; `Queue` selects
  nothing today.
- **Not** an `MPI_ANY_TAG` / `MPI_ANY_SOURCE` receive. `ReapAny` completes one of an
  explicit handle set, not a wildcard receive over an open communicator.
- **Not** an ABI change. Every field documented here already exists, frozen, in
  `internal/abi/types.go`; this doc adds nothing to the struct.
- **Not** a scheduler. There is no behavior to verify at runtime — the seam is inert by
  design, and that inertness is the honest claim.
- **Not** an MPI progress thread. The overlap arm's concurrent `Reap` is host goroutines
  running an *inline* `Complete`, not a non-blocking transport with a background progress
  engine; its `overlap-efficiency` is a locally measured wall-time ratio, never an
  MPI/InfiniBand overlap or message-rate number.

---

## See also

- [`internal/abi/types.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/abi/types.go) — `SubmissionHandle`, `Completion`, `Ext`/`ExtKey` (the frozen source).
- [`internal/kernel/kernel.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/kernel/kernel.go) — `ReapAny`, the first (and today only) consumer of a handle.
- [`cmd/fanbench/overlap_test.go`](https://github.com/anthony-chaudhary/fak/blob/main/cmd/fanbench/overlap_test.go) — the comm/compute-overlap arm (issue #651): concurrent `Reap` over an atomic offline engine, reporting `overlap-efficiency` as a `-race`-clean go-test fixture.
- [vdso-revoke-as-comm-revoke.md](../explainers/vdso-revoke-as-comm-revoke.md) — the sibling MPI-analogue mapping (`vdso.Revoke` ≈ `MPI_Comm_revoke`) in the same epic.
- [abi+architest.md](abi+architest.md) — the proof that the ABI's additive-only / human-owned discipline is enforced.
