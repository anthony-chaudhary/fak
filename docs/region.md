---
title: "Region: one-sided shared-result pool as an RMA analogue"
description: "Maps fak's shipped Resolver.Put/Resolve shared-result pool, vDSO tier-2 shared-read cache, coherence fence, and ShareScope/taint metadata onto MPI RMA vocabulary without claiming RDMA, zero-copy, or hardware atomics."
slug: region-rma-analogue
keywords:
  - MPI RMA
  - one-sided communication
  - shared-result pool
  - Resolver
  - ShareScope
  - vDSO coherence
date: 2026-06-25
---

# Region: one-sided shared-result pool as an RMA analogue

> TL;DR: fak's shipped `abi.Resolver` surface is a one-sided, addressable payload
> pool: `Put` stores bytes and returns an `abi.Ref`; `Resolve` materializes bytes
> from that ref. The vDSO tier-2 cache uses that pool as a shared-read window:
> one adjudicated read can fill a result `Ref`, and later readers can receive the
> same `Ref` without re-running the producer. `vdso.bumpAndPublish` and
> `vdso.Revoke` are the coherence fences that make stale refs miss. This is an
> MPI RMA vocabulary analogy, not an RDMA or hardware-memory claim.

This page documents the already-shipped shape tracked by
[#654](https://github.com/anthony-chaudhary/fak/issues/654). The code paths are
`abi.Ref` / `abi.Resolver` in [`internal/abi/types.go`](../internal/abi/types.go),
the default content-addressed backend in [`internal/blob/store.go`](../internal/blob/store.go),
the optional tier router in [`internal/storedrv/storedrv.go`](../internal/storedrv/storedrv.go),
and the tier-2 shared-read cache plus coherence fences in
[`internal/vdso/vdso.go`](../internal/vdso/vdso.go),
[`internal/vdso/scope.go`](../internal/vdso/scope.go), and
[`internal/vdso/revoke.go`](../internal/vdso/revoke.go).

## Why RMA is the right analogy

MPI remote-memory access (RMA) names three useful ideas:

- a **window**: a region of exposed state addressed by handles, not by a request
  to the original producer;
- **one-sided operations**: an origin can put or get data without the producer
  participating in that operation;
- a **synchronization epoch**: a fence or flush boundary that says which view of
  the window a reader is allowed to observe.

fak has the same structural shape at the shared-result layer. It does not expose
raw process memory. It exposes `Ref` handles whose bytes live behind the active
`Resolver`, and the vDSO decides when a cached `Ref` is still coherent enough to
serve.

## The shipped window

`abi.Ref` is the address. It carries:

- `Kind`: inline bytes, content-addressed blob bytes, or a backend region handle;
- `Digest` / `Handle` / `Len`: the identity and backend locator;
- `Taint`: `TaintTrusted`, `TaintTainted`, or `TaintQuarantined`;
- `Scope`: `ScopeAgent`, `ScopeFleet`, or `ScopeTenant`.

`abi.Resolver` is the one-sided access surface:

```go
type Resolver interface {
    Resolve(ctx context.Context, r Ref) ([]byte, error)
    Put(ctx context.Context, b []byte) (Ref, error)
}
```

The default backend is the in-memory `blob` CAS. Small payloads ride inline on
the `Ref`; larger payloads are stored by digest. `storedrv.Router` can compose
hot and durable tiers behind the same `Resolver` interface. In both cases, the
handle shape stays `Ref`, so the vDSO and context-MMU can pass addresses instead
of copying payloads through every envelope.

## The tier-2 shared-read pool

The vDSO tier-2 cache is the shared-result pool over those refs:

1. A read-shaped, idempotent tool call misses the vDSO and reaches the engine.
2. On successful completion, `vdso.Emit` fills tier-2 with the result `Ref`,
   keyed by tool, canonical argument hash, and the current coherence epoch.
3. A later matching read builds the same key. If it is still coherent, `Lookup`
   returns a result whose payload is the stored `Ref` and whose metadata says
   `served_by=vdso`, `tier=2`.
4. The consumer materializes the bytes through `Resolver.Resolve`.

That is the one-sided part: a later consumer reads the already-addressed result
through the shared pool. The producer does not run again, and the consumer does
not need a callback to the original producer. It is still a kernel-mediated
serve: the read had to be eligible, the stored result keeps its taint and scope
metadata, principal isolation can force a miss instead of a cross-tenant share,
and downstream gates still enforce the scope boundary.

## RMA vocabulary map

| MPI RMA vocabulary | fak analogue | What is actually shipped |
|---|---|---|
| RMA window | active `RegionBackend` / `Resolver` plus the vDSO tier-2 map | A process-local shared-result pool addressed by `abi.Ref`, not raw memory |
| Exposed memory | bytes behind a `Ref` | Inline bytes, CAS blobs, or backend-issued `RefRegion` handles |
| Origin | the caller that warms or consumes the pool | A tool call mediated by the kernel, not an MPI rank |
| Target | the active resolver and cache entry | A registered backend, not a remote process' memory window |
| `MPI_Put` | `Resolver.Put(ctx, bytes)` and vDSO fill on completion | Store bytes, return a `Ref`, keep taint/scope on the handle |
| `MPI_Get` | `Resolver.Resolve(ctx, ref)` and tier-2 hit reuse | Materialize bytes from a `Ref`; the default backends return copies |
| Window scope | `abi.Ref.Scope` / `ShareScope` | `ScopeAgent` is private by default; `ScopeFleet` and `ScopeTenant` are explicit wider scopes |
| Window synchronization / fence | `vdso.bumpAndPublish` | Write-shaped completions bump root or scoped epochs and publish a `Mutation`; stale keys miss |
| Integrity invalidation | `vdso.Revoke(witness)` | Refuted witnesses evict matching entries, bump `TrustEpoch`, and refuse re-admission |
| `MPI_Accumulate` | gap | No shipped first-class `internal/region.Accumulate` yet |

## Coherence fence

The shared pool has two fences, one for consistency and one for integrity.

`bumpAndPublish` is the consistency fence. When the vDSO observes a write-shaped
completion, it bumps the relevant epoch: globally by default, or by namespace /
resource tag when finer invalidation is configured. Tier-2 keys include the epoch
stamp, so old keys become unreachable after the bump. The same locked step
publishes a `Mutation{Tool, Tags, WorldVer, Seq, Principal}` to subscribers.

`Revoke` is the integrity fence. If a later witness refutes an entry that was
already pooled, `Revoke(witness)` evicts every resident entry admitted under that
witness, records the witness as refuted, bumps `TrustEpoch`, and publishes a
revocation event. That is separate from `worldVer` because a refutation can happen
without any real-world write.

Together they preserve the vDSO invariant: a hit must be equivalent to a fresh
call. If either fence says the entry is stale or poisoned, the cache turns the
would-be hit into a miss.

## What fak adds beyond ordinary RMA vocabulary

fak's shared-result pool is not just a byte window. It adds policy metadata and
adjudication to the handle:

- **Adjudication:** fills originate from kernel-observed successful completions,
  and cache serves are available only to read-shaped, idempotent calls whose
  vDSO gates can prove they are safe to reuse. Write-shaped, resource-misnamed, or
  witness-refuted shapes miss instead of serving stale bytes.
- **Taint:** `Ref.Taint` rides with the payload. Sharing a result does not launder
  quarantined or tainted bytes.
- **ShareScope:** `Ref.Scope` names the maximum intended sharing boundary.
  `ScopeAgent` is the fail-closed zero value. Wider scopes are explicit.
- **Principal-aware sharing:** a named principal can be folded into the tier-2 key
  so private identity-dependent results do not cross tenants; a tool must be
  declared shareable before principal-scoped entries are shared across principals.
- **Witness revocation:** a cached result can be invalidated because its external
  witness was refuted, even if the underlying bytes and consistency epoch did not
  change.

Those additions are the fak-specific part. MPI RMA gives a vocabulary for an
exposed window and one-sided operations; fak layers trust, scope, and coherence
on the handle before a result becomes reusable context.

## What this does not claim

This analogy is deliberately narrow:

- It is **not RDMA**. No NIC, verbs provider, registered remote memory region, or
  MPI transport is involved.
- It is **not remote DMA**. A caller cannot write into another process' memory by
  address. It can only use the registered `Resolver` methods and vDSO serve path.
- It is **not a zero-copy claim for the shared-result pool**. The default `blob`
  backend returns copies, and `storedrv` materializes bytes through drivers. The
  separate `xenginekv` backend has an opt-in zero-copy arena, but that is not the
  default shared-result pool described here.
- It has **no hardware atomics**. There is no compare-and-swap, fetch-add, or
  hardware reduction against remote memory.
- It is **not a performance claim**. This page maps vocabulary and correctness
  boundaries only; it makes no throughput, bandwidth, latency, or scaling claim.

## The Accumulate gap

MPI RMA's `Accumulate` is the important missing verb for a true mutable shared
object: many origins can update one target using a defined operation, and the
window's synchronization rules define the result.

fak does not ship that as a first-class `internal/region` package today. The
current pool supports addressable result reuse: `Put`, `Resolve`, scoped sharing,
and coherence invalidation. It does not provide a named mutable cell with a base
revision, deterministic fold rule, conflict value, or `Accumulate` operation.

That gap is why the shared-state ladder still marks live shared objects as planned
first-class region/window work. Closing it means adding a typed region layer over
`Ref` + `ShareScope` with at least:

- a stable cell/window name independent of one producer;
- a base version or digest for each update;
- an adjudicated write capability;
- taint and scope propagation through the update;
- a deterministic `Accumulate` fold, or a typed conflict when no safe fold exists;
- coherence events that distinguish "the cache entry is stale" from "the shared
  object value advanced by this fold."

Until that lands, use this page's term precisely: fak has a shipped one-sided
shared-result pool. It does not yet have a general mutable RMA window.
