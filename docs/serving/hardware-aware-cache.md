---
title: "Hardware-Aware KV Cache: Tiers, Zero-Copy, Demote-Not-Evict"
description: "fak's cachemeta plane plans where a KV span lives across HBM, DRAM, NUMA-far, CXL, disk, and remote tiers, with per-tier TTL and demote-not-evict placement."
---

# Hardware-aware cache: tiers, zero-copy, per-tier TTL, demote-not-evict

> Status: the **policy/metadata plane** is shipped and tested (`internal/cachemeta`,
> `cmd/hwcachedemo`). It decides *where a cached span should live and when it should
> move*, and emits the existing `KVTransfer` directives. The *physical* byte movement
> (a CXL HDM map, an RDMA transfer, a disk spill) is performed by the engine adapter
> that consumes those directives — this plane touches no bytes. Numbers in the demo are
> a deterministic cost model over representative tier profiles, not a hardware
> measurement; an operator overrides the profiles with values measured for their box.

## Why

A KV cache that is co-optimized with the hardware from day one needs to know more than
"is this entry hot." It needs to know the *physical character* of every place a span can
live, and it needs to make the one decision a blind LRU cache cannot: under memory
pressure, **relocate** a hot prefix one tier colder rather than **drop** it and pay a
full re-prefill later.

`internal/cachemeta` already named *where* a payload lived (`ResidencyTier`) and *how* a
KV span moved between tiers (`KVTransfer`: offload / restore / route / migrate). What it
could not express is the part a modern memory hierarchy turns on — the latency,
bandwidth, capacity, byte-addressability, and zero-copy shareability of each tier, and a
freshness/lifecycle policy fine-grained enough to act per tier. This layer adds exactly
that, additively, on the tier-1 foundation plane (no payloads, no new dependencies).

## The three pieces

### 1. The tier model — `hardware.go`

Two first-class tiers join the ladder between local DRAM and disk:

- **NUMA-far** — byte-addressable, cache-coherent DRAM on another socket: same
  load/store semantics as local DRAM, a NUMA hop's worth of extra latency.
- **CXL** — CXL-attached memory (CXL.mem / a Type-3 expander or a fabric pool):
  byte-addressable and coherent like DRAM, a few times the latency and a fraction of
  the bandwidth, in exchange for very large, poolable, **shareable** capacity. This is
  the tier that makes "relocate, don't evict" pay: a span demoted here stays attendable
  in place — never recomputed — and, on a coherent CXL fabric, can be shared zero-copy
  across hosts.

The full ladder, hottest to coldest:

```
HBM → DRAM → NUMA-far → CXL → Disk → Remote
```

Each tier carries a `TierProfile` — `ReadLatencyNanos`, `BandwidthMBPerSec`,
`CapacityBytes`, `ByteAddressable`, `Coherent`, `Persistent`, and the native zero-copy
`Share` kind. `AttendableInPlace()` (byte-addressable **and** coherent) is the property
that makes a demote cheap: HBM/DRAM/NUMA-far/CXL are attendable in place, disk/remote
are not (a read must stage them back first).

**Zero-copy sharing** is first-class metadata. A `ShareDescriptor` on `Residency` says
how a resident payload can be handed to another consumer without a memcpy:

| `ShareKind` | meaning |
|---|---|
| `ShareCopy` (zero value) | must be copied — the fail-safe default |
| `ShareMmap` | shared mapping, zero-copy across processes on one host |
| `ShareCXLHDM` | coherent CXL region, zero-copy load/store across sockets/hosts |
| `ShareRDMA` | RDMA-registered region, zero-copy over the wire by the NIC |
| `ShareDmabuf` | exported GPU dma-buf, zero-copy GPU↔GPU / GPU↔NIC |

The zero value is `ShareCopy`, so an entry that has not *declared* a zero-copy
capability is never aliased by accident.

### 2. Per-tier TTL + a multi-state lifecycle — `lifecycle.go`

A single global TTL answers only "is this stale yet." A tiered cache needs freshness
expressible **per tier** — a span is cheap to keep in CXL for a long time and expensive
to keep in scarce HBM for even a short one — and explicit states so a policy can act on
"expired in HBM, demote it" without conflating it with "expired everywhere, drop it."

- `TierTTL` is a per-tier budget in millis: e.g. `{HBM: 2s, DRAM: 60s, CXL: 0 (forever)}`
  keeps the scarce hot tier turning over fast while letting a span rest indefinitely in
  cheap far memory. Each tier's clock is measured from when the entry *entered that tier*,
  so a demote resets the freshness window for the colder tier — TTL "down to the lowest
  levels."
- `EntryState` is the flexible state set: `filling → resident → expiring →
  expired/spilled → evicted` (a relocation is atomic here — the engine adapter owns any
  in-flight tracking). An access during the *expiring* grace window
  **revives** the entry to resident — a hot span proves itself and stays.
- `Advance(policy, nowMillis)` is the time-driven transition: pure, deterministic, and
  **wall-clock-free** (the caller injects `nowMillis`, the same testable posture the
  benchmark harness takes with an injected `--now`), so a workload replays identically.

`Advance` decides *when* a span stops being fresh; placement decides *where* it then
goes. Keeping the two separate is what lets the time policy be tested without a hardware
profile and the placement policy without a clock.

### 3. The hardware-cost-driven placement policy — `placement.go`

`PlanPlacement` is the decision a blind LRU cache cannot make. Given the entry's
lifecycle, its size and token count, the tier profiles, and the live per-tier pressure,
it returns one of: **keep / promote / demote / spill / evict**, plus the `KVTransfer`
directive for the move.

The core move is **demote-instead-of-evict**:

- Find the nearest colder profiled tier with room.
- If it is attendable in place (NUMA-far / CXL) and **retaining there beats recompute**,
  `demote` to it (a one-time `bytes / bandwidth` stage instead of a `tokens × per-token
  prefill` rebuild).
- If only a non-attendable tier (disk) has room and retaining still beats recompute,
  `spill`.
- Otherwise — nothing colder has room, or the span is so cheap to rebuild that holding
  it in scarce memory is not worth it — `evict` and recompute on demand.

The quantified comparison is `RetainCheaperThanRecompute`: stage cost into the colder
tier vs the recompute cost it avoids. With realistic memory bandwidth, staging almost
always wins for a large, expensive prefix — *which is exactly why demote-not-evict is the
right default*. The exception the cost model still gets right: a small span whose only
colder home is slow (disk) is cheaper to rebuild than to read back, so it is evicted.

## What you can run

```
go run ./cmd/hwcachedemo
```

A no-model, no-GPU, deterministic proof that prints the tier ladder with each tier's
character, which tiers share zero-copy, a hot 4000-token prefix relocating one tier at a
time under escalating pressure (`demote → demote → demote → spill → evict`), the
cheap-span exception, and the head-to-head tally:

```
== Blind LRU vs hardware-aware tiering (8 turns sharing a 4000-token prefix) ==
  blind LRU:      re-prefilled 28000 tokens (evict+recompute every reuse)
  tiered (fak):   re-prefilled 0 tokens (demote to dram, stage back)
  -> 28000 prefill tokens saved by demoting instead of evicting
```

## How it connects to the rest of fak

- The directives `PlanPlacement` emits (`KVOffload` / `KVRestore`) are the **same**
  vocabulary `internal/engine`'s `CacheEventRecorder` already normalizes into the one
  cache-entry stream and exposes as `fak_engine_cache_*` Prometheus metrics — so a
  placement decision is observable end-to-end, and a failed restore is a typed fault,
  never a silent recompute.
- `internal/radixkv` (the RadixAttention prefix cache) evicts the LRU leaf today; this
  layer is the policy that turns that eviction into a *demotion* when a colder tier can
  still hold the prefix attendably — the bridge from "drop the coldest" to "relocate the
  coldest."
- It is the placement counterpart of `materialization.go`'s correctness gate: that file
  decides *whether* a materialized KV span may be reused (model/tokenizer/position must
  match); this file decides *where it should live* and *when it should move*.

## Honest boundaries

- cachemeta is **tier-1 foundation** and **owns no cache** — it names objects and plans
  moves; it never touches a KV tensor. The physical CXL map / RDMA transfer / disk spill
  is the engine adapter's job. This plane makes the decision *legible and testable*; it
  does not itself perform DMA.
- The default tier profiles are **representative order-of-magnitude** values, not a
  measurement of any particular machine. The placement math is identical against measured
  profiles; an operator supplies real numbers (the same posture
  `experiments/benchmark/catalog.json` takes for the machine table).
