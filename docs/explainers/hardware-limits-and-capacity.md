---
title: "fak explainer: how fak interfaces with hardware capacity limits"
description: "fak's HAL lifts seven hardware-SHAPE assumptions; finite device CAPACITY is the eighth. The capacity-aware policy lives in the cachemeta plane; the physical allocators panic blind. This explains the two planes and the bridge between them."
---

# How fak interfaces with hardware limits — capacity is the eighth assumption

> **Status:** this is a *design* explainer, not a shipped feature. It names a real,
> cross-cutting gap and the seam that starts closing it (`compute.DeviceCapacity`, shipped
> with this note), and it credits the capacity mechanisms that already exist so the picture
> stays honest. Nothing here claims that OOM is solved, that a span spills under live
> pressure, or that a model that does not fit is refused on the live path yet. Those are
> tracked planks, listed at the end.

*Who this is for:* contributors reasoning about what fak does when it runs out of memory —
on a GPU, on the host, across a tier hierarchy. Prerequisites: the
[hardware-portability explainer](hardware-portability.md) (the `internal/compute` HAL seam)
and the [hardware-aware cache doc](../serving/hardware-aware-cache.md) (the `cachemeta`
placement plane). This explainer sits *between* them.

## 1. Shape versus limit

The [hardware-portability explainer](hardware-portability.md) makes a precise claim: the
`internal/compute` HAL neutralizes **seven** host-CPU assumptions — float32 monoculture,
host-pointer aliasing, x86 build-tag dispatch, synchronous return, goroutine-only
parallelism, row-major layout, and eager full-RAM residency. Read the list again and notice
what they have in common: every one describes the **shape** of the hardware — its dtype, its
address space, its ISA, its execution model, its memory layout, how it obtains weights. Lift
all seven and a new backend is a *registration*, not a fork.

None of the seven describes the one thing every accelerator actually runs out of: **finite,
exhaustible memory.** A GPU is not just a different *shape* of compute; it is a *smaller* pool
of it. The faithful-model ceiling the [hardware matrix](../HARDWARE-MATRIX.md) reports —
"`fak` faithful ≤ 7B on the 36 GB Mac" — is not a shape fact. It is a **capacity** fact. And
capacity is the assumption the HAL has not yet lifted.

This matters more for `fak` than for a typical serving engine, because `fak` owns the KV
cache as a kernel object rather than renting it from vLLM or SGLang. When you own the cache,
you own its residency — and residency is governed by capacity. So "how much fits" is not a
detail `fak` can delegate; it is load-bearing for the central design claim.

## 2. Two planes that do not touch

Search the tree for how `fak` reasons about running out of memory and you find **two
separate worlds**, each real, with no control path between them.

### Plane A — the policy plane (`internal/cachemeta`)

This plane is genuinely good, and it is *built and tested*. It names where a cached span can
live (`ResidencyTier`: HBM → DRAM → NUMA-far → CXL → Disk → Remote) and it carries the
physical character of each tier in a `TierProfile` — `CapacityBytes`, `BandwidthMBPerSec`,
`ReadLatencyNanos`, `ByteAddressable`, `Coherent`. Its `PlanPlacement` makes the one decision
a blind LRU cannot: under pressure, **demote** a hot prefix one tier colder rather than
**evict** it and pay a full re-prefill (`RetainCheaperThanRecompute` quantifies the
trade-off; `Lifecycle.DemoteOnExpiry` ages a span down per-tier). The
[hardware-aware cache doc](../serving/hardware-aware-cache.md) explains it in full, and
`cmd/hwcachedemo` runs it: a 4000-token prefix relocating `demote → demote → demote → spill →
evict` under escalating pressure, with `0` tokens re-prefilled versus `28000` for blind LRU.

The honest boundary that doc states about itself is the whole point of *this* one:

> the policy/metadata plane … decides where a cached span should live and when it should
> move … The physical byte movement … is performed by **the engine adapter that consumes
> those directives** — this plane touches no bytes.

So the plane plans. Something else is supposed to act. Read on for what that something is.

### Plane B — the physical plane (the `compute` HAL allocators)

This is where bytes are actually allocated and actually run out: `cudaBackend.dalloc`,
`metalBackend.dalloc`, `vulkanBackend.dalloc`. Here is what they do when the device is full:

- **CUDA** (`cuda.go: dalloc`) — `panic("compute: cuda device allocation failed")`. No
  capacity check before the alloc, no fallback, no typed error.
- **Metal** (`metal.go: dalloc`) — `panic("compute: metal device allocation failed")`. Same.
- **Vulkan** (`vulkan.go: dalloc`) — the one that tries: it falls back to a host-visible
  allocation, and only panics if *both* device-local and host-visible are exhausted.

And the capacity *information* the physical plane has, it throws away. `fcuda_init` probes
`p.totalGlobalMem` and returns it; `cuda.go: init` read it into a local and **discarded it**
(this note's change keeps it). `Caps.DeviceMemory` is a boolean — it means "resident tensors
are not host-addressable," a *shape* fact — not "the device has N bytes." Nothing in the
`Backend` interface lets a caller ask "will this fit?"

### The gap: the planes meet only at the meter, never at the control

```
   POLICY PLANE (cachemeta)                 PHYSICAL PLANE (compute HAL)
   ┌───────────────────────────┐           ┌───────────────────────────┐
   │ PlanPlacement             │           │ cudaBackend.dalloc  panic │
   │  demote / spill / evict   │           │ metalBackend.dalloc panic │
   │ TierProfile.CapacityBytes │           │ vulkanBackend.dalloc spill│
   │ TierPressure (fullness)   │           │ totalGlobalMem  discarded │
   │ Lifecycle.DemoteOnExpiry  │           │ Caps: no capacity field   │
   └───────────┬───────────────┘           └─────────────┬─────────────┘
               │                                          │
        plans against                              allocates against
   DefaultTierProfiles (placeholder              the real device, blindly,
   GB) + SYNTHETIC pressure                      and PANICS when it's full
   (cmd/hwcachedemo feeds it)
               │                                          │
               └──────────────► (no control path) ◄───────┘
                         meet only here:
               internal/engine CacheEventRecorder  (metrics)
```

The policy plane plans demote-not-evict beautifully — against `DefaultTierProfiles`, whose
own comment calls them "representative order-of-magnitude defaults, not measurements of any
particular box," and against `TierPressure` values that, on the live serving path, **nothing
computes from real device state.** `cmd/hwcachedemo` injects escalating pressure by hand; the
live model engine never does. Meanwhile the physical allocators that *do* know the real
pressure — they hold `totalGlobalMem`, they are the ones that hit `nullptr` — never report it
into the policy plane and never consume a demote directive out of it. The two planes are
connected only through the *observability* layer: `PlanPlacement`'s `KVOffload`/`KVRestore`
directives flow into `CacheEventRecorder` as `fak_engine_cache_*` metrics. That makes a
placement decision *legible*. It does not make it *happen*.

## 3. The eighth assumption

Stated in the form the portability explainer uses for the first seven:

| # | Assumption | Where it lives today | What it shuts out |
|---|---|---|---|
| 8 | **uniform / infinite capacity** — a backend can hold whatever it is handed; "out of memory" is a process abort, not a value | `cuda.go`/`metal.go` `dalloc` panic; `Caps.DeviceMemory` is a shape bool; `cuda.go: init` discards `totalGlobalMem` | any device too small for the model — i.e. every real serving target above its faithful ceiling |

Lifting it follows the same discipline as the other seven: **neutralize it in the type
system first**, even though only one backend can satisfy it today. That is what
`compute.DeviceCapacity` does (§4). The portability doc's own framing applies verbatim — the
contract "already assumes none of [the assumptions], even though only the CPU reference is
implemented today." Capacity is now one more capability a backend *may* report, discovered the
same way `CollectiveBackend` is (a `Caps` bool plus a type-assert), absent by default, with the
core path failing **open** when it is absent so the portable floor is never blocked.

## 4. The bridge, plank by plank

The fix is not one change; it is a bridge with several planks. Only the first ships with this
note. The rest are named here so the deferral is tracked, not forgotten.

**Plank 1 — backends REPORT their ceiling (`compute.DeviceCapacity`). _Shipped._**
A backend that can probe its own memory implements `DeviceMemory() (total, free int64, known
bool)`. `DeviceMemoryInfo(b)` and `FitsOnDevice(b, wantBytes, headroom)` let any caller ask
without knowing the concrete backend, and they **fail open**: a backend that cannot answer
(the pure-Go `cpu-ref` floor, a wasm target) reports `known=false`, and the fit check returns
`FitUnknown` ("proceed"), never `FitTooBig`. The `cuda` backend is the first producer — it now
*keeps* `totalGlobalMem` and reports it (free is `FreeUnknown` until a `cudaMemGetInfo` query
is wired, so `FitsOnDevice` checks against the total ceiling, which already catches a model
that cannot fit the whole device). This is the report half of the bridge: the wire by which a
real backend's capacity can one day feed `cachemeta.TierPressure` instead of placeholder
profiles.

**Plank 2 — OOM becomes a typed value, not a panic.** `dalloc` should return a typed
allocation fault — the exact shape `abi.KVResidencyFault` already models for the off-box KV
direction ("a failed restore is NEVER a silent recompute … typed, never a hang"). A caller
that gets a fault can spill, evict, or refuse with a sizing message. Today it gets a
stack trace.

**Plank 3 — feed real pressure into the policy plane.** Replace the `DefaultTierProfiles`
placeholders and the demo's hand-injected `TierPressure` with live numbers from
`DeviceMemoryInfo`, so `PlanPlacement` plans against the device that actually exists.

**Plank 4 — the engine adapter that EXECUTES a placement directive.** The
[hardware-aware cache doc](../serving/hardware-aware-cache.md) already names this missing
piece ("the engine adapter that consumes those directives"). It is the control path the
diagram in §2 is missing: a `PlanPlacement` `demote`/`spill` decision turned into a real
`KVStore.Evict` plus a stage to the colder tier, against the kernel-owned cache. The
mechanical primitives exist — `kvmmu.Context.ApplyPlan` already does bit-exact eviction to
match a plan, and `model/paging.go: pagedKernel` already pages a weight in on demand — but
neither is driven by live capacity pressure on the serving loop.

**Plank 5 — a load-time fit pre-check.** `FitsOnDevice` before the `make`/`append` in the
model loaders (`model: Load`, `LoadSafetensors`, `ggufload.LoadModel`) turns "OOM panic
mid-load" into "this needs ~W GB, the device has ~A GB" *before* a byte is allocated. Today
the Go load paths allocate optimistically; the only fit gate that exists, `tools/memgate.py`,
is an operator script outside the Go process.

**Plank 6 — the capacity ESCAPE when one device is not enough.** When the model does not fit
*any* single device, capacity is solved by spreading it: multi-GPU tensor/expert parallelism
and CPU/NVMe offload. That track is real and separately planned in
[the native-753B staged plan](../notes/native-753b-track-staged-plan.md) (Pillars 3 and 4),
with its own honesty ledger: the collective seams exist (`CollectiveBackend`,
`LocalCollective`, `DistComm`) but a real cross-**device** NCCL collective does not yet, and
today's "CPU-offload" is compute-placement, not tensor paging. This explainer is the
single-device capacity story; that plan is the multi-device one.

## 5. What already exists (so the gap is stated honestly)

Capacity is not *un*handled in `fak` — it is handled in scattered, isolated places that do not
add up to a HAL contract. Crediting them is what keeps the gap claim honest:

| Mechanism | What it does | Why it is not the bridge |
|---|---|---|
| `vulkan.go: dallocWeight` (`FAK_GPU_BUDGET_MB`) | spills the cold weight tail host-visible when a device-local budget is exceeded | Vulkan-only; weights-only; env-set, not a device query; not in the HAL contract |
| `cachemeta.PlanPlacement` + `TierPressure` | full demote-not-evict placement policy | wired into `cmd/hwcachedemo`/`cmd/cxlpooldemo` + tests, **not** the live model/serving allocation path |
| `residency.Manager` + `polymodel.Pool` (`ErrTooLarge`) | a backend-agnostic resident-weight-byte budget with LRU page-out | policy-only; caller-supplied budget, not VRAM-derived; not wired to CUDA/Metal `Upload` |
| `model/paging.go: pagedKernel` | upload → compute → free, an on-demand page-in primitive | standalone; bit-equal to resident, but not integrated into the live weight HAL |
| `tools/memgate.py` | an operator pre-flight RAM gate | a Python script outside the Go load path |
| `glm52_serve_preflight.py: required_vram_gb` | VRAM-vs-quant fit + per-rank TP shard sizing | models the fit of **external** engines (SGLang/vLLM), not `fak`'s own backends |

Every one is real. None lets the forward loop, or the layer above it, ask the backend it is
running on whether the next allocation will fit — and act on the answer. That is the bridge,
and it is what `DeviceCapacity` begins.

## 6. Honest boundaries

- This note ships **Plank 1 only** — the report capability and the fit helper, lifted into
  the type system, with `cuda` as the first producer. It enforces nothing on the live path.
- `FitsOnDevice` is a *pre-check*, not an allocator. Nothing yet calls it before a load or an
  upload; wiring it in is Plank 5.
- The `cuda` producer reports `total` only; `free` is `FreeUnknown` until `cudaMemGetInfo` is
  wired, so the check catches "too big for the whole device," not "too big for the current
  free headroom."
- The capacity numbers in `cachemeta`'s `DefaultTierProfiles` remain representative defaults,
  not measurements. Plank 3 is what would make them real.
- The industry scorecard ([`docs/industry-scorecard/memory.md`](../industry-scorecard/memory.md))
  honestly lists `fak`'s capacity gaps (paged KV, KV-offload hierarchy, memory-utilization %,
  fleet KV-aware routing) as out-of-scope / no-claim today. This explainer harmonizes with
  that: it does not turn any of those into a claim — it draws the map and lays the first plank.

## See also

- [Hardware portability — the `internal/compute` HAL seam](hardware-portability.md) — the
  seven *shape* assumptions; this explainer adds the eighth (capacity).
- [Hardware-aware cache — tiers, demote-not-evict](../serving/hardware-aware-cache.md) — the
  policy plane (Plane A) in full.
- [Native 753B serving — staged plan](../notes/native-753b-track-staged-plan.md) — the
  multi-device capacity escape (Plank 6).
- [Hardware matrix](../HARDWARE-MATRIX.md) — the faithful-model ceilings that *are* capacity
  facts.
