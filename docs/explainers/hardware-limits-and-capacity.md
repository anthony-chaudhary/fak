---
title: "fak explainer: how fak interfaces with hardware capacity limits"
description: "fak's HAL lifts seven hardware-SHAPE assumptions; finite device CAPACITY is the eighth. The capacity-aware policy lives in the cachemeta plane; the physical allocators panic blind. This explains the two planes and the bridge between them."
---

# How fak interfaces with hardware limits — capacity is the eighth assumption

> **Status:** this is an incremental bridge, not a finished OOM subsystem. `compute`
> now exposes a classed capacity fit contract (`DeviceCapacity`, optional `HostCapacity`,
> `MemoryPlan`), the
> policy plane has a real device-pressure seam, the engine has a demote/spill executor,
> and `fak serve --gguf --backend` refuses a known-too-large GGUF before the device load
> allocates. For quantized-upload backends that fit check is mixed precision: lean-Q8
> resident weights plus f32 HAL KV, activation, and scratch estimates; f32 weights are now
> only the fallback for backends that cannot consume quantized uploads yet.
> `--cpu-offload-experts` uses a split plan: dense/router/attention weights plus KV and HAL
> transients consume device capacity while expert weights are host-scoped offload bytes.
> Host-scoped offload/DDR bytes can refuse when a native GPU backend explicitly advertises
> `HostCapacityProbe` from the OS host-memory probe; otherwise they stay visible and fail open.
> On a successful eager GGUF device load, the same classed admission profile is visible on
> `/metrics` (`fak_model_load_memory_plan_bytes`,
> `fak_model_load_memory_plan_dtype_bytes`, `fak_model_load_memory_capacity_*`,
> `fak_model_load_memory_fit_bytes`) and
> `/debug/vars` (detailed rows with dtype/storage labels plus the capacity snapshot).
> Runtime device allocation failures now recover as a typed,
> classed in-kernel OOM on the served path, and served backend requests now run a
> prompt+`max_tokens` KV/HAL-transient capacity precheck before device decode when capacity
> is known. Both failure paths publish classed counters on `/metrics` plus last-site drilldown on
> `/debug/vars`, and the last served backend request's successful admission plan is visible
> as `fak_gateway_in_kernel_request_memory_*` gauges, including per-scope
> want/budget/margin rows, plus `/debug/vars.request_memory`.
> Nothing here claims that allocator OOM can
> already spill/retry, that the serving loop demotes KV under live pressure, or that
> every model loader has a complete memory plan yet. Those remain tracked
> planks below.

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

- **CUDA** (`cuda.go: dalloc`) — it now falls back from `cudaMalloc` to
  `cudaMallocManaged`, and `FAK_GPU_BUDGET_MB` deliberately sends over-budget explicit
  weight uploads to managed memory before a device-local OOM.
- **Metal** (`metal.go: dalloc`) — raises a typed `DeviceAllocError`, but still has
  no capacity check before the alloc and no fallback.
- **Vulkan** (`vulkan.go: dalloc`) — it falls back to a host-visible allocation, and only
  raises a typed `DeviceAllocError` if *both* device-local and host-visible are exhausted.

Historically the capacity *information* the physical plane had was not part of the HAL
contract: CUDA's `totalGlobalMem` was read and discarded, and Vulkan's device-local heap
size stayed behind the shim. `compute.DeviceCapacity` is the typed way out: `Caps.DeviceMemory`
keeps its old shape meaning ("resident tensors are not host-addressable"), while
`Caps.CapacityProbe` plus `DeviceMemory()` says "this backend can report N bytes."

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
particular box," and against `TierPressure` values that used to be synthetic only:
`cmd/hwcachedemo` injected escalating pressure by hand. Plank 3 now derives HBM pressure from
the real backend and `RunCapacityPressureSweep` binds that signal to the Plank-4 adapter for
caller-supplied candidates, but the live serving loop still does not invoke that sweep
automatically. Meanwhile the physical allocators that *do* know the real pressure — they hold
`totalGlobalMem`, they are the ones that hit `nullptr` — still do not spill/retry from an OOM
site by themselves. The reusable loop exists; automatic live admission/retry policy is the
remaining integration.

## 3. The eighth assumption

Stated in the form the portability explainer uses for the first seven:

| # | Assumption | Where it lives today | What it shuts out |
|---|---|---|---|
| 8 | **uniform / infinite capacity** — a backend can hold whatever it is handed; "out of memory" is a process abort, not a value | legacy raw `dalloc` panics; `Caps.DeviceMemory` is a shape bool, not a capacity report; unprobed backends still fail open | any device too small for the model — i.e. every real serving target above its faithful ceiling |

Lifting it follows the same discipline as the other seven: **neutralize it in the type
system first**, even though only some backends can satisfy it today. That is what
`compute.DeviceCapacity` does (§4). The portability doc's own framing applies verbatim — the
contract "already assumes none of [the assumptions], even though only the CPU reference is
implemented today." Capacity is now one more capability a backend *may* report, discovered the
same way `CollectiveBackend` is (a `Caps` bool plus a type-assert), absent by default, with the
core path failing **open** when it is absent so the portable floor is never blocked.

## 4. The bridge, plank by plank

The fix is not one change; it is a bridge with several planks. Several planks now ship, and
the remaining ones are named here so the deferral is tracked, not forgotten.

**Plank 1 — backends REPORT their ceiling (`compute.DeviceCapacity`). _Shipped._**
A backend that can probe its own memory implements `DeviceMemory() (total, free int64, known
bool)`. `DeviceMemoryInfo(b)` and `FitsOnDevice(b, wantBytes, headroom)` let any caller ask
without knowing the concrete backend, and they **fail open**: a backend that cannot answer
(the pure-Go `cpu-ref` floor, a wasm target) reports `known=false`, and the fit check returns
`FitUnknown` ("proceed"), never `FitTooBig`. CUDA now *keeps* `totalGlobalMem` and reports
it, and the Windows Vulkan backend reports the sum of its device-local heaps; both leave
free memory as `FreeUnknown` until a backend-specific free-memory query is wired, so
`FitsOnDevice` checks against the total ceiling and catches a model that cannot fit the
whole device. This is the report half of the bridge: the wire by which a real backend's
capacity can feed `cachemeta.TierPressure` instead of placeholder profiles. The same
contract now has a classed form: `MemoryPlan` / `MemoryDemand` keeps
weights, KV cache, DDR cache, offload staging, activations, and scratchpad bytes distinct,
while `FitsMemoryPlan` / `RefuseMemoryPlanIfTooBig` still fail open on unknown capacity and
carry the class breakdown into the typed `FitError`. Device-scoped demands are checked
against `DeviceCapacity`; host-scoped demands are checked against optional `HostCapacity`
only when `Caps.HostCapacityProbe` is set, otherwise those host bytes remain visible but do
not refuse. The native GPU backends now advertise that host side when the stdlib OS probe
succeeds: Windows uses `GlobalMemoryStatusEx`, Linux uses `sysinfo`, and Darwin reports the
physical-memory ceiling via `hw.memsize` with `free=FreeUnknown`.

**Plank 2 — OOM becomes a typed value, not a panic. _Partially shipped._** `dalloc` should
return a typed allocation fault — the exact shape `abi.KVResidencyFault` already models for
the off-box KV direction ("a failed restore is NEVER a silent recompute … typed, never a
hang"). The shipped slice is the in-kernel served path: `compute.DeviceAllocError` carries
the failed byte count, allocator site, and memory class (`weights`, `kv_cache`, `offload`,
`scratchpad`, `activation`, or `unknown`); CUDA, Metal, and Vulkan allocation choke points
raise it instead of a raw string panic where the backend allocator returns nil;
`agent.InKernelPlanner` recovers only that typed allocation fault; and the gateway maps it
to a 503 `in_kernel_oom` with a client-safe, class-visible remedy. The same served planner
also refuses a known-too-large request before allocation when the prompt+`max_tokens` KV
window plus HAL activation/scratch/logit transients exceed the reported device budget; it
also includes resident weights when the backend reports total capacity but not current free
memory. That refusal returns `agent.InKernelCapacityError` with site
`capacity-precheck`. The visibility slice is also live: `/metrics` publishes
`fak_gateway_in_kernel_oom_total`,
`fak_gateway_in_kernel_oom_failed_bytes_total`, and
`fak_gateway_in_kernel_oom_last_failed_bytes` by memory class for both allocation OOMs and
capacity precheck refusals, while `/debug/vars` includes the last site for drilldown without
turning dynamic sites into Prometheus labels.
Honest fences: allocation OOM recovery is still panic/recover below the HAL, unknown
capacity still fails open, generic allocation sites can still be `unknown`, resource-cap
validation panics are not OOM, and no caller yet spills, demotes, or retries based on the
class.

**Plank 3 — feed real pressure into the policy plane. _Shipped_ (#707).**
`internal/engine.PlanPlacementForDevice` is the report→policy wire: it derives a live
`cachemeta.TierPressure` (and the HBM tier's real `CapacityBytes`) from the active backend's
device memory via `compute.DeviceMemoryInfo(b)`, folds them into the request, and calls
`PlanPlacement` — so the policy plans against the device that actually exists instead of
`DefaultTierProfiles`' representative numbers and the demo's hand-injected `TierPressure`. It
**fails open**: a backend that cannot probe (the `cpu-ref` floor, a wasm target;
`known=false`) contributes no pressure and no capacity override, so the profile default
stands and no path that works today regresses. While the CUDA/Vulkan producers report `total`
but not `free` (the `cudaMemGetInfo` / backend free-memory follow-ups), HBM pressure derives
from `total` vs fak's tracked-resident bytes. `engine.RunCapacityPressureSweep` now binds
that report→policy wire to the Plank-4 adapter for a bounded list of KV candidates: it scales
observed pressure to an operator high-water mark, plans each span, executes demote/spill/evict
through `CapacityAdapter`, and stops once estimated HBM pressure falls below target or the
candidate set is exhausted. Witness: `go test ./internal/engine -run "Test(CapacityPressureSweep|PlanPlacementForDevice)"`
(a high-water device flips a hot prefix
from `keep` → `demote`, stages it to DRAM, and records the move as `ddr_cache`; `cpu-ref` is
unchanged). What remains is the live serving loop supplying real pressured KV candidates and
calling the sweep at the right admission/pressure points.

**Plank 4 — the engine adapter that EXECUTES a placement directive. _Shipped_ (#708).**
`internal/engine.CapacityAdapter.Execute` is the control path the diagram in §2 was
missing: it turns a `PlanPlacement` `demote`/`spill` decision into a real
`abi.KVBackend.StageSpan` (stage to the colder tier, addressed by digest) PLUS an
`abi.KVBackend.Evict` (the re-RoPE/renumber eviction from the live KV tier) against the
kernel-owned cache. It is fail-safe — the live copy is staged before it is dropped, so a
typed staging MISS/FAULT retains the span rather than losing it — and it records every
transition through the same `CacheEventRecorder` as a typed offload event, so a staging
fault is never a silent recompute. Witness: `go test ./internal/engine -run
TestCapacityAdapter`. The mechanical primitives it leans on were already in place —
`kvmmu.Context.ApplyPlan` does bit-exact eviction to match a plan, and
`model/paging.go: pagedKernel` pages a weight in on demand — but neither was driven by
live capacity pressure. This adapter is the demote/spill executor that consumes a
placement directive, and the reusable pressure sweep now performs the Plank-3 → Plank-4 bind
for caller-supplied candidates. What remains is wiring the served decode path to invoke that
sweep automatically.

**Plank 5 — load-time and request-time fit pre-checks. _Partially shipped._** `FitsOnDevice` before the
`make`/`append` in the model loaders (`model: Load`, `LoadSafetensors`, `ggufload.LoadModel`)
turns "OOM panic mid-load" into "this needs ~W GB, the device has ~A GB" *before* a byte is
allocated. The shipped slice is the GGUF path: `ggufload.WeightSource` can estimate the raw
payload plan (`EstimateLoadMemoryPlan`) and the f32-resident plan used by `LoadModel`
(`EstimateF32LoadMemoryPlan`) from the header, and `FitOnDevice` / `FitF32OnDevice` route
that plan through the classed `compute` refusal. The HAL also exposes
`EstimateKVStoreMemoryPlan`, which counts the current device KV layout (`Kraw`, `K`, and `V`
as f32 rows) from `KVConfig` plus a planned token window, and
`EstimateHALTransientMemoryPlan`, which adds a conservative per-token HAL activation and
scratchpad estimate from the same model geometry. `fak serve --gguf --backend` now resolves
the backend before eager model load and, for the non-offload device path, refuses a
known-too-large classed plan. If the backend advertises quantized upload, the plan uses the
lean/raw Q8 proxy plus f32 KV, activation, and scratch estimates from the GGUF
`context_length` (or the smaller `--context-budget-tokens` window); a backend without
quantized upload falls back to f32-resident weights with the same f32 runtime demands.
`--cpu-offload-experts` uses the same header-only path but partitions the GGUF tensor
directory through the runtime offload predicate: dense/router/attention weights stay
device-scoped, routed/shared experts become host-scoped `offload` demands, and KV plus HAL
transients stay device-scoped. If a backend also advertises `HostCapacityProbe`, the same
classed pre-check can reject an expert-offload plan that exceeds reported host capacity; if
the OS probe is unavailable the host-scoped bytes still fail open. The remaining 15% device
headroom is still reserved for allocator fragmentation, backend pools, and unmodeled runtime
uploads. After a successful eager device load, the gateway publishes the exact admission
profile it used: `/metrics` aggregates `fak_model_load_memory_plan_bytes` by class+scope,
adds `fak_model_load_memory_plan_dtype_bytes` by class+scope+dtype so mixed-precision
plans are explicit,
exports device/host capacity known/free-known/byte gauges, records the headroom ratio, and
adds `fak_model_load_memory_fit_bytes` want/budget/margin rows by scope so the
headroom-adjusted fit is visible without operator arithmetic; `/debug/vars` keeps the detailed
memory-plan rows, dtype labels, capacity snapshot, and `memory_fit` rows for drilldown.
The served in-kernel backend path now reuses the same classed fit contract per request:
after tokenizing the transcript and applying the request's `max_tokens`, it checks the
planned KV window plus HAL activation/scratchpad demand before entering `Prefill`/`Step`.
When the backend reports current free bytes, resident weights are not counted again because
that free value already reflects the loaded model; when only total capacity is known
(`FreeUnknown`) or capacity is unknown, the check includes the resident-weight estimate or
fails open respectively.
The gateway also exposes the last observed backend request plan as
`fak_gateway_in_kernel_request_memory_plan_bytes` by backend+class+scope+dtype, token-window
gauges, capacity-known/free-known gauges, `fak_gateway_in_kernel_request_memory_fit_bytes`
want/budget/margin rows by backend+scope, and `/debug/vars.request_memory`, so ordinary
successful requests are inspectable before a refusal occurs.
Honest fences: unknown capacity still fails open, host memory is physical/OS memory rather
than cgroup/container quota, safetensors and generic `model.Load` are not wired yet, runtime
upload pre-checks are still allocator-side, and the telemetry is not spill/retry.

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
| `vulkan.go: dallocWeight` / `cuda.go: dallocWeight` (`FAK_GPU_BUDGET_MB`) | spills the cold weight tail when a device-local budget is exceeded: host-visible on Vulkan, managed memory on CUDA | weights-only; env-set, not a device query; not in the HAL contract |
| `cachemeta.PlanPlacement` + `TierPressure` | full demote-not-evict placement policy | wired into `cmd/hwcachedemo`/`cmd/cxlpooldemo` + tests, **not** the live model/serving allocation path |
| `residency.Manager` + `polymodel.Pool` (`ErrTooLarge`) | a backend-agnostic resident-weight-byte budget with LRU page-out | policy-only; caller-supplied budget, not VRAM-derived; not wired to CUDA/Metal `Upload` |
| `model/paging.go: pagedKernel` | upload → compute → free, an on-demand page-in primitive | standalone; bit-equal to resident, but not integrated into the live weight HAL |
| `engine.CacheEventMetrics` memory-class projection | KV residency events expose `memory_class` alongside `to_tier`: HBM is `kv_cache`, DRAM/NUMA-far/CXL are `ddr_cache`, and Disk/Remote/Provider are `offload`; byte/token breakdowns are exported as `fak_engine_cache_*_breakdown_total` series | visibility only; it does not yet drive spill/retry policy or allocate a DRAM cache |
| `engine.RunCapacityPressureSweep` + `CapacityAdapter` | bounded report→policy→execute loop for caller-supplied KV candidates: derives HBM pressure from the backend, treats an operator high-water mark as "full", plans demote/spill/evict with `cachemeta.PlanPlacement`, executes via `StageSpan` then `Evict`, and records DRAM/CXL moves as `ddr_cache` and disk/remote/provider moves as `offload` | reusable engine primitive, not yet called by the live serving loop; candidate ordering, resident-byte accounting, restore, and retry policy still belong to the serving/cache owner |
| `compute.DeviceAllocError` / `agent.InKernelCapacityError` → gateway `in_kernel_oom` + `/metrics` | runtime device allocation failure becomes a typed served 503 with bytes + memory class; served backend requests also refuse a known-too-large prompt+`max_tokens` KV/HAL plan before allocation with site `capacity-precheck`; both are counted by class on `/metrics` and drillable by last site on `/debug/vars`; the last served backend request plan is separately visible as `fak_gateway_in_kernel_request_memory_*`, including per-scope want/budget/margin fit rows; device weight, KV-cache, offload, scratchpad, HAL activation upload, GLM-DSA backend activation, and paged-weight page-in sites carry dedicated classes where the backend knows the purpose | classification/admission and visibility only; no spill/retry policy yet, ambiguous generic alloc sites stay `unknown`, the last-request gauges are not cumulative histograms, and the request precheck fails open when capacity is unknown |
| `ggufload.WeightSource.FitOnDevice` / `FitF32OnDevice` / `FitCPUOffloadExpertsOnDevice` + `compute.EstimateKVStoreMemoryPlan` / `EstimateHALTransientMemoryPlan` + gateway model-load memory telemetry | header-only GGUF load refusal before the Go process allocates resident weights, with classed HAL KV-cache, activation, and scratchpad estimates when GGUF config exposes context geometry; expert-offload plans keep host expert bytes visible without counting them against device capacity, and optional `HostCapacity` can refuse known-too-large host-scoped `offload`/`ddr_cache` plans; successful eager device loads expose the admission profile on `/metrics` and `/debug/vars`, including bounded dtype/storage labels and per-scope want/budget/margin fit rows | shipped for GGUF serve's device load paths; quantized-upload backends are explicitly mixed precision (Q8 resident weights, f32 runtime state); f32 weights are only the fallback for backends without quantized upload; host capacity is OS physical memory, not cgroup/container quota; telemetry is visibility only, not spill/retry; not yet safetensors, generic `model.Load`, upload pre-checks, or backend-specific transient peaks |
| `agent.KVMemoryReporter` + gateway `fak_gateway_kv_memory_*` | local in-kernel KV-prefix residency reports as `kv_cache`: bytes per KV position from `compute.EstimateKVStoreBytes`, true resident `PrefixTokens`, estimated resident bytes, configured radix LRU budget tokens, current LRU edge-token count, tree shape, splits, and LRU vs policy evictions; it also emits capacity-known/free-known, capacity bytes, headroom, and resident KV `want`/`budget`/`margin` fit rows; `fak_gateway_kv_memory_dtype_info` and `/debug/vars` carry the KV row dtype (`f32` today) and the same fit snapshot | visibility over the CPU-backed radix KV prefix cache and its known budget-vs-true-footprint gap; the host budget includes `free + resident` when free memory is known so the current cache is not counted against itself; device-HAL serve currently reports per-token geometry/capacity with `enabled=false` because it uses per-request backend sessions, not device-side radix reuse; no pressure-driven demote/spill, quantized KV, or retry yet |
| `tools/memgate.py` | an operator pre-flight RAM gate | still useful outside the Go process, but no longer the only fit gate |
| `glm52_serve_preflight.py: required_vram_gb` | VRAM-vs-quant fit + per-rank TP shard sizing | models the fit of **external** engines (SGLang/vLLM), not `fak`'s own backends |

Every one is real. None lets the forward loop, or the layer above it, ask the backend it is
running on whether the next allocation will fit — and act on the answer. That is the bridge,
and it is what `DeviceCapacity` begins.

## 6. Honest boundaries

- This note now covers shipped pieces of **Plank 1 (capacity reporting + classed memory
  plans), Plank 2 (typed/classed runtime allocation OOM and request precheck refusal on the served in-kernel path),
  Plank 3 (real device-pressure planning seam), Plank 4 (the engine adapter that executes
  a demote/spill, #708), and Plank 5 slices for GGUF load-time fit refusal plus served
  request-time KV/HAL fit refusal**.
- `FitsOnDevice` / `FitsMemoryPlan` are *pre-checks*, not allocators. The live
  `fak serve --gguf --backend` now calls a GGUF plan before load: Q8/raw weights plus f32
  HAL KV/activation/scratch estimates on quantized-upload backends, f32 weights plus the
  same runtime demands on f32-only backends, and a dense/device vs host/expert split for
  `--cpu-offload-experts`. Host-scoped `offload`/`ddr_cache` bytes are checked only when a
  backend advertises `HostCapacityProbe` from the OS host-memory probe; without that they
  remain visible and fail open. Successful eager device loads expose that same classed plan
  and capacity snapshot on `/metrics` and `/debug/vars`; dtype-specific rows make the
  mixed-precision state explicit without implying quantized KV, and fit-summary rows show
  the headroom-adjusted want/budget/margin by scope. On the served in-kernel backend
  path, each request also checks the prompt+`max_tokens` KV/HAL transient plan before device
  decode when the backend reports capacity, including resident weights only when current
  free memory is unknown; unknown capacity still proceeds. The request-memory gauges are
  last-request snapshots for operator visibility, not long-term distributions, and include
  the same want/budget/margin rollup by scope.
  Safetensors, generic `model.Load`, runtime upload pre-checks, and backend-specific
  transient peaks are still unwired.
- The CUDA and Vulkan producers report `total` only; `free` is `FreeUnknown` until
  backend-specific free-memory probes are wired, so the check catches "too big for the whole
  device," not "too big for the current free headroom." Metal still advertises device
  residency but not `CapacityProbe`.
- Plank 3 (#707) makes the HBM tier's pressure and `CapacityBytes` real on a probing backend
  (`engine.PlanPlacementForDevice`), and `engine.RunCapacityPressureSweep` now binds that
  report→policy decision to the Plank-4 `CapacityAdapter` for caller-supplied KV candidates.
  The live serving loop still does not invoke the sweep automatically, and the other tiers'
  (`DRAM`/`NUMA-far`/`CXL`/`Disk`) numbers in `DefaultTierProfiles` remain representative
  defaults until measured per box.
- The local KV-prefix memory surface reports the CPU-backed radix tree's true resident
  `PrefixTokens` separately from its LRU edge-token budget (`Tokens`), because those can
  diverge sharply on deep chains. It also reports host capacity and headroom-adjusted fit
  rows when the OS probe succeeds; this is physical host memory, not a cgroup/container
  limit. Device-HAL serve does not yet use the radix tree, so it reports geometry/capacity
  with `enabled=false`, not a fake device-resident cache.
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
