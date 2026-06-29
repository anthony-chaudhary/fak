---
title: "Blackwell-class hardware readiness map: where fak's serving seams light up on GB200/GB300 NVL72"
description: "A grounded plan for running fak better on the latest NVIDIA datacenter hardware (Blackwell NVL72: NVFP4, a 72-GPU coherent NVLink domain, Grace coherent CPU memory). fak's serving seams are already NVL72-shaped — coherent KV pool, residency tiers, device-mesh collectives, bit-exact paged evict — but every one is a policy-plane / CPU-ref / single-device stub today. Maps each Blackwell capability onto the exact seam it lights up, names the contexts where performance is much better, quantifies how far the value envelope expands, and gives a hardware-ungated-first rung ladder."
---

# Blackwell-class hardware readiness map (GB200/GB300 NVL72)

_Date: 2026-06-29. Planning note. Asserts **no** measured fak number on Blackwell — every
economic figure below is a **MODELED** cost-model projection or a published hardware spec,
labelled as such. The parity-must-be-measured rule (`#44`) still binds: a throughput claim is
real only once the `cmd/paritybench` harness runs it on the hardware. Connects
[`dual-track-serving-plan`](../serving/dual-track-serving-plan.md),
[`THROUGHPUT-TRUST-SHARED-SPINE`](THROUGHPUT-TRUST-SHARED-SPINE-2026-06-24.md),
[`native-device-mesh-collectives`](../serving/native-device-mesh-collectives.md),
[`hardware-aware-cache`](../serving/hardware-aware-cache.md),
[`cxl-memory-pool`](../serving/cxl-memory-pool.md), and
[`multi-node-compute`](../serving/multi-node-compute.md) to one concrete target platform._

## TL;DR

fak's whole scale-out architecture was designed single-box and payload-free: a coherent KV
**pool** plane (`internal/cachemeta/pool.go`), a hardware-aware residency **ladder**
(`internal/cachemeta/hardware.go`), a device-**mesh** collective seam
(`internal/compute/compute.go`, CPU-ref only), and a bit-exact middle-span **evict** that
survives paging (`internal/model/paged_evict.go`). A Blackwell GB300 NVL72 is, almost line for
line, the physical substrate those planes were written against: **one coherent 20.7 TB HBM3e
NVLink domain across 72 GPUs + 36 Grace CPUs, 130 TB/s of in-domain NVLink, NVFP4 at 0.5
bytes/param**. Today every one of those seams is a stub — a cost model, a CPU reference, a
`cudaSetDevice(0)` — so the hardware is the thing that turns them from "tested mechanism" into
"realized value."

The strategic point is sharper than "fak gets faster." fak is **not** a faster model server —
vLLM/SGLang/TRT-LLM win raw tokens/sec and you run `fak serve` in front of them. fak's value is
the **governance / reuse / disaggregation referee** layer, and that layer's worth is a
*function* of pool size, coherence, tenant density, and context length — exactly the four knobs
NVL72 maxes out. On a 36 GB laptop the moat is a nicety; on a 20.7 TB coherent pool shared by an
agent fleet the questions fak answers (whose KV may be reused across tenants, can a poisoned span
be **provably** evicted from the middle of shared pages, where should each span live, how is
disaggregated P/D KV moved at NVLink speed) stop being optional and become the control plane the
hardware structurally requires.

## 1. The target hardware (what actually changed)

GB300 NVL72 (Blackwell Ultra), the current flagship rack, and the GB200 NVL72 before it, with
the Rubin generation on the H2-2026 horizon. The numbers that matter to **fak specifically**:

| Capability | GB300 NVL72 | Why it matters to fak (not to a generic server) |
|---|---|---|
| **One coherent NVLink domain** | 72 GPUs + 36 Grace CPUs, 130 TB/s aggregate NVLink, 1.8 TB/s per GPU, 9 NVSwitch, 576-way, ~300 ns switch latency | This **is** the `pool.go` `FabricShareable()` fabric: `Hosts>1` + coherent + zero-copy share. The plane that prices N-prefill→1 collapse finally has a real coherent pool under it. |
| **HBM3e capacity** | 288 GB/GPU @ 8 TB/s → **20.7 TB in one domain** | The single-box ceiling (`fak` faithful ≤7B on 36 GB) is gone by ~575×. After FP4 weights, ~20 TB of HBM is free for **resident KV** — fak's home turf (prefix reuse, demote-not-evict). |
| **Grace coherent CPU memory** | NVLink-C2C @ 900 GB/s, ~480 GB LPDDR5X/Grace → ~17 TB coherent, byte-addressable | A real, fast, huge **attendable-in-place** tier — exactly the `TierNUMAFar`/`TierCXL` rung in `hardware.go`, but at 900 GB/s and coherent. Demote-not-evict wins overwhelmingly here. |
| **NVFP4** | 0.5 bytes/param (vs FP8 1 B, FP16 2 B); 5th-gen tensor cores; 5 precision tiers TF32/FP16-BF16/FP8/FP6/NVFP4 | 4× weight shrink vs FP16 frees HBM for KV. But it **collides** with the bit-exact-evict moat (§4) — the one load-bearing tension this note resolves. |
| **2× attention / GPU** vs GB200 | 2× attention-layer throughput, 1.5× FP4 | Directly speeds the paged-attention gather fak's KV kernel owns. |
| **ConnectX-8** | 800 Gb/s RDMA per GPU | The cross-domain `StageTransport` backend (when a model spans more than one NVL72). |

Sources: [GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) ·
[Inside Blackwell Ultra (NVIDIA dev blog)](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) ·
[GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/).

## 2. The capability → seam map (current state, file-anchored)

Each row: a Blackwell capability, the fak seam it lights up, the **current state** of that seam,
and the rung (§5) that lands it. State tags follow the house grammar: `[SHIPPED]` / `[SEAM-ONLY]`
(the algebra/policy exists, no bytes/device) / `[GAP]` (greenfield).

| Blackwell lever | fak seam (anchor) | State today | Lit by |
|---|---|---|---|
| NVFP4 weights (0.5 B/param) | `Dtype` enum `internal/compute/compute.go:56-63` — F32/F16/BF16/Q8_0/I8/I4/**FP8**/Q4_K | `[GAP]` — **FP4/MXFP4 is not even an enum value**; FP8 and BF16 are enum-only with no device path; device GEMMs today are F32/F16 + Q8/Q4_K weight-narrowing (`internal/compute/cuda_kernels.cu`) | R0, R5 |
| NVFP4 KV | KV-precision tiers `internal/compute/kvprecision.go:28-113` — `KVPrecisionF32` vs `KVPrecisionQ8` (post-RoPE K/V quantized, **pre-RoPE `Kraw` stays f32** for exact evict) | `[SEAM-ONLY]` — only F32 and Q8 tiers; the f32-`Kraw` invariant is the exact pattern FP4-KV must reuse (§4) | R1 |
| 20.7 TB HBM + 17 TB Grace coherent | residency ladder `internal/cachemeta/hardware.go:100-161` — HBM→DRAM→NUMA-far→CXL→Disk→Remote, `AttendableInPlace()`, per-tier `TierProfile` | `[SHIPPED]` policy plane; profiles are **representative**, not measured; **Grace is mentioned nowhere** | R2 |
| 130 TB/s coherent NVLink pool | fleet-reuse pool `internal/cachemeta/pool.go:41-100` — `Reachable()`, `FabricShareable()`, cross-tenant `PoolReuseVerdict` trust gate | `[SHIPPED]` payload-free; emits **no bytes**; no NVL72 `PoolProfile` (`Hosts=72`, coherent) | R2 |
| NVLink/NVSwitch as the byte mover | KV→bytes serializer (`#29`) + `StageTransport` (`internal/xenginekv/arena.go`) | `[GAP]` — a span has **no portable byte form**; `cachemeta` `BytesMoved` is a reported counter, nothing copies KV in-tree | R3, R7 |
| 72-GPU TP/EP inside one NVLink island | device mesh `internal/compute/compute.go:346-408` (`CollectiveBackend`) + `internal/model/tensor_parallel.go` (`TPShard`/`TPPlan`) | `[SEAM-ONLY]` — **CPU-ref collectives only**; `DistComm` is host-f32 TCP, **not** NCCL/multi-GPU; `cudaSetDevice(0)` hardcoded; **zero** NVLink/NVSwitch topology code; `ForwardTP` not wired to live Forward | R6 |
| Cross-node host collective (floor) | `fak cluster` over `model.DistComm` | `[SHIPPED]` — real cross-node AllReduce/AllGather, bit-exact vs `LocalCollective`, runnable today | done |
| Provable eviction over a shared pool | bit-exact middle-span evict `internal/model/paged_evict.go` + `KVCache.Evict`; deletion cert `internal/deletioncert` | `[SHIPPED]` (proof, `#33` GO, `max\|Δ\|=0`); `[GAP]` the production paged allocator (`#34`) + the L3 region backend (`#55`/`#79`) | R8 |

## 3. Where performance is *much* better there — and the mechanism

Not "Blackwell is fast." These are the regimes where fak's **specific** levers, which barely
move on a single small box, compound on NVL72. Each labelled with what is **MODELED** vs a
hardware spec; none is a measured fak result.

1. **Fleet prefix reuse across a coherent domain.** fak's reuse value is `(tenants sharing a
   prefix) × (pool coherence)`. NVL72 maxes both: one coherent 20.7 TB domain that
   `pool.go::FabricShareable()` recognizes as the single regime that collapses *N* prefills **and**
   *N* resident copies into **one**. `cmd/cxlpooldemo` already computes this (8 tenants × a
   4000-token prefix → **28,000 prefill tokens + 448 MB of copies saved**, MODELED); on NVL72 that
   stops being a hypothetical profile and becomes the physical fabric. Agent fleets are the ideal
   workload — prefix-heavy (shared system prompt + tools), latency-tolerant, unattended — so the
   reuse multiple is largest exactly where fak is pointed.

2. **Demote-not-evict against a 900 GB/s coherent tier.** `placement.go::RetainCheaperThanRecompute`
   compares stage-cost-into-a-colder-tier vs recompute-cost-avoided. Grace LPDDR at 900 GB/s,
   coherent and byte-addressable, is an `AttendableInPlace()` tier where staging always beats a
   re-prefill for any non-trivial prefix — so the demote-not-evict default is *right by a wide
   margin* instead of marginal. `cmd/hwcachedemo` already shows the blind-LRU-vs-tiered gap
   (28,000 re-prefilled tokens → 0, MODELED); a real Grace tier is the substrate that earns it.

3. **KV disaggregation at memory-bus speed.** The shared-spine plan builds the KV byte mover
   "TCP-first" (~10s of GB/s). In-domain NVLink is **1.8 TB/s** — ~18–36× the PCIe/NIC the plan
   assumed — so native prefill/decode disaggregation and an L3 KV tier become nearly free *inside*
   one NVL72. The disaggregation referee fak wants to be is cheap precisely where the pool is
   biggest.

4. **The frontier-model class comes in-reach, weights stop dominating HBM.** A 753B model is
   ~1506 GB at FP16 but **~377 GB at NVFP4** — under two GPUs of a 20.7 TB domain. The ~1.1 TB of
   HBM that FP16 weights would have burned is redirected to **resident KV**, i.e. to fak's layer.
   The single-box "faithful ≤7B" ceiling is replaced by "weights are a rounding error; KV
   governance is the scarce resource." (Honest caveat: linear-attention/hybrid families —
   Gated-DeltaNet, part of GLM-5.2 — reject mid-span evict via `RecurrentEvictUnsupportedError`,
   so the evict moat covers the dense/attention families, not those. See §6.)

## 4. The one load-bearing tension: NVFP4 vs the bit-exact-evict moat

NVFP4 is Blackwell's headline lever, and it **collides** with fak's headline moat. The bit-exact
middle-span evict re-derives each shifted survivor's post-RoPE K in a *single* rotation from its
**pre-RoPE `Kraw`** at the survivor's new logical position (`paged_evict.go`,
`EVICT-ON-PAGED-KV-DESIGN`). Re-rotating from a *lossy* FP4 `Kraw` is no longer bit-exact, and
composing two rotations is worse. So you cannot have both "FP4 everything" and "`max|Δ|=0`
evict" on the same span.

**Decision — tier precision per *plane*, not per model** (the FP4 analogue of the Q8-KV choice
already made in `kvprecision.go` and `#1047`):

- **FP4 the weights.** Pure win — 4× HBM saving, no interaction with evict (weights are not the
  KV).
- **FP4/FP6 the V plane and the attention-read K.** V is never rotated; the read-K is lossy-OK for
  the attention dot-product.
- **Keep an f32 `Kraw` shadow** for any span that must be *governably evictable*. That preserves
  `max|Δ|=0` middle-span evict + the deletion certificate. The cost is a denser K-plane for
  evictable spans — affordable against 20.7 TB HBM + 17 TB Grace, and a span that will never be
  quarantined skips the shadow and **degrades honestly** to whole-prefix flush
  (`SupportsExactSpan=false`), never silently.

This makes NVFP4 and the moat coexist by construction. It is a new `Dtype` enum value + the same
f32-`Kraw` invariant the code already enforces — not a redesign.

## 5. The rung ladder (hardware-ungated first)

Worst-regret-minimizing order: everything that can be proven **without** a Blackwell box is
front-loaded (so the NVL72 lease, when it comes, is spent only on the genuinely hardware-blocked
rungs). Each rung is a `dos`-verifiable prove-or-refute with a named witness; a rung is `[SHIPPED]`
only with a committed witness, never asserted.

**Hardware-UNGATED (do now, on this box / any box):**

- **R0 — FP4/BF16 as first-class `Dtype` + a CPU-ref oracle.** Add `NVFP4` (and wire the dormant
  `BF16`/`FP8`) to `compute.go` with a `QuantSpec` variant and a **cpu-reference** quant/dequant.
  Witness: a bit-exact FP4 round-trip test — the oracle every device kernel (R5) is later graded
  against. *Refutes the "no FP4 anywhere" gap.*
- **R1 — FP4 KV tier that the moat survives.** Add `KVPrecisionFP4` to `kvprecision.go` (V + read-K
  in FP4, `Kraw` f32 shadow). Witness: a `paged_evict_test`-shaped test showing a middle-span evict
  on the FP4-KV tier is still `max|Δ|=0` (because re-RoPE reads the f32 `Kraw`). This is the §4
  decision, proven — the FP4 analogue of `#33`.
- **R2 — Model the NVL72 in the cost planes.** Add a Grace/NVLink-C2C `TierProfile` to `hardware.go`
  (coherent, byte-addressable, ~900 GB/s, ~480 GB/host) and an NVL72 `PoolProfile` to `pool.go`
  (`Hosts=72`, coherent, `FabricShareable`). Witness: `hwcachedemo`/`cxlpooldemo` re-run over the
  NVL72 profile, emitting the demote-not-evict and fleet-reuse economics against real Blackwell
  numbers (still MODELED — a cost model over a real spec, not a hardware measurement).
- **R3 — KV→bytes serializer + `StageTransport` TCP path (`#29`).** Give a span a portable,
  materialization-keyed byte form that fails closed under a wrong model/tokenizer/position regime;
  move it over TCP first. Witness: serialize→ship→deserialize→bit-exact restore; fail-closed on a
  wrong key. This is the byte form NVLink later carries at 1.8 TB/s.

**Hardware-GATED (need a real GB200/GB300 NVL72):**

- **R5 — NVFP4 device GEMM** behind the HAL on 5th-gen tensor cores (`sm_100`+), witnessed
  within-tolerance vs the R0 cpu-ref oracle. The first Blackwell-native compute.
- **R6 — Device `CollectiveBackend` over NCCL.** Lift `cudaSetDevice(0)`, bind per-rank device,
  NVLink/NVSwitch all-reduce, **NVLink-topology-aware** placement (TP/EP inside the island per
  `native-device-mesh-collectives`), witnessed bit-exact vs cpu-ref (multi-node rung 5; `#25`/`#305`).
- **R7 — `StageTransport` NVLink/RDMA backend swap.** KV byte mover over 1.8 TB/s NVLink +
  ConnectX-8; native P/D disaggregation in-domain; the `L3RegionBackend` (`#55`/`#79`) over the
  Grace/NVLink tier.
- **R8 — The governed mega-pool (value capture).** Wire the placement plane + the cross-tenant
  trust gate + per-span deletion certificates over the *real* NVL72 coherent pool. fak as the
  KV-governance control plane for 20.7 TB shared across a fleet — the rung where the architecture
  pays.

## 6. Honest fences

- **No Blackwell-native compute exists today.** FP4 is not an enum value; FP8/BF16 are enum-only;
  the device path is F32/F16 + Q8/Q4_K weight-narrowing. R5 is greenfield **and** hardware-gated.
- **No device communicator.** No NCCL/RCCL anywhere; `cudaSetDevice(0)` is hardcoded; no NVLink
  topology code. The collective *contracts* (`CollectiveBackend`, `TPShard`/`TPPlan`, `DistComm`)
  exist and are bit-exact on one box — what is greenfield is the **device backend + wire**, not the
  architecture.
- **The cost planes move no bytes.** `cachemeta`/`pool.go` are payload-free; the NVL72 economics in
  R2 are a cost model over a real spec, **not** a fak-on-hardware measurement.
- **Grace is modeled, not integrated.** No coherent-CPU-memory path in-tree; R2 adds the profile,
  R7 adds the real mover.
- **The moat does not cover linear-attention families.** Gated-DeltaNet / hybrid (part of GLM-5.2)
  reject mid-span evict (`RecurrentEvictUnsupportedError`); FP4 + governed evict is a
  dense/attention-family claim. State it that way.
- **No throughput number is asserted.** Every figure here is a published hardware spec or a MODELED
  cost-model projection. A fak parity/throughput claim is real only via `cmd/paritybench` on the
  hardware (`#44`), and "we orchestrated SGLang to parity" ≠ "native fak hit parity" ≠ "fak governs
  this engine's KV" — three claims that must never collapse into one.

## 7. See also

- [Dual-track serving plan](../serving/dual-track-serving-plan.md) — the authoritative S1–S6 sequencing this overlays.
- [Throughput/trust shared spine](THROUGHPUT-TRUST-SHARED-SPINE-2026-06-24.md) — why the KV byte mover is one shared dollar both tracks need.
- [Native device mesh & collectives](../serving/native-device-mesh-collectives.md) — the `#25` design gate R6 consumes (TP/EP inside the NVLink island).
- [Hardware-aware KV cache](../serving/hardware-aware-cache.md) — the residency ladder R2 extends with a Grace tier.
- [Multi-tenant CXL memory pool](../serving/cxl-memory-pool.md) — the fleet-reuse economics R2 re-runs over the NVL72 profile.
- [Multi-node compute](../serving/multi-node-compute.md) — the rung ladder R6 sits on (`fak cluster` is the floor).
- [Evict-on-paged KV design (#33)](EVICT-ON-PAGED-KV-DESIGN-2026-06-28.md) — the bit-exact-evict invariant §4's FP4-KV decision preserves.
- [Affordable-hardware agent fleets](AFFORDABLE-HARDWARE-AGENT-FLEETS-2026-06-28.md) — the opposite end of the same axis (the doable tier vs the frontier rack).
