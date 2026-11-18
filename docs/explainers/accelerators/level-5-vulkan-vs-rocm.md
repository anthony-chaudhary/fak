---
title: "Level 5 — Vulkan vs ROCm, head to head"
description: "The honest head-to-head between fak's cross-vendor Vulkan backend (measured on AMD silicon) and its AMD-native ROCm scaffold (host-witnessed only): the real axes of difference, the per-dispatch tax, and the open problems."
slug: accelerators-level-5
keywords:
  - vulkan vs rocm
  - amd gpu compute comparison
  - cross-vendor vs native gpu api
  - spir-v vs hip
  - gpu backend portability tradeoff
  - fak accelerator ladder
date: 2026-11-18
---

# Level 5 — Vulkan vs ROCm, head to head

*You are on **Level 5 of 5** of the [fak accelerator ladder](README.md).*

> **Audience.** A senior engineer or researcher who has read [Level 3 — Vulkan](level-3-vulkan-how-it-works.md) and [Level 4 — ROCm](level-4-rocm-how-it-works.md). You know what a command buffer is and what a HIP kernel launch is. This page does not re-teach either; it puts them on the same axes and says what fak's evidence actually supports today.

> **Short answer.** Vulkan is a **cross-vendor standard** reached through a driver; ROCm is **one vendor's native stack**. Everything else — abstraction level, kernel language, code you must write, hardware you can tune for — follows from that sentence. In fak today, Vulkan is **shipped and witnessed on real AMD silicon** and ROCm is a **host-witnessed scaffold with no silicon run**. So the honest comparison here is of *designs and evidence classes*, not of measured throughput.

> **Evidence warning.** Vulkan has been measured on a real AMD Radeon RX 7600 (RDNA3 / gfx1102, 8 GB): full SmolLM2-135M forward pass, argmax-exact greedy decode, prefill-logit cosine = 1.0, and a decode rate ~60× slower than llama.cpp. ROCm has **not** been run on any GPU. Its device-architecture taxonomy is unit-witnessed on any host; its HIP kernel compilation and **every acceptance bullet that runs on silicon are deferred** to an AMD-GPU-on-Linux node. **No ROCm throughput or correctness number was run, estimated, or fabricated.** There is therefore **no measured ROCm-vs-Vulkan throughput comparison on this page**, and every ROCm number below is labeled `declared, unmeasured`.

---

## What is the one-sentence difference?

Vulkan is a **cross-vendor standard**; ROCm is **one vendor's native stack**.

Vulkan is specified by the Khronos Group and implemented by every GPU vendor's driver: the same program runs on AMD, NVIDIA, and Intel. ROCm is AMD's own compute platform — AMD's compiler (`hipcc`), AMD's runtime, AMD's libraries — and it does not pretend to run anywhere else.

That one sentence generates the whole page. A cross-vendor API must present a machine that *all* vendors can implement, so it cannot expose vendor-specific hardware. A native stack can expose everything the vendor's silicon has, at the cost of running on exactly one vendor. Vulkan trades vendor-specific tuning for portability; ROCm trades portability for AMD-specific hardware.

---

## What are the real axes of difference?

Each row is an axis; the "why" column is the part that matters, because the *what* is usually obvious and the *why* is where the design pressure lives.

| Axis | Vulkan | ROCm | Why the difference exists |
|---|---|---|---|
| **Portability** | AMD, NVIDIA, Intel | AMD only | A standard must be implementable by everyone, so it can only expose the intersection of what everyone has. |
| **Abstraction level** | You build Instance → Device → Queue → CommandBuffer → Pipeline → Buffer yourself | Looks like CUDA: allocate, launch, synchronize | Vulkan is a *graphics-and-compute* API; its object model is the price of explicit control. HIP mirrors CUDA, so the mental model is a kernel launch, not a command recording. |
| **Kernel language** | **SPIR-V** bytecode, compiled ahead of time from GLSL | C++ compiled by `hipcc` | Vulkan consumes a vendor-neutral IR so one shader binary can be re-targeted by each driver. HIP compiles C++ straight to the vendor's ISA because it only ever targets one vendor. |
| **Code reuse** | Hand-written shim (`internal/compute/vulkan_shim.cpp`, ~9,157 lines) plus 116 GLSL compute shaders | Plan: **hipify** the existing CUDA kernels rather than write a second set | Vulkan could not reuse the CUDA kernels at all — different language, different execution model. HIP is source-compatible with CUDA by design, so `internal/compute/cuda_kernels.cu` is expected to compile through `hipify`/`hipcc`. |
| **Dispatch model** | Synchronous, one command buffer per op, serialized by `vulkanMu` (`internal/compute/vulkan.go:69`) | A single serialized HIP stream behind `rocmMu` (`internal/compute/rocm_backend.go:22`) | Both are single-stream first cuts — command-buffer model vs kernel-launch model. Neither is multi-stream or multi-queue yet. |
| **Vendor-specific hardware access** | None: the API cannot expose what not every vendor has | Can expose AMD-specific features | `internal/compute/rocm_arch.go` models **wavefront width** — 64 lanes on GCN/CDNA, 32 on RDNA — and warns that picking the wrong family "silently mistunes occupancy and LDS" (`internal/compute/rocm_arch.go:27`). A cross-vendor API has no such knob. |
| **Correctness class** | `Approx` (`internal/compute/vulkan.go:587`) | `Approx` (`internal/compute/rocm_backend.go:62`) | Both are graded against the `cpu-ref` `Reference` path: argmax held to EXACT, everything else to cosine plus a small max\|Δ\|. Neither is bit-exact. |
| **Evidence class today** | **Measured on silicon** | **Host-witnessed taxonomy only** | The honest row. Vulkan has a full forward pass on a real RX 7600; ROCm has a unit-tested arch table and nothing that has touched a GPU. |

The last row decides how you should read the other seven. Seven of these axes are *design* claims, and design claims are cheap. One is an *evidence* claim, and evidence is expensive.

---

## Why does the per-dispatch tax matter so much?

Because it is the shared enemy, and it is the reason ROCm exists at all.

Vulkan's fak backend is synchronous: one command buffer per op, serialized by `vulkanMu` (`internal/compute/vulkan.go:69`). Every op pays a CPU → driver → GPU round trip. On the RX 7600 that round trip measures **~2 ms/op**, and it is *fixed per op* — not per byte, not per FLOP. A decode token is still roughly 17 dispatches per layer, so the tax is paid ~17 times per layer, every token.

This is why fak's Vulkan backend reaches **numerical parity** (argmax-exact, prefill cosine 1.0) and still runs **~60× slower than llama.cpp**. The numbers are not a numeric defect; they are a scheduling defect. The results doc records the diagnosis by elimination: command-buffer batching (record a whole token into one buffer, submit once) barely moved decode, and a shared-memory matmul shader did not move it either. When both obvious levers fail, the floor is the per-op CPU↔GPU overhead itself.

ROCm's *point* is to remove that tax. A native runtime with a kernel-launch model pays a launch cost, not a command-buffer recording and submission cost per op. The CUDA-on-WSL path on the same class of problem measures a host-launch tax of **~0.07 ms/op** — roughly 28× lower than Vulkan's ~2 ms/op. That is the direction ROCm is expected to move.

**The bar, and the honesty.** The ROCm benchmark gate does not invent a threshold. It reuses the existing `WithinTarget(measured, baseline, factor)` helper (`internal/compute/prefill.go:281`), so the device node would feed `WithinTarget(rocmSeconds, vulkanSeconds, 1.5)`: ROCm is expected to be **within 1.5× of Vulkan**, i.e. to beat it. **This has not been measured.** The 1.5× is a *declared bar*, not a result — a hypothesis with a number attached, and the number is what the silicon run is meant to falsify or confirm.

```
   VULKAN PATH (cross-vendor)          ROCM PATH (AMD-native)
   --------------------------          ----------------------

   Go: backend op                      Go: backend op
        |                                   |
        v                                   v
   cgo -> vulkan_shim.cpp              cgo -> rocm.go (//go:build rocm)
        |                                   |
        v                                   v
   record CommandBuffer                HIP runtime call
   (bind pipeline, descriptors,        (hipLaunchKernel)
    push constants, dispatch)               |
        |                                   v
        v                              kernel launch -> GPU
   vkQueueSubmit + fence                    |
        |                                   v
        v                              (single serialized stream,
   driver -> GPU                       rocmMu)
        |
        v
   one round trip PER OP: ~2 ms/op

   cost: per-dispatch CPU/driver tax   cost: per-launch tax
   benefit: AMD, NVIDIA, Intel         benefit: AMD-specific tuning
```

The two paths differ in *where* the per-op cost is paid, not in whether one is paid. Vulkan pays it in the driver on every command buffer; ROCm pays it in the runtime on every launch. The bet is that the second is far cheaper.

---

## Why can't a cross-vendor API be the whole answer?

This is the deepest technical point on the page, and it is worth slowing down for.

A cross-vendor API must present a machine that **all** vendors can implement — by construction, the *intersection* of what the vendors have. AMD's datacenter parts (GCN, CDNA — the Instinct line) execute **64-lane wavefronts**; AMD's consumer parts (RDNA — the Radeon line) execute **32-lane** ones. NVIDIA's and Intel's execution widths are different again. A standard implementable by all of them cannot say "the wavefront is 64 lanes," because for most of its implementers that would be false.

So the standard hides the machine. And hiding the machine costs you the ability to tune for it.

fak's ROCm taxonomy encodes exactly the distinction the standard cannot. `ROCmFamily` (`internal/compute/rocm_arch.go:30`) is the architecture generation; the `rocmArches` table (`internal/compute/rocm_arch.go:105`) carries each target's native wavefront width; and `Datacenter()` (`internal/compute/rocm_arch.go:92`) is the predicate a fleet planner keys on to tell an Instinct from a Radeon. The file's doc comment states the stake directly: picking the wrong family "silently mistunes occupancy and LDS" (`internal/compute/rocm_arch.go:27`). Occupancy and LDS (local data share, AMD's shared memory) are exactly the knobs a 64-vs-32-lane decision moves.

The general principle, stated once:

> **Portability is achieved by hiding the machine, and hiding the machine costs you the ability to tune for it.**

That is the real tradeoff — not "Vulkan is worse" or "ROCm is better," but two APIs optimizing different objective functions that are in tension. It is also why a serious accelerator stack usually wants **both** a portable path and a native path: the portable path runs everywhere and is the safe default; the native path runs fast on the hardware you actually own.

```
   PORTABILITY  <---------------------------------------->  TUNING

   Vulkan (cross-vendor)                    ROCm (AMD-native)
   |                                        |
   runs on AMD, NVIDIA, Intel               runs on AMD only
   cannot expose wavefront width            models 64- vs 32-lane
   cannot expose per-vendor matrix cores    can expose AMD matrix cores
   one shader binary, re-targeted           one kernel, one ISA
   |                                        |
   "the machine is the intersection"        "the machine is the silicon"
```

---

## What is the hipify bet, and what does it risk?

ROCm's plan is to compile the **existing** `internal/compute/cuda_kernels.cu` through `hipify`/`hipcc` rather than hand-write a second kernel set. HIP is source-compatible with CUDA by design (`hipMalloc` ↔ `cudaMalloc`, and so on), and AMD ships `hipify` to translate CUDA to HIP. This is the mature path — the one llama.cpp's ROCm backend takes.

**The upside** is one kernel source for two vendors. You maintain `cuda_kernels.cu` once; the CUDA build compiles it with `nvcc`, the ROCm build with `hipcc`, and the two backends stay in lockstep because they are literally the same source. Fix a kernel once, fix it for both.

**The risk** is that you inherit CUDA's assumptions and depend on hipify's translation fidelity. A kernel that is correct CUDA is not automatically correct HIP: warp-level primitives, memory-model assumptions, and intrinsic names all have HIP equivalents that are *close* but not always *identical*. If hipify's output drifts numerically, the argmax-exact gate is where it shows up — and that gate is unforgiving, because a single flipped bit in a tie-break can change the token.

Now the asymmetry that makes this genuinely interesting. Compare the two backends' code surfaces:

- **Vulkan** could not reuse the CUDA kernels at all — different language (GLSL → SPIR-V, not C++), different execution model (command buffers, not kernel launches). So it needed a hand-written shim (`internal/compute/vulkan_shim.cpp`, ~9,157 lines) **plus** 116 GLSL compute shaders, built from scratch.
- **ROCm** is expected to reuse the CUDA kernels through hipify, so its hand-written surface is a thin cgo wrapper (`rocm_backend.go`), a C ABI header (`rocm_backend.h`), and a build script (`build_rocm.sh`).

**The portable path cost more code than the native path is expected to.** That is the asymmetry, and it is not the direction intuition points. The intuition is "portable = less work, because it runs everywhere." The reality is "portable = more work, because you build the portable machine yourself, then build your program on top of it." The native path leans on the vendor's existing stack — including the vendor's ability to compile the CUDA kernels you already wrote.

---

## What is the honest scorecard?

This table is the page's most important artifact. The `Evidence class` column is the one that makes it trustworthy: every ROCm performance row reads `declared, unmeasured`, because that is what it is.

| Axis | Vulkan | ROCm | Evidence class |
|---|---|---|---|
| Runs on AMD silicon | Yes — RX 7600 (gfx1102) | Not yet | Vulkan `measured on silicon`; ROCm `declared, unmeasured` |
| Numerical parity vs cpu-ref | argmax-exact, prefill cosine 1.0 | Not yet | Vulkan `measured on silicon`; ROCm `declared, unmeasured` |
| Decode throughput | ~2.5 tok/s (SmolLM2-135M f32), ~60× behind llama.cpp | No number exists | Vulkan `measured on silicon`; ROCm `declared, unmeasured` |
| Throughput vs the other backend | Baseline | Expected within 1.5× of Vulkan | Vulkan `measured on silicon`; ROCm `declared, unmeasured` |
| Device-arch taxonomy | n/a (cross-vendor) | `ROCmFamily` + `rocmArches` + `Datacenter()` | ROCm `host-witnessed` |
| Offload-target selection | n/a | `ROCmOffloadArch(gfx)` normalization | ROCm `host-witnessed` |
| Backend registration seam | Registers as `Approx` `"vulkan"` | Registers as `Approx` `"rocm"` (planned) | Vulkan `measured on silicon`; ROCm `declared, unmeasured` |
| Multi-GPU | Not implemented | Planned over RCCL via `CollectiveBackend` | ROCm `declared, unmeasured` |

Read the table by column, not by row. The Vulkan column is almost entirely `measured on silicon`. The ROCm column is almost entirely `declared, unmeasured`, with two `host-witnessed` rows for the parts that are pure computation and need no GPU. That split is the whole story of the two backends' maturity.

```
   THE EVIDENCE LADDER

   strongest
      |
      |  MEASURED ON SILICON
      |    - Vulkan: full forward on RX 7600, argmax-exact, cosine 1.0
      |    - Vulkan: ~2.5 tok/s decode, ~60x behind llama.cpp
      |    - Vulkan: ~2 ms/op per-dispatch tax, diagnosed by elimination
      |
      |  HOST-WITNESSED  (unit-tested on any host, no GPU)
      |    - ROCm: ROCmFamily / rocmArches / Datacenter() taxonomy
      |    - ROCm: ROCmOffloadArch(gfx) offload-target normalization
      |
      |  DECLARED, UNMEASURED  (a plan with a number attached)
      |    - ROCm: "within 1.5x Vulkan throughput"
      |    - ROCm: "bit-exact results"
      |    - ROCm: "multi-GPU support"
      |    - ROCm: "runs on AMD GPU on Linux"
      |
   weakest
```

---

## What would change the answer?

Everything that would move ROCm from scaffold to silicon is enumerated in `internal/compute/ROCM-C002-NOTES.md`. In order:

1. **An AMD-GPU-on-Linux node with ROCm installed** — the precondition for all the rest; the current dev host has no AMD GPU and no `hipcc`.
2. **Confirm the toolchain.** `hipcc --version` and `rocminfo | grep gfx` to get the reported `gfxNNNN`, then feed it to `ROCmOffloadArch` — which accepts the `:sramecc+:xnack-` feature suffix `rocminfo` may print.
3. **Write the cgo half.** `rocm_backend.go` (`//go:build rocm`), `rocm_backend.h`, and `build_rocm.sh` — a near-clone of `cuda.go`'s wrapper, registering an `Approx` backend named `"rocm"` over a flat C ABI.
4. **Witness op-by-op parity, then the full forward.** Mirror `vulkan_test.go`'s 7-op suite, then the full forward pass; hold argmax to EXACT, the rest to cosine plus a small max\|Δ\|.
5. **Benchmark against the Vulkan baseline on the same card**, graded with `WithinTarget(rocmSeconds, vulkanSeconds, 1.5)`.
6. **Multi-GPU** via `CollectiveBackend` over RCCL; the single-rank identity is witnessable on one card before a second GPU is needed.

The four acceptance bullets that are **unwitnessed** — quoted verbatim from the notes — are:

- [ ] Run on AMD GPU on Linux
- [ ] Within 1.5× Vulkan throughput
- [ ] Bit-exact results
- [ ] Multi-GPU support

All four are measurements on AMD silicon under ROCm, and none can be witnessed on a host with no AMD GPU and no `hipcc`. Until they are, the ROCm column of the scorecard stays `declared, unmeasured`.

---

## What are the open problems?

Being a researcher here means naming the questions that are genuinely unresolved, not the ones with tidy answers.

**How much of the ~60× Vulkan gap is recoverable by batching, versus by moving to a native runtime?** The results doc already tried the batching lever and it barely moved decode: recording a whole token into one command buffer preserved correctness but left throughput roughly flat. That suggests the floor is per-op overhead, not submit count — which would mean batching alone cannot close the gap and a native runtime is *necessary*, not merely nice. But "suggests" is not "proves." The clean experiment — same card, same model, Vulkan-with-maximal-fusion versus ROCm — has not been run.

**Is hipify's output numerically faithful enough to hold the argmax-exact gate?** The load-bearing correctness question for the whole hipify bet. If hipify's translation of `cuda_kernels.cu` drifts even slightly, the argmax-exact gate catches it — and that gate is unforgiving by design. We will not know until the kernels compile and run on silicon.

**Is a single serialized stream a real ceiling for decode?** Both backends are single-stream first cuts: Vulkan serializes on `vulkanMu` (`internal/compute/vulkan.go:69`), ROCm on `rocmMu` (`internal/compute/rocm_backend.go:22`). For batch-1 decode a single stream may be fine — there is little to overlap. For batched prefill or multi-request serving it is almost certainly a ceiling. *Where* it bites is an empirical question about the workload, not a design question.

**How should a portable path and a native path coexist in one backend registry?** The subtlest open problem, and a *product* problem as much as a technical one. fak's registry lets a backend register itself and be selected explicitly (`FAK_BACKEND=vulkan`, `--backend rocm`), with `cpu-ref` staying the `Reference` default so nothing runs on the GPU by accident. That is the right safety property. But it leaves a trap: if the portable path is always available and the native path only sometimes, the portable path can silently become the slow default — the thing that runs because it is there, not because it is right. A serious stack needs a policy for *when* to prefer the native path, and that policy is not written yet.

---

## Checkpoint — test yourself

**1. In one sentence, why do Vulkan and ROCm differ on every other axis?**
Vulkan is a cross-vendor standard and ROCm is one vendor's native stack. Portability forces Vulkan to present the intersection of all vendors' machines; the native stack presents AMD's actual silicon.

**2. Vulkan reaches numerical parity on the RX 7600 and is still ~60× slower than llama.cpp. Is that a correctness bug?**
No — a scheduling defect. The per-dispatch CPU↔GPU round trip is ~2 ms/op and is fixed per op, so a token with ~17 dispatches per layer pays it ~17 times per layer. The numbers are right; the schedule is wrong.

**3. Why can't Vulkan expose AMD's 64-lane wavefront?**
Because a cross-vendor API must be implementable by every vendor, and most vendors do not have a 64-lane wavefront. Exposing it would make the standard unimplementable for them. The standard hides the machine, and hiding the machine costs the ability to tune for it.

**4. What is the hipify bet, and what is the asymmetry it creates?**
The bet is to compile the existing `cuda_kernels.cu` through `hipify`/`hipcc` instead of hand-writing a second kernel set. The asymmetry: the *portable* path (Vulkan) could not reuse the CUDA kernels and needed a ~9,157-line shim plus 116 shaders, while the *native* path (ROCm) is expected to reuse them through a thin cgo wrapper. The portable path cost more code than the native path is expected to.

**5. What does `WithinTarget(rocmSeconds, vulkanSeconds, 1.5)` mean, and what is its evidence class?**
It is the declared bar for the ROCm benchmark: ROCm is expected to be within 1.5× of Vulkan, i.e. to beat it. Its evidence class is `declared, unmeasured` — a hypothesis with a number attached, not a result.

**6. Why does the scorecard's `Evidence class` column matter more than its other columns?**
Because the other columns are design claims, and design claims are cheap. The evidence column is the only one that distinguishes "we built this and measured it on silicon" from "we plan to build this." It refuses to let a plan read like a result.

---

## See also

- [The accelerator ladder (README)](README.md) — the index for all five levels.
- [Level 2 — the HAL seam](level-2-the-hal-seam.md) — how a backend registers and gets picked.
- [Level 3 — Vulkan, how it works](level-3-vulkan-how-it-works.md) — the command-buffer model in detail.
- [Level 4 — ROCm, how it works](level-4-rocm-how-it-works.md) — the HIP model and the arch taxonomy.
- [Hardware portability](../hardware-portability.md) — the broader portability story.
- [Multi-GPU tensor parallelism](../multi-gpu-tensor-parallelism.md) — the `CollectiveBackend` seam ROCm's multi-GPU bullet targets.
- [Vulkan AMD results](../../benchmarks/VULKAN-AMD-RESULTS.md) — the measured-on-silicon numbers this page cites.

*[Full ladder (README)](README.md) · prev ← [Level 4 — ROCm](level-4-rocm-how-it-works.md)*
