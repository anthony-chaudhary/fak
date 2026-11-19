---
title: "The fak accelerator ladder: five rungs from the CPU to Vulkan and ROCm"
description: "One subject — how fak runs its in-kernel forward pass on a GPU — explained five times, from a first CPU-vs-GPU question up to a head-to-head of Vulkan and ROCm for AMD hardware."
slug: accelerators-ladder
keywords:
  - fak accelerator ladder
  - vulkan compute explained
  - rocm hip explained
  - gpu backend for llm inference
  - compute HAL seam
  - vulkan vs rocm
  - beginner gpu inference guide
  - advanced gpu inference guide
date: 2026-11-18
---

# The fak accelerator ladder

> **What this is.** The same subject — how fak's in-kernel forward pass runs on a GPU
> instead of a CPU — written **five times**, once for each kind of reader. Each rung
> assumes exactly what the one below it taught, so you can start at the bottom and climb,
> or drop in at the rung that matches what you already know. **Vulkan** and **ROCm** get
> the most attention: they are the paths most readers can run on hardware they own.

## Why does this subject need a ladder?

fak runs the model itself, in-process. The same forward pass — the loop that turns a
prompt into the next token — can run on four machines behind one interface: a **CPU
reference** (`cpu-ref`, the correctness oracle, always available), **CUDA** for NVIDIA,
**Vulkan** (cross-vendor: AMD, Intel, NVIDIA), and **ROCm** (AMD's native HIP path,
silicon-witnessed once, not reproducible here — see below).

Vulkan and ROCm are genuinely different machines. Vulkan is a *graphics* API pressed into
compute service: portable, but it pays a per-dispatch tax. ROCm is AMD's *native* compute
stack: potentially faster, but only on AMD silicon under Linux. One flat page either
drowns a newcomer or bores a senior engineer who wants the tradeoff table.

## The five rungs

| Level | Who it's for | What you'll get |
|---|---|---|
| **[1 · Why not just the CPU?](level-1-why-not-just-cpu.md)** | A complete newcomer. | Why a CPU is the wrong shape for a transformer forward pass, what a GPU actually is, and the one idea — many slow lanes beating few fast ones — that makes the rest make sense. |
| **[2 · The HAL seam](level-2-the-hal-seam.md)** | A competent Go developer. | How a backend plugs in: the `Backend` interface, `Register`/`Lookup`/`Pick`, the `Reference` vs `Approx` classes, and why adding a GPU is a registration, never a forward-loop edit. |
| **[3 · Vulkan: how it actually works](level-3-vulkan-how-it-works.md)** | A systems or graphics dev. | The Vulkan compute pipeline end to end — instance, device, queues, SPIR-V shaders, descriptor sets, command buffers — and where the per-dispatch overhead comes from. |
| **[4 · ROCm: how it actually works](level-4-rocm-how-it-works.md)** | A systems engineer, AMD-curious. | HIP as CUDA's source-compatible twin, the `hipcc`/`hipify` build, the CDNA-vs-RDNA wavefront split, and the arch taxonomy that ships before any kernel runs. |
| **[5 · Vulkan vs ROCm, head to head](level-5-vulkan-vs-rocm.md)** | A senior engineer or researcher. | The tradeoff table: portability vs native speed, the per-dispatch tax, what each path can and cannot do today, and how to choose for a card and workload. |

Each rung ends with a nav footer linking its neighbours and back here.

## How to read it

- **Just want it to work?** Read **Level 1**, then **Level 2** — what a backend is and how to select one.
- **Writing a backend?** **Level 2** is the contract; **Level 3** and **Level 4** are two worked implementations of it, one portable and one native.
- **Choosing AMD hardware?** **Level 5** is the decision page — read it after **Level 3** and **Level 4** so the tradeoffs have something concrete to attach to.
- **Doing research?** **Level 5** names the open edges; the "What is proven" section below says exactly which numbers exist and which do not.

## What is proven, and what is not

**Vulkan is shipped and witnessed on real AMD silicon.** The full SmolLM2-135M forward
pass runs on an **AMD Radeon RX 7600** (RDNA3 / gfx1102, 8 GB) through the native Vulkan
loader, reaching **numerical parity** with the CPU reference: argmax-exact greedy decode,
prefill-logit cosine = 1.0. It is **not** fast — still roughly **60× slower than
llama.cpp**, from per-dispatch CPU/driver overhead, a known and fixable architectural
limit, not a numeric defect. The measured record is `docs/benchmarks/VULKAN-AMD-RESULTS.md`.

**ROCm is written, and it has run on AMD silicon — once, and not on this host.** Three different
things are true here, and the ladder keeps them apart. The device-architecture taxonomy in
`internal/compute/rocm_arch.go` is shipped **and unit-witnessed on any host** — it classifies
every supported AMD target (`gfxNNNN`) as GCN5 / CDNA1-3 / RDNA1-3 and returns the canonical
`hipcc --offload-arch` token. The rest of the backend — `rocm_backend.go`, `rocm_kv.go`,
`rocm_kernels.hip`, and a 1,455-line test file — is **in the tree, and it has been compiled and
run on real AMD hardware**: `docs/tickets/native-rocm-backend/TICKET-01.md:64` records the
functional witness on 2026-09-10, HIP 7.2 on a `gfx1151` part, required-device tests passing
without skips. What it has **not** done is run on *this* dev host, which has no AMD GPU and no
`hipcc` — every one of those files carries `//go:build linux && rocm && cgo`, so a default
`go build ./...` skips them and the run is not reproducible here. And **no ROCm throughput number
exists**: correctness on one card is not a speed measurement, and the four acceptance bullets that
need measurement remain open in `internal/compute/ROCM-C002-NOTES.md`. The ladder never invents a
ROCm number: where a rung needs one, it says the figure is unmeasured, not that it is small or
large.

## The shape of the stack

```
   ┌────────────────────────────────────────────────┐
   │  device backends  (Approx — argmax-exact)      │
   │   cuda      vulkan      rocm (silicon-run)     │
   └───────────────────────┬────────────────────────┘
                           │  Register / Lookup / Pick
  ═════════════════════════╪═════════════════════════  ← the HAL seam
                           │  compute.Backend (whole-op methods)
   ┌───────────────────────┴────────────────────────┐
   │  cpu-ref  (Reference — the correctness oracle) │
   └────────────────────────────────────────────────┘
```

The seam is the whole design. Above it, a backend implements whole operations — `MatMul`,
`BatchedMatMul`, `Attention`, `Argmax` — and registers itself; below it, the forward loop
calls those operations and never knows which machine answered. The interface is
`internal/compute/compute.go:450`, the correctness classes `internal/compute/compute.go:279`,
the registration verbs `internal/compute/compute.go:639`,
`internal/compute/compute.go:661`, `internal/compute/compute.go:673`. Vulkan registers at
`internal/compute/vulkan.go:214` from its `init()` at `internal/compute/vulkan.go:171`;
ROCm at `internal/compute/rocm_backend.go:57` from its `init()` at
`internal/compute/rocm_backend.go:47`.

## See also

- [The explainers index](../README.md) — every ladder and single-page explainer, indexed by what you already know.
- [Hardware portability](../hardware-portability.md) — the HAL-seam explainer this ladder extends.
- [The inference ladder](../inference/README.md) — the sibling ladder: what the forward pass *is*, before this one asks where it runs. Read it first if "the loop that turns a prompt into the next token" is not yet concrete.
- [The caching ladder](../caching/README.md) — the structural precedent: another subject written five times, once per reader.
- [Multi-GPU tensor parallelism](../multi-gpu-tensor-parallelism.md) — when one device is not enough and the seam must span ranks.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.
- [Vulkan AMD results](../../benchmarks/VULKAN-AMD-RESULTS.md) — the measured RX 7600 record cited above.
- [Vulkan on Linux](../../compute/vulkan-linux.md) — building and running the Vulkan backend on a Linux host.
- [The Vulkan host backend](../../architecture/vulkan-host-backend.md) — the architecture note behind the Vulkan path.

*[Full ladder (README)](README.md) · next → [Level 1 — Why not just the CPU?](level-1-why-not-just-cpu.md)*
