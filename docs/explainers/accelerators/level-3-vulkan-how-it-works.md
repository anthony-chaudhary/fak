---
title: "Level 3 — Vulkan: how it actually works"
description: "The Vulkan compute path in fak end to end: instance, device, queue, SPIR-V shaders, command buffers, the byte-level admission check, and where the per-dispatch tax comes from."
slug: accelerators-level-3
keywords:
  - vulkan compute explained
  - spir-v shader
  - vulkan command buffer
  - vulkan compute pipeline
  - glsl compute shader
  - vulkan for machine learning
  - fak accelerator ladder
date: 2026-11-18
---

# Level 3 — Vulkan: how it actually works

*You are on **Level 3 of 5** of the [fak accelerator ladder](README.md).*

> **Audience.** A systems engineer who knows what a GPU is and has never written a Vulkan
> compute program. You are comfortable with C, C++, and the idea of a driver. You do not
> need to have read Level 2, though the seam it describes is where this page plugs in.

> **Short answer.** Vulkan is a cross-vendor standard, not a vendor's product. Before the
> GPU can multiply anything you must build an instance, pick a physical device, create a
> logical device and a queue, compile your shader to SPIR-V bytecode ahead of time, and
> record a command buffer for every operation. fak does all of that in a C++ shim behind a
> flat C ABI, and it does it **synchronously, one command buffer per op** — which is exactly
> why the backend reaches numerical parity on real AMD silicon and is still ~60× slower
> than llama.cpp.

## What is Vulkan, and why is it not CUDA?

Vulkan is a **standard**, not a product. It is a specification for talking to graphics and
compute hardware at a low level, and AMD, NVIDIA, Intel and others each ship a driver — an
*installable client driver*, or ICD — that implements it. Your program talks to the
specification; the vendor's ICD talks to the silicon.

That is the whole difference from CUDA and ROCm. CUDA is NVIDIA's own stack; ROCm is AMD's
own stack. Each is one vendor's answer for one vendor's hardware, and each is free to expose
whatever that hardware can do. Vulkan is a contract that *every* vendor must be able to
satisfy, so it can only expose what they all have in common.

The trade is explicit and it cuts both ways:

- **You gain portability.** One backend, written once, can in principle run on an AMD card,
  an NVIDIA card, and an Intel integrated GPU. fak's Vulkan backend is the only device
  backend in the tree that is not tied to a single vendor.
- **You pay in verbosity and in tuning.** Vulkan makes you build the machine yourself, and
  it cannot tell you that AMD's datacenter parts execute 64-lane wavefronts while its
  consumer parts execute 32-lane ones — because a cross-vendor API is not allowed to know
  that. Level 5 is about what that costs.

## The four things you must build before any math happens

This is the part that surprises people coming from CUDA. In CUDA you write a kernel and
launch it. In Vulkan you first construct a small universe.

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │  VkInstance            the connection between your program and the  │
   │                        Vulkan loader + the installed ICDs           │
   └───────────────────────────────┬─────────────────────────────────────┘
                                   │  enumerate
   ┌───────────────────────────────▼─────────────────────────────────────┐
   │  VkPhysicalDevice      one real GPU. You inspect its properties and │
   │                        pick one (fak prefers a DISCRETE_GPU)        │
   └───────────────────────────────┬─────────────────────────────────────┘
                                   │  create
   ┌───────────────────────────────▼─────────────────────────────────────┐
   │  VkDevice  +  VkQueue  the logical device you submit work to, and   │
   │                        the queue that work is submitted on          │
   └───────────────────────────────┬─────────────────────────────────────┘
                                   │  record into
   ┌───────────────────────────────▼─────────────────────────────────────┐
   │  VkCommandBuffer       the recorded list of work: bind this         │
   │                        pipeline, bind these buffers, dispatch N     │
   └───────────────────────────────┬─────────────────────────────────────┘
                                   │  built from
   ┌───────────────────────────────▼─────────────────────────────────────┐
   │  VkPipeline            the compiled program + its fixed state,      │
   │                        built from a SPIR-V module                   │
   └─────────────────────────────────────────────────────────────────────┘
```

And separately, where the numbers live:

```
   VkBuffer  +  VkDeviceMemory   ──►  an opaque handle, NOT a host pointer
```

That last line is the one that matters most for a compute backend. A device buffer is not
memory your Go program can dereference. It is a handle that names memory on the other side
of the bus. fak's shim says so directly in its own header comment: the opaque handle the C
ABI hands to Go is a `Buffer*`, **not** a host pointer, matching the CUDA device-pointer
contract (`internal/compute/vulkan_shim.cpp:13`).

fak builds all of this in one place. `fvk_init` — declared at
`internal/compute/vulkan_backend.h:33` — creates the instance, picks a physical device
(preferring a discrete GPU, falling back to the first available), creates the logical device
and a compute queue, and builds one compute pipeline per kernel from the SPIR-V modules in
`spirv_dir`. It also reports the device name and whether the device is discrete, so a
software rasterizer such as Mesa's lavapipe can never silently masquerade as the real card.

## Why is the shader a compiled artifact?

Vulkan does not accept GLSL at runtime. GLSL is compiled **ahead of time** to **SPIR-V**
bytecode, and the driver consumes the bytecode. This is a deliberate design choice with two
consequences that matter here.

First, the shipped binary contains no shader compiler. fak keeps its compute shaders as GLSL
in `internal/compute/shaders/` — 119 files, of which 116 are `.comp` compute shaders — and
compiles them offline. The build recipe is a loop, from `docs/compute/vulkan-linux.md`:

```sh
build="$PWD/_scratch/vulkan-linux"
mkdir -p "$build/spirv"
for shader in internal/compute/shaders/*.comp; do
  name=$(basename "$shader" .comp)
  glslc -O --target-env=vulkan1.2 -fshader-stage=comp "$shader" -o "$build/spirv/$name.spv" || exit
done
c++ -O3 -std=c++17 -fPIC -c internal/compute/vulkan_shim.cpp -o "$build/vulkan_shim.o" || exit
ar rcs "$build/libfakvulkan.a" "$build/vulkan_shim.o" || exit
```

Second — and this is the interesting part — because the shader is a **byte stream** before it
ever reaches a driver, you can inspect it without a GPU.

## The admission check: reading bytes instead of running on silicon

fak does not hand the SPIR-V blob to the driver on faith.
`internal/compute/vulkan_spirv_admission.h` is a **pure byte-level validator**: it parses the
SPIR-V word stream and checks narrow ABI properties with no Vulkan handles, no device state,
and no file I/O. The entry point for one such check is `swigluPushConstantABI` at
`internal/compute/vulkan_spirv_admission.h:17`.

Why bother? Because a mismatched push-constant layout is a **silent memory-corruption bug**.
The host writes a struct into a push-constant block; the shader reads fields at offsets it
believes are correct. If the two disagree, the shader reads the wrong bytes and produces
plausible-looking garbage. You can catch that by reading the SPIR-V's declared member offsets
and comparing them to the host struct — no GPU required, no flaky test, no "works on my
driver".

The file is honest about its own limits, and so is this page: these checks establish narrow
ABI properties, **never numerical GPU qualification**. Passing admission means the shader and
the host agree on a layout. It says nothing about whether the arithmetic is right. That is
what the argmax-exact witness is for.

## The dispatch path, end to end

Here is one operation, `MatMul(w, x)`, traced from Go to silicon and back.

```
   Go:  MatMul(w, x)
     │
     ├─► vulkanMu.Lock()                    ← one mutex, whole backend
     │
     ├─► cgo call into the flat C ABI       ← fvk_* functions, extern "C"
     │
     ├─► shim records a ONE-SHOT command buffer
     │      bind pipeline · bind descriptor sets · push constants · vkCmdDispatch
     │
     ├─► vkQueueSubmit  →  vkWaitForFences   ← the synchronous model
     │
     ├─► GPU executes the SPIR-V
     │
     └─► fence signals  →  buffer is Ready()==true  →  vulkanMu.Unlock()
```

Three design decisions are visible in that picture, and each is a deliberate trade.

**One-shot command buffers, synchronous wait.** The shim's header comment states it plainly:
every op records a one-shot command buffer, submits, and waits on a fence — the synchronous
`Ready()==true` model (`internal/compute/vulkan_shim.cpp:16`). Nothing is pipelined. The Go
side gets a finished answer or it gets nothing.

**A buffer pool.** `vkAllocateMemory` is slow, so the shim recycles allocations and
steady-state decode pays roughly zero allocation cost
(`internal/compute/vulkan_shim.cpp:17`). This is the same reasoning CUDA applies to
`cudaMalloc`.

**One mutex for the whole backend.** `internal/compute/vulkan.go:69` declares
`var vulkanMu sync.Mutex`, and the shim's comment says all entry points are serialized by
that Go-side mutex, so the single command pool and queue need no internal locking
(`internal/compute/vulkan_shim.cpp:19`). This is simplicity chosen over concurrency, and it
is worth naming the cost: with one serialized queue there is no overlap of compute with
transfer, and no multi-stream pipelining. It is the right first cut — correctness first, and
one lock is easy to reason about — but it is a ceiling, not a floor.

## The per-dispatch tax — the honest headline

Now the most important engineering lesson on this page.

fak's Vulkan backend reaches **numerical parity** on a real **AMD Radeon RX 7600**
(RDNA3 / gfx1102, 8 GB): argmax-exact greedy decode, prefill-logit cosine = 1.0. The model
runs *correctly* on AMD hardware. And it is still roughly **60× slower than llama.cpp**.

The cause is the picture above. Every single operation pays a full CPU → driver → GPU round
trip: record a command buffer, submit it, wait on a fence. For a 30-layer model doing many
small ops per token, that overhead dominates the actual arithmetic. The measured record is
`docs/benchmarks/VULKAN-AMD-RESULTS.md`, and it is explicit that this is a **known, fixable
architectural limitation, not a numeric defect**.

```
   what the GPU is doing:      ████░░░░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░████
   what the CPU is doing:      ░░░░████████████████░░░░████████████████░░░░
                                   ↑ record+submit+wait   ↑ record+submit+wait
   the GPU is idle for most of the wall clock, waiting to be told what to do next
```

The fix direction is equally clear and equally unglamorous: batch many operations into fewer
command buffers, and stop waiting on a fence after every one. That work is not claimed here
as done. The honest statement is: parity yes, throughput no, and the gap is understood.

## What the shaders actually are

A flavour of the real set, so the names are not a mystery:

- **Core primitives** — `matmul.comp`, `attention.comp`, `argmax.comp`, `add.comp`,
  `add_bias.comp`. These mirror the `Backend` interface one-for-one.
- **Quantized-weight matvec kernels** — a large family: `iq1s_matvec.comp` through
  `iq4xs_matvec.comp`, `q2k_matmul.comp`, `q3k_matmul.comp`, `q4k_matmul.comp`. Each is a
  different **weight storage format**. A quantized matvec kernel dequantizes weights on the
  fly, so the model's weights stay small in memory and the arithmetic happens at full
  precision. That is why there are so many of them: one per format, not one per operation.
- **Cooperative-matrix variants** — `coopmat_wave32_wmma.comp`, `q4k_matmul_coopmat.comp`.
  These use the GPU's matrix-multiply hardware rather than doing the multiply-accumulate by
  hand. fak probes for this capability at init via `fvk_have_cooperative_matrix`
  (`internal/compute/vulkan_backend.h:359`) rather than assuming it.
- **Fused kernels** — `matmul_add.comp`, `matmul_argmax.comp`, `attention_split_kv.comp`.
  Fusing two ops into one dispatch is the direct answer to the per-dispatch tax, applied
  where it pays most.

The directory also holds `attention.metal` — the same shader family serves the Metal
backend, which is a useful reminder that "one shader per operation" is a portable idea even
when the bytecode format is not.

## How do you know it works?

Not by reading the code. By running a bounded witness on a physical device. From
`docs/compute/vulkan-linux.md`:

```sh
FAK_VULKAN_SPIRV="$build/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 \
  FAK_VULKAN_EXPECT_DEVICE=8060S \
  timeout 30s "$build/compute.test" -test.run '^TestVulkanArgmaxExact$' -test.v -test.timeout 25s
```

The design point worth noticing is `FAK_VULKAN_REQUIRE_DEVICE=1`. It makes a **missing**
device a test *failure* rather than a skip. Without it, a machine with no GPU would report a
green run — and a green run that silently skipped the GPU is not a witness, it is a
decoration. The same doc is careful about the other direction too: a passing primitive
witness proves only that dispatch, and full-model quality and throughput require separate
model-bound witnesses.

## Checkpoint

**1. Why can't Vulkan expose AMD's 64-lane wavefront distinction?**
Because Vulkan is a cross-vendor contract. It may only expose what every vendor can
implement, so vendor-specific execution geometry is invisible to it. That is the price of
portability, and it is why a native path like ROCm exists alongside it.

**2. What is SPIR-V, and why compile ahead of time?**
It is the bytecode Vulkan drivers consume. Compiling GLSL to SPIR-V offline means the shipped
binary needs no shader compiler, and — more usefully — the bytecode can be validated as bytes
before a driver ever sees it.

**3. What does the admission check prove, and what does it not prove?**
It proves narrow ABI properties, such as a push-constant block's member layout matching the
host struct. It does **not** prove the arithmetic is correct. Numerical qualification needs a
device witness.

**4. Where does the ~60× gap come from?**
Per-dispatch CPU/driver overhead. Every op records a one-shot command buffer, submits, and
waits on a fence, so the GPU sits idle between operations. It is an architectural limit, not
a numeric defect — the results are argmax-exact.

**5. Why is there one mutex for the whole backend?**
`vulkanMu` (`internal/compute/vulkan.go:69`) serializes every entry point so the single
command pool and queue need no internal locking. It is a simplicity-over-concurrency choice,
and it costs pipelining.

**6. Is the Vulkan backend the default?**
No. It registers as `"vulkan"` in the `Approx` class
(`internal/compute/vulkan.go:587`), so `cpu-ref` stays the `Reference` default and nothing
runs on the GPU unless it is explicitly selected.

## See also

- [The accelerator ladder (README)](README.md) — the index for all five rungs
- [Level 2 — the HAL seam](level-2-the-hal-seam.md) — the interface this backend implements
- [Level 4 — ROCm: how it works](level-4-rocm-how-it-works.md) — AMD's native path, and what
  it is meant to fix
- [Level 5 — Vulkan vs ROCm](level-5-vulkan-vs-rocm.md) — the head-to-head
- [Hardware portability](../hardware-portability.md) — the wider portability story
- [Vulkan on Linux](../../compute/vulkan-linux.md) — the full build and witness recipe
- [The Vulkan host backend](../../architecture/vulkan-host-backend.md) — the architecture note
- [Vulkan AMD results](../../benchmarks/VULKAN-AMD-RESULTS.md) — the measured RX 7600 record

*[Full ladder (README)](README.md) · prev ← [Level 2 — The HAL seam](level-2-the-hal-seam.md) · next → [Level 4 — ROCm](level-4-rocm-how-it-works.md)*
