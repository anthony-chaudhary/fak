---
title: "Level 2 — The HAL seam: how a backend plugs in"
description: "Where a new GPU backend attaches in fak, the compute.Backend contract it must satisfy, and why adding one is a registration rather than an edit to the forward loop."
slug: accelerators-level-2
keywords:
  - compute HAL seam
  - compute.Backend interface
  - gpu backend registration
  - reference vs approx correctness
  - go build tags for gpu
  - fak accelerator ladder
date: 2026-11-18
---

# Level 2 — The HAL seam: how a backend plugs in

*You are on **Level 2 of 5** of the [fak accelerator ladder](README.md).*

> **Audience.** A competent Go developer who has never written a GPU backend. You know
> interfaces, `init()`, and build tags. You do not need to know SPIR-V, HIP, or how a shader
> is dispatched — that is Levels 3 and 4.

> **Short answer.** You do not touch the forward loop. You write a type that satisfies one
> interface, `compute.Backend`, and register it in an `init()` behind a build tag. The forward
> loop already calls that interface; it has never heard of your hardware. Adding a backend is
> a *registration*, never an edit.

## The problem: seven invisible assumptions

fak's in-kernel forward pass (`internal/model`) was written as **one hardware target wearing
seven assumptions** — invisible because they are not *config*. They are baked into the
**types and the call sites**. You cannot turn them off with a flag; you would have to change
the code that has them.

Here are the seven, with where each one lives and the hardware class it shuts out. Source:
[`docs/explainers/hardware-portability.md`](../hardware-portability.md).

| # | Assumption | Where it lives | Hardware it shuts out |
|---|---|---|---|
| 1 | **float32 monoculture** — `[]float32` is the only currency; Q8 is a *duplicated* forward pass gated by a `bool`, not a dtype | every op signature; `q8Tensor`/`q8Vec`; `Session.Quant` | f16/bf16/fp8/MX/int4-native GPU/XPU/NPU/dataflow |
| 2 | **host-pointer aliasing** — `unsafe.Slice((*float32)…)` reinterprets a host blob; ops pass and return host slices | `internal/model/weights.go:690` | any device with a separate address space (GPU VRAM, NPU SRAM) |
| 3 | **x86 build-tag dispatch** — AVX2/512 hand-asm gated by `//go:build amd64` + CPUID; the only other path is slow scalar | `internal/model/quant_amd64.{go,s}`, `internal/model/quant_noasm.go` | ARM/RISC-V CPUs, every accelerator, WASM |
| 4 | **synchronous return-by-value** — every op computes and returns *now* | `matRows`, `qMatRows`, the layer loop | async accelerators (enqueue → fence) |
| 5 | **goroutine-only parallelism** — `parFor` splits output rows across CPU workers | `internal/model/parallel.go`, `internal/model/prefill_attn.go` | intra-kernel-lane (GPU) / pinned-graph (dataflow) HW |
| 6 | **row-major only** — `w[o*in+i]` index math everywhere; no layout descriptor | all matmuls + the KV cache | tiled/blocked/col-major device-native layouts |
| 7 | **eager full-RAM residency + LE host** — read the whole weight blob into RAM up front; "amd64 is little-endian" | `internal/model/weights.go` | small-SRAM NPU, browser/WASM, big-endian, pre-staged device weights |

Read the right-hand column as a wall: each assumption is a brick, and behind it is a class of
hardware that cannot be reached while the brick stands.

```
   the forward loop as written
   +--------------------------------------------------------------+
   | 1 float32 only   2 host pointers   3 x86 asm   4 sync return  |
   | 5 goroutines     6 row-major       7 full-RAM + LE host       |
   +--------------------------------------------------------------+
        |        |         |         |        |        |        |
        v        v         v         v        v        v        v
     f16/bf16  separate  ARM/RISC  async   GPU lanes tiled   small-SRAM
     fp8/int4  address   -V/WASM   accel   /graphs   layouts NPU, BE
```

Assumption 1 is not hypothetical: Q8 already forked the forward pass once — the Q8 loops
(`tokenHiddenQ`, `prefillBatchedQ`, `stepBatchQ`) are hand-copies of the f32 loops, selected
by a `bool`. That fork is the evidence for the next section.

## Why editing in place is the wrong move

The tempting move is to add your backend by editing those seven places. Do that and you have
re-forked the forward pass a **third** time — Q8 was the second. Now every future format
multiplies against every future piece of hardware: **O(formats × hardware)** edits to hot
loops that are proven and bit-exact, each risking the CPU path that already works.

The seam inverts the multiplication. Write the loop **once** against an interface; a new
backend is then a registration, never an edit. The seven assumptions do not disappear — they
get *named* and given a boundary in the type system, so a backend that does not share them
can still plug in.

```
   internal/model  -- the forward loop (written ONCE, hardware-blind)
   +--------------------------------------------------------------+
   |  for each layer:  x = RMSNorm(x); x = Attention(x, kv); ...   |
   +---------------------------┬----------------------------------+
                               |  calls only these methods
   ============================|==================================  <- the seam
                               |   compute.Backend interface
   +-----------+-----------+---+-------+-----------+--------------+
   | cpu-ref   |  cuda     |  vulkan   |  rocm     |  metal       |
   | Reference |  Approx   |  Approx   |  Approx   |  Approx      |
   +-----------+-----------+-----------+-----------+--------------+
     always      -tags       -tags       -tags       -tags
     built       cuda        vulkan      rocm        metal
```

## The contract: the `Backend` interface

Everything a backend must satisfy is one interface, declared at
`internal/compute/compute.go:450`:

```go
type Backend interface {
	Name() string            // stable id, e.g. "cpu-ref"
	Tier() string            // private capability probe result, e.g. "scalar","avx512","sm90"
	Class() CorrectnessClass // Reference or Approx (harness-enforced)
	Caps() Caps              // optional capabilities advertised

	// residency / dtype. CPU reference: Upload/Read are identity over the host slice.
	Upload(t Tensor, as Dtype) Tensor // host -> resident, optionally narrowing dtype
	Host(t Tensor) ([]float32, bool)  // host-addressable f32 view, or (nil,false) on a device
	Read(t Tensor) []float32          // resident -> host f32 (the only host fence on an async backend)
	Free(t Tensor)                    // release device storage (no-op on CPU)
	NewKV(cfg KVConfig) KVStore

	// primitives — dtype dispatched on the WEIGHT tensor where applicable.
	MatMul(w, x Tensor) Tensor               // y = x @ Wᵀ ; w [out,in], x [in] -> y [out]
	BatchedMatMul(w, X Tensor, P int) Tensor // prefill GEMM ; X [P,in] -> Y [P,out]
	RMSNorm(x, weight Tensor, eps float32) Tensor
	RoPE(x Tensor, pos, nHeads, headDim int, theta float64) Tensor // rotate each head
	SwiGLU(gate, up Tensor) Tensor                                 // silu(gate)*up
	AddInPlace(dst, src Tensor)                                    // residual: dst += src
	AddBias(dst, bias Tensor)                                      // dst += bias
	// Attention is one fused op: scores = (q·k)*scale over the causal window in kv,
	// softmax, ΣwV. causal/grp/scale are attributes a device lowers to flash/paged attn.
	Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor
	// Argmax is the scalar reduction so greedy decode never pulls the full logits host-ward.
	Argmax(logits Tensor) int
}
```

The methods fall into three groups.

**Identity** — `Name()`, `Tier()`, `Class()`, `Caps()`. `Name` is the string a user selects
(`"vulkan"`, `"rocm"`). `Tier` is a *private* capability probe result — what the backend
actually found on this machine (`"scalar"`, `"avx512"`, `"sm90"`) — and the forward loop
never branches on it. `Class` and `Caps` are the two honesty mechanisms, below.

**Residency** — `Upload`, `Host`, `Read`, `Free`, `NewKV`. This is where the host-pointer
assumption (number 2) is neutralized. A `Tensor` is opaque. On the CPU reference, `Upload`
and `Read` are identity over the host slice — the tensor *is* the slice. On a device, the
tensor is a handle to memory your Go program cannot address, `Host` returns `(nil, false)`,
and `Read` is the one call that copies bytes back. `NewKV` hands back the device-resident
attention cache, so the KV store is never a raw `[][]float32`.

**Primitives** — the nine ops. The design choice that matters most: **the ops are whole-op,
not element-wise.** There is no `Add(a, b)` over scalars, no `Softmax` you call yourself.
`Attention` is *one call* that lowers to a fused flash/paged-attention kernel on a device —
the Go loop never sees the softmax or the score matrix. And `Argmax` is a scalar reduction:
greedy decode asks "which index won?" and gets back an `int`, so the full logits vector never
travels host-ward. That matters because a host round-trip per op is the tax you are trying to
avoid; whole-op methods let a device keep intermediates resident and dispatch one kernel per
op.

## `Reference` vs `Approx` — the honesty mechanism

A GPU does **not** produce the same bits as the CPU. Floating-point addition is not
associative — `(a + b) + c` and `a + (b + c)` can differ in the last decimal place — and a
GPU sums a reduction in a different order than a CPU loop does. So a device backend
legitimately differs from the reference in the low-order bits. fak does not pretend
otherwise; it makes the difference a *typed* property (`internal/compute/compute.go:279`):

```go
type CorrectnessClass uint8

const (
	Reference CorrectnessClass = iota
	Approx
)
```

The CPU path is `Reference`: bit-exact, the answer key. A device backend is `Approx`. The
test harness asserts the class (`RequireReference`), so it is mechanically impossible to hold
a device to bit-identity, or to promote a device to reference by accident.

So what *is* the gate for an `Approx` backend, if not bit-identity? Two things that matter
for a language model:

- **argmax-exact greedy decode** — the token the model picks must be the *same token* the
  reference picks, every step. If the last decimal places wobble but the winner is unchanged,
  the model's output is unchanged.
- **a logit cosine threshold** — the full logit vector must point in nearly the same
  direction as the reference, measured by cosine similarity, above a per-backend threshold.

That is the right gate because it is the gate the *user* cares about. Nobody reads the
logits; they read the generated tokens. The concrete ROCm wording is in
`internal/compute/ROCM.md`: the complete-forward fixture "requires exact greedy token
selection and prefill logit cosine of at least 0.999."

## `Caps` — advertising what you can do

The core interface assumes *none* of the optional capabilities. A backend that has one
advertises it in `Caps` (`internal/compute/compute.go:301`); the forward loop type-asserts
for it and falls back to the synchronous core when it is absent. That is what makes every
backend combination correct-by-construction. The fields that matter:

- **`Async`** — methods enqueue and return *unready* buffers. The sole host fence is
  `Read`/`Argmax`. This is how assumption 4 (synchronous return-by-value) is lifted: an async
  backend can keep a queue full instead of blocking at every op.
- **`DeviceMemory`** — resident tensors are **not** host-addressable; `Host` returns false.
  This is the honest flag for assumption 2.
- **`UploadDtype`** — `Upload` honors the requested dtype, narrowing weights at H2D (host to
  device). A backend that can store Q8 natively sets this; the CPU reference ignores it.
- **`FusedAttn`** — the backend can lower `Attention` to one flash/paged-attention kernel.
- **`CapacityProbe`** — the backend can *report* the finite, exhaustible size of the memory
  it owns (total/free bytes). This is the report half of the hardware-*capacity* bridge, the
  eighth assumption the HAL lifts at the type level.
- **`Collective`** — implements the tensor-parallel cross-rank seam (all-reduce, all-gather,
  reduce-scatter over device tensors). See
  [`multi-gpu-tensor-parallelism.md`](../multi-gpu-tensor-parallelism.md).

The design detail worth internalizing is **fail-open**. A backend that does not set
`CapacityProbe` is treated as "unknown, proceed" — `DeviceMemoryInfo`/`FitsOnDevice` do not
block. The portable floor never refuses to run just because a backend cannot report its size.
Optional means optional both ways: absent capability degrades to the synchronous core, it
does not fail.

## Registration, concretely

The registry replaces the old `//go:build amd64` fork as the dispatch mechanism. Backends
self-register in `init()`; the host picks one by name. Three functions, all in
`internal/compute/compute.go`:

- `Register(b Backend)` at `internal/compute/compute.go:639` — appends to the registry. A
  second registration with the same `Name` replaces the first, so a test double can override.
- `Lookup(name string) (Backend, bool)` at `internal/compute/compute.go:661` — returns the
  named backend *without* falling back, so a typo cannot silently run on the reference floor
  and masquerade as an accelerator result.
- `Pick(name string) Backend` at `internal/compute/compute.go:673` — returns the named
  backend, or `Default()` if the name is empty or unknown.

Here is the real pattern from a shipped backend. Vulkan's `init()` is at
`internal/compute/vulkan.go:171`, and it ends by calling `Register(vulkanDev)` at
`internal/compute/vulkan.go:214`. ROCm has the identical shape: `init()` at
`internal/compute/rocm_backend.go:47`, `Register(b)` at `internal/compute/rocm_backend.go:57`.

The consequence is worth stating plainly: **`init()` runs at package load.** So merely
*building* with `-tags vulkan` is what makes the backend exist. And `cpu-ref` stays the
`Reference` default (`Default()` returns the first registered `Reference` backend), so
nothing runs on the GPU unless it is explicitly selected.

## The build-tag fence

A device backend is excluded from the default build by a build tag, deliberately: the default
build needs no GPU toolchain. The real tags:

- Vulkan — `//go:build vulkan && (windows || linux) && cgo` (`internal/compute/vulkan.go:1`)
- ROCm — `//go:build linux && rocm && cgo` (`internal/compute/rocm_backend.go:1`)

Both require `cgo`, because both cross a C ABI to reach the driver. Neither is in a default
`go build ./...`.

Now the subtle part. The ROCm arch taxonomy (`internal/compute/rocm_arch.go`) is deliberately
**pure Go with no build tag** — its first line is just `package compute`
(`internal/compute/rocm_arch.go:1`). It compiles and is unit-tested on *any* host, including
one with no AMD GPU. That is how a hardware-dependent backend still ships a host-witnessed
half: the part that needs no silicon (which AMD generations exist, whether each is CDNA
datacenter or RDNA consumer, and the canonical `gfxNNNN` offload target) is always compiled
and tested; only the part that needs a device is fenced behind the tag.

## Checkpoint

**1. Where does a new backend attach — the forward loop, or somewhere else?**
Somewhere else. It satisfies `compute.Backend` (`internal/compute/compute.go:450`) and
registers itself in `init()`. The forward loop is never edited.

**2. Why is a device backend `Approx` and not `Reference`?**
Floating-point addition is not associative, and a GPU sums in a different order, so the last
decimal places legitimately differ. The gate is argmax-exact greedy decode plus a logit
cosine threshold, not bit-identity.

**3. What does `Caps.Async` change about how the forward loop calls the backend?**
Methods enqueue and return unready buffers; the only host fence is `Read`/`Argmax`. The loop
can keep a queue full instead of blocking at every op.

**4. Why is `Attention` one method instead of a `Softmax` you call yourself?**
Whole-op methods let a device keep intermediates resident and dispatch one fused kernel. An
element-wise interface would force a host round-trip per op — the exact tax a device backend
exists to avoid.

**5. What happens if a backend does not set `CapacityProbe`?**
Nothing bad. The capacity checks fail *open* — "unknown, proceed" — so the portable floor
never blocks on a backend that cannot report its memory size.

**6. Why is `rocm_arch.go` pure Go with no build tag?**
So the hardware-independent half of the backend (the arch taxonomy and offload-target
selection) compiles and is unit-tested on any host. Only the cgo HIP half is fenced behind
`//go:build linux && rocm && cgo`.

## Honesty note: what is shipped, and what is not

Two backends, two very different maturity levels — and the page must not blur them.

**Vulkan is shipped and witnessed on real AMD silicon.** A full SmolLM2-135M forward pass
runs on an AMD Radeon RX 7600 (RDNA3 / gfx1102), argmax-exact, with prefill cosine 1.0. It is
still roughly 60× slower than llama.cpp, from per-dispatch CPU/driver overhead — a known,
fixable architectural limit, not a numeric defect. See `docs/benchmarks/VULKAN-AMD-RESULTS.md`.

**ROCm is a scaffold.** The arch taxonomy is shipped and unit-witnessed on any host; the HIP
kernel compilation and every on-silicon acceptance bullet are deferred to an
AMD-GPU-on-Linux node. **No ROCm performance number exists** — none was run, estimated, or
fabricated. See `internal/compute/ROCM-C002-NOTES.md`.

## See also

- [Level 1 — Why not just the CPU?](level-1-why-not-just-cpu.md) — the arithmetic and the
  memory wall, for readers who want the motivation first.
- [Level 3 — Vulkan](level-3-vulkan-how-it-works.md) — the shipped, witnessed backend, in
  depth.
- [hardware-portability.md](../hardware-portability.md) — the seven assumptions, the
  neutrality ledger, and the type contract that lifts them.
- [multi-gpu-tensor-parallelism.md](../multi-gpu-tensor-parallelism.md) — the `Collective`
  cap and the cross-rank seam.
- [glossary.md](../glossary.md) — terms used across the ladder.

*[Full ladder (README)](README.md) · prev ← [Level 1 — Why not just the CPU?](level-1-why-not-just-cpu.md) · next → [Level 3 — Vulkan](level-3-vulkan-how-it-works.md)*
