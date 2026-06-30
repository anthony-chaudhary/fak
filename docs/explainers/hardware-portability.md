---
title: "fak explainer: hardware portability via the compute HAL seam"
description: "Explains the internal/compute HAL seam that lets fak's in-kernel forward pass add CUDA and Vulkan backends by registration rather than re-forking the hot loops."
---

# Hardware portability for the in-kernel forward pass — the `internal/compute` HAL seam

> **Status:** the seam is shipped and can carry **two real device backends** beside the
> pure-Go CPU reference. `internal/compute` (the contract) registers `cpu-ref` (Reference),
> plus `cuda` (Approx, `//go:build cuda`) and `vulkan` (Approx, `//go:build vulkan`) — each
> proven on actual silicon: CUDA runs the in-kernel Llama decode on this box's RTX 4070
> (argmax-exact, logit cosine 1.0 — `../../GPU.md`) and Vulkan runs the full
> SmolLM2-135M forward pass on a real AMD Radeon RX 7600 (argmax-exact, prefill cosine 1.0 —
> `../benchmarks/VULKAN-AMD-RESULTS.md`). The model package routes through the seam via
> `Model.NewBackendSession(compute.Backend)`, and `TestHALSessionMatchesLegacyCPUReference`
> proves the `cpu-ref` path is byte-identical to the legacy session path on a deterministic
> synthetic model. The optimized legacy prefill/batch path is still the default until full
> adoption. `cmd/modelbench -backend <name> -require-non-reference` is the production gate:
> it fails closed on a CPU-only build (only `cpu-ref` registered) and **passes when built
> with `-tags cuda`/`-tags vulkan` on a box with that device** — which is exactly how the two
> witnesses above were captured. The original seam design came from a 19-agent
> audit→design→adversarial-verify→synthesis pass (CUDA / edge-NPU / dataflow-wafer / WASM
> lenses); two of those four lenses (CUDA, and Vulkan as the discrete-GPU case) are now
> built and witnessed on real hardware, not hypothetical.

*Who this is for:* contributors adding or reasoning about a non-CPU backend (CUDA, Vulkan,
NPU, dataflow, WASM) for fak's in-kernel forward pass. Prerequisites: familiarity with the
`internal/model` forward pass and Go build tags. By the end you'll understand the seven
host-CPU assumptions the `internal/compute` HAL neutralizes, how its type contract lets a
new backend be a *registration* rather than a fork, and where each hardware class plugs in.
Vendor-facing onboarding lives in
[`docs/vendor/neo-silicon-onboarding.md`](../vendor/neo-silicon-onboarding.md), including the
minimum compiling backend example.

## 1. Why a *seam*, not a *port*

The in-kernel forward pass (`internal/model`) is correct and, on CPU, fast. But it was
written as one hardware target wearing seven invisible assumptions. They are invisible
because they are not *config* — they are baked into the **types and the call sites**:

| # | Assumption | Where it lives today | Hardware it shuts out |
|---|---|---|---|
| 1 | **float32 monoculture** — `[]float32` is the only currency; Q8 is a *duplicated* forward pass gated by a `bool`, not a dtype | every op signature; `q8Tensor`/`q8Vec`; `Session.Quant` | f16/bf16/fp8/MX/int4-native GPU/XPU/NPU/dataflow |
| 2 | **host-pointer aliasing** — `unsafe.Slice((*float32)…)` reinterprets a host blob; ops pass/return host slices | `weights.go:96` | any device with a separate address space (GPU VRAM, NPU SRAM) |
| 3 | **x86 build-tag dispatch** — AVX2/512 hand-asm gated by `//go:build amd64` + CPUID; the only other path is slow scalar | `quant_amd64.{go,s}`, `quant_noasm.go` | ARM/RISC-V CPUs, every accelerator, WASM |
| 4 | **synchronous return-by-value** — every op computes and returns *now* | `matRows`, `qMatRows`, the layer loop | async accelerators (enqueue → fence) |
| 5 | **goroutine-only parallelism** — `parFor` splits output rows across CPU workers | `parallel.go`, `prefill_attn.go` | intra-kernel-lane (GPU) / pinned-graph (dataflow) HW |
| 6 | **row-major only** — `w[o*in+i]` index math everywhere; no layout descriptor | all matmuls + the KV cache | tiled/blocked/col-major device-native layouts |
| 7 | **eager full-RAM residency + LE host** — `os.ReadFile` the whole ~537 MB blob (SmolLM2-135M f32: 135M params × 4 B); "amd64 is little-endian" | `weights.go` | small-SRAM NPU, browser/WASM, big-endian, pre-staged device weights |

Adding *any* non-CPU backend by editing these in place would mean re-forking the forward
pass a third time (Q8 already forked it once — `tokenHiddenQ`/`prefillBatchedQ`/`stepBatchQ`
are hand-copies of the f32 loops). That is O(formats × hardware) edits to proven, bit-exact
hot loops. The seam inverts it: **write the loop once against an interface; a new backend is
a registration, never an edit.**

### Hardware-shape neutrality ledger

This is the competitive buyer view of the same table. A backend matrix can still be
GPU-biased if it counts device names but leaves the model loop shaped like a host CPU. fak's
claim is narrower and fenced: the `internal/compute` contract names the seven host-shape
assumptions, gives each a boundary or fallback, and keeps unsupported regimes `FENCED`
rather than silently `UNDEFINED` in the support-maturity sense
([honesty fence](../standards/support-maturity-honesty-fence.md),
[matrix](../HARDWARE-MATRIX.md)).

| Assumption | Porting tax on non-GPU / neo-silicon | HAL fence | Current witness |
|---|---|---|---|
| float32 monoculture | Native bf16/fp8/MX/int4 hardware needs a new loop clone or lossy host expansion. | `Dtype` + `QuantSpec`; dtype dispatch lives on `Tensor` and weight ops. | `cpu-ref` f32/Q8 lanes and device backend parity tests exercise dtype-dispatched `MatMul`; broader low-precision coverage is still a scored gap. |
| host-pointer aliasing | Device SRAM/VRAM cannot be passed as a Go `[]float32` without staging everything through host RAM. | Opaque `Tensor`/`Buffer`, `Host(t)` opt-in view, and `Read(t)` as the explicit fence. | CUDA, Vulkan, Metal, and CPU backends register behind the same contract; `DeviceMemory` remains cap-advertised only when true. |
| x86 build-tag dispatch | A new accelerator becomes another build fork rather than a runtime backend. | `Register`, `Lookup`, `Pick`, and private `Tier()` probing. | `cpu-ref`, CUDA, Vulkan, and Metal are selected by registration/build tag without editing the forward loop. |
| synchronous return-by-value | Async command queues must block at every op boundary, losing overlap and graph capture. | `Caps.Async`, `Buffer.Ready()`, and host fences at `Read` / `Argmax`. | The contract is fenced; production async depth is backend-specific and advertised only when implemented. |
| goroutine-only parallelism | Device kernels must emulate row-splitting instead of using their native lanes, tiles, or graphs. | Whole-op `Backend` methods such as `MatMul`, `BatchedMatMul`, `Attention`, and `Argmax`. | CUDA/Vulkan/Metal lower whole ops behind the interface; CPU keeps goroutines private to the reference. |
| row-major only | Tiled, blocked, sparse, or compiler-native layouts pay repack cost at every call site. | `Layout` on `Tensor`; backends repack at `Upload` and keep layout private after that. | The shipped path is row-major-first, but the layout boundary is explicit rather than an implicit host assumption. |
| eager full-RAM residency + LE host | Small-SRAM, pre-staged, streaming, big-endian, or browser/WASM targets inherit a full host-blob requirement. | `WeightSource.Weight(name, want)` and `Upload(t, as)` move residency and narrowing behind the backend. | `WeightSource` is a shipped seam; broad vendor staging remains not-yet until backend conformance and scaffold tooling land. |

Provenance: the contract and CPU/CUDA/Vulkan/Metal registrations are WITNESSED where their
support rows say so. A specific future NPU, dataflow chip, PIM target, TPU, or vendor SDK
remains ASPIRED until its backend passes the relevant conformance rung. This is why the
industry scorecard reports hardware-shape neutrality separately from hardware breadth.

## 2. The type contract — assumptions neutralized in the types

`internal/compute` lifts all seven assumptions **in the type system**, even though only the
CPU reference is implemented today. The point is that the *contract* a future GPU/NPU
implements already assumes none of them.

- **Dtype is first-class** (`Dtype` enum on every `Tensor`, plus `QuantSpec`). The model's
  `tensorMeta.Dtype` string — parsed then *discarded* today — becomes real dispatch. A
  weight's `Dtype` selects the kernel, so the f32/Q8 "forward pass exists twice"
  duplication collapses into one `MatMul` that switches on `w.Dtype`. fp8/MX/int4/asymmetric
  schemes are new `Dtype` + `QuantSpec` values, **not a third clone**. *(lifts #1)*
- **A `Tensor` holds no host pointer.** Storage is an opaque `Buffer`; host addressability
  is reachable *only* by type-asserting to `HostBuffer` (implemented solely by the CPU
  backend) or via `Backend.Host(t) → (slice, ok)` which returns `(nil,false)` on a device.
  A device tensor therefore **cannot be silently reinterpreted as a host slice** — the
  compile/assert kills the `unsafe.Slice` hazard. The contract exposes no `unsafe.Pointer`,
  so it stays wasm-clean. *(lifts #2)*
- **Dispatch is a runtime registry** (`Register`/`Pick`), not a build-tag fork. `Tier()` is
  each backend's *private* capability probe (CPUID on x86, a driver query on a GPU),
  generalizing the existing `resolveTier()`/`FAK_QKERNEL` mechanism across the whole device
  boundary. Build tags then gate only *which backends compile in*; the registry picks which
  one *runs*. The package never reads `os.Getenv` (empty on wasm) — the host passes the
  name. *(lifts #3)*
- **Execution can be async** without forcing it on anyone. `Buffer.Ready()` + `Caps.Async`
  let a device enqueue and return an unready buffer, fencing only inside `Read`/`Argmax`.
  `Argmax` is a first-class scalar-reduction op so greedy decode returns a 4-byte token id
  instead of copying the full ~49 K-vocab logits host-ward every step. *(lifts #4)*
- **Parallelism is the backend's business.** The interface exposes *whole ops* (`MatMul`,
  `Attention`), never "split these rows across workers", so a device expresses its own
  intra-kernel parallelism; the reference's fork-join stays private. *(lifts #5)*
- **Layout is a descriptor** (`Layout` on every `Tensor`). The CPU reference honors only
  `RowMajor`; a tensor-core backend declares `Tiled`/`ColMajor` and repacks at `Upload`
  without the loop seeing it. *(lifts #6)*
- **Residency is pluggable.** `WeightSource.Weight(name, want)` lets a backend stream or
  pre-stage weights instead of slurping one host blob, and `Upload(t, as)` narrows dtype at
  H2D. *(lifts #7, at the type level)*

Two cross-cutting guard rails (judge grafts):

- **`CorrectnessClass{Reference, Approx}` is typed and harness-enforced.** Only a `Reference`
  backend may be subjected to the exact rungs (max|Δ|=0 R2/R14, the HF argmax oracle);
  `RequireReference(b)` gates every such assertion. Every `Approx` backend (the Q8 lane, and
  every future device) is held to the looser argmax-exact + logit-cosine gate, with a
  per-backend cosine threshold. It is *mechanically impossible* to expect bit-identity of a
  device or to silently promote one to reference.
- **`Caps`** (`Async`, `FusedAttn`, `FusedFFN`, `GraphCompile`, `UploadDtype`, `DeviceMemory`,
  `Collective`, `CapacityProbe`) are optional capabilities a backend advertises; the core
  interface assumes none, the loop falls back to the core when a cap is absent → every backend
  combination is correct by construction. (`CapacityProbe` is the newest — it reports the
  device's *size*, the eighth assumption; see
  [hardware-limits-and-capacity.md](hardware-limits-and-capacity.md).)

## 3. The CPU reference is *verbatim*

The day-1 backend (`cpuref.go`, `Class()==Reference`) reproduces the model's arithmetic
exactly, so adoption is byte-identical:

| Backend method | reproduces (model) | reduction order preserved |
|---|---|---|
| `MatMul` (F32) | `matRows`/`parMatRows` | `fdot` 8-accumulator fixed tree |
| `MatMul` (Q8_0) | `qMatRows` | `qdot8scalar` 4-acc per-block |
| `BatchedMatMul` (F32 / Q8_0) | `matMulBatch` / `qGemm8scalar` | `fdot` / `qgemm8cell` (lanes=16) |
| `RMSNorm` | `rmsnorm` | serial in-order sum-of-squares (the load-bearing one) |
| `RoPE` | `ropeRow`+`applyRopeRow` | non-interleaved rotate_half |
| `Attention` | `tokenHidden` attn loop | single-acc score `dot`, in-order ΣwV |
| `SwiGLU` / `AddInPlace` / `AddBias` | the MLP/residual loops | elementwise |
| `Argmax` | `argmaxF32` | first-max |
| `KVStore` (`AppendKV`/`Evict`/`Clone`) | `KVCache` | single-rotation re-RoPE on evict |

It is pure-Go, scalar, stdlib-only — **no unsafe, no asm, no cgo, no `os.Getenv`** — so it is
*also* the portable floor every other target degrades to (it compiles to wasm unchanged). A
real CPU backend may later expose the model's x86 AVX kernels via `Tier()`; that is a private
acceleration of this same reference contract, picked by the registry, not a fork of the loop.
*(This is now concrete on two ISAs: the model package's accelerated Q8 lane is amd64
AVX2/AVX-512 **and** arm64 NEON SDOT — measured head-to-head vs llama.cpp in
`../benchmarks/LLAMACPP-HEADTOHEAD-RESULTS.md` (Zen5) and `../benchmarks/M3-LLAMACPP-RESULTS.md`
(Apple M3). Both stay bit-identical to the scalar reference — exactly the "private
acceleration, not a fork" the `Tier()` seam describes. So assumption #3's "ARM/RISC-V CPUs"
gap above is now closed for arm64.)*

## 4. What day-1 buys

- A buildable, tested cross-platform contract (`go test ./internal/compute/` green): the
  Backend self-test (each op == the model function, `Float32bits`-equality), the
  reduction-order pin, the device-tensor type contract, the registry/capability gates, the
  Q8 Approx gate, and the **evict == never-saw (max|Δ|=0)** KV-quarantine witness.
- The f32/Q8 *kernel* duplication expressed as one dtype dispatch (`MatMul` on `w.Dtype`),
  demonstrating the collapse the audit ranked hardest.
- A `KVStore` seam shipped from day 1 (the verifiers' unanimous "do not defer this") so a
  device-resident / paged KV is an added impl, not a forward-loop rewrite later.

## 5. The known-open ledger (tracked deferrals, not blind spots)

Each open assumption is named with the seam that will close it. Honesty graft from the
design panel: the deferrals are deliberate, not forgotten.

| Open assumption | Why deferred | Closing seam |
|---|---|---|
| eager full-RAM `os.ReadFile` of the ~537 MB blob (SmolLM2-135M f32) | CPU policy unchanged day-1 | `WeightSource` (stream/stage per tensor) |
| little-endian `unsafe.Slice` (big-endian broken) | lives inside CPU `Upload` only | device-native repack in `Upload`/`WeightSource` |
| per-op host alloc (`make([]float32)` for q/k/v/scores) | not needed to ship the CPU seam | an `Alloc(shape,dtype)` scratch-pool cap |
| row-major only on CPU | reference honors `RowMajor` | a backend that honors the `Layout` field |
| bf16→f32 widening at load | `Dtype` field now present; end-to-end narrow is future | `ReadAs(Dtype)` + native-narrow `WeightSource` |
| synchronous return-by-value | day-1 simplicity + bit-identity | `Caps.Async` + `Buffer.Ready()` futures; `GraphCompile` record-replay |
| optimized model package not yet fully wired to the seam | the safe first slice is a per-token HAL session path; the legacy batched/Q8 paths remain the production default | fold `prefillBatched`, Q8, and batch decode through `Backend` once the per-token gate stays green |
| **finite device capacity** — OOM is a `dalloc` panic, `Caps.DeviceMemory` is a *shape* bool (not a size), and `cuda.go` discarded the `totalGlobalMem` it probed | the seven lifts above are all hardware *shape*; capacity is a hardware *limit* — a different category, treated in its own explainer | `compute.DeviceCapacity` (report) + `FitsOnDevice`, bridging to the `cachemeta` placement plane and an engine adapter. See **[hardware-limits-and-capacity.md](hardware-limits-and-capacity.md)** — the eighth assumption |

## 6. How each hardware class plugs in (and what each adversarial lens demanded)

- **CUDA GPU** (separate VRAM, async streams, native f16/bf16/fp8): implements `Upload` as
  H2D DMA narrowing per `as`, `Host`→`(nil,false)`, `Caps{Async,FusedAttn,UploadDtype,
  DeviceMemory}`; `Attention` lowers to FlashAttention; `Argmax` is a device reduction (4
  bytes out). *Lens verdict: **built and witnessed** — `internal/compute/cuda.go`
  (+`cuda_kernels.cu`, `//go:build cuda`) runs a real in-kernel Llama decode on this box's
  RTX 4070, argmax-exact with logit cosine 1.0 vs cpu-ref (`TestHALDeviceForwardMatchesNative`;
  `../../GPU.md`). The shipped v1 advertises the `DeviceMemory` cap and a device-resident
  KV cache; the remaining `Async`/`FusedAttn`/`UploadDtype` caps above are the still-open
  optimization surface, not a correctness gap.*
- **Vulkan compute GPU** (AMD/RDNA3 and any Vulkan 1.x device; separate VRAM, SPIR-V
  compute shaders, native Windows loader): a structural mirror of the CUDA backend
  (`internal/compute/vulkan.go` + `vulkan_shim.cpp` + `shaders/*.comp`, `//go:build vulkan`).
  `Host`→`(nil,false)`, device-resident weights/KV, `Argmax` as a two-pass block reduction
  (4 bytes out), and a fused decode graph (RMSNorm+Q/K/V, RMSNorm+gate/up, FFN-tail, residual
  matmul-add, op-level Q8_0 GEMM). *Lens verdict: **built and witnessed on real AMD silicon**
  — the full SmolLM2-135M forward pass on a Radeon RX 7600 is argmax-exact with prefill
  cosine 1.0 across all 30 layers (`../benchmarks/VULKAN-AMD-RESULTS.md`, Rung 1). Throughput is
  the honest open gap: ~9× behind llama.cpp CPU and climbing with each op-fusion (Rung 2),
  bounded by per-dispatch CPU/driver overhead, not numerics. This is the discrete-GPU lens
  made concrete on a card without CUDA.*
- **Intel XPU / OneDNN (SYCL)** (Intel Arc discrete GPUs and Data-Center GPU Max XPUs —
  separate device memory via SYCL USM, the oneAPI runtime, native f16/bf16 and int8 XMX
  matrix engines): a structural sibling of the CUDA and Vulkan lenses, reached through Intel's
  oneAPI stack — the oneDNN primitive library (matmul/inner-product, softmax, normalization)
  lowered onto a SYCL queue. It maps onto the seam exactly as the other discrete GPUs do:
  `Upload` is an H2D copy into SYCL USM narrowing per `as` (`Caps.UploadDtype` once int8/`Q8_0`
  rides the XMX engines), `Host`→`(nil,false)` with `Caps.DeviceMemory`, the SYCL queue makes
  it an `Async` backend (`Buffer.Ready()` + `Caps.Async`, fencing only inside `Read`/`Argmax`),
  `Attention` lowers to a oneDNN fused-SDPA primitive (`Caps.FusedAttn`), `Argmax` is a device
  reduction (4 bytes out), and the Level-Zero device-memory query feeds `DeviceCapacity`
  (`Caps.CapacityProbe`). It compiles only behind `//go:build onednn` (cgo to the oneAPI/SYCL
  runtime + an offline-built oneDNN shim — the CUDA/Vulkan shim pattern), so the default
  `go build ./cmd/fak` stays one pure-Go binary, and it registers an **Approx** backend (held
  to argmax-exact + logit-cosine, never bit-identity). *Lens verdict: **designed, not yet
  built** (#264) — the contract already carries everything this lens needs (`Dtype`/`QuantSpec`
  for int8 XMX, async via `Caps.Async`, device residency, the capacity probe), so the XPU
  backend is a *registration*, not a forward-loop edit. What remains is host-gated and cannot
  be witnessed on a CPU-only box: the cgo oneDNN/SYCL shim, then the four acceptance rungs on
  real Intel Arc silicon — runs on Arc, ≥ 5× CPU throughput, argmax-exact vs `cpu-ref` (the
  Approx class's bit-exactness rung), and the device-memory-efficiency report.
  `cmd/modelbench -backend onednn -require-non-reference` is the gate that will record that
  evidence, the same way the CUDA and Vulkan witnesses above were captured.*
- **OpenVINO (Intel CPU/GPU/NPU)** (Intel's inference runtime: ingest an IR, dispatch the whole
  model across the CPU, integrated/discrete GPU, or NPU plugins): distinct from the oneDNN-SYCL
  XPU lens above — that hand-lowers oneDNN primitives onto a SYCL queue op-by-op on an Arc GPU,
  whereas OpenVINO is the higher-level runtime whose load-bearing decision is *device selection*
  and whose unique reach is the **Intel NPU** (the AI-Boost accelerator on Meteor/Lunar/Arrow
  Lake) that oneDNN-SYCL does not target. It maps onto the seam by registering an **Approx**
  backend named `"openvino"` (`//go:build openvino`) that exports fak's in-process op-list to an
  OpenVINO IR and `core.compile_model(model, device)`s it: a discrete GPU advertises
  `Caps.DeviceMemory`, the NPU advertises `Caps.GraphCompile` (it compiles the whole IR to a
  static device blob ahead of time), and the CPU plugin is the programmable parity floor — the
  "within 1.5× native CPU" baseline. The native precision is a real `Dtype` (F32 on the CPU
  plugin, F16 on GPU/NPU), and AUTO/HETERO/MULTI/BATCH are recognized as virtual meta-plugins that
  delegate to physical devices, never a compile target. *Lens verdict: **designed, not yet built**
  (#257) — the always-compiled device-plugin taxonomy is shipped and unit-witnessed on any host
  (`internal/compute/openvino_arch.go`: `LookupOVDevice`/`OVDeviceToken`/`IsVirtualOVDevice`, the
  CPU/GPU/NPU split, the native-precision-per-device invariant). What remains is host-gated: the
  cgo `//go:build openvino` half, then runs-via-OpenVINO + within-1.5×-CPU + NPU-support on real
  Intel silicon — see `internal/compute/OPENVINO-C006-NOTES.md`.*
- **Edge NPU** (fixed vendor op menu, native int8/int4, must pre-stage weights): uses
  `QuantSpec` (asymmetric, per-channel, int4, static-act) for its weights, `Caps.FusedFFN`
  to map a whole MLP block to one vendor primitive, and `WeightSource` to stage a
  device-native packed layout. *Lens verdict: needs the WeightSource + richer QuantSpec the
  contract now carries; full native-narrow end-to-end is on the ledger.*
- **Dataflow / wafer (Groq/Cerebras/Tenstorrent)** (whole graph compiled & pinned ahead of
  time): advertises `Caps.GraphCompile`, runs the Backend methods in record-only mode to
  capture the op sequence as a portable **in-process op-list** (no ONNX/StableHLO importer),
  then compiles+places it; the CPU reference eagerly interprets that *same* op-list through
  its exact kernels, so the recorded-graph replay stays bit-identical. *Lens verdict: the
  one class needing whole-graph visibility — reachable via the GraphCompile cap without
  taxing the day-1 eager path.*
- **TPU / Neural Engine** (two accelerators, two compiler lanes): the issue title (#261, C-004)
  lumps Google's TPU and Apple's Neural Engine, but they lower through different lanes and the
  split is load-bearing. A **Google TPU** (v2–v6e Trillium) is a whole-graph, ahead-of-time part:
  it reuses the *Dataflow* mechanism above — record the in-process op-list, lower it to StableHLO,
  hand it to XLA/PJRT which compiles & places it (`Caps.GraphCompile`), native tier **bf16** on the
  MXU. An **Apple Neural Engine** (A17/M3-family … M4, *not* the Metal GPU backend in `metal.go`)
  is an *Edge-NPU* fixed-op-menu part reached through CoreML: map a whole MLP block to one CoreML
  op (`Caps.FusedFFN`), stage weights device-native via `WeightSource`, native tier **fp16**. fak
  does **not** import an external ONNX/StableHLO graph — it lowers its own recorded op-list, so the
  scope's "ONNX import" is reframed as that in-process path (an external-graph importer is the
  inverse direction and out of the seam's architecture). Both register an **Approx** backend
  (argmax-exact + logit-cosine, never bit-identity). *Lens verdict: **designed, not yet built**
  (#261) — the always-compiled accelerator→lane taxonomy is shipped and unit-witnessed on any host
  (`internal/compute/tpu_arch.go`: `LookupAccelArch`/`AccelTarget`, the XLA/CoreML split, the
  native-tier-per-lane invariant). What remains is host-gated: the cgo `//go:build xla` (PJRT) and
  `//go:build coreml` halves, then runs-on-the-accelerator + forward-parity + baseline on real TPU
  / Apple silicon — see `internal/compute/TPU-C004-NOTES.md`.*
- **WASM / browser** (no threads/asm/env/unsafe by default, bounded memory, WebGPU optional):
  runs the pure-Go scalar reference as the floor unchanged; selection comes through a host
  config channel (not `os.Getenv`), parallelism defaults to serial, weights stream via
  `WeightSource`, WebGPU is an `Async` backend. *Lens verdict: the reference already compiles
  here; the env-free `Pick` and no-unsafe `HostBuffer` were the fixes this lens forced.*

## 7. Bit-identity, and the adoption diff

**Preserved by construction + scoping.** The CPU backend's methods *are* the model
functions, so no reduction is reordered and no kernel rewritten — the bytes out equal the
bytes in; the only change is a method indirection. The `KVStore` is interface extraction
only, so the kvmmu evict-vs-never-saw witness is untouched. `CorrectnessClass` makes the
two-tier gate a typed, harness-enforced invariant so the scoping cannot rot.

The model-package **adoption** is now partially executable: `NewBackendSession` builds a
HAL-owned `KVStore` and routes the f32 per-token path through `Backend.RMSNorm`, `MatMul`,
`RoPE`, `Attention`, `SwiGLU`, `AddInPlace`, and `Argmax`-compatible logits. The exactness
gate is `TestHALSessionMatchesLegacyCPUReference`: prefill, decode, and greedy generation
match the legacy path byte-for-byte under `cpu-ref`.

What remains is the production adoption diff: collapse `tokenHidden`/`tokenHiddenQ` and
the batched prefill/decode paths into one loop taking a `Backend`; the f32-vs-Q8 choice
becomes the weight `Tensor`'s `Dtype` (resolved from `Session.Quant`), not a `bool` branch;
and `cmd/modelbench -backend <non-reference> -require-non-reference` records real backend
evidence. The existing R2/R14/oracle tests in `internal/model` remain the equivalence proof
for the reference path — they must stay max|Δ|=0, argmax-exact. Run the suite via WSL
(`.\fak\test.ps1`) for full verification on Windows when native WDAC policy flakes unsigned test
binaries on this host.
