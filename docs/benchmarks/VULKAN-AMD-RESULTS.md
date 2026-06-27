---
title: "Vulkan AMD Backend on Radeon RX 7600"
description: "Witnesses fak's Vulkan compute backend reaching numerical parity on a real AMD RX 7600 while still ~60x slower than llama.cpp from per-dispatch overhead."
---

# VULKAN-AMD-RESULTS — the AMD GPU backend, witnessed on a real Radeon RX 7600

> An AMD GPU compute backend for the in-kernel model, behind the existing `compute.Backend`
> seam, targeting **Vulkan compute** (SPIR-V shaders). Every number below is measured on the
> actual hardware on this box — **AMD Radeon RX 7600** (RDNA3 / gfx1102, 8 GB), reached
> through the native Windows Vulkan loader. Nothing here is `[SIMULATED]`.
>
> **Bottom line, stated plainly:** the backend reaches **numerical parity** with the CPU
> reference on real AMD silicon (argmax-exact greedy decode, prefill-logit cosine = 1.0). It
> does **not** reach **throughput parity** with llama.cpp — it is still ~60× slower — and the
> cause is a known, fixable architectural limitation (per-dispatch CPU/driver overhead),
> not a numeric defect. This doc records both halves honestly.

## What was built

A new `//go:build vulkan` backend (`internal/compute/vulkan.go` + `vulkan_shim.cpp` +
`shaders/*.comp`), a structural mirror of the CUDA backend. It registers as `"vulkan"`
(`Approx` class), so `cpu-ref` stays the `Reference` Default and nothing runs on the GPU
unless explicitly selected (`FAK_BACKEND=vulkan` / `-backend vulkan`). The whole forward pass
(`model.Session.tokenHAL`) runs through it unchanged — adding the backend was a registration,
never an edit to the forward loop, exactly as the HAL was designed for.

Toolchain (installed via winget this session): Vulkan SDK 1.4.350.0 (`glslc`, headers,
loader) + MinGW-w64 g++ (the cgo C++ compiler — native clang here has no C++ STL). Builds and
runs NATIVELY on Windows (the GPU is unreachable from WSL — no `/dev/dri` passthrough), unlike
the rest of fak's suite. See `internal/compute/build_vulkan.ps1`.

## Rung 1 — numerical parity (PASS, witnessed on the RX 7600)

| Witness | Gate | Result |
|---|---|---|
| 7 op-level kernels vs cpu-ref (`go test -tags vulkan -run Vulkan ./internal/compute/`) | Approx: cosine + small max\|Δ\|; argmax EXACT | **PASS** (all 7) |
| Full SmolLM2-135M forward pass (`-run HALVulkan ./internal/model/`) | greedy argmax-exact + prefill cosine ≥ 0.999 | **PASS** — argmax-exact / 10 tokens, **prefill cosine = 1.00000000** |

The op suite (`vulkan_test.go`) isolates each primitive — matmul, RMSNorm, RoPE, SwiGLU,
add/add_bias, the fused causal-GQA attention (4-phase softmax), and the first-max argmax — so
a shader bug surfaces at the op, not as a mysterious end-to-end drift. `argmax` is held to
EXACT equality (the cpuref first-max tie-break is reproduced bit-for-bit). The full forward
pass through all 30 layers on the GPU is numerically indistinguishable from the reference
(cosine 1.0), which is the load-bearing correctness claim: **the model runs correctly on AMD.**

The backend tier string self-reports `discrete:AMD Radeon RX 7600`, so a software rasterizer
(lavapipe) or the integrated GPU can never silently masquerade as the discrete card.

## Rung 2 — throughput parity vs llama.cpp (NOT met; honest numbers)

Measured on the RX 7600, SmolLM2-135M, f32, via `cmd/modelbench -backend vulkan`:

| Engine | Precision | Decode ms/tok | Decode tok/s | Source |
|---|---|---:|---:|---|
| **fak Vulkan (RX 7600), + fused SwiGLU·down-proj·residual (FFN-tail fusion)** | f32 | **394** | **2.5** | `experiments/parity/vulkan-rx7600-fused-smollm2-135m.json` |
| fak Vulkan (RX 7600), batched + staging + descriptor-set pool + stack descriptor writes + in-place HAL RoPE + fused residual matmul-add | f32 | 420 | 2.4 | `experiments/parity/vulkan-rx7600-inplace-rope-matmuladd-smollm2-135m.json` |
| fak Vulkan (RX 7600), batched + staging + descriptor-set pool + stack descriptor writes + in-place HAL RoPE | f32 | 449 | 2.2 | `experiments/parity/vulkan-rx7600-inplace-rope-smollm2-135m.json` |
| fak Vulkan (RX 7600), batched + staging + descriptor-set pool + stack descriptor writes | f32 | 538 | 1.9 | `experiments/parity/vulkan-rx7600-stackdesc-smollm2-135m.json` |
| fak Vulkan (RX 7600), batched + staging + descriptor-set pool | f32 | 588 | 1.7 | `experiments/parity/vulkan-rx7600-descpool-smollm2-135m.json` |
| fak Vulkan (RX 7600), batched + persistent staging | f32 | 571–908 | 1.1–1.8 | `experiments/parity/vulkan-rx7600-staging-smollm2-135m.json` |
| fak Vulkan (RX 7600), unbatched | f32 | 325–804 | 1.2–3.1 | measured (decode-steps 16) |
| fak CPU optimized (Zen5) | f32 | 17.6 | 56.9 | `MODEL-BASELINE-RESULTS.md` |
| llama.cpp (CPU, this box) | Q8_0 | 6.91 | ~145 | `comparison.json` |

**The trend now points at the real lever: op-count, via kernel FUSION.** The ladder above is
monotonic — every op removed from the per-token path helps, and fusing the FFN tail (SwiGLU +
down-proj matmul + residual add → ONE dispatch) is the latest step (2.4 → 2.5 tok/s). This is
the opposite of the earlier "per-op CPU overhead is a fixed floor" read: it IS fixed *per op*,
so the way through is fewer, fatter ops. A decode token is still ~17 dispatches/layer; the
remaining fusions (Q/K/V projections → one, RMSNorm+matmul, attention+O-proj) each remove
more. Extrapolating the ~per-op cost, fusing to ~4–5 ops/layer projects to ~8–10 tok/s —
better, but still NOT parity.

Honest status: the current best is **2.5 tok/s — ~23× slower than fak's own CPU f32 path and
~58× slower than llama.cpp CPU**, climbing as fusion lands. Throughput parity is NOT reached
and is not imminent: closing 58× needs the full fusion set PLUS Q8 device GEMM PLUS an async
multi-buffered submission model — a sustained multi-session effort, not a single change. The
framing must not pretend 2.5 tok/s is parity; it isn't. But the path is now empirically
validated, not speculative.

A like-for-like llama.cpp *Vulkan-on-this-GPU* number is not captured here: the installed
`llama-cpp-python` 0.3.30 is a CPU-only build (`llama_supports_gpu_offload() == False`), and
no Vulkan-enabled llama.cpp CLI is exposed on this box (LM Studio bundles one internally). So
the honest reference is llama.cpp's CPU number, against which we are still ~60× behind — a
Vulkan llama.cpp on this card would be faster still, widening the gap.

### Why it is slow (diagnosed by trying the obvious fixes — they didn't close it)

The natural first hypothesis was **one `vkQueueSubmit`+fence per primitive op** (~300/token).
Two fixes were implemented and measured:

1. **Command-buffer batching** (`fvk_batch_begin/flush`): record a whole token's forward pass
   into ONE command buffer with compute→compute barriers, submit once. Shipped (`9fc670f`),
   correctness preserved (cosine still 1.0). **Throughput barely moved** (decode ~1.7 tok/s).
2. **Shared-memory matmul shader**: stage the input vector once per workgroup instead of
   re-reading it per output. Shipped (`76fd1dc`), correct. **Also no meaningful change.**

That both levers failed to move decode is itself the finding: at **~2 ms/op** the floor is
**fixed per-op CPU↔GPU overhead**, not submit count or compute efficiency.

> **This ~2 ms/op is the Vulkan/native-Windows per-op tax — it is backend- and OS-dependent.**
> The CUDA-on-WSL path measures a far lower **~0.07 ms/op** host-launch tax
> (`GPU-QWEN-RESULTS.md` §4), so this Vulkan floor is **~28× higher** than CUDA-on-WSL's. A
> reader projecting GPU decode on another box must apply the per-op tax for *that* backend —
> ~2 ms/op is Vulkan/native, 0.07 ms/op is CUDA-on-WSL — not assume one number crosses backends.

A follow-on cgo
cleanup now removes the fresh host-staging allocation/map/free cycle by reusing one mapped
HOST_VISIBLE transfer buffer; correctness is witnessed on the RX 7600, and the restored
SmolLM2 export measures decode at **570.8 ms/tok**. A follow-on descriptor-set pool removes
the per-dispatch `vkAllocateDescriptorSets`/`vkFreeDescriptorSets` churn and improves short
prefill (P16: **11.7s → 9.0s**, P64: **51.8s → 37.7s**), but decode remains **588.2 ms/tok**.
A final cgo cleanup moves per-dispatch descriptor write structs from heap `std::vector`s to
fixed stack arrays; it improves the same short run again (P16: **9.0s → 8.5s**, P64:
**37.7s → 33.3s**, P256: **167.8s → 145.8s**, decode: **588.2 → 538.1 ms/tok**). The next
safe dispatch-count cleanup avoids value-semantics RoPE copies where the HAL can prove the
source is disposable: Q rotates in place, and Kraw is copied into the KV cache before being
rotated in place and appended as the post-RoPE K. That removes two device-to-device copies
and two transient allocations per layer, moving decode **538.1 → 449.1 ms/tok** on the same
single-rep benchmark. Prefill remains mixed/noisy (P16 regressed in this run; P64 improved
versus the prior stack-descriptor run; P256 is roughly flat), so the honest claim is a
decode-path improvement, not a general throughput-parity fix. A further decode-path cleanup
adds a `matmul_add` kernel for the two residual projections per layer (`o_proj` and
`down_proj`): the projection accumulates directly into the residual tensor, removing the
temporary projection tensor and the following `AddInPlace` dispatch. It moves the same
single-rep run to **420.3 ms/tok** and improves prefill to **P16 8.5s, P64 28.9s, P256
114.4s**. The real remaining measured cost is still per-dispatch descriptor **updates/binds**
plus command recording across hundreds of ops/token. Closing the ~60× llama.cpp CPU gap
needs an architectural change, not another shader tweak:

1. **Pre-bake stable descriptor bindings / reduce dispatch count** instead of updating and
   binding one descriptor set per primitive op.
2. **One device allocation backing many tensors** (sub-buffer offsets); today a bucketed pool
   + `drainPool()`-on-pressure prevents `maxMemoryAllocationCount` exhaustion but the churn is real.
3. Quantized (Q8_0) device GEMM (device MatMul is f32-only today; the Go side refuses
   quantized weights with a clear message).

These are honestly-scoped follow-ups. The seam, correctness, device residency, weight cache
(`Session.halW`), per-token batching, persistent transfer staging, descriptor-set pooling,
stack-allocated descriptor writes, safe in-place HAL RoPE, and fused residual matmul-add are
all in place; what remains is removing the per-dispatch CPU overhead — genuinely a larger
effort than one session, not a defect.

## Honest status line

- **AMD GPU support: real and witnessed.** The model executes correctly on a Radeon RX 7600
  through Vulkan, behind the typed backend seam, gated so it never runs by accident.
- **llama.cpp parity: numerical yes, throughput no.** ~60× off, with a named, fixable cause.
  Claiming throughput parity here would be false; this doc exists so no one does.

Reproduce: `pwsh fak/internal/compute/build_vulkan.ps1 test` (op + forward-pass witnesses);
the benchmark is `modelbench -backend vulkan -dir internal/model/.cache/smollm2-135m`.

## 2026-06-20 — the Q8 decode path now gets the f32 path's kernel fusion (dispatch-count lever)

> Context: the "most common use case" for a GPU-served LLM is single-user, batch-1 decode of a
> small **quantized** model on a consumer GPU. On this box that is fak-Vulkan vs llama.cpp-Vulkan,
> Q8, on the RX 7600. Measured parity gap before this change (Qwen2.5-1.5B-Instruct Q8_0):
> **fak 12.4 tok/s vs llama.cpp Vulkan 130.2 tok/s (~10.5×).** Per the diagnosis above the
> Vulkan bottleneck is **per-dispatch CPU overhead**, so the lever is **fewer dispatches**.

**Root cause closed.** The HAL's three fused decode kernels — `RMSNormMatMul3` (RMSNorm+Q/K/V),
`RMSNormMatMul2` (RMSNorm+gate/up), `SwiGLUMatMulAddInPlace` (SwiGLU+down+residual) — were each
guarded `&& !useQ8Weights` (`internal/model/hal.go`), so a quantized model fell back to the
**unfused** path: ~**12 dispatches/layer** vs the f32 path's ~**7**. The Vulkan backend's Q8
fused kernels (`rmsnorm_q8_matmul2/3.comp`, `swiglu_q8_matmul_add.comp`) now exist and the HAL
guards are dropped (the fused branches pass `matWeightHAL`, so they get the Q8 tensor), bringing
the Q8 per-layer dispatch count down to the f32 ~7/layer.

**A real defect the new tests caught.** `fvk_rmsnorm_q8_matmul3_f32` binds **11** storage
buffers (3 Q8 weight code+scale pairs + X + NormW + Q/K/V), but the shim's `MAX_DISPATCH_BUFS`
was **10** — so that dispatch was *silently skipped* (`dispatch skipped; kernel has 11 buffers,
max 10`), zeroing the Q/K/V output (cosine 0.0). Raised the cap to 12; well under the device's
storage-buffer binding limit and the 32768-descriptor pool.

**Witnessed on the RX 7600 (cosine ≥ 0.9999, max|Δ| ≤ 1e-2; argmax/cosine-exact e2e):**

| Witness | Result | Test |
|---|---|---|
| Q8 RMSNorm+Q/K/V fused, incl. wide in=3072/8960 | **PASS** | `TestVulkanQ8RMSNormMatMul3Approx` |
| Q8 RMSNorm+gate/up fused, incl. wide in=3072/8960 | **PASS** | `TestVulkanQ8RMSNormMatMul2Approx` |
| Q8 SwiGLU+down+residual fused, incl. in=8960 | **PASS** | `TestVulkanQ8SwiGLUMatMulAddInPlaceApprox` |
| **Q8 real-model forward through the fused path** | **prefill cosine 1.0, step cosine 1.0** | `TestHALVulkanQ8ForwardMatchesComputeF32` (`…Q8ForwardMatchesComputeQ8`) |
| f32 path unchanged (no regression) | **argmax-exact /10, prefill cosine 1.0** | `TestHALVulkanForwardMatchesNative` |

**Throughput re-measurement is pending a free GPU:** a peer fleet session is holding the card
(`llama-server` serving Qwen3.6-27B-Q4_K_M, `--n-gpu-layers 20 --fit on`), so a fresh
`modelbench` OOMs. The dispatch reduction (~12 → ~7/layer) is a structural, countable fact and
per the per-dispatch-overhead diagnosis should raise decode tok/s materially from the 12.4
baseline; the measured number will be filled in here once the GPU frees. This lands the single
biggest lever — it does **not** by itself reach 130 tok/s; the residual is per-dispatch
descriptor update/bind overhead (pre-baked descriptor sets are the additive follow-up).

Reproduce: `go test -tags vulkan -run 'VulkanQ8RMSNorm|VulkanQ8SwiGLU|HALVulkan' ./internal/compute/ ./internal/model/`;
throughput: `go run -tags vulkan ./cmd/modelbench -lean -gguf <Qwen2.5-1.5B…Q8_0.gguf> -backend vulkan -decode-steps 32 -decode-reps 3`
vs `llama-bench -m <same.gguf> -p 16 -n 64 -r 2`.
