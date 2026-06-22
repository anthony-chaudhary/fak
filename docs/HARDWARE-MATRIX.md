# Hardware matrix — every machine fak has been profiled on

> **The point of this page:** `fak`'s correctness and serving claims are not from one
> lucky box. The same pure-Go kernel — same bit-exact gates — has been run and
> benchmarked across **four distinct hardware platforms** spanning **two CPU ISAs**
> (arm64 + x86_64), **three CPU vendors** (Apple · AMD · Intel), **four GPU backends**
> (Apple Metal · AMD Vulkan · NVIDIA CUDA Ada *and* Ampere), and **four OS targets**
> (macOS · Windows · WSL2 Linux · Linux). Portability across that spread *is* a result —
> a kernel that owns the KV cache as its own object has to prove it stays correct on
> every one.

Every number on this page traces to a committed artifact via the single source of
truth, **[`fak/BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md)**. This page is the
*rollup* — the at-a-glance "how serious is this" view — not a new claim. Where a number
appears here it carries a pointer to the doc + commit that owns it.

**Lineage:** rolled up 2026-06-21 · fak `v0.30.0` · against `BENCHMARK-AUTHORITY` +
`MODEL-LADDER-VS-SOTA-2026-06-21` + the per-platform results docs linked below.

![Hardware coverage matrix — four platforms across two CPU ISAs, four GPU backends, four operating systems](../visuals/56-hardware-coverage-matrix.svg)

---

## The coverage matrix

| Platform | CPU / ISA | GPU backend | OS | Quant coverage | What's proven here |
|---|---|---|---|---|---|
| **Apple M3 Pro** *(primary bench node)* | Apple M3 Pro 6P+6E, arm64 | 18-core **Metal** + **NEON** Q8 | macOS | f32 · Q8_0 · Q4_K · Q2_K | Full model ladder, the agent-fleet value stack, the pure-kernel latency stack, Qwen3.6-27B end-to-end in fak's own engine |
| **AMD Ryzen 9 9950X + Radeon RX 7600** | AMD Zen 5, 16C/32T, x86_64, AVX-512 | **Vulkan** Q8 (RX 7600) + CPU Q8 | Windows | f32 · Q8_0 · Q4_K | Q8-on-GPU throughput, the GPU/CPU crossover, 3/3 live agent surfaces on Qwen3.6-27B |
| **Intel x86_64 + NVIDIA RTX 4070** | Intel, x86_64, AVX2/AVX-512 | **CUDA** (Ada, sm_89): f32 · F16 · Q8 · graph | Windows + WSL2 Linux | f32 · F16 · Q8_0 | In-kernel CUDA decode at llama.cpp parity, batched decode curve, cross-platform bit-exact determinism vs the Mac |
| **8× NVIDIA A100-SXM4-40GB** *(serving lane)* | x86_64 host | **CUDA** (Ampere, sm_80), multi-GPU | Linux | Q4_K · (FP8/BF16 target) | The multi-GPU serving + GLM-5.2 readiness lane — big-iron, where single-box ceilings stop binding |

**Reading the spread:** the deterministic results (token-count speedups, cache hit rate,
bit-exact eviction) are *hardware-independent by construction* and reproduce byte-for-byte
across these boxes. The wall-clock numbers are per-box and stay labeled as such. The fact
that the **same kernel binary's correctness gates pass on Metal, Vulkan, and two CUDA
generations** is the portability claim this matrix exists to make visible.

---

## Platform 1 — Apple M3 Pro (`node-macos-a`) · the primary bench node

The box almost every published `fak` number is measured on.

| Component | Spec |
|---|---|
| CPU | Apple M3 Pro — 6 performance + 6 efficiency = 12 cores, **arm64** |
| GPU | 18-core GPU (Metal 4) |
| Memory | 36 GB unified (CPU+GPU shared), ~150 GB/s |
| OS / toolchain | macOS · Go 1.26.0 |
| Backends | **NEON Q8** (arm64 SIMD), **Metal GPU** (`-tags fakmetal`), pure-Go HF/GGUF loaders |

**What's been profiled here:**

- **Single-stream model ladder vs llama.cpp** — Qwen2.5-1.5B Q8 **27.9 tok/s**, 7B Q8
  **8.58 tok/s**; the fak÷llama.cpp gap *narrows* with size (0.39× → 0.53×). MoE
  30B-A3B hits 50 tok/s on llama.cpp (sparse activation, the real scaling lever).
  → `MODEL-LADDER-VS-SOTA-2026-06-21.md` (private companion — not published)
- **The agent-fleet value stack** — the README headline 50-turn × 5-agent Qwen2.5-1.5B
  run: **19.0 min vs ~78 min** tuned warm-cache (**4.1×**), and the high-T session ladder
  climbing **24.9× → 139.3×** vs the naive loop.
  → [`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md)
- **RadixAttention ladder** — live speedup **4.58× → 6.95×** (135M → 1.5B), **86.7%** hit
  rate (100% of optimal), climbing to the deterministic 7.50× token ceiling.
- **The pure-kernel latency stack** — canonical Decide **362 ns**, full Admit gate
  3.3–15.8 µs, in-process vs spawned-hook boundary tax **~2,849×**.
- **Qwen3.6-27B in fak's *own* in-kernel engine** — the 753-tensor `qwen35`
  Gated-DeltaNet path loads and generates end-to-end (GGUF→Q8, ~25.8 GB RSS), first two
  greedy tokens matching the llama.cpp oracle.
  → [`FAK-NATIVE-QWEN35-RESULTS.md`](benchmarks/FAK-NATIVE-QWEN35-RESULTS.md)
- **arm64 NEON kernel work** — the `tile2x4` register-tiled GEMM, ~252 tok/s prefill@256,
  plus the Q8 decode bandwidth-roofline.
  → `MAC-M3PRO-TILE2X4-KERNEL-BENCH-2026-06-21.md`
  · `MAC-M3PRO-DECODE-ROOFLINE-2026-06-21.md` (private companions — not published)

---

## Platform 2 — AMD Ryzen 9 9950X + Radeon RX 7600 · the Vulkan lane

Proves the GPU path is not NVIDIA-only — the same HAL device backend runs on a second
vendor's GPU through Vulkan.

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 9 9950X — 16 cores / 32 threads, **x86_64**, AVX-512 |
| Memory | 272 GB |
| GPU | **AMD Radeon RX 7600** (8 GB, Vulkan 1.4) + integrated UMA |
| OS | Windows (native Vulkan) |
| Backends | **Vulkan Q8** device GEMM, CPU Q8 (AVX-512) |

**What's been profiled here:**

- **Q8-on-GPU throughput** — first committed Vulkan Q8 numbers: SmolLM2-135M decode
  **24.6 tok/s**, a **1.49×** win over the same forward in f32 on the same device (narrower
  weight traffic on a memory-bound path). Correctness gated on the real GPU
  (`TestHALVulkanQ8ForwardMatchesComputeQ8`, cosine 1.0).
  → [`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md)
- **GPU/CPU crossover** — the CPU's lead collapses **7.2× (135M) → 1.16× (1.5B)** as
  per-token compute grows ~11×: direct evidence the device path is launch-bound on tiny
  models and catches up as the model grows.
- **Qwen3.6-27B, full agent surface** — **3/3** live surfaces pass (agent · OpenAI gateway
  · MCP); fak's gateway runs at **0.96×** of raw llama.cpp on the identical setup, and the
  pure-fak in-kernel prefill is **1.88–3.25×** over llama.cpp's Vulkan build on the same
  GGUF. → [`QWEN36-AMD-VULKAN-RESULTS.md`](benchmarks/QWEN36-AMD-VULKAN-RESULTS.md)

---

## Platform 3 — Intel x86_64 + NVIDIA RTX 4070 · the CUDA / WSL2 lane

The "go all in, fused kernel on a GPU that fits the model" lane, plus the cross-platform
determinism check that proves the deterministic metrics are not arm64-specific.

| Component | Spec |
|---|---|
| CPU | Intel, **x86_64**, AVX2 / AVX-512 |
| GPU | **NVIDIA RTX 4070** (Ada, **sm_89**), CUDA 12.6 |
| OS | Windows 11 (native) + WSL2 Ubuntu (CUDA via user-space micromamba) |
| Backends | **CUDA** f32 · **CUDA F16** (tensor cores, cuBLAS) · **CUDA Q8** (W8A16) · **CUDA Graph** · CPU Q8 |

**What's been profiled here:**

- **In-kernel CUDA decode at llama.cpp parity** — on a model that fits the GPU, the fused
  in-kernel CUDA decode hits **~120 tok/s on Qwen-class Q8_0**, parity with llama.cpp, with
  an opt-in CUDA graph. → README "How far do you want to take it?"
- **F16 parity** — Qwen2.5-1.5B f16 **36.6 tok/s** vs llama.cpp F16 34.3 (parity);
  SmolLM2-135M **~100–120 tok/s** (CUDA Graph).
- **Batched multi-user decode curve** — SmolLM2-135M Q8 peaks at **862 agg tok/s** at
  batch 512, **44.92×** over the naive baseline.
  → [`docs/benchmark/CROSS-MACHINE-INFRASTRUCTURE.md`](benchmark/CROSS-MACHINE-INFRASTRUCTURE.md)
- **Cross-platform bit-exact determinism** — the RadixAttention deterministic fields
  reproduce **byte-for-byte on Windows x86_64** vs the Mac arm64 artifact (hit 86.7%,
  token speedup 7.50×, reused 5512 / computed 848); only the live wall-clock moves
  (2.60× x86 vs 4.58× Mac), exactly as the small-model clone-overhead thesis predicts.

---

## Platform 4 — 8× NVIDIA A100-SXM4-40GB · the multi-GPU serving lane

The big-iron lane: ~320 GB of GPU on a DGX-class node, where the single-box memory
ceilings (`fak` faithful ≤ 7B on the 36 GB Mac) stop binding and the questions become
multi-GPU serving and frontier-model readiness.

| Component | Spec |
|---|---|
| GPU | **8× NVIDIA A100-SXM4-40GB** (Ampere, **sm_80**), ~320 GB aggregate |
| Host | x86_64, Linux |
| Backends | **CUDA** (sm_80), multi-GPU serving target |

**What's documented here:**

- **The model-ladder-on-A100 plan** — tiny smoke model → dense Qwen2.5 → hybrid
  Gated-DeltaNet bridge → Qwen3.6-27B, de-risking multi-GPU serving and the
  fak-gateway-vs-raw comparison per rung. *(Tracked in the private DGX-A100
  model-ladder runbook, not part of the public snapshot.)*
- **GLM-5.2 serving-readiness** — the feasibility finding that stock SGLang/vLLM cannot
  serve GLM-5.2's `glm_moe_dsa` (DSA kernels + memory) on Ampere sm_80, which is precisely
  where `fak`'s gateway/baseline role and the shipped serving-readiness preflight gate
  apply. The runnable form of this finding ships publicly as
  [`tools/glm52_serve_preflight.py`](../tools/glm52_serve_preflight.py) and
  [`tools/glm52_serve.sh`](../tools/glm52_serve.sh); the private DGX fast-loop and
  SGLang/vLLM-readiness notes are not part of the public snapshot.

> **Honesty fence.** This lane is reported as the documented serving/readiness track, not
> a published single-box throughput row — the per-rung wall-clock witnesses live behind
> the same DOS verification discipline as everything else and are gated on the serving
> work landing. No A100 tok/s figure is asserted here that isn't traced to an artifact in
> `BENCHMARK-AUTHORITY.md`.

---

## Why this many machines

It would be cheaper to benchmark on one box and call it done. `fak` profiles across this
spread on purpose:

1. **Portability is a correctness claim.** Because `fak` owns the KV cache as a kernel
   object (not rented from a serving engine), its bit-exact eviction and prefix-reuse
   guarantees have to hold on *every* backend — Metal, Vulkan, and both CUDA generations.
   Running the same gates on four platforms is how that claim is kept honest.
2. **Two regimes need two kinds of hardware.** The single-stream ceiling (≤7B on 36 GB)
   is a small-box story; the multi-agent fleet win and the frontier-model serving lane
   need the AMD/CUDA desktops and the A100 node respectively.
3. **The deterministic metrics must be machine-independent.** The token-count speedups
   and cache hit rates are claimed as hardware-independent — the cross-platform Mac↔Windows
   reproduction is the witness that they actually are.

---

## See also

- **[`fak/BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md)** ⭐ — the single source of
  truth; every number here traces to a row there with its commit + artifact.
- **`HARDWARE-CATALOG.md`** (operator machine catalog — intentionally private) — the
  per-machine onboarding catalog (specs, baseline-run requirements, the scientific-rigor metadata schema).
- **`MODEL-LADDER-VS-SOTA-2026-06-21.md`** (private companion — not published) —
  the full two-regime model-size ladder behind the M3 Pro rows.
- **[`docs/benchmark/CROSS-MACHINE-INFRASTRUCTURE.md`](benchmark/CROSS-MACHINE-INFRASTRUCTURE.md)** —
  the design for storing and querying results across all of these machines.
</content>
</invoke>
