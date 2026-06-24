---
title: "Native 753B GLM-5.2 Serving on fak: Staged glm_moe_dsa Plan"
description: "The dependency-ordered, multi-month plan to serve GLM-5.2 753B (glm_moe_dsa) natively on fak: GGUF parse, quantized device GEMM, multi-GPU TP, and CPU offload."
---

# Native 753B serving — staged plan (GLM-5.2 `glm_moe_dsa`)

_2026-06-23._ The track to make fak serve the **real** GLM-5.2 753B model natively,
on the pure fak engine, end to end. This note records the plan and the first slice
that landed today; it is a living map, not a finished product. The track is
multi-month and will not finish in one session.

## Where we start

fak already runs GLM-5.2's forward **bit-exact on GPU kernels** (cosine 1.0) and
its real-oracle `Generate` matches HF greedy — but only at small scale and in f32.
The wall between that and serving the actual 753B model is four pillars:

1. **GGUF `glm_moe_dsa` config + weight parse** — read a real GLM-5.2 checkpoint.
2. **Quantized (non-f32) device GEMM** — run the matmuls at Q4_K/Q6_K/Q8_0, not f32.
3. **Multi-GPU NCCL / tensor-parallel sharding** — the model does not fit one GPU.
4. **CPU-offload serving** — experts that dwarf VRAM live in host RAM (and beyond).

A survey of the existing code (four parallel readers + a synthesis pass) found that
the substrate is much further along than greenfield, and that the honest gaps are
specific. The good and the missing, per pillar:

### Pillar 1 — GGUF parse
- **Have:** a fully generic GGUF v3 reader (metadata KV + tensor directory, split
  shards, mmap, load profiler) and a complete dequant suite (Q4_K/Q5_K/Q6_K/Q8_0/
  MXFP4/…). `model.Config` already carries every MoE + MLA + DSA field. The native
  `glm_dsa.go` forward already consumes the canonical names. `isGLMMoeDsa()` already
  fires off `ModelType == "glm_moe_dsa"`.
- **Missing (before today):** `(*File).Config` had only qwen35/gemma4 branches and
  read **none** of the MoE/DSA metadata, so a `glm_moe_dsa` GGUF loaded silently as
  dense and wrong. `CanonicalTensorNameArch` maps no MoE expert tensors and no DSA
  tensors. No GGUF batched-expert splitter.

### Pillar 2 — quantized device GEMM
- **Have:** a real CUDA dequant-fused Q4_K GEMV/GEMM (`k_q4k_gemm`) and a Q8_0 kernel,
  graded against a bit-exact cpu-ref Reference; `metalgemm` has a Q4_K kernel.
- **Missing:** Vulkan is Q8_0-only (no Q4_K on AMD); Metal's `compute.Backend` is
  f32-only at the HAL; the CUDA Q4_K/Q8_0 cosines have **never been run to a recorded
  number** on a GPU node; no end-to-end mixed-precision full-model forward witness.

### Pillar 3 — multi-GPU
- **Have:** a real, bit-exact in-process tensor-parallel decomposition
  (`tensor_parallel.go`) over a `Collective` seam; `LocalCollective` (host-slice
  all-gather/all-reduce) and a `compute.CollectiveBackend` HAL contract (AllReduce/
  ReduceScatter/AllToAll) — both proven, but **cpu-ref only and not connected**.
- **Missing:** any real cross-device/cross-process collective (NCCL/RCCL/TCP). The
  existing head-parallel TP is the **wrong** decomposition for GLM's MLA + shared
  latent KV; `forwardTPSupported` fails closed on `glm_moe_dsa`, MoE, and
  quant-resident weights. MLA-aware TP + expert-parallel placement is mostly unbuilt.

### Pillar 4 — CPU offload
- **Have:** a genuine, well-tested `--n-cpu-moe` analogue (`splitKernel` /
  `CPUOffloadExperts`) that runs expert GEMMs on the host CPU kernel while dense +
  router + DSA-attend run on the device; a CUDA hybrid witness; whole-**model** LRU
  tiering (`residency.Manager`); memory-lean quant-on-load.
- **Missing:** this is **compute-placement, not tensor paging** — the expert weight
  stays host-resident and never moves to VRAM. No per-weight VRAM budget/ring, no
  async/pinned H2D staging, no on-demand page-in, no NVMe tier.

## First slice landed today (P1, foundationOrder 1)

**`applyGLMMoeDsaConfig` in `internal/ggufload/gguf.go`** — a `glm_moe_dsa` branch in
`(*File).Config` that reads the MoE + MLA + DSA-indexer metadata KV into the existing
`model.Config` fields, mirroring `applyGemma4Config`. Plus a `(*File).Bool` scalar
accessor and a table-driven golden test (`gguf_glm_test.go`).

- **Maps:** `expert_count`→NumExperts, `expert_used_count`→NumExpertsPerTok,
  `expert_feed_forward_length`→MoEIntermediateSize, `expert_shared_count`→NSharedExperts,
  `expert_shared_feed_forward_length`→SharedIntermediateSize,
  `leading_dense_block_count`→FirstKDenseReplace, `expert_group_count`→NGroup,
  `expert_group_used_count`→TopKGroup, `expert_weights_scale`→RoutedScalingFactor,
  `expert_weights_norm`→NormTopKProb; `attention.q_lora_rank`/`kv_lora_rank`→QLoraRank/
  KVLoraRank; the MLA head dims (explicit `qk_nope`/`qk_rope`/`v_head_dim`, else the
  deepseek2 `key_length`/`value_length`/`rope.dimension_count` derivation); and the DSA
  indexer scalars `index_n_heads`/`index_head_dim`/`index_topk`/`indexer_types`.
- **Test:** a `glm_moe_dsa` GGUF header round-trips into a `model.Config` whose
  `ModelType`/`IsMoE()` and every MoE/MLA/DSA scalar match what was written — both with
  explicit head-dim keys and via the deepseek2 derivation fallback. Green under WSL.
- **Why this first:** it is the strict prerequisite for both downstream Pillar-1 slices
  (you cannot split E experts or size the indexer until NumExperts/IndexNHeads are read)
  and for Pillars 2/3/4 on the real model (they all need a correctly-sized, MoE-true
  config). It is the smallest of the four pillar first-slices — pure metadata parsing,
  no kernel, no GPU — and duplicates nothing.
- **Caveat (read this):** no real GLM-5.2 GGUF exists on disk and upstream llama.cpp may
  not yet ship a `glm_moe_dsa` converter, so the **exact key spellings are PROVISIONAL**.
  They are modeled on llama.cpp's `deepseek2.*` convention (GLM-DSA attention IS DeepSeek
  MLA + an indexer) and pinned as one named-constant block, so the closing follow-on — a
  golden against a real GGUF header — only re-pins that block.

## Staged plan (dependency-ordered)

Each milestone is a shippable green step with a one-line acceptance test. "ships now"
means it needs nothing downstream and can land before the heavy pillars.

| Phase | Milestone | Acceptance | Depends on |
|---|---|---|---|
| **P1 Config parse** ✅ | `applyGLMMoeDsaConfig` reads MoE+MLA+DSA metadata into Config | `go test ./internal/ggufload -run TestGLMMoeDsaConfig` green | — (landed) |
| P1 Tensor names (1:1 ✅) | `CanonicalTensorNameArch` maps the MLA/indexer attn tensors, `ffn_gate_inp` router, `exp_probs_b`, `ffn_*_shexp` — **shipped `b1c0f04`**; the batched `ffn_*_exps` stay unmapped (fail loud) for the splitter | the 1:1 names resolve (golden `TestGLMMoeDsaCanonicalTensorNames`); `ffn_*_exps` still "no canonical mapping" until the splitter | P1 Config parse |
| **P1 Expert splitter ✅** | a `[E,out,in]` `ffn_*_exps` blob splits into per-expert canonical 2-D tensors on load (wired into both loader loops) — **shipped `f673dca`** | synthetic `*_exps` loads bit-equal to manual slicing ✅ + an e2e GGUF→`F32Tensors`→per-expert gate | P1 Tensor names (1:1) |
| **P1 E2E ✅ (load + forward + oracle)** | a complete `glm_moe_dsa` GGUF resolves its full tensor set + splits experts (**`2e632ef`**), `LoadModel`→`Forward` RUNS the native glm_dsa forward (**`d3e1c1d`**), AND it forwards **BIT-FOR-BIT == a model built directly from the same canonical tensors** (**`431b52a`**, `...ForwardMatchesReference`) — so the round-trip is correctness-faithful, not just structural. **P1 is COMPLETE**: fak loads AND forwards glm_moe_dsa from GGUF, proven. | GGUF-loaded argmax + logits == reference ✅ (bit-exact) | P1 Expert splitter |
| P2 Vulkan Q4_K GEMM | Vulkan `compute.Backend` gains a Q4_K dequant-fused GEMV→GEMM | Vulkan Q4_K MatMul vs cpu-ref at cosine floor + argmax-exact (AMD node) | ships now (kernel); real bytes need P1 |
| P2 Metal HAL + **CUDA witness (CUDA ✅)** | expose `metalgemm` Q4_K via the unified HAL (open); the `-tags cuda` Q8_0/Q4_K gates **run + recorded on an sm_80 node, 2026-06-24** | Metal Q4_K via HAL host-witnessed cosine (open); **CUDA recorded: Q8_0 `0.99999980`, Q4_K `1.00000000`, argmax-exact** — [witness](glm52-quant-device-gemm-on-gpu-witness.md) | P2 Vulkan Q4_K |
| P2 Full-model quant forward (Q8 ✅) | end-to-end **Q8** memory-lean `glm_moe_dsa` forward from GGUF — `LoadModelQuant`→`Forward`+`Session` decode RUN to finite logits (**shipped `0c2d74f`**, `...QuantLoadForwards`); the mixed-precision **Q4_K** lane + a faithful argmax-match (needs a non-degenerate fixture — Q8-vs-f32 is uncorrelated on random tiny weights, by design) remain | Q8 full-model forward finite ✅; Q4_K mixed-precision ◻ | P1 E2E load; P2 Vulkan Q4_K |
| **P3 Collective bridge ✅** | `BackendCollective`: `model.Collective` wrapping `compute.CollectiveBackend` (the NCCL plug-in seam) — **shipped `41017e3`** | `BackendCollective == LocalCollective` at max\|Δ\|=0 ✅; `ForwardTP` equal both ways (cpu-ref) ✅ | ships now (de-risks seam); real use needs P1+P2 |
| P3 MLA-aware TP + EP | an MLA-aware (not head-parallel) TP decomposition + expert-parallel placement; quant-aware sharding | `ForwardTP` sharding-invariant on a synthetic `glm_moe_dsa`+MoE quant model | P3 Collective bridge; P1 E2E; P2 Full-model |
| P3 Real cross-process NCCL | a non-cpu-ref `CollectiveBackend` (NCCL/RCCL or a TCP transport mirroring `pipeline_transport.go`) | a 2-GPU/2-process all-reduce of a device tensor matches cpu-ref — **only now may "multi-GPU" be claimed** | P3 Collective bridge; P3 MLA-aware TP |
| **P4 Device paging primitive ✅ (standalone)** | an upload→compute→free `pagedKernel` with an observable `pageIn` counter — **shipped `f54e01a`** (the first honest "paged to device on demand") | GEMM bit-equal to resident ✅; paged weight absent from `halW` ✅; `pageIn` counts each page-in ✅ — standalone primitive; halW-integration is P4 Async streaming | ships now (existing fixture); real win needs P2 |
| P4 Async expert streaming | per-weight VRAM ring + async/pinned H2D so host-resident experts stream per-layer; serve loop auto-sizes the split | a >VRAM `glm_moe_dsa` serves on the GPU node at a measured tok/s within budget | P4 paging primitive; P2 Full-model; P1 E2E |
| **Integration 753B serve** | real GGUF → mixed-precision device GEMM → multi-GPU TP/EP → CPU/NVMe-tiered offload, with a real-GGUF golden | native 753B `Generate` matches the real-oracle greedy at the agreed bar on the GPU server | P3 Real NCCL; P4 Async streaming; all P1/P2 |

## Risks / honesty ledger

- **Provisional GGUF keys (P1):** no real GLM-5.2 GGUF on disk; the exact metadata key
  spellings are the single most fragile assumption. Mitigated by the one-block named
  constants + a deferred real-header golden. Do not treat the parse as validated against
  a real checkpoint until that golden exists.
- **Q4_K bit-exact port hazard (P2):** the `getScaleMinK4` 6-bit packing + f16 d/dmin
  decode must be reproduced exactly in GLSL/SPIR-V; a wrong port silently collapses the
  cosine. The cpu-ref Reference catches it — it is the trap the CUDA port already hit.
- **Wrong TP decomposition (P3):** head-parallel TP is wrong for GLM's MLA shared latent
  KV; `forwardTPSupported` fails closed on `glm_moe_dsa`/MoE/quant. MLA-aware + EP +
  quant-aware sharding is the biggest multi-month unknown.
- **Claims discipline:** today's "CPU-offload" is compute-placement (the `--n-cpu-moe`
  equivalent), and all collectives are in-process. **Do not claim multi-GPU** until a
  non-cpu-ref `CollectiveBackend` exists. The `pageIn` primitive HAS landed (`f54e01a`,
  standalone, bit-equal to resident), so the upload→compute→free rung is real — but **do not
  claim a >VRAM serve** until it is wired into the live weight HAL with async streaming.
- **Scale gap:** every correctness witness is tiny-oracle and f32. (The CUDA quant
  device-GEMM cosines are now **recorded on sm_80 hardware** — 2026-06-24, Q8_0 `0.99999980`
  / Q4_K `1.00000000`, both argmax-exact, weight 3.56×/7.11× smaller; see
  [the witness](glm52-quant-device-gemm-on-gpu-witness.md) — but that is a single-GEMM rung,
  not the full model, and the correctness-first quant kernels are slower than f32 SGEMM in
  raw FLOPs.) The real cost is still multi-month 753B serving on the GPU server.

## Load-bearing existing code (reuse, don't duplicate)

`ggufload` generic reader + full dequant suite · `applyGemma4Config` (the template P1
mirrored) · `model.Config` MoE/MLA/DSA fields · `glm_dsa.go` native forward (the
downstream consumer that fixes the target canonical names) · CUDA `k_q4k_gemm`/`k_q8_gemm`
+ the cpu-ref Q4_K/Q8_0 Reference (the bit-exact grader for every device lane) ·
`tensor_parallel.go` TP decomposition · `collective.go` `CollectiveBackend` seam ·
`moe_offload.go` `splitKernel` · `pipeline_transport.go` `TCPTransport` (the cross-process
pattern to mirror for a real all-reduce).
