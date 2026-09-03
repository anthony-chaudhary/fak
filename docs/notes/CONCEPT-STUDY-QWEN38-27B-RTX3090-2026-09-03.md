---
title: "CONCEPT-STUDY: syv-ai/qwen38-27b-rtx3090 consumer 24GB GPU serving optimizations, speculative block drafting, and hybrid GDN state management"
description: "Exhaustive, pinned study of syv-ai/qwen38-27b-rtx3090: Lookup-Augmented Block Drafting (LABD, 381 tok/s decode), split-KV multi-query speculative attention, AutoRound negative-scale sign-folding for Marlin W4A8, 16-bit GDN recurrent state, and 40k draft vocab truncation on consumer 24 GB GPUs."
date: 2026-09-03
---

# CONCEPT-STUDY: syv-ai/qwen38-27b-rtx3090 consumer 24GB GPU serving optimizations, speculative block drafting, and hybrid GDN state management (2026-09-03)

**Verdict:** `syv-ai/qwen38-27b-rtx3090` is an exceptionally disciplined, empirical engineering tour de force designed to serve [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (a hybrid 48-layer Gated DeltaNet + 16-layer full-attention model) on a single consumer 24 GB GPU (GeForce RTX 3090 / RTX 4090) within a 250 W power envelope. It tackles the severe memory, bandwidth, and kernel underutilization cliffs inherent to running 27B hybrid models under tight VRAM margins. Its key contributions span:

1. **Lookup-Augmented Block Drafting (LABD) & n-gram Chains:** Extends block drafters (MTP, DFlash2) by searching prompt/dialog history for suffix occurrences; proposes verbatim continuations with point-mass distributions; dynamically scales verify blocks up to 16 tokens; and executes drafter-free n-gram chains during sustained verbatim copying, reaching **381 tok/s decode** on context-reproducing tasks (document quoting, structured code editing, diffing).
2. **Split-KV Speculative Verify Attention Kernel:** Solves the severe GPU underutilization of FlashAttention-2 during speculative multi-query verification ($q_{\text{len}} > 1$, where FA2 leaves 58 of 82 SMs idle on RTX 3090) by tiling query rows and parallelizing along the KV dimension across SM segments, dropping verify attention latency from 1.3 ms to 120 us at 16k context.
3. **Marlin W4A8 Negative-Scale Sign-Folding:** Identifies and repairs a silent numerical corruption defect in Marlin INT8 Tensor Core GEMMs where unsigned `uint16_t` interpretation of scales corrupts AutoRound symmetric exports (~50% negative scales), folding the sign into int4 weight nibbles ($s \to -s, q \to 16-q$) at load time without kernel recompiles.
4. **Empirical Draft Vocabulary Truncation:** Replaces the 248k full-vocabulary projection head in speculative drafters with a dense 40k subset derived from 5.4M tokens of empirical model generation counts (`draft_vocab_ids.json`), preserving 97.5% coverage (96% on code) while slashing draft head latency from 3 ms to 0.5–1 ms per token.
5. **16-Bit Recurrent State Storage & Concurrency Scaling:** Halves the resident recurrent state of Qwen3.8's 48 Gated DeltaNet layers from float32 (~150 MB/request) to float16 (~75 MB/request), lifting concurrency from 37 to 64 concurrent streams on 24 GB VRAM with perplexity unchanged.
6. **Sort-Free Small-k Sampling & Multi-Block Row Softmax:** Replaces full $O(V \log V)$ vocabulary sorting and map allocation with bounded $O(V \log k)$ selection for $k \le 64$, and replaces single-block 248k row softmax with a 64-chunk multi-block Triton kernel, cutting sampling latency by 6x–14x.
7. **Hybrid Prefix Caching Checkpoint Order Preservation:** Prevents periodic 0% cache hit collapse on hybrid linear-attention models by prioritizing prompt-region recurrent state snapshots against premature LRU eviction under interleaved traffic.
8. **Dedicated Bulk-Copy Vision Tower CPU Offloading:** Keeps 878.8 MiB of vision transformer weights in pinned host RAM and bulk-copies per-module to GPU only during image forward passes (296 ms -> 333 ms), saving ~870 MiB resident VRAM to prevent OOM without the 10x penalty of UVA zero-copy.

---

## 1. Scope, Provenance, and Durable Receipt

Observed and pinned on **2026-09-03**:

| Repository | Pinned Revision | License | Notes |
|---|---|---|---|
| [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090) | `8d832f8758ae4fd36c29a15d3c45888922bc4377` | Apache-2.0 | Serving stack for Qwen3.8-27B on single consumer 24 GB GPUs with vLLM, speculative drafting, custom Triton kernels, and 28 production patches. |

- **Parent Epics:**
  - #10193: `epic(perf): prioritize the cross-backend Qwen3.8 native performance critical path`
  - #8011: `epic(model): first-class Qwen3.8-27B support across MacBook Metal and A100 CUDA`
  - #2236: `epic(superset): fak > best of vLLM + SGLang + Dynamo + TRT-LLM/LMDeploy/llama.cpp`
- **Filed Issues:**
  - #11099: `feat(model): adaptive context-lookup drafting (LABD) and verify block expansion for speculative decoding`
  - #11100: `feat(compute): split-KV multi-query speculative verify attention kernel for CUDA`
  - #11101: `fix(model): fold negative group scales into int4 nibbles at load time for Marlin W4A8 INT8 GEMMs`
  - #11102: `feat(model): empirical output-frequency draft vocabulary truncation for Qwen3.8 speculative draft heads`
  - #11103: `feat(model): 16-bit recurrent state storage and capacity pricing for hybrid Qwen3.8 GDN layers`
  - #11104: `perf(agent): sort-free small-k logit truncation without full-vocabulary sorting for wide vocabularies`
- **Durable Study Receipt:** `study_69a87c504d53efb41a243ad7f233c19d5293ecbb611869ed64292ad229fd7c2d` (persisted via `fak study add`)

**License Boundary:** Apache-2.0. Clean-room porting and algorithmic adaptation into `fak`'s native Go / CUDA / Vulkan codebase (`internal/model/`, `internal/compute/`, `internal/agent/`) is fully permitted with attribution in `NOTICE`.

---

## 2. Worldview Reconstruction: Who They Built It For & Tradeoffs

1. **Who they built this for:**
   - **Single developers and local operators on consumer hardware:** Users running Qwen3.8-27B on a single 24 GB GPU (NVIDIA GeForce RTX 3090 or RTX 4090), frequently on desktop workstations running X11/Wayland compositors where available VRAM is restricted to ~23–23.5 GB.
   - **Interactive coding agent and chat workloads:** Users whose prompts contain long code files, documents, system prompts, or tool call definitions, where token generation speed directly impacts developer productivity and where responses frequently repeat or edit prompt text.
2. **What they optimized:**
   - **Interactive single-stream throughput:** Pushing single-sequence decode rate from stock 46 tok/s up to 121–133 tok/s for ordinary generation, and up to **381 tok/s** for document reproduction/editing.
   - **Razor-thin memory engineering:** In a 24 GB envelope, 27B weights (W4A16 AutoRound) take ~14.7 GB. That leaves under 9 GB for everything else: KV cache, recurrent states, activation buffers, CUDA graphs, and workspace. Every megabyte matters: requantizing embedding tables saves 2.6 GB; halving GDN recurrent states saves 75 MB/request; offloading vision tower saves 878.8 MiB.
   - **Turn-to-turn latency via hybrid prefix caching:** Turning multi-thousand token follow-up prompt processing from 23s cold down to 0.5s–1s warm.
3. **Tradeoffs vs. fak:**
   - *Architecture Ownership:* `syv-ai` achieves their results by patching vLLM 0.27.1 via 28 sequential `.patch` files and Triton kernels; `fak` is a self-contained Go kernel that owns its compute, scheduler, memory managers, and models natively end-to-end (`docs/native-inference-goal.md`).
   - *Porting Strategy:* Rather than importing vLLM or Python scripts, `fak` adopts the underlying mathematical mechanisms and invariants directly into `internal/model/`, `internal/compute/`, and `internal/agent/`.

---

## 3. Subsystem Analysis & Load-Bearing Mechanisms

### A. Lookup-Augmented Block Drafting (LABD) & n-gram Chains
*Source:* `patches/dflash2-lookup-drafting.patch:1-120@8d832f8758ae4fd36c29a15d3c45888922bc4377` and `patches/dflash2-ngram-chains.patch:1-60@8d832f8758ae4fd36c29a15d3c45888922bc4377`

Neural draft models only see a limited context window (2,048 tokens). When generating answers that quote documents, apply git diffs, or reproduce instructions, the continuation tokens already exist verbatim in the prompt buffer.

LABD scans token history for the longest suffix match ($n_{\text{min}} \le n \le n_{\text{max}}$) using a GPU/host scan, proposing continuation tokens as point-mass draft probabilities. Key innovations:
- **Decoupled verify block size:** The target verify block is decoupled from the drafter's block size (e.g. 7 drafts $\to$ 16 verify positions). Extra positions cost the drafter nothing.
- **Adaptive scheduling:** Long verify blocks are scheduled only when consecutive saturated copy steps are observed, avoiding verify attention overhead on ordinary prose.
- **Agreement gating:** Short lookup matches are only proposed if the neural drafter independently agrees on the first 2 tokens.
- **Drafter-free n-gram chains:** When verbatim reproduction continues, forward passes and CUDA graph replay of the draft model are skipped entirely until the first rejected token, hitting **381 tok/s decode** on RTX 3090.

### B. Split-KV Multi-Query Verify Attention for Speculative Decoding
*Source:* `patches/spec-decode-attn.patch:38-68@8d832f8758ae4fd36c29a15d3c45888922bc4377`

FlashAttention-2 only splits the KV sequence across thread blocks when $q_{\text{len}} = 1$. When speculative verification evaluates a batch of $k+1$ tokens (e.g. 5 tokens for MTP, 8–16 tokens for DFlash2), FA2 assigns only one thread block per query head ($24$ blocks on RTX 3090's 82 SMs).
At 16k context, attention latency jumps to 1.3 ms per layer.
The `SpecDecodeAttention` kernel tiles query rows into `BLOCK_M` rows and splits the KV sequence into `NUM_SEGMENTS` blocks with online softmax partial reductions and a combine step, reducing verify attention latency to 120 us.

### C. Load-Time Sign-Folding for Negative Scales in Marlin W4A8 INT8 GEMMs
*Source:* `patches/marlin-int8-negative-scales.patch:23-54@8d832f8758ae4fd36c29a15d3c45888922bc4377`

Marlin W4A8 INT8 Tensor Core kernels reinterpret int16-requantized group scales as unsigned `uint16_t` (`marlin_template.h: reinterpret_cast<uint16_t*>`). AutoRound symmetric exports have ~50% negative group scales. Loading such checkpoints without sign correction turns those groups into nonsense, causing perplexity collapse.
The load-time fix folds the scale sign directly into int4 weights ($s \to -s, q \to 16-q$ for uint4 nibbles) before repacking into Marlin layout, ensuring exact numerical equivalence.

### D. Empirical Output-Frequency Draft Vocabulary Truncation
*Source:* `prepare/build_draft_vocab.py:1-127@8d832f8758ae4fd36c29a15d3c45888922bc4377`

Qwen3.8 has a vocabulary of 248,320 tokens. Evaluating the draft head's lm_head projection across all 248k tokens takes ~3 ms per token.
By profiling 5.4M tokens generated by Qwen3.8 on code, math, and dialogue, the authors extracted the top 40,000 most frequent tokens (`draft_vocab_ids.json`), covering 97.5% of generated outputs (96% on code). Truncating the draft projection matrix to 40k cuts draft head latency from 3 ms to 0.5–1 ms, providing a ~10% net throughput gain.

### E. 16-Bit Recurrent State Storage & Capacity Accounting
*Source:* `docs/optimizations.md:54-62@8d832f8758ae4fd36c29a15d3c45888922bc4377`

Qwen3.8 has 48 hybrid Gated DeltaNet (linear recurrent) layers. Default fp32 state requires 150 MB of state per request, which bounds concurrent requests to 37 on a 24 GB card. Halving state precision to float16 halves memory consumption to 75 MB/request and doubles concurrency headroom to 64 requests with zero perplexity loss.

### F. Sort-Free Small-k Sampling & Multi-Block Row Softmax
*Source:* `patches/sampler-small-topk-fast-softmax.patch:1-20@8d832f8758ae4fd36c29a15d3c45888922bc4377`

For wide vocabularies ($V = 248\text{k}$), standard `apply_top_k_top_p` sorts the entire 248k vocabulary ($O(V \log V)$) per row, taking 1–2 ms on GPU and causing excessive heap allocations in Go CPU runtimes. When $k \le 64$, using bounded $O(V \log k)$ selection or min-heap partition avoids full sorting and map allocation. In GPU kernels, splitting wide row softmax into 64-chunk multi-block Triton kernels drops latency from 140 us to 10 us.

### G. Hybrid Prefix Caching Checkpoint Eviction Order
*Source:* `patches/mamba-align-checkpoint-order.patch:1-60@8d832f8758ae4fd36c29a15d3c45888922bc4377`

In hybrid linear-attention prefix caching, a cache hit requires both KV blocks and a matching GDN state snapshot. Under memory pressure, standard LRU queues can evict recurrent state snapshots before attention blocks, causing sudden 0% cache hit collapse on multi-turn conversations (TTFT spiking from 2s to 30s). Retaining prompt-region state checkpoints until turn completion stabilizes hit rates.

### H. Dedicated Bulk-Copy Vision Tower CPU Offloading
*Source:* `patches/vision-tower-cpu-offload.patch:1-100@8d832f8758ae4fd36c29a15d3c45888922bc4377`

Keeping 878.8 MiB of vision transformer weights resident on a 24 GB card leaves no transient margin for speculative verification graphs. Unified virtual addressing (UVA) zero-copy offloading slows vision forwards to 3327 ms due to repeated PCIe tile reads. A dedicated bulk-copy offloader in pinned host RAM transfers modules only during image forward passes (333 ms, only +37 ms over resident), keeping resident VRAM usage under 10 MiB.

### I. Structured Output Verification Past Grammar Termination
*Source:* `patches/xgrammar-spec-terminated.patch:1-80@8d832f8758ae4fd36c29a15d3c45888922bc4377`

Speculative verification windows frequently accept tokens that cross the grammar termination boundary (closing braces, newlines after tool tags, stop tokens). Older parsers treated post-termination tokens as syntax errors and aborted valid tool calls. Ignoring tokens after termination resolves this issue.

---

## 4. Current fak Witness & Gap Matrix

| Upstream Mechanism | fak Equivalent | Current fak Witness | On-Axis Gap & Disposition | Filed Issue |
|---|---|---|---|---|
| **Adaptive Context-Lookup Drafting (LABD)** | `internal/model/ngram.go` | `ngram.go:27-55`, `ngram_test.go:16-52` | **PARTIAL → DEFAULT**. Fak has basic CPU NgramDrafter with earliest tie-break; lacks recency preference, MTP fusion, adaptive verify block expansion, and drafter-free chains. | #11099 |
| **Split-KV Multi-Query Verify Attention** | `internal/compute/cuda_kernels.cu` | `cuda_kernels.cu:1908`, `decode_occupancy.go:13-27` | **ABSENT → DEFAULT**. `k_flash_attention` launches 1 block per query head; lacks split-KV 3D reduction kernel for multi-query verify steps. | #11100 |
| **Marlin W4A8 Negative-Scale Sign-Folding** | `internal/vllmquant/`, `internal/model/` | `internal/vllmquant/contract.go:502-512` | **ABSENT → DEFAULT**. Fak checks Marlin requirements but does not fold negative scale signs at load time, risking silent corruption on AutoRound W4A8 checkpoints. | #11101 |
| **Empirical Draft Vocab Truncation** | `internal/model/qwen35_mtp_draft.go` | `qwen35_mtp_draft.go:88-96` | **PARTIAL → DEFAULT**. Fak MTP drafter projects across full 248k vocabulary (~3ms); truncating to 40k empirical tokens cuts draft projection time to <1ms. | #11102 |
| **16-Bit Recurrent State Storage** | `internal/model/qwen35_recurrent_capacity.go` | `qwen35_recurrent_capacity.go:48-54`, `cuda_qwen35_gdn.go:39` | **PARTIAL → DEFAULT**. Fak hardcodes 4 bytes/element (float32) for GDN recurrent state pricing and validation; float16 halves memory footprint per request. | #11103 |
| **Sort-Free Small-k Sampling for Wide Vocab** | `internal/agent/inkernel_sampling.go` | `inkernel_sampling.go:204-248` | **PARTIAL → DEFAULT**. `descProbOrder` sorts all 248k logits using `sort.Slice` and allocates maps; bounded $O(V \log k)$ selection avoids full sorting. | #11104 |
| **Hybrid Prefix Cache State Retention** | `internal/model/paged_prefix_cow.go` | `paged_prefix_cow.go:28-31`, `prefix_snapshot_codec.go:10-40` | **PARTIAL → WATCH**. Fak requires exact hybrid boundaries (`ErrNonExactHybridPrefix`); retaining prompt state checkpoints avoids periodic cache drop lottery. | — |
| **Bulk-Copy Vision Tower CPU Offloader** | `internal/model/safetensors.go` | `safetensors.go:368-373` | **ABSENT → WATCH / WORLDVIEW-FINDING**. Fak intentionally drops `model.visual.*` for text agents; bulk-copy offload is the blueprint when vision is added. | — |
| **Structured Output Window Early Termination** | `internal/model/constraint.go` | `constraint.go:194-206`, `fastforward.go:41-69` | **PRESENT-on-axis**. Fak's `DecodeConstraint` explicitly permits unconstrained tokens past the end of `PerStep` without error. | — |

---

## 5. Candidate Disposition Matrix

| Candidate Borrow | Source Anchor | Axis Optimized | fak Seam | Disposition | Filed Issue |
|---|---|---|---|---|---|
| Adaptive Context-Lookup Drafting (LABD) | `patches/dflash2-lookup-drafting.patch:1-120` | Decode throughput on context-quoting tasks (up to 381 tok/s) | `internal/model/ngram.go:27-55` | **DEFAULT** | #11099 |
| Split-KV Multi-Query Verify Attention | `patches/spec-decode-attn.patch:38-68` | GPU SM occupancy during speculative verify steps (1.3ms $\to$ 120us) | `internal/compute/cuda_kernels.cu:1908` | **DEFAULT** | #11100 |
| Marlin W4A8 Negative-Scale Sign-Folding | `patches/marlin-int8-negative-scales.patch:23-54` | Numerical accuracy of INT8 Tensor Core GEMMs | `internal/vllmquant/contract.go:502-512` | **DEFAULT** | #11101 |
| Empirical 40k Draft Vocabulary Truncation | `prepare/build_draft_vocab.py:1-127` | Draft head projection latency (<1ms vs 3ms) | `internal/model/qwen35_mtp_draft.go:88-96` | **DEFAULT** | #11102 |
| 16-Bit Recurrent State Storage | `docs/optimizations.md:54-62` | Recurrent state VRAM footprint (150MB $\to$ 75MB) and concurrency | `internal/model/qwen35_recurrent_capacity.go:48-54` | **DEFAULT** | #11103 |
| Sort-Free Small-k Logit Sampling | `patches/sampler-small-topk-fast-softmax.patch:1-20` | Sampling CPU/GPU latency over 248k vocabulary | `internal/agent/inkernel_sampling.go:204-248` | **DEFAULT** | #11104 |
| Hybrid Prefix Cache State Retention | `patches/mamba-align-checkpoint-order.patch:1-60` | Multi-turn TTFT cache hit stability | `internal/model/paged_prefix_cow.go:28-31` | **WATCH** | — |
| Dedicated Bulk-Copy Vision Tower Offloading | `patches/vision-tower-cpu-offload.patch:1-100` | Transient VRAM allocation for multimodal serving | `internal/model/safetensors.go:368-373` | **WATCH** | — |
| Structured Output Speculative Termination | `patches/xgrammar-spec-terminated.patch:1-80` | Request survivability under grammar verification | `internal/model/constraint.go:194-206` | **PRESENT** | — |

---

## 6. Completeness-Critic Review

- **Subsystems Inspected:**
  1. Quantization & Format Conversions (`prepare/quant_*.py`, `drafter/quant_*.py`, `patches/marlin-*.patch`).
  2. Speculative Drafter Architecture & Vocabulary (`drafter/`, `prepare/build_draft_vocab.py`, `patches/dflash2-backport.patch`, `patches/qwen3_5-mtp-draft-vocab.patch`).
  3. Lookup-Augmented Block Drafting & n-gram Chains (`patches/dflash2-lookup-drafting.patch`, `patches/dflash2-ngram-chains.patch`, `bench/labd_bench.py`).
  4. Attention Kernels & Verify Paths (`patches/spec-decode-attn.patch`, `patches/triton-prefill-attn-int8.patch`, `bench/test_spec_decode_attn.py`).
  5. Hybrid Recurrent State & KV Allocation (`patches/hybrid-sw-block-promote.patch`, `patches/hybrid-kv-groups-v2-cudagraph.patch`, `patches/mamba-align-checkpoint-order.patch`, `kvarn/`).
  6. Serving Governors & Hardware Boundaries (`patches/vision-tower-cpu-offload.patch`, `patches/xgrammar-spec-terminated.patch`, `docs/gotchas.md`).
- **Completeness Verdict:** Exhaustive coverage across code, test benches, and documentation; no relevant subsystem or patch was bypassed. Every identified borrow has been grounded at `path:line@sha` and compared on its specific axis against `fak`'s implementation.
