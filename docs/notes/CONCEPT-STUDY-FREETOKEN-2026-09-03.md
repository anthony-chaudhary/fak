---
title: "CONCEPT-STUDY: FreeToken edge-native MoE serving — Qwen3.8-Flash-Next PLE NVMe direct streaming, GLM-5.3-Flash KDA/DSA hybrid indexer, and bandwidth-adaptive MoE co-execution"
description: "Follow-up study of FlashML-org/FreeToken at revision 03c28d2, triggered by Shuo Yang's 2026-09-03 announcement of Qwen3.8-Flash-Next running at 68.3 tok/s on an RTX 5090 with 63GB host RAM. Mines the 51GB PLE direct NVMe streaming engine with CUDA stream memops, GLM-5.3-Flash 45-layer hybrid KDA/DSA architecture, and bandwidth-adaptive MoE miss partitioning."
date: 2026-09-03
---

# CONCEPT-STUDY: FreeToken edge-native MoE serving (2026-09-03)

**Verdict:** FreeToken (`FlashML-org/FreeToken`) has evolved from an initial edge MoE offloader into a high-performance, edge-native inference engine for frontier mixed-architecture models. This follow-up study (extending `docs/notes/CONCEPT-STUDY-FREETOKEN-2026-08-22.md`) mines three new production-validated mechanisms:
1. **Direct NVMe streaming of 51GB PLE n-gram tables (`--ple-backend disk`):** In Qwen3.8-Flash-Next (`qwen4_exp`), only 16 rows are accessed per token. Direct O_DIRECT / `io_uring` batched reads (0.1–0.4ms) directly into pinned staging buffers with CUDA stream memop synchronization (`cuStreamWaitValue64` / `cuStreamWriteValue64`) eliminate DRAM caching entirely, reducing host RAM requirements from >114GB to 63GB at ~0.5% throughput cost and enabling 68.3 tok/s decode on a single RTX 5090.
2. **Bandwidth-adaptive MoE miss partitioning ($q^\star$ co-execution):** Instead of coarse layer-level spilling or serialized PCIe paging on GPU cache misses, FreeToken dynamically splits expert misses using the measured bandwidth ratio ($q^\star = \text{PCIe\_BW} / \text{CPU\_BW}$). The top $\approx q^\star \times \text{misses}$ (by recency) stream over PCIe while the remaining overflow misses execute concurrently on host CPU via AVX-512/AVX2 kernels, finishing both transfers and compute simultaneously.
3. **GLM-5.3-Flash 45-layer hybrid KDA/DSA execution:** Native support for 34 KDA (linear-attention) and 11 DSA (sparse attention) layers with 288 NVFP4 experts, featuring a kpool DSA indexer backend (64-token pages, 1/ratio shadow index slab, CUDA-graph safe decode).

---

## 1. Scope, Provenance, and Durable Receipt

Observed and pinned on **2026-09-03**:

- **Repository:** <https://github.com/FlashML-org/FreeToken>
- **Pinned Revision:** `FlashML-org/FreeToken@03c28d2b154484f84397e42731a9b97f340322b5` (commit `03c28d2`, 2026-09-03).
- **Triggering Event:** Public announcement and video demonstration by Shuo Yang (@Andy_ShuoYang) on 2026-09-03 ([x.com/Andy_ShuoYang/status/2095579848723050842](https://x.com/Andy_ShuoYang/status/2095579848723050842)) showcasing Qwen3.8-Flash-Next executing on an RTX 5090 at 68.3 tok/s with 63GB host RAM.
- **Paper:** Yang et al., *FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution*, arXiv:2608.16157v1.
- **License:** Apache-2.0. Direct porting and adaptation into fak-native Go/C++ runtime are permitted with attribution.
- **Durable Study Receipt:** `study_75aef7f39f666f166991072b47a5083c2bc150bf38c0fc3c3f6e963d9d4120a9` (supersedes `study_8fc6130243ee492e8b2934b77ab93b267f3bacd24792a96f62186627bcb6e391`, persisted in local receipt store via `fak study add`).

---

## 2. Worldview Reconstruction and Tradeoffs

- **Who they built this for:** Developers, researchers, and agentic workflows running frontier open-weight models (Qwen3.8-Flash-Next, GLM-5.3-Flash, DeepSeek-V4) on high-end consumer hardware (single/dual RTX 4090/5090 desktops and laptops with 32–64GB host RAM).
- **What they were optimizing:** Eliminating the "VRAM/DRAM wall" without sacrificing interactive tokens/sec. Rather than aggressive 1-bit/2-bit weight degradation or full model CPU offload (which drops decode to 2–5 tok/s), FreeToken preserves production precision (NVFP4, FP8) by exploiting model structural sparsity:
  1. **Sparse n-gram table:** 51GB total parameters, but only 16 rows (~4KB) touched per token. Storing this table on fast NVMe SSD with Direct I/O cuts DRAM usage by 51GB with ~0.5% throughput overhead.
  2. **Sparse MoE routing:** Only top-8 or top-10 experts activate per token. The GPU caches the hot working set; misses are split across PCIe transfer and multithreaded CPU compute to avoid idling either processor.
- **Divergence from fak:** FreeToken is implemented in Python, PyTorch, and Triton, targeting end-user consumer deployment with an integrated desktop app. fak is a single Go binary agent kernel prioritizing verifiable security, determinism, in-kernel native inference, and fleet management. Therefore, borrows are adapted natively into Go and CUDA/C++ kernels rather than importing Python dependencies.

---

## 3. Evidence Surface and What Was Read

Deep subsystem inspection covered:
- **PLE NVMe Direct Streaming:** `python/freetoken/models/qwen4_exp/ple_disk.py` (lines 1–270), `python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp` (lines 1–621), `tests/models/qwen4_exp/test_ple_disk.py`.
- **Bandwidth-Adaptive Offload:** `python/freetoken/moe/benchbw.py` (lines 1–997), `python/freetoken/moe/offload_kernels.py` (lines 1–431), `python/freetoken/moe/offload_cache.py`, `python/freetoken/moe/cpu_executor.py`.
- **GLM-5.3-Flash Hybrid Stack:** `python/freetoken/models/glm5_next/` (`model.py`, `kda.py`, `attention.py`, `weight.py`), `python/freetoken/attention/dsa_indexer_kpool.py`, `python/freetoken/kernel/fla/kda.py`, `tests/attention/test_dsa_kpool.py`.
- **Sampling & Kernels:** `python/freetoken/kernel/triton/sampling.py` (commit `03c28d2`), `python/freetoken/kernel/triton/fp8_pertensor_linear.py` (commit `58f4b9e`).
- **Semantic Anchors & Dynamic Reallocation:** `python/freetoken/scheduler/cache.py` (`snapshot_toolcall_anchor`, `maybe_free_swa_out_of_window`), `python/freetoken/kvcache/dsv4_cost_model.py`.

*Completeness Critic:* All load-bearing modules touched in commits between `0ab982f` and `03c28d2` were inspected. No relevant model execution or kernel path was skipped.

---

## 4. Current-fak Witness and Candidate Ledger

| Borrow | Source `path:line@sha` | Axis | Their-Worldview Reason | fak Witness on-axis | Portfolio / License | Filed # |
|---|---|---|---|---|---|---|
| **Direct NVMe streaming of 51GB PLE n-gram table** | `python/freetoken/models/qwen4_exp/ple_disk.py:101-240@03c28d2`, `python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp:1-200@03c28d2` | Stream 51GB predictive table from NVMe using O_DIRECT + io_uring, batching 16 rows/token with CUDA stream memop graph synchronization (`cuStreamWaitValue64`) | 16 rows/token is tiny; NVMe random read is 0.1–0.4ms; cuts host RAM requirement from >114GB to 63GB at ~0.5% throughput cost | **PARTIAL.** `internal/qwen4exp/placement.go:18,28` declares `NGram3PLEEmbeddings` on `TierSSD` and validates capacity in admission, but the native runtime (`internal/model/hal.go`) lacks an SSD row-streaming engine or stream memop synchronization | **DEFAULT** for edge/consumer deployment; ADAPT (Apache-2.0) | #11037 |
| **Intra-step bandwidth-adaptive miss partitioning ($q^\star$ policy)** | `python/freetoken/moe/benchbw.py:1-35,660-675@03c28d2`, `python/freetoken/moe/offload_kernels.py:43-125@03c28d2` | Dynamically split top-k expert misses between PCIe fetch and host CPU execution by measured bandwidth ratio ($q^\star = \text{PCIe\_BW} / \text{CPU\_BW}$) and recency | PCIe and host CPU DRAM are independent channels; co-executing both minimizes decode latency when GPU cache misses | **PARTIAL.** `docs/MOE-ACTIVATED-OFFLOAD-PLAN.md` and `internal/model/expert_spill_placement.go:279` implement only coarse layer-level spilling or static ring page-in; no intra-token concurrent split exists | **OPTIONAL-MODULE** for heterogeneous edge hosts; ADAPT (Apache-2.0) | #11038 |
| **Double-buffered prefill layer streaming with D2D hit compaction** | `python/freetoken/moe/offload_cache.py:588-611,723-794@03c28d2`, `python/freetoken/layers/moe.py:413-424@03c28d2` | Overlap layer $L+1$ PCIe DMA transfer with layer $L$ GEMM execution in double-buffer staging window; compact and gather resident experts in HBM device-to-device | Streaming entire offloaded layers over PCIe serializes transfer and compute; double-buffering hides transfer latency and D2D gather bypasses PCIe | **ABSENT.** `internal/model/moe_offload.go` implements static `splitKernel` with no double-buffering or asynchronous prefetch pipeline between layers | **DEFAULT** for offloaded MoE prefill; ADAPT (Apache-2.0) | #11115 |
| **GLM-5.3-Flash KDA/DSA hybrid execution** | `python/freetoken/models/glm5_next/model.py:1-207@03c28d2`, `python/freetoken/attention/dsa_indexer_kpool.py:1-288@03c28d2` | Native execution of 34 KDA linear-attention and 11 DSA layers with 288 NVFP4 experts and kpool indexer | GLM-5.3-Flash hybrid architecture achieves frontier capabilities within consumer compute budgets | **PARTIAL.** `internal/model/glm5next_spine.go:21` recognizes config but explicitly returns `GLM5NextUnsupportedError` ("KDA/DSA/indexer/mHC kernels are not implemented") | **DEFAULT**; tracked under existing issue | #9441 |
| **Semantic tool-call anchor snapshotting** | `python/freetoken/scheduler/cache.py:151-216@03c28d2`, `python/freetoken/scheduler/scheduler.py:357-362@03c28d2` | Snapshot GDN recurrent state and cap SWA eviction at tool-call opener tokens during decode | Normalizing tool calls invalidates exact token prefixes; anchored snapshots preserve reusable state | **PARTIAL.** Gateway has prompt-cache breakpoints, but model cache lacks opener-keyed GDN/SWA state checkpoints | **OPTIONAL-MODULE**; tracked under existing issue | #8601 |
| **Idle-only in-place cache-geometry rebuild without model reload** | `python/freetoken/engine/engine.py:778-923@03c28d2`, `python/freetoken/kvcache/base.py:86-121@03c28d2` | Dynamically re-allocate VRAM between MoE expert cache slots and KV cache pages under live traffic with automated rollback | Workload shifts between long-context dialogue and compute-heavy batch turns; restarting or reloading models wastes minutes | **PARTIAL.** `internal/model/paging_ring.go` fixes expert ring at load time; fak has no idle-gated transactional resize endpoint | **OPTIONAL-MODULE**; tracked under existing issue | #8603 |
| **FTW 4096-byte aligned chunked storage with Direct I/O streaming** | `python/freetoken/checkpoint/ftw.py:52-86,192-344@03c28d2`, `python/freetoken/moe/host_banks.py:74-150,284-347@03c28d2` | 4096-byte block alignment for O_DIRECT NVMe reads, streaming directly into anonymous memory with asynchronous post-load pinning | Bypasses OS page cache pollution and avoids `cudaHostAlloc` zero-fill latency (~47s for 137GB) | **PARTIAL.** `internal/ggufload/` has fast mmap loading, but lacks 4096-byte aligned Direct I/O streaming and pin-after-fill pipeline | **WATCH** (portfolio evaluation) | — |
| **Exact single-launch Triton sampling** | `python/freetoken/kernel/triton/sampling.py:293-433,528-598@03c28d2` | Multi-CTA split-vocab and 4-round uint32 bit-histogram exact top-k / top-p sampling preserving boundary ties | Saturate GPU compute at batch-1 without host roundtrips or FlashInfer CUB header dependency | **PRESENT/DIVERGENT.** fak uses native Go/C++/Vulkan samplers (`internal/model/sample.go`); Triton DSL is excluded from fak runtime | **EXCLUDE** (runtime divergence) | — |

---

## 5. Summary of Filed Work

1. **#11037: `feat(qwen4exp): stream 51GB PLE n-gram table from NVMe with CUDA stream memop synchronization`**
   - **Seam:** `internal/qwen4exp/placement.go:18,28` and `internal/model/qwen4exp_spine.go:55`.
   - **Parent Epic:** #9204 (`epic(model): QWEN38-FLASH-NEXT fak-native support`), Milestone: 20 (`QWEN38-FLASH-NEXT F2 — Hardware & net-true performance`).
   - **Deliverable:** Direct I/O safetensors shard reader with batched 16-row gather into pinned staging memory and CUDA graph stream memop handshake (`cuStreamWaitValue64` / `cuStreamWriteValue64`).

2. **#11038: `perf(moe): intra-step bandwidth-adaptive miss partitioning between PCIe fetch and host CPU compute`**
   - **Seam:** `internal/model/expert_spill_placement.go:279` and `internal/model/paging_ring.go`.
   - **Parent Epic:** #4207 (`epic(inference-radar-study): mine RunAnywhere's Inference Radar (Hy3 / MoE-serving roundup) for fak — witnessed borrows`), Milestone: 4 (`Decode parity across every backend (GPU/Metal/CPU)`).
   - **Deliverable:** Intra-step bandwidth partition solver calculating optimal integer miss allocation between PCIe gather and host CPU AVX-512 GEMV based on measured transfer and compute throughput.

3. **#11115: `perf(moe): double-buffered prefill layer streaming with D2D hit compaction for offloaded MoE weights`**
   - **Seam:** `internal/model/moe_offload.go:25-45` and `internal/model/moe_split.go`.
   - **Parent Epic:** #4207 (`epic(inference-radar-study): mine RunAnywhere's Inference Radar (Hy3 / MoE-serving roundup) for fak — witnessed borrows`), Milestone: 4 (`Decode parity across every backend (GPU/Metal/CPU)`).
   - **Deliverable:** Double-buffered prefill layer staging buffer with ping-pong synchronization events prefetching layer $L+1$ on an asynchronous transfer stream while layer $L$ compute proceeds, with device-side hit compaction to gather resident experts at HBM bandwidth.

---

## 6. Companions and Cross-References

- Field borrow skill: `.claude/skills/field-borrow/SKILL.md`
- Study repo skill: `.claude/skills/study-repo/SKILL.md`
- Prior study: `docs/notes/CONCEPT-STUDY-FREETOKEN-2026-08-22.md`
- Epics: #9204 (Qwen3.8-Flash-Next native support), #9433 (GLM-5.3-Flash native support), #4207 (MoE serving roundup)
- Issues: #11037, #11038, #11115, #9441, #8601, #8603
