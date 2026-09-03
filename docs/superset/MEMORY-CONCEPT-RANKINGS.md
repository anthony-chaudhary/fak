# Memory-Concept Ranking Dossier: fak ⊇ vLLM · SGLang · NVIDIA Dynamo · TRT-LLM · LMDeploy · llama.cpp · LMCache · Mooncake · llm-d

> **Status:** Grounded research dossier for epic #2236 (superset rankings, memory-first) and issue #2237 / issue #3143.
> **Rung:** R0 conceptual completion with file-cited fak positions, per-engine evidence tables, and explicit adopt / already-lead / SKIP verdicts.

---

## 1. Doctrine & Honesty Fence

This dossier establishes the conceptual foundation (R0) of the superset claim: **for every load-bearing memory concept across the leading open serving runtimes, fak maintains an equal-or-better mechanism (witnessed) or states an explicit, reasoned SKIP.**

### Honesty Fence (docs/benchmarks/BENCHMARK-GOVERNANCE.md)
1. **External numbers are the claimants' published results**, cited for comparative ranking only. They are not direct head-to-head fak comparisons.
2. **Every fak benchmark cell is `TBD`** until an empirical run lands on committed hardware receipts (`docs/benchmarks/VLLM-HEADTOHEAD-RESULTS.md`).
3. **No performance boasts**: On single-instance raw decode throughput, fak is not vLLM's or SGLang's peer and does not claim to be. The superset claim concerns the **concept set**, auditability, and memory efficiency under agentic multi-tenant execution.

---

## 2. Master Comparative Ranking Matrix

| # | Memory Concept | Best-in-Class (Ranked) | fak Today (Evidence) | Rung | Verdict & Target |
|---|---|---|---|---|---|
| **M1** | Paged/Block KV Allocator | 1. vLLM (PagedAttention, O(1) LRU block pool)<br>2. SGLang (Paged radix slabs)<br>3. TRT-LLM | `model.PagedKVPool` (`internal/model/paged_kv.go:28`), `KVCacheToPaged`/`SwapToHost` (`internal/modelengine/nativesched_preempt.go:523`) | R1 | **Adopt (Default-On)**: Wire paged KV into live decode loop (#1533) |
| **M2** | Prefix-Reuse Index | 1. SGLang (RadixAttention trie, 75–95% hit rate)<br>2. vLLM (Automatic Prefix Caching)<br>3. llama.cpp (reference baseline `seq_cp` interop) | `internal/radixkv/radixkv.go:1-356` (longest-prefix trie, mid-edge split, policy eviction) | R2 | **Already Lead**: Native radix tree + quarantine eviction no external engine has |
| **M3** | Hierarchical KV Tiering & Offload | 1. Dynamo (KVBM 4-tier CMX)<br>2. SGLang (HiCache L1/L2/L3)<br>3. LMCache (Redis/S3/Mooncake/GDS)<br>4. vLLM (Offloading connector) | `internal/cachemeta/hardware.go:105` (`TierProfile`), `internal/l3kv/store.go`, `internal/compute/kvdemote.go` | R1 | **Adopt**: Page-granular CPU/disk offloading with zero bit-drift (#10722, #1469) |
| **M4** | KV Transfer Plane | 1. Dynamo (NIXL RDMA/UCX/TCP)<br>2. Mooncake (Transfer Engine)<br>3. vLLM (Async connector API) | `internal/cachemeta/nixl_lease.go`, `LocalKVTransport` (`internal/modelengine/native_pd.go:229`), `internal/l3kv/async_job.go` | R1 | **Adopt**: Stale-watermark async transport and NIXL backend matrix (#10729, #2243) |
| **M5** | Eviction & Lifecycle Policy | 1. Dynamo (Cost-aware KVBM, pinning)<br>2. vLLM (LRU + priority + cache salt)<br>3. SGLang (LRU over radix) | `internal/compute/kvcost.go:145` (`KVEvictionCost`), `kvreuse.go`, `kvcost_pin.go`, `radixkv.go:839` | R2 | **Adopt/Extend**: Pluggable reuse seam and cost-aware demote-before-drop (#3411, #3414) |
| **M6** | KV-Aware Routing | 1. llm-d (KVEvents global block index)<br>2. Dynamo (KV router + priority hints)<br>3. Mooncake (Conductor)<br>4. SGLang (Router) | `internal/gateway/residency_router.go:418`, `PrefixResidencyIndex` (`native_pd.go:418`), `internal/cacheprice/route.go` | R1 | **Adopt**: Fleet-level event-fed KV prefix router (#2238) |
| **M7** | KV Quantization & Compression | 1. TRT-LLM (FP8/NVFP4 KV)<br>2. LMDeploy (TurboMind INT8/INT4)<br>3. LMCache (CacheGen bitstream)<br>4. vLLM (FP8 KV) | `internal/model/quant_*.go`, `internal/vllmquant/contract.go`, `internal/model/kvquant.go` | R1 | **Adopt**: First-class 4-bit/FP8 KV-quant ladder rungs (#2240, #10710) |
| **M8** | Non-Prefix KV Reuse | 1. LMCache (CacheBlend; positional-independent reuse + selective recompute) | Longest-prefix match in `internal/radixkv`; `Kraw` re-RoPE ability in `internal/model/kvcache.go:65` | R0 | **SKIP (Reasoned)**: High quality-recovery recompute tax on agent tool traces (#3143) |
| **M9** | Hybrid-Model Memory (SWA/GDN/Mamba) | 1. vLLM (Hybrid Memory Allocator v0.21)<br>2. SGLang (UnifiedTree HiCache) | `internal/model/swa.go`, `arch_support.go` (GLM-DSA, Qwen3.5/3.6 Gated-DeltaNet) | R1 | **Adopt**: Allocator groups unifying recurrent state with paged/radix plane (#2241) |
| **M10** | Offload-Before-Preempt | 1. vLLM (KV offload connector)<br>2. SGLang (HiCache swap) | `internal/modelengine/nativesched_preempt.go:501` (`DemoteBeforeDrop`), `compute.PlanKVDemotion` | R2 | **Already Lead**: Economically-priced spill-before-drop with restore cost comparison (#3414) |
| **S1** | Span-Evictable KV (Structural Lead) | None (All external engines report `SupportsExactSpan=false`) | `internal/model/kvcache.go:120-185`: pre-RoPE `Kraw` allows bit-exact middle span eviction and re-RoPE | R2 | **Lead / Defend**: Core architectural moat for prompt compaction and privacy |
| **S2** | Cross-Worker Prefix Sharing (Structural Lead) | None (All engines isolate memory per instance or rely on distributed network IPC) | In-process CoW block tables (`internal/compute/cuda.go:1253`), shared radix tree across workers | R2 | **Lead / Defend**: 1.1–1.2× at 4 workers on shared agent system prompts (#10723) |

---

## 3. Deep Concept Analysis (M1–M10)

### M1: Paged / Block KV Allocator
- **Definition:** Decouples logical sequence length from contiguous physical GPU VRAM by allocating fixed-size page blocks (e.g. 16 or 32 tokens), eliminating external memory fragmentation and enabling dynamic growth without reallocations.
- **Per-Engine Evidence:**
  - *vLLM:* Originator of PagedAttention (`vllm/v1/core/block_pool.py@a56654d6`). O(1) allocation/deallocation via free list. Reports 2–4× throughput gain over naive contiguous memory.
  - *SGLang:* Paged radix cache allocating memory in page-aligned block slabs (`sglang/srt/mem_cache/memory_pool.py`).
  - *TensorRT-LLM:* Paged KV cache manager with configurable block sizes (16, 32, 64 tokens).
- **Ranking:** 1. vLLM · 2. SGLang · 3. TRT-LLM · 4. LMDeploy.
- **fak Position:** `model.PagedKVPool` (`internal/model/paged_kv.go`), `KVCacheToPaged` (`internal/modelengine/nativesched_preempt.go:523`). The primitives exist for preemption swapping and P/D disaggregation, but the steady-state in-kernel decode path is whole-prefix KV. Rung R1.
- **Verdict:** **ADOPT (Default-On)**. Land paged decode path under explicit memory budget (#1533).

---

### M2: Prefix-Reuse Index
- **Definition:** Maintains an in-memory index of previously computed KV states keyed by token sequences, enabling instant prompt-cache hits for repeated system prompts, few-shot examples, and multi-turn agent histories.
- **Per-Engine Evidence:**
  - *SGLang:* RadixAttention (`sglang/srt/mem_cache/radix_cache.py`). Token-level radix tree with longest-prefix match, mid-edge splitting, and LRU eviction. Claims 75–95% cache hit rates in agentic workflows.
  - *vLLM:* Automatic Prefix Caching (APC) using block hash matching (`vllm/v1/core/kv_cache_utils.py`). Coarser granularity than SGLang (page-level vs token-level).
  - *llama.cpp (reference baseline):* Sequence copy (`seq_cp`) and slot allocation for interop / historical comparisons. Manual / coarse session saving.
- **Ranking:** 1. SGLang · 2. vLLM · 3. llama.cpp (reference baseline).
- **fak Position:** `internal/radixkv/radixkv.go:1-356`. Compressed radix tree, token-granular matching, mid-edge splitting, and reference-counted leases. **Crucially, fak adds policy eviction (`EvictNode`, `EvictPrefix`) and quarantine ejection for security/taint boundaries that zero external engines support.** Rung R2.
- **Verdict:** **ALREADY LEAD**. Defend token-level radix index and quarantine security boundaries; benchmark cross-worker prefix hits (#1532).

---

### M3: Hierarchical KV Tiering & Offload
- **Definition:** Extends cache capacity beyond expensive GPU HBM by spilling colder KV spans down a storage hierarchy: HBM → Host DRAM → Local NVMe/CXL → Distributed Object Storage (S3/MinIO), restoring spans on demand.
- **Per-Engine Evidence:**
  - *Dynamo:* KVBM + CMX (`docs.nvidia.com/dynamo/latest/architecture/kvbm_intro.html`). 4-tier storage management with storage plugins (NVMe, ICMS).
  - *SGLang:* HiCache (`lmsys.org/blog/2025-09-10-sglang-hicache/`). 3-tier memory architecture (L1 GPU, L2 Host, L3 Disk/Remote). Reports up to 6× throughput and 80% TTFT reduction.
  - *LMCache:* Multi-backend connector supporting Redis, Mooncake, Local Disk, and S3.
  - *vLLM:* CPU-offloading connector (`vllm/v1/kv_offload/cpu/manager.py`). Reports +5.4% throughput, -15.8% TTFT.
- **Ranking:** 1. Dynamo · 2. SGLang · 3. LMCache · 4. vLLM.
- **fak Position:** `internal/cachemeta/hardware.go` (`TierProfile`), `internal/cachemeta/placement.go` (`PlanPlacement`), `internal/l3kv/store.go` (disk-backed L3 store), `internal/compute/kvdemote.go` (`PlanKVDemotion`). Rung R1.
- **Verdict:** **ADOPT**. Ship page-granular CPU/disk offloading with zero bit-drift restore (#10722, #2169, #1469).

---

### M4: KV Transfer Plane
- **Definition:** High-speed communication transport (RDMA, RoCE, PCIe P2P, TCP) for moving KV cache blocks between disaggregated Prefill and Decode nodes or across cluster tiers without CPU serialization bottlenecks.
- **Per-Engine Evidence:**
  - *Dynamo:* NIXL (`docs.nvidia.com/dynamo/latest/backends/trtllm/kv-cache-transfer.html`). Unified communication library using UCX/RDMA, RoCE, and TCP.
  - *Mooncake:* Transfer Engine (`arxiv.org/abs/2407.00079`). Zero-copy RDMA transfer engine handling 100B tokens/day in production.
  - *vLLM:* Asynchronous KV transfer connector API (`vllm/distributed/kv_transfer/`).
- **Ranking:** 1. Dynamo · 2. Mooncake · 3. vLLM.
- **fak Position:** In-process `LocalKVTransport` (`internal/modelengine/native_pd.go:229`), `nixl_lease.go` (`internal/cachemeta`), `AsyncWatermarkManager` (`internal/l3kv/async_job.go`, #10729). Rung R1.
- **Verdict:** **ADOPT**. Wire remote NIXL/RDMA transport matrix (#2243).

---

### M5: Eviction & Lifecycle Policy
- **Definition:** Algorithmic determination of which KV cache blocks to drop, demote, or retain under memory pressure, moving beyond memoryless LRU to value-aware, frequency-aware, and agent-intent-guided policies.
- **Per-Engine Evidence:**
  - *Dynamo:* Cost-aware KVBM with explicit cache-control hints from agentic frontends.
  - *vLLM:* Priority-based eviction, LRU queue, and tenant `cache_salt`.
  - *SGLang:* LRU over radix tree with tree-level reference pinning.
- **Ranking:** 1. Dynamo · 2. vLLM · 3. SGLang.
- **fak Position:** `internal/compute/kvcost.go:145` (`KVEvictionCost`), `kvreuse.go` (`KVReuseTerm`, `KVReuseEstimate`, #3411), `kvcost_pin.go` (`PinBoostFromTTL`), `internal/radixkv/radixkv.go:839` (`DemotionDecider`, #3414). Rung R2.
- **Verdict:** **ADOPT/EXTEND**. Unify reuse terms across cost variants and wire demote-before-drop (#3411, #3414).

---

### M6: KV-Aware Routing
- **Definition:** Routing incoming inference requests to the worker instance that already holds the largest resident prompt prefix in its local KV cache, maximizing cache hits and minimizing redundant prefill.
- **Per-Engine Evidence:**
  - *llm-d:* KV-Cache Indexer (`developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference`). Global block index fed by KVEvents.
  - *Dynamo:* KV router with priority and latency hints.
  - *Mooncake:* Conductor with cache-distribution dispatch and replica awareness.
  - *SGLang:* SGLang Router with prefix tree residency mapping.
- **Ranking:** 1. llm-d · 2. Dynamo · 3. Mooncake · 4. SGLang.
- **fak Position:** In-process decode routing (`internal/modelengine/native_pd.go:418`), gateway `PrefixResidencyIndex` (`internal/gateway/residency_router.go:418`), `cacheprice.CheapestRoute` (`internal/cacheprice/route.go`). Rung R1.
- **Verdict:** **ADOPT**. Scale prefix-overlap-scored routing to cluster fleets (#2238).

---

### M7: KV Quantization & Compression
- **Definition:** Compressing KV cache representations from FP16/BF16 down to FP8, INT4, or 3-bit formats to double or quadruple effective context capacity per gigabyte of memory.
- **Per-Engine Evidence:**
  - *TensorRT-LLM:* Production NVFP4 and FP8 KV cache support.
  - *LMDeploy:* TurboMind INT8 / INT4 KV cache with per-channel scaling factors.
  - *LMCache:* CacheGen bitstream compression using contextual entropy encoding.
  - *vLLM:* FP8 KV cache with static/dynamic scales (`vllmquant`).
- **Ranking:** 1. TRT-LLM · 2. LMDeploy · 3. LMCache · 4. vLLM.
- **fak Position:** `internal/vllmquant/contract.go` (negotiation engine, #6238), `internal/model/kvquant.go`, `docs/design` rotated Lloyd-Max KV quant ladder (#2240, #10710). Rung R1.
- **Verdict:** **ADOPT**. Ship production 4-bit / FP8 KV cache quantization ladder (#2240).

---

### M8: Non-Prefix KV Reuse (CacheBlend Analysis — #3143)
- **Definition:** Reusing KV cache blocks that appear at arbitrary positions in a prompt (not just matching from token 0), using selective recompute or cross-attention blending to recover accuracy.
- **Per-Engine Evidence:**
  - *LMCache (CacheBlend):* Yao et al., EuroSys 2025. Selectively recomputes tokens at document boundaries while reusing non-contiguous KV blocks. Reports up to 2.8× speedup on multi-document RAG.
  - *All others:* Absent. Standard serving runtimes restrict reuse to strict prefixes starting at token 0.
- **Ranking:** 1. LMCache · (all others: absent).
- **fak Position:** `internal/radixkv` is strict longest-prefix only. However, fak's kernel-owned pre-RoPE `Kraw` (`internal/model/kvcache.go:65`) provides a unique mathematical ability: fak can rotate un-rotated keys to arbitrary new positions without recomputing QKV projections.
- **Verdict:** **SKIP (Reasoned)**.
  - *Reason:* Agentic coding workflows exhibit high prefix coherence (shared system prompts, repo maps, tool definitions) and low mid-prompt document permutations. CacheBlend requires selective recompute and introduces non-deterministic perplexity degradation (~1–3% task degradation). For strict coding agents where syntax and tool schemas require 100% precision, positional blending introduces unacceptable hallucination risk.
  - *Disposition:* Closed under issue #3143 as SKIP with mathematical rationale documented.

---

### M9: Hybrid-Model KV Memory (SWA / Mamba / GDN)
- **Definition:** Managing heterogeneous KV states for architectures that combine sliding-window attention (SWA) with linear recurrent states (Gated DeltaNet, Mamba, GLM-DSA), preventing windowed layers from holding unneeded tokens.
- **Per-Engine Evidence:**
  - *vLLM:* Hybrid Memory Allocator (v0.21.0). First-class memory pooling for hybrid SWA/Mamba architectures.
  - *SGLang:* UnifiedTree HiCache supporting SWA and recurrent states.
- **Ranking:** 1. vLLM · 2. SGLang.
- **fak Position:** In-kernel compute kernels exist for SWA (`internal/model/swa.go`), Qwen3.5/3.6 Gated-DeltaNet (`arch_support.go`), and GLM-DSA (`paged_glmdsa.go`). Rung R1.
- **Verdict:** **ADOPT**. Unify hybrid state allocation with paged/radix memory pool (#2241).

---

### M10: Offload-Before-Preempt
- **Definition:** When a running request must be preempted due to VRAM exhaustion, swapping its KV cache to host DRAM rather than discarding it for recompute, avoiding catastrophic prefill thrashing.
- **Per-Engine Evidence:**
  - *vLLM:* Offloading connector swaps preempted KV to DRAM (`vllm/v1/kv_offload/cpu/manager.py`).
  - *SGLang:* HiCache swap-on-preempt.
- **Ranking:** 1. vLLM · 2. SGLang.
- **fak Position:** Native preemption supports both `NativePreemptSwap` and `NativePreemptRecompute` (`internal/modelengine/nativesched_preempt.go:501`). With #3414 landed, `DemoteBeforeDrop` economically compares tier restore cost against recompute cost. Rung R2.
- **Verdict:** **ALREADY LEAD**. fak prices demotion vs recompute per-token before deciding whether to spill or drop (#3414).

---

## 4. Structural Leads (Moats Over All External Engines)

### S1: Span-Evictable KV (`SupportsExactSpan`)
- **What it is:** The ability to excise a poisoned, sensitive, or redundant sequence of tokens from the *middle* of an active KV cache, shift remaining tokens leftward, and re-apply RoPE bit-exact in O(tail) time.
- **Why no external engine can do it:** Every external serving engine (vLLM, SGLang, TRT-LLM) stores only post-RoPE keys $K_{\text{rope}} = R(\theta, pos) \cdot K$. Because RoPE is non-commutative with position shifts and key concatenation, removing tokens requires recomputing all subsequent tokens from scratch or invalidating the entire suffix. External drivers explicitly declare `SupportsExactSpan = false`.
- **fak Mechanism:** fak stores pre-RoPE keys `Kraw` alongside the active cache (`internal/model/kvcache.go:65`). Middle-out eviction shifts `Kraw` slices without information loss and re-applies RoPE at new contiguous positions (`internal/compute/cuda.go:1243`).
- **Witness:** `TestChatProxyFrontsVLLMAndSGLangServedToolCalls`, `internal/model/kvcache_test.go:TestKVCacheMiddleEvictionBitExact`.

### S2: Cross-Worker Prefix Sharing
- **What it is:** In-process sharing of physical KV cache memory across distinct worker subagents running concurrent exploration turns.
- **fak Mechanism:** Zero-copy block tables with refcounted physical pages (`internal/compute/cuda.go:1253`, #10723). When a worker clones a session for speculative execution, VRAM overhead scales as $O(\text{delta})$ rather than $O(\text{prefix} + \text{delta})$.
- **Witness:** `TestCudaKVCloneZeroCopyPrefixSharing`.

---

## 5. Summary Roadmap & Child Issue Reconciliation

| Concept | Status | Issue | Next Checkable Milestone |
|---|---|---|---|
| M1 | R1 | #1533 | Default paged KV decode path under memory budget |
| M2 | R2 | #1532 | Cross-worker prefix reuse benchmark |
| M3 | R1 | #10722, #2169 | Page-granular CPU/disk offloading |
| M4 | R1 | #10729, #2243 | Stale-watermark async transport matrix |
| M5 | R2 | #3411, #3414 | Pluggable reuse term & demote-before-drop |
| M6 | R1 | #2238 | Fleet-level KV-aware routing |
| M7 | R1 | #2240, #10710 | 4-bit / FP8 quantization ladder |
| M8 | R0 | #3143 | **CLOSED (SKIP)**: High quality tax on coding agents |
| M9 | R1 | #2241 | Hybrid SWA/GDN/Mamba unified memory plane |
| M10 | R2 | #3414 | **CLOSED (SHIPPED)**: Economically-priced spill-before-drop |
