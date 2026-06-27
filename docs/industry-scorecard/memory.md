---
title: "fak industry scorecard — memory"
description: "The memory dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# memory — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## KV cache & prefix reuse (`kv-cache`)

### ○ KV-cache memory efficiency and max concurrent sequences (paging / fragmentation) — fak: **no-claim**

*Why it matters:* GPU memory left for KV cache, after weights, sets the hard ceiling on how many sequences can run concurrently, which caps batch size and therefore throughput. Pre-reserving max-context KV cache wastes 60-80% of memory to fragmentation; paged allocation cuts waste to under 4%, letting the same GPU hold far more concurrent sequences.

- **SOTA bar:** PagedAttention keeps KV-cache memory waste under 4% (near-zero internal/external fragmentation), enabling 2-4x throughput at the same latency; this remains the canonical paged-KV efficiency reference as of mid-2026.
- **Leading systems:** vLLM PagedAttention, SGLang RadixAttention (radix-tree paged KV)
- **Source:** [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180) (2023-09)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE / disclosed capability gap: fak owns its own in-kernel KV cache via copy-CAS (that IS the fusion), with no PagedAttention-style block paging to drive KV-memory waste toward <4% or to maximize concurrent-sequence count on fixed VRAM. CLAIMS marks external zero-copy KV co-residence an explicit [STUB] (~120h, out of scope). No memory-efficiency or max-concurrency number exists; reuses the named-gap row vllm-zero-copy-kv.
- **Trace:** No PagedAttention-style memory-waste or max-concurrency number in BENCHMARK-AUTHORITY.md. CLAIMS.md L161 [STUB] 'No zero-copy KV co-residence with an external serving engine' (copy-CAS path, paging into the working set is residency decomposition, not anti-fragmentation KV paging).

### ≈ Prefix caching / automatic KV reuse across requests (shared-context workloads) — fak: **parity**

*Why it matters:* Real workloads (long system prompts, RAG, few-shot, multi-turn chat, agent fleets) repeat large prefixes. Automatic prefix caching reuses the already-computed KV for shared prefixes, skipping that prefill entirely. On these workloads it is a large, distinct throughput and TTFT multiplier that is invisible to single-request benchmarks but dominates production economics.

- **SOTA bar:** SGLang RadixAttention delivers ~29% higher end-to-end throughput than vLLM on H100 (16,200 vs 12,500 tok/s, Llama 3.1 8B) on shared-prefix workloads, with up to 6.4x on prefix-heavy RAG/multi-turn traffic; the 29% SGLang-vs-vLLM edge still holds in 2026 benchmarks.
- **Leading systems:** SGLang RadixAttention, vLLM Automatic Prefix Caching
- **Source:** [https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison) (2026)
- **fak:** parity — 6.95 × (shipped)
- **fak note:** A ceiling is approached, never led: live wall-clock climbs toward the 7.50× token bound with model size (135M 4.58× → 360M 5.40× → 0.5B 6.20× → 1.5B 6.95×), confirming the residual gap is clone/memcpy overhead that vanishes on larger models, not an architectural limit.
- **Trace:** 92896a4 · experiments/radixattention/radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json · data.json rows 'radix-live-vs-ceiling' (live 6.95x vs 7.50x token ceiling, artifact workloads[0].live_prefill_speedup=6.951) and 'kv-hit-rate' (86.7% in SGLang band); BENCHMARK-AUTHORITY.md RadixAttention Model Ladder (135M 4.58x -> 1.5B 6.95x), commit 92896a4. Split-reuse == recompute max|delta|=0.

### ≈ Prefix/KV-cache reuse impact on TTFT (cache-hit latency) — fak: **parity**

*Why it matters:* Real workloads (shared system prompts, RAG documents, multi-turn chat) share long prefixes. Automatic prefix caching skips recomputing the shared KV, slashing prefill tokens and therefore TTFT on cache hits. The cache-hit vs cache-miss TTFT gap, and the hit rate under a realistic prompt mix, are decisive for chatbot/RAG/agent latency yet are absent from cold synthetic benchmarks.

- **SOTA bar:** Qualitative: APC reuses exact KV pages via hash-based block matching, cutting prefill tokens and TTFT for repeated/long-prefix prompts with negligible steady-state overhead; overhead from hash compute grows with input length and concurrency. Distributed prefix-cache scheduling (llm-d) extends the win across replicas.
- **Leading systems:** vLLM Automatic Prefix Caching (APC), SGLang RadixAttention, llm-d distributed prefix cache
- **Source:** [https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) (2025-01)
- **fak:** parity — 6.95 × (shipped)
- **fak note:** A ceiling is approached, never led: live wall-clock climbs toward the 7.50× token bound with model size (135M 4.58× → 360M 5.40× → 0.5B 6.20× → 1.5B 6.95×), confirming the residual gap is clone/memcpy overhead that vanishes on larger models, not an architectural limit.
- **Trace:** 92896a4 · experiments/radixattention/radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json · experiments/radixattention/radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json (commit 92896a4): live_prefill_speedup=6.95x climbing to the 7.50x exact token-reuse ceiling; cache hit 86.7% (FCFS 62.1% -> cache-aware), inside SGLang's published 50-99% band; split-reuse proven bit-identical to recompute (max|delta|=0).

### ○ Paged KV block management and memory utilization (anti-fragmentation) — fak: **no-claim**

*Why it matters:* KV cache is the dominant runtime memory consumer in LLM serving. Before paging, engines wasted 60-80% of KV memory to internal fragmentation and over-reservation, capping batch size and throughput. Block-based (paged) allocation determines how close an engine gets to theoretical KV capacity, which directly sets max concurrency and tokens/sec on fixed hardware. This is the foundational dimension every serious operator evaluates first.

- **SOTA bar:** PagedAttention achieves ~96.3% KV-cache memory utilization (the complement of <4% waste); no 2026 system reports a higher published paged-KV utilization figure.
- **Leading systems:** vLLM PagedAttention, SGLang RadixAttention
- **Source:** [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180) (2023-09)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. fak does not implement a PagedAttention-style block allocator; it owns a contiguous kernel-owned KVCache (CLAIMS 'In-kernel model') and a radix tree over it for reuse, not anti-fragmentation block management. No utilization % is measured against vLLM's ~96.3%, so no parity is claimed.
- **Trace:** No memory-utilization figure in BENCHMARK-AUTHORITY.md or CLAIMS.md. CLAIMS line 85: in-kernel model owns its KV cache as a Go structure; internal/radixkv rebuilds RadixAttention (edge-split via Evict) over it, but there is no paged-block utilization / fragmentation % measured.

### ≈ Automatic prefix caching / RadixAttention prefix reuse and cache hit rate — fak: **parity**

*Why it matters:* Shared prompt prefixes (system prompts, few-shot exemplars, multi-turn history, RAG templates) are extremely common. Reusing their KV instead of recomputing prefill is the single largest throughput lever for prefix-heavy workloads. The hit rate an engine actually attains under a given workload, and whether reuse is automatic (no app changes), is a primary buyer differentiator between serving stacks.

- **SOTA bar:** Automatic prefix caching now reaches ~85-95% cross-request hit rates on shared-prefix workloads (SGLang RadixAttention few-shot 85-95%, agentic ~88%; vLLM APC ~87% warm-cache hit, ~88% faster TTFT); the original RadixAttention 6.4x throughput claim (LMSYS 2024-01) remains the canonical headline number.
- **Leading systems:** SGLang RadixAttention, vLLM Automatic Prefix Caching
- **Source:** [https://www.lmsys.org/blog/2024-01-17-sglang/](https://www.lmsys.org/blog/2024-01-17-sglang/) (2024-01)
- **fak:** parity — 86.7 % (shipped)
- **fak note:** fak measured on-box: cache-aware (≡ DFS) scheduling recovers FCFS 62.1% → 86.7% = 100% of optimal, inside SGLang's published band. Split-reuse proven bit-identical to recompute (max|Δ|=0), so the hit is a real reuse, not a numerics shortcut. FENCE: 86.7% is on radixbench's synthetic 'agents' workload (deliberately cache-favorable); on a REAL tau2-airline trace fak's measured addressable purity is ~0.7% (CLAIMS vDSO unit 33/83) — so this is a same-workload-as-SGLang reuse-mechanism parity, not a promise of 86.7% on every production corpus.
- **Trace:** 92896a4 · experiments/radixattention/radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json · BENCHMARK-AUTHORITY.md 'RadixAttention hit rate 86.7% (FCFS 62.1% -> cache-aware)' commit 92896a4, artifact radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json; CLAIMS line 93 (radixkv 77.2-88.2% across shapes, inside SGLang's verified 50-99% band, reuse bit-identical to recompute max|Δ|=0).

### ○ Non-prefix / cross-document KV reuse (RAG chunk caching) — fak: **no-claim**

*Why it matters:* Prefix caching only helps when the shared text is a literal prefix. RAG concatenates retrieved chunks in varying order, so prefix reuse misses most of the opportunity. Engines that can reuse KV for arbitrary cached chunks (recomputing only the small fraction needed to fix cross-attention) unlock a second, larger tier of reuse for the most common enterprise workload. This is a sharp differentiator that a naive prefix-only scorecard omits.

- **SOTA bar:** LMCache CacheBlend reuses NON-prefix KV caches (any RAG chunk, not just the first) and selectively recomputes ~15% of tokens, achieving near-100% KV-cache hit rate with 2.2-3.3x lower TTFT and 2.8-5x higher throughput vs full recompute; won ACM EuroSys'25 Best Paper.
- **Leading systems:** LMCache CacheBlend, Mooncake (non-prefix KV pool)
- **Source:** [https://dl.acm.org/doi/10.1145/3790254](https://dl.acm.org/doi/10.1145/3790254) (2025)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure. fak's reuse is strictly prefix-shaped (radix longest-prefix + byte-identical PrefixDigest share gate); it has nothing analogous to CacheBlend's ~15%-recompute reuse of non-prefix RAG chunks. For a kernel whose entire thesis is KV reuse, cross-document/non-prefix reuse is an adjacent capability worth measuring, not an out-of-scope axis.
- **Trace:** No CacheBlend-style non-prefix / cross-document KV reuse in CLAIMS.md or BENCHMARK-AUTHORITY.md. internal/radixkv (CLAIMS line 93) is longest-PREFIX match only; CanShare (line 90) requires a byte-identical PrefixDigest. No partial-recomputation cross-chunk KV fusion exists.

### ○ KV-cache offloading to CPU/host/NVMe/remote (multi-tier hierarchy depth) — fak: **no-claim**

*Why it matters:* GPU HBM holds only a small KV working set. Offloading cold KV to CPU DRAM, local SSD/NVMe, and remote object/RDMA stores expands effective cache capacity by orders of magnitude, raising hit rates and enabling 'trade storage for compute'. The depth of the tier hierarchy, supported backends, and load bandwidth from each tier are what separate a production KV layer from a single-GPU cache.

- **SOTA bar:** LMCache (graduated to production Jan 2026, OSDI/arXiv paper Oct 2025) delivers up to 15x throughput improvement over vanilla vLLM by tiering KV cache across HBM/DRAM/NVMe with a global prefix index; llm-d production benchmarks report up to 57x faster TTFT and 2x throughput vs round-robin under high prefix reuse on 16 H100s.
- **Leading systems:** LMCache (vLLM, production Jan 2026), llm-d native KV offload (57x TTFT), Mooncake Store
- **Source:** [https://arxiv.org/abs/2510.09665](https://arxiv.org/abs/2510.09665) (2025-10)
- **fak:** no-claim — no number (in-flight)
- **fak note:** OUT OF SCOPE as a measured head-to-head, though the policy plane is shipped. fak models a multi-tier hierarchy (incl. CXL/NUMA-far) and decides demote-vs-evict, but it moves no KV bytes and has no access-bandwidth or capacity-multiple number against LMCache's claimed ~7x faster CPU access / ~100x capacity or Mooncake's pooled store. The demote demo's '28000 tokens saved' is vs blind LRU on its own engine, not a cross-tier bandwidth comparison.
- **Trace:** CLAIMS line 142: internal/cachemeta models HBM->DRAM->CXL/NUMA-far->Disk tiers with per-tier TTL, demote-not-evict, and a hardware-cost-driven PlanPlacement; cmd/hwcachedemo is deterministic (28000 prefill tokens saved by demoting vs blind LRU). Explicitly the payload-free POLICY plane: cachemeta touches no bytes; physical movement is left to an engine adapter.

### ○ KV compression / token eviction under a fixed cache budget — fak: **no-claim**

*Why it matters:* For very long contexts, even quantized KV is too large. Eviction/sparsity methods (keep attention-heavy or recent/sink tokens) shrink the resident KV to a fixed token budget, trading a controlled accuracy loss for memory and decode speed. Operators serving long-context or reasoning workloads must evaluate how aggressively a system can compress before quality breaks, and which algorithm it uses.

- **SOTA bar:** Reasoning-aware eviction now beats the old H2O/Quest budget cliff: R-KV achieves lossless compression at 34% KV budget on MATH-500 and 10% on AIME-2024, and at 16% budget exceeds FullKV at 105% accuracy; Apple's EpiCache (2026) gets near-full-cache accuracy at 4-6x compression with up to 40% higher accuracy than eviction baselines and 3.5x lower peak memory.
- **Leading systems:** R-KV (redundancy-aware, reasoning, arXiv 2505.24133), EpiCache (Apple, 2026), SnapKV / H2O / Quest (baselines now superseded)
- **Source:** [https://arxiv.org/pdf/2505.24133](https://arxiv.org/pdf/2505.24133) (2025-05)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP on the lossy-compression axis. fak does provable EXACT eviction (the deletion-cert / quarantine-evict story, a containment strength), not budget-bounded token-dropping that trades quality like H2O/SnapKV/Quest. There is no fak accuracy-at-256/1024-token-budget number, so no parity/lead is claimed; the H2O-class quality-under-budget question is unmeasured for fak. (correctness-oracle reasoning: where fak IS exact-vs-oracle it is parity, but that is on eviction equivalence, not on this compression-quality dimension.)
- **Trace:** CLAIMS line 85/87/93: KVCache.Evict and the quarantine-evict bridge are bit-EXACT (evicted == never-saw, max|Δ|=0), and internal/radixkv EvictNode adds policy-driven span eviction over LRU. But every fak eviction is exact/lossless removal of a chosen span, not lossy compression of a retained cache under a fixed budget; no quality-vs-budget curve is measured.

### ○ KV-cache transfer for prefill/decode disaggregation (NIXL / NCCL / UCX) — fak: **no-claim**

*Why it matters:* Disaggregated serving splits prefill and decode onto separate GPUs, requiring the prefill KV to be shipped to the decode engine. The transport library, supported fabrics (NVLink/InfiniBand/RoCE/NVMe-oF/S3), and whether transfer is non-blocking and overlapped layer-by-layer with compute decide whether disaggregation adds TTFT or hides the cost. This is the connective tissue that makes large-scale disaggregation viable.

- **SOTA bar:** NVIDIA Dynamo 1.0 (GA March 16, 2026 at GTC) delivers up to 7x higher throughput with disaggregated prefill/decode serving combined with wide expert parallelism, measured on DeepSeek-R1 on GB200 NVL72; KV blocks move prefill->decode over RDMA via NIXL at wire speed.
- **Leading systems:** NVIDIA Dynamo 1.0 + NIXL (GA Mar 2026), Mooncake Transfer Engine, SGLang EPD disaggregation
- **Source:** [https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/) (2026-03)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a single-process reuse kernel. fak has no prefill/decode disaggregation and moves no KV over the wire; the KVTransfer directives are policy-plane intents consumed by an external adapter. It claims nothing against NIXL/Dynamo point-to-point RDMA transfer or layerwise overlap.
- **Trace:** No prefill/decode disaggregation or NIXL/NCCL/UCX KV transfer in CLAIMS.md. cachemeta emits KVTransfer offload/restore DIRECTIVES (line 142) and a coherence bus broadcasts refutations (BENCHMARK-AUTHORITY causal-invalidation row), but no cross-GPU KV transport, no layerwise overlap, no measured transfer bandwidth or TTFT-inflation figure.

### ○ KV-cache-aware / prefix-aware request routing across a fleet — fak: **no-claim**

*Why it matters:* A cache is only as good as the routing that lands a request on the GPU that already holds its KV. Across a multi-node fleet, a cache-aware router computes prefix/block overlap and balances it against load to maximize cluster-wide hit rate. Without it, replicas duplicate KV and recompute prefill. This is the dimension that turns per-engine caching into fleet-level economics.

- **SOTA bar:** KV-cache-aware routing across a replica fleet now delivers ~50% lower TTFT, ~34% lower TPOT, ~61% more RPS and ~62% more output tok/s (Baseten on NVIDIA Dynamo, Qwen3 480B), with llm-d reporting 3x throughput / 2x TTFT vs round-robin and GKE Inference Gateway 13.9x faster TTFT than a 3rd-party service.
- **Leading systems:** Baseten + NVIDIA Dynamo KV router, llm-d v0.5 (GAIE/EPP prefix-cache-aware), GKE Inference Gateway prefix-cache-aware routing
- **Source:** [https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/](https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/) (2025-10)
- **fak:** no-claim — no number (in-flight)
- **fak note:** REAL GAP at the fleet axis. fak has cache-aware SCHEDULING within one kernel instance (the FCFS->optimal recovery folded into the kv-hit-rate parity row) and a coherence bus, but not Dynamo-style prefix-aware ROUTING that scores block overlap across many GPUs to pick a worker. No analog of Baseten's ~2x routing speedup is measured; for a fleet-serving kernel this cross-instance router is a gap worth measuring, not out of scope.
- **Trace:** CLAIMS / data.json kv-hit-rate: fak's cache-aware (DFS) SCHEDULING recovers FCFS 62.1% -> 86.7% (100% of optimal) on one instance; the causal-invalidation witness (BENCHMARK-AUTHORITY) broadcasts refutations on a coherence bus. But there is no fleet-level request-to-cached-block overlap ROUTER across GPUs, and no routing speedup number.

### ○ Attention architecture impact on KV footprint (MQA/GQA/MLA bytes-per-token) — fak: **no-claim**

*Why it matters:* The model's attention design sets the floor on KV size before any serving trick applies. MQA/GQA share K/V heads; MLA compresses K/V into a low-rank latent. A 3-5x smaller KV footprint per token compounds with every other lever (paging, offload, quant) to multiply servable concurrency and context. Operators selecting models for serving cost must weigh this, and engines must implement the architecture's KV layout efficiently.

- **SOTA bar:** Multi-head Latent Attention (MLA, DeepSeek-V3) holds ~70 KB/token KV footprint vs 192-328 KB/token for GQA models (a 2.7-4.7x reduction, ~9x vs MHA, ~2x vs 8-group GQA) at higher quality; the frontier is now MLA + learned sparsity (DeepSeek NSA, 2025) pushing toward sub-100 floats/token for long contexts.
- **Leading systems:** MLA (DeepSeek-V3), MLA + Native Sparse Attention (NSA, DeepSeek 2025), Qwen-Latent / MiniMax-Text-01 MLA-style
- **Source:** [https://arxiv.org/pdf/2502.14837](https://arxiv.org/pdf/2502.14837) (2025-02)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE. KV footprint per token is a MODEL-architecture property (MLA vs GQA), and fak is a serving kernel, not a model designer; it faithfully serves whatever attention shape the checkpoint defines (MHA/GQA/MQA argmax-exact, GLM-5.2 DSA dense path on-GPU) but claims no KB/token figure against DeepSeek MLA's ~70 KB/token. Analogous to the swebench-resolve-rate fence: the kernel inherits, it does not set, this number.
- **Trace:** CLAIMS line 146: fak's CUDA flash-attention is correct for MHA/GQA/MQA (cosine=1.0) and line 147 serves GLM-5.2's DSA attention (dense path pure-GPU, sparse DSA still host-side). fak SERVES these architectures faithfully but measures no bytes-per-token KV footprint and ships no MLA-style footprint reduction of its own.

### ○ KV-cache quantization (FP8 / INT8 / KIVI sub-4-bit) accuracy at long context — fak: **no-claim**

*Why it matters:* KV-cache dominates memory at long context and high concurrency; halving it (FP8) or going to 2-4 bits (KIVI/KVQuant) directly buys longer contexts or more concurrent requests at fixed hardware. The buyer question is exactly where it stays lossless vs where it degrades (prefill-heavy, head_dim=256, hybrid sliding-window layers).

- **SOTA bar:** Ultra-low-bit KV quant frontier (2026): Together AI's OSCAR reaches 2.28 effective bits within 1.42 points of BF16 (Qwen3-8B; ~0.02-pt gap on Qwen3-32B) with ~8x KV memory reduction at 100K context and up to 7.83x job-level throughput; TurboQuant (ICLR 2026) is near-lossless at 3-4 bits / full-precision NIAH recall at 4x.
- **Leading systems:** OSCAR (Together AI, 2026), TurboQuant (Google/NYU, ICLR 2026), Kitty 2-bit dynamic channel-wise (arXiv 2511.18643)
- **Source:** [https://www.marktechpost.com/2026/06/18/the-kv-cache-compression-race-turboquant-vs-oscar-vs-epicache/](https://www.marktechpost.com/2026/06/18/the-kv-cache-compression-race-turboquant-vs-oscar-vs-epicache/) (2026-06)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should be aware of, but currently OUT OF SCOPE. fak owns its KV cache (that IS the reuse/eviction moat) but stores it at f32 — there is no FP8/INT8/sub-4-bit KV quantization and therefore no long-context accuracy-retention number. fak's adjacent KV claims (bit-exact Evict/quarantine, 86.7% radix hit-rate) are reuse/correctness properties on an UNQUANTIZED cache, not KV-quant accuracy, so no parity is implied. KV quantization is the natural lever a serving-stack-facing fak would eventually measure.
- **Trace:** none for KV QUANTIZATION accuracy — fak's KV cache is a kernel-owned f32 structure; its KV claims are bit-exact eviction (max|Δ|=0, KV-quarantine bridge) and radix prefix reuse, NOT a quantized (FP8/INT8/KIVI) KV cache with a long-context accuracy study

### ≈ Automatic cross-request prefix / KV-cache reuse (shared system prompts, agent scaffolds, few-shot) — fak: **parity**

*Why it matters:* Agent and fleet workloads repeat large identical prefixes (system prompts, tool schemas, few-shot exemplars). Reusing their KV across requests is the single largest lever on prefill cost and TTFT, and is now table-stakes; buyers compare hit-rate and the throughput gain it delivers.

- **SOTA bar:** RadixAttention's automatic cross-request KV reuse via an LRU radix tree gives up to 6.4x throughput on shared-prefix workloads with 50%-99% cache hit rates; this remains the canonical cross-request prefix-reuse bar in 2026.
- **Leading systems:** SGLang RadixAttention, vLLM Automatic Prefix Caching, Mooncake KV pool
- **Source:** [https://www.lmsys.org/blog/2024-01-17-sglang/](https://www.lmsys.org/blog/2024-01-17-sglang/) (2024-01)
- **fak:** parity — 86.7 % (shipped)
- **fak note:** fak measured on-box: cache-aware (≡ DFS) scheduling recovers FCFS 62.1% → 86.7% = 100% of optimal, inside SGLang's published band. Split-reuse proven bit-identical to recompute (max|Δ|=0), so the hit is a real reuse, not a numerics shortcut. FENCE: 86.7% is on radixbench's synthetic 'agents' workload (deliberately cache-favorable); on a REAL tau2-airline trace fak's measured addressable purity is ~0.7% (CLAIMS vDSO unit 33/83) — so this is a same-workload-as-SGLang reuse-mechanism parity, not a promise of 86.7% on every production corpus.
- **Trace:** 92896a4 · experiments/radixattention/radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json · BENCHMARK-AUTHORITY.md RadixAttention ladder; experiments/radixattention/radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json (commit 92896a4); CLAIMS.md 'RadixAttention parity vs SGLang' [SHIPPED]

