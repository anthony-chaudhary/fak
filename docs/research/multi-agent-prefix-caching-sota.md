# Multi-Agent Prefix Caching: SOTA Survey and Architectural Gap Analysis

> TL;DR: Multi-agent teams build branching dialogue trees that degrade standard linear KV caches. This survey examines SGLang, Mooncake, vLLM, and academic systems to blueprint tree caching in FAK.

Multi-agent prefix caching is the technique of sharing key-value tensors across branching agent execution graphs.

```bash
python tools/doc_appeal_scorecard.py --target docs/research/multi-agent-prefix-caching-sota.md
```

## 1. Executive Summary and Problem Statement

Modern agent tools group complex software tasks into coordinated multi-agent teams with distinct roles for planning, coding, and verification. A team coordinator sends jobs to four to sixteen leaf workers. Each worker runs in parallel across tree-disjoint lanes to inspect code, run tests, and report results back to the team. Tasks run fast. The workers share large prompt contexts.

Shared context includes base instructions, repository maps, and workspace rules that describe how tools behave during typical execution cycles. When you execute agents concurrently, standard engines recompute these shared tokens for each worker from scratch on every single turn. Compute costs spike. VRAM fills up fast.

Standard model servers assume linear token sequences during batch decoding passes where requests arrive independently from disconnected end users. That assumption fails on branching workloads where many subagents share long system prompts and distinct task histories. Duplicate prefill passes choke GPU tensor cores across all workers. Delays climb. TTFT spikes. Queues stall.

This survey evaluates modern serving engines across the open-source community and research literature. We study tree-structured caches, split worker pools, and memory fabrics. Finally, we map these tools to concrete upgrades for FAK.

## 2. Multi-Agent Workload Dynamics and Tree Branching

Standard chat sessions follow a linear path. Each turn adds new tokens to a single sequence. The engine caches past tokens and prefills only the newest user prompt. Simple caches work well here.

Multi-agent workloads form branching trees across parallel worker sessions that share identical system prompts but diverge during tool execution. The coordinator creates a root context with shared rules and tool schemas. Workers fork from this root to run tree-disjoint tasks. Workloads branch fast.

```
       [System Prompt + Repo Map + Workspace Rules] (Root)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Worker A: Refactor]              [Worker B: Test Suite]
            │                                 │
     ┌──────┴──────┐                   ┌──────┴──────┐
     ▼             ▼                   ▼             ▼
[Tool Call]  [Tool Output]       [Tool Call]  [Tool Output]
```

Consider sixteen workers branching from a twenty thousand token root context that contains system policies and large workspace schemas. Naive engines repeat prefill sixteen times across identical prefix tokens. That burns over three hundred thousand tokens of redundant compute before any agent can return an answer.

Queue times climb steeply. Memory bandwidth stalls. Tree caching solves this problem cleanly by storing the shared root tensor once across all active worker sessions in the cluster. Branching workers attach as child nodes and compute only their unique suffix tokens during prefill. Cache hits rise. Prefixes stay intact. Tokens stay warm.

## 3. Dynamic Tree Caching: SGLang and RadixAttention

SGLang adds RadixAttention to automate KV cache reuse across diverse agent requests that share common instruction prefixes in their prompts. Older engines force users to mark shared prefixes by hand. SGLang treats the entire cache pool as a compressed radix tree. Cache lookup is automatic.

In this radix trie, edges hold lists of token IDs while nodes store handles to physical key-value tensor buffers in device VRAM. The tree updates on the fly as requests enter and finish. Prefixes stay hot.

### 3.1 Prefix Matching and Edge Splitting

Incoming prompts walk the tree from the root node to locate the longest matching prefix and skip redundant matrix multiplication passes. SGLang matches prompt tokens against edge labels to find the longest cached prefix. The engine reuses matching key-value tensors and computes only the remaining suffix.

If a prompt diverges midway along an existing edge, SGLang splits that edge into two pieces. The shared segment becomes a new parent node. The divergent paths become child edges linked to the new parent. Tree splits stay fast.

### 3.2 Eviction and Upward Collapsing

GPU RAM bounds cache capacity across long multi-agent sessions. Under memory pressure, SGLang evicts leaf nodes that lack active request pins using a standard least recently used replacement policy.

When a leaf drops, its parent remains intact. SGLang collapses single-child parent nodes upward when appropriate to save index space while keeping shared token blocks available. Hot root prefixes survive long serving sessions. Root tensors remain resident.

### 3.3 Cache-Aware Routing and Speculative Decoding

In cluster setups, front routers track worker cache state across all backend nodes to direct requests toward existing warm prefixes. The router sends incoming requests to the GPU node that hosts the longest prefix match. This routing policy raises cluster cache hit rates.

SGLang also adapts RadixAttention for tree decoding across speculative draft branches. Speculative decoders test multiple candidate token branches in parallel. The radix tree verifies draft tokens without duplicate tensor copies. Speed goes up.

## 4. Disaggregated Prefill and Decode: Mooncake

Mooncake powers long-context inference for Moonshot AI and Kimi by splitting the serving pipeline into distinct prefill and decode stages. The system resolves resource conflicts between prefill and decode phases. Running both phases on shared chips causes delay spikes. Tail times suffer. Jitter grows.

Prefill operations are compute-heavy. Large matrix multiplications flood tensor cores at high arithmetic intensity. Decode operations are memory-bandwidth bound, reading large caches to generate one token per step. Hardware needs differ.

### 4.1 Phase Separation

Mooncake splits prefill nodes from decode nodes into separate server pools. Prefill pools run batch prefill jobs on compute-dense chips. Decode pools focus entirely on token generation speed.

This split stops batch interference between heavy prompt inputs and lightweight token steps, ensuring smooth delivery across active connections. Large prefill bursts cannot pause active decode streams. Time to first token drops. Token delivery stays smooth.

### 4.2 Decentralized Memory Fabric and Streaming

Mooncake builds a decentralized store for key-value tensors across cluster hardware. It organizes GPU RAM, host RAM, and local SSD drives into one tiered cache. A distributed metadata plane tracks page allocation across cluster nodes.

Prefill nodes stream key-value tensors to decode nodes over high-speed RDMA links during prefill computation to overlap network and compute. Mooncake chunks tensors into transfer units during prefill. Decode instances start generating output tokens before prompt prefill finishes. Work runs in parallel.

## 5. Paged Attention and Block Hashing: vLLM and Sarathi-Serve

The vLLM engine provides the foundation for open-source model serving across modern cloud infrastructure through its PagedAttention memory manager. Its core tool, PagedAttention, organizes key-value cache memory into non-contiguous physical pages.

PagedAttention cuts out external memory fragmentation across dynamic request batches by mapping virtual sequence spaces to physical page tables. Each block stores sixteen or thirty-two tokens. Pages map cleanly.

### 5.1 Automatic Prefix Caching

The vLLM engine adds Automatic Prefix Caching using chained cryptographic hashes that link each token block to its ancestral context. The runtime hashes token blocks in sequence from the prompt origin. Each block hash depends on its own tokens and the prior block hash.

```
Block 0: Hash(Tokens[0..15]) ──────────────────────────┐
                                                       ▼
Block 1: Hash(Tokens[16..31] + Hash(Block 0)) ─────────┐
                                                       ▼
Block 2: Hash(Tokens[32..47] + Hash(Block 1)) ─────────┘
```

Block hashing introduces quantization boundaries into prefix matching logic across subagent prompts. If a shared system prompt contains fifty-five tokens with a sixteen-token block size, three blocks cache forty-eight tokens. The remaining seven tokens require recomputation. Tail tokens miss cache.

### 5.2 Chunked Prefill Mechanics

Sarathi-Serve and vLLM use chunked prefill to smooth out scheduling pauses. Long prefill prompts stall waiting decode requests. Chunked prefill splits incoming prompts into fixed batches of five hundred tokens.

The scheduler interleaves prefill chunks with active decode steps to keep GPU tensor cores saturated while maintaining steady token timing. This balance ensures steady GPU use while keeping decode token timing predictable across concurrent requests. Pipeline bubbles disappear.

## 6. Academic Literature and the RoPE Misalignment Problem

Academic teams have built several caching tools:

- PromptCache reuses attention states across modular prompt segments using position-independent coordinate frames.
- DistServe separates prefill and decode clusters to satisfy distinct latency objectives.
- Splitwise assigns compute-optimized GPUs for prefill and memory-rich chips for decode.
- AgentKV compresses agent scratchpad tokens while retaining shared dialogue subtrees.

### 6.1 The Rotary Position Embedding Formula

Rotary Position Embedding encodes token position by rotating query and key vectors. For a vector at sequence index $m$, RoPE applies orthogonal rotation matrix $R_m$:

$$k_m = R_m W_k x_m, \quad q_n = R_n W_q x_n$$

The attention dot product depends strictly on relative distance:

$$q_n^T k_m = (W_q x_n)^T R_n^T R_m (W_k x_m) = (W_q x_n)^T R_{m-n} (W_k x_m)$$

Relative coordinate invariance holds when positions grow one by one from zero. Angles stay aligned.

### 6.2 The Coordinate Misalignment Problem

Multi-agent engines often try non-prefix middle-segment caching to share reusable subroutines. For example, an agent might cache tool definitions at positions one hundred to three hundred. Later, it might paste those tokens into a different prompt at position five hundred.

This operation causes severe coordinate misalignment. Cached key vectors retain rotation angles computed from their original positions. When new queries at position six hundred attend to these keys, the relative offset becomes incorrect.

The shifted angular distance distorts attention scores across the entire sequence context. The model attends to the wrong tokens, causing total output collapse. Output text turns to garbage.

```
Original Context:  [Pos 0 .. 99] [Cached Block: Pos 100 .. 300]
New Context:       [Pos 0 .. 499] [Cached Block Injected Here!]
Problem:           Keys carry rotation (100 .. 300) instead of (500 .. 700).
Outcome:           Relative distance (600 - 100 = 500) corrupts attention.
```

### 6.3 Positional Rebinding: PRCR and Kamera

Two primary strategies address rotary coordinate shifts:

1. Strict Prefix Preservation: Caching only contiguous prefixes anchored at position zero. Because positions never shift, rotary coordinates remain valid without adjustment.
2. Positional Rebinding: Algorithms like PRCR and Kamera multiply stored key vectors by compensatory rotation matrices. This transformation converts position $m$ to position $k$ during cache transfer.

Positional rebinding adds matrix multiplication work during cache retrieval. In practice, strict prefix caching remains the dominant production standard. Prefix caching wins.

## 7. Comparative Matrix: Radix Trees versus Block Tables

Modern systems balance structural precision against implementation complexity. The following table contrasts key architectural trade-offs:

| Dimension | SGLang (Radix Tree) | vLLM (Block Table) | Mooncake (Fabric) | FAK (Target) |
|---|---|---|---|---|
| Cache Structure | Compressed token trie | Chained hash table | Distributed page index | Hybrid radix-paged |
| Matching Boundary | Token-exact boundary | Block-aligned boundary | Page-aligned boundary | Token-exact boundary |
| Eviction Policy | Tree LRU with collapsing | Block LRU / reference count | Multi-tier LRU (RAM/SSD) | Policy-gated tree LRU |
| Branching Overhead | Zero-copy pointer fork | Block table duplication | Remote transfer setup | Zero-copy COW fork |
| Disaggregation | Worker routing proxy | Co-located engine | Full RDMA P/D split | Local UMA zero-copy |
| RoPE Strategy | Strict prefix anchor | Strict prefix anchor | Strict prefix anchor | Prefix anchor + checks |

Radix trees excel at branching multi-agent workloads. They eliminate the quantization penalties inherent in block tables. However, block tables simplify memory paging across non-contiguous hardware allocations.

## 8. Structural Prompt Normalization and Suffix Isolation

Dynamic metadata represents the greatest threat to prefix cache efficiency across multi-agent workflows. Multi-agent harnesses frequently generate volatile tokens during execution. Examples include request identifiers, wall-clock timestamps, and random seeds.

Consider a harness that places a dynamic timestamp at token position ten. That single timestamp changes every request. Every subsequent token receives a new cryptographic hash and a divergent radix path.

A twenty-thousand token prompt containing tools and policies loses all cache benefit. The prefix cache hit rate plunges from ninety-five percent down to zero. Cache value vanishes.

```
Unnormalized Prompt (Cache Hit = 0%):
[Preamble] -> [Timestamp: 10:42:01] -> [System Policy 20k tokens] -> [User Task]

Normalized Prompt (Cache Hit = 95%):
[Preamble] -> [System Policy 20k tokens] -> [User Task] -> [Timestamp: 10:42:01]
```

To maximize cache hits, you must normalize prompt geometry into four distinct zones:

- Zone 1: Immutable root identity and core system contracts.
- Zone 2: Static tool declarations and workspace taxonomy.
- Zone 3: Long-lived conversation history and shared file snapshots.
- Zone 4: Volatile execution context, timestamps, and turn parameters.

Placing volatile tokens strictly in Zone 4 preserves identical prefix tokens across all subagents. Cache hits stay high.

## 9. Hardware Topologies: RDMA Streaming versus Unified Memory

Hardware architecture dictates optimal cache distribution strategy across cluster and workstation deployments. Cluster deployments face different bottlenecks than unified workstation environments.

### 9.1 Data Center RDMA Networks

Data center clusters connect discrete GPUs over high-speed networks. Modern nodes utilize high-speed InfiniBand or RoCE network interfaces. These links deliver approximately fifty gigabytes per second of network bandwidth.

Transferring key-value tensors across network links incurs latency. For a thirty-two billion parameter model, key-value state requires over one megabyte per token layer. Transferring four thousand cached tokens consumes nearly one hundred milliseconds.

When network transfer time approaches local prefill latency, disaggregation loses its performance benefit. Systems must balance network transfer cost against accelerator compute throughput. Latency budgets matter.

### 9.2 Unified Memory Architectures

Workstation setups feature Unified Memory Architectures like Apple Silicon. High-end chips share unified memory pools across CPU cores and GPU cores. Memory bandwidth ranges up to eight hundred gigabytes per second.

In unified memory systems, prefill and decode workers share physical DRAM. Workers exchange tensor pointers through shared memory mappings without copying bytes across bus links. FAK uses this zero-copy path to eliminate network delays. Zero copy saves time.

## 10. Actionable Blueprint for FAK Subsystems

FAK contains foundational building blocks in `internal/radixkv` and `internal/ctxmmu`. This survey identifies three concrete engineering initiatives to achieve state-of-the-art multi-agent serving.

### 10.1 Paged Subtrees for internal/radixkv

The current `internal/radixkv` package stores complete prefix cache objects at each node. While this ensures strict correctness, it consumes excessive memory during extensive branching.

FAK should evolve `internal/radixkv` to manage paged tensor blocks. Nodes should store references to shared underlying pages using copy-on-write semantics. This transition enables instant subagent forks with negligible memory allocation. Forks become cheap. Nodes fork fast.

### 10.2 Normalization Filters for internal/ctxmmu

FAK must expand `internal/ctxmmu` with automated prompt normalization. The normalizer should parse incoming messages and detect volatile tokens such as timestamps and UUIDs.

The system will reorder message components before tokenization. It pushes dynamic metadata into terminal prompt suffixes while keeping core policies anchored at token zero. This transformation guarantees maximal prefix cache sharing across subagents. Prefixes stay safe.

### 10.3 Cache-Aware Multi-Agent Scheduling

FAK's lane arbiter should incorporate prefix affinity into scheduling decisions. When launching parallel worker cohorts, the scheduler should group subagents sharing common ancestor context.

Assigning related subagents to the same engine instance minimizes cache duplication. Combined with unified memory zero-copy sharing, FAK achieves optimal serving efficiency for multi-agent workloads. Swarms run fast. VRAM overhead drops. Throughput jumps.

## 11. References and Prior Art

| System or Paper | Authors and Year | Venue | Primary Focus |
|---|---|---|---|
| SGLang | Zheng et al. (2023) | arXiv:2312.07104 | RadixAttention and dynamic tree cache |
| Mooncake | Qin et al. (2024) | arXiv:2407.00079 | Disaggregated KV store with RDMA streaming |
| PagedAttention | Kwon et al. (2023) | SOSP 2023 | Non-contiguous virtual memory paging |
| Sarathi-Serve | Agrawal et al. (2024) | MLSys 2024 | Chunked prefill for steady decode timing |
| PromptCache | Gim et al. (2024) | MLSys 2024 | Modular prompt state reuse across tasks |
| DistServe | Zhong et al. (2024) | OSDI 2024 | Disaggregated prefill and decode clusters |
| Splitwise | Patel et al. (2024) | ISCA 2024 | Phase-matched hardware allocation |
| RoFormer | Su et al. (2021) | arXiv:2104.09864 | Rotary position embeddings formula |
