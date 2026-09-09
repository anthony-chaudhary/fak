---
title: "Native Harness: Default Subagents, Unified Effort Slider, and Fused Serving Co-Location"
description: "Architectural specification of fak's default-on native subagents, the 4-tier unified effort/size slider, and hardware-accelerated co-location with Context MMU, RadixKV prefix sharing, and iteration-level continuous batching."
---

# Native Harness: Default Subagents, Unified Effort Slider, and Fused Serving Co-Location

> **Summary:** In conventional LLM architectures, multi-agent decomposition is an economic and latency penalty because every subagent duplicates prompt prefills, bloats KV cache VRAM, and serializes on remote provider queues. In **`fak`**, because model serving, capability enforcement, and the agent harness reside within a single co-located runtime, **subagents on by default is an asymmetric performance advantage**. This document details the architectural foundation, the 4-tier unified effort slider (`low`, `med`, `high`, `ultra` / UltraCode), the serving-side prefix-sharing mechanics, and empirical performance receipts.

---

## 1. The Serving-Side Asymmetry: Why Subagents are Fast in FAK

### The Multi-Agent Cost Paradox in Traditional Architectures

In standard external agent frameworks (LangGraph, AutoGen, CrewAI, naive Claude/Codex multi-agent wrappers), running $N$ subagents imposes severe penalties:

1. **Redundant Prefill Overhead ($O(N \cdot P)$)**: Each subagent is treated as an independent request. For a 16,000-token repository context $P$, running $N=16$ subagents requires calculating **256,000 prefill tokens** repeatedly across separate network calls.
2. **Interactive Time-to-First-Token (TTFT) Queue Blowout**: Remote provider inference queues and local backends without shared prefix trees serialize parallel requests, exploding TTFT from milliseconds into minutes (e.g. $>120\text{ s}$ under multi-slot `llama.cpp`).
3. **KV Cache VRAM Exhaustion**: Each slot allocates independent physical KV memory for identical system prompts, repository files, and tool definitions, resulting in $(N-1) \cdot P \cdot \beta$ wasted memory bytes.
4. **Tool I/O Cache Eviction**: When subagents run local bash commands or unit tests, external serving frameworks evict their idle contexts, requiring full re-prefill upon tool return.

As a result, traditional agent systems are forced to default to **single-agent, sequential execution**.

### FAK's Inverted Economics: Concurrency is Hardware-Accelerated

In `fak`, model serving (`internal/model`, `internal/compute`), capability enforcement (`internal/adjudicator`), and the agent harness (`internal/agent`) share unified process memory. Subagents on by default accelerates execution:

```
STANDARD / REMOTE MULTI-SLOT SERVING (vLLM / llama.cpp / Cloud APIs)
====================================================================
Subagent 1 Slot:  [ Preamble KV (4k) ][ Suffix 1 (512) ][ Decode 1 (64) ]  -> 4,672 tokens VRAM
Subagent 2 Slot:  [ Preamble KV (4k) ][ Suffix 2 (512) ][ Decode 2 (64) ]  -> 4,672 tokens VRAM
Subagent 3 Slot:  [ Preamble KV (4k) ][ Suffix 3 (512) ][ Decode 3 (64) ]  -> 4,672 tokens VRAM
Subagent 4 Slot:  [ Preamble KV (4k) ][ Suffix 4 (512) ][ Decode 4 (64) ]  -> 4,672 tokens VRAM
Total VRAM Footprint: 4 * 4,672 * 128 B = 2.39 GB redundant allocation
Queue Effect: Slot 4 waits behind Slots 1, 2, 3 prefills -> TTFT blows out to 120s-250s.


FAK-NATIVE CONTEXT MMU + RADIXKV SINGLEFLIGHT
=============================================
Unified Memory / UMA DRAM Page Blocks (internal/ctxmmu/cow.go, internal/radixkv)

                  +----------------------------------------------+
                  |  Shared Root Preamble Page Blocks (4,096 tok)|
                  |  refCount = 4, immutable = true, cost ~ 4x   |
                  +----------------------------------------------+
                    /             |              |             \
                   /              |              |              \
             (COW Pointer)  (COW Pointer)  (COW Pointer)  (COW Pointer)
                 /                |              |                \
    +------------------+ +------------------+ +------------------+ +------------------+
    | Subagent 1 Leaf  | | Subagent 2 Leaf  | | Subagent 3 Leaf  | | Subagent 4 Leaf  |
    | Suffix 1 + Dec 1 | | Suffix 2 + Dec 2 | | Suffix 3 + Dec 3 | | Suffix 4 + Dec 4 |
    | refCount = 1     | | refCount = 1     | | refCount = 1     | | refCount = 1     |
    +------------------+ +------------------+ +------------------+ +------------------+
Total VRAM Footprint: [4,096 + 4 * (512 + 64)] * 128 B = 0.82 GB (65.7% physical memory saved)
```

1. **Context MMU Zero-Copy COW Branching (`internal/ctxmmu/cow.go`)**:
   Forking a child subagent from a coordinator is an $O(1)$ pointer copy taking **$<0.03\text{ ms}$** and allocating **$0\text{ bytes}$** of duplicate physical memory. Physical DRAM blocks are marked `immutable = true` with atomic reference counting (`blk.Retain()`). Private pages are allocated only when a child emits distinct tokens.
2. **RadixKV Trie Singleflight Coalescing (`internal/radixkv/singleflight.go`)**:
   When $N$ subagents launch simultaneously with the same repository instructions and tool schemas, the first subagent acts as **Leader** and performs the prefill GEMM. Subagents $2..N$ coalesce on the leader's in-flight promise and bind to the resident prefix instantaneously. Prefill compute work drops from $N \cdot P$ to $P$.
3. **Stationary KV Cache During Tool I/O (`internal/ctxmmu/checkpoint.go`)**:
   When a subagent pauses to execute a tool (file edit, test run, git commit), its KV cache pages remain pinned in UMA DRAM (`SlotStateYieldedIO`). Uncommitted decode headroom is yielded to active peers. When tool output returns, execution resumes immediately at **$0\text{ B}$ bus transfer** and **$0\text{ reprefill tokens}$**.
4. **Batched Autoregressive Weight Amortization (`internal/model/batch.go`)**:
   Autoregressive generation is strictly memory-bandwidth bound ($t_{\text{step}} \approx W / B_{\text{mem}}$). Running $N$ subagents concurrently reads model weights from DRAM **once** per decode step across all $N$ tokens, converting idle memory bus bandwidth into effective token throughput ($N \times \text{tokens/second}$).
5. **vDSO Userspace Read Deduplication (`internal/vdso`)**:
   File reads, repository greps, and directory listings executed by Subagent 1 are cached in the process-global Tier-2 vDSO cache and served directly to Subagents $2..N$ without filesystem thrashing.

---

## 2. Subagents On By Default Across the Native Harness

Subagent task tools are armed and available **by default** across all native execution surfaces (`fak agent`, `fak up`, and `fak harness`).

### The Kernel-Mediated Tool Suite

When subagents are active, the coordinator agent possesses native in-process delegation primitives:

| Tool | Engine | Description |
|---|---|---|
| `task_spawn` | `agent.task_spawn` | Spawns a child task with prompt-scoped intent admission, capacity check, and stable handle return. |
| `task_wait` | `agent.task_wait` | Blocks until target child tasks reach terminal state (`completed`, `failed`) or timeout. |
| `task_status` | `agent.task_status` | Non-blocking inspection of active, pending, completed, and backlog tasks. |
| `task_cancel` | `agent.task_cancel` | Cancels or aborts an in-flight child task by ID. |

### CLI Ergonomics & Configuration

```bash
# Subagents active by default with balanced capacity:
fak agent --task "Fix data race in gateway session table"

# Explicitly disable subagents for single-agent sandbox mode:
fak agent --subagents=false --task "Format documentation"

# Select an explicit subagent effort tier:
fak agent --effort ultra --task "Refactor compute backend to support Vulkan sub-kernels"
```

---

## 3. The 4-Tier Unified Subagent Effort / Size Slider

Rather than treating "effort" solely as a 1D token counter for single-agent internal monologue, `fak` parameterizes effort across **five dimensions**:
1. Concurrency / Fanout Width
2. Active and Backlog Task Capacity Limits
3. Reasoning Token Budget (Thinking Tokens)
4. Pipeline Lifecycle Cohorts
5. Lease and Verification Invariants

```
SLIDER LEVEL:    [ Low / Fast ] ──────► [ Medium / Standard ] ──────► [ High / Rigor ] ──────► [ Ultra / UltraCode ]
Concurrency:     1 Worker               1 - 2 Workers                 3 - 4 Workers             4 - 16 Disjoint Workers
Routing:         Fast / Small (7B/Flash) Balanced (Sonnet/27B)        Heavy Coordinator + Fast  Astra / Frontier Lead + Fleet
Pipeline:        Single-Pass (Direct)   2-Stage (Plan -> Exec)        3-Stage Linear Pipeline   3-Tier Parallel Cohorts
Depth:           Depth 0                Depth 0 -> 1                  Strict Depth 0 -> 1       Depth 0 -> 1 + Detached Trees
Witness Gate:    Exit Code / Build      Unit Tests (go test)          Repro-First + Smoke       Adversarial + DOS Kernel Gate
```

### The Effort Profile Matrix

| Level | Aliases | Thinking Budget | Max Active | Max Backlog | Default Fanout | Pipeline Cohorts | Lease Rules | Primary Workload |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Low / Fast** | `low`, `fast`, `none` | 256 tokens | 1 | 2 | 1 | `false` | Advisory | Mechanical edits, typos, lint cleanup, bounded script generation. |
| **Med / Standard (Default)** | `medium`, `med`, `standard`, `balanced`, `adaptive` | 1024 tokens *(adaptive)* | 2 | 4 | 2 | `false` | Advisory | General feature implementation, bug fixes, package-level tests. |
| **High / Rigor** | `high`, `rigor`, `deep-reason` | 2048 tokens | 4 | 8 | 4 | `true` | Required | Concurrency races, subtle edge cases, failing repro test first. |
| **Ultra / UltraCode** | `ultra`, `ultracode` | 4096 tokens | 16 | 64 | 8 | `true` | Mandatory (`dos arbitrate`) | Subsystem migrations, fleet waves, multi-issue tranches, zero-conflict trunk landing. |

### How "Ultra" Naturally Embodies UltraCode

In `fak`, **UltraCode is not a marketing buzzword or an oversized token budget**—it is the operational mode where problem-solving transitions from monolithic prompt pondering into an **arbitrated, multi-agent distributed system**:

1. **Decoupled 3-Tier Parallel Cohorts**:
   - *Cohort 1: Scoping & Research* (`researcher`, `scout`, `explore`) to isolate prior art and contracts.
   - *Cohort 2: Disjoint Implementation* (`worker`, `deep-reason`) executing across pairwise non-overlapping directory lanes.
   - *Cohort 3: Adversarial Verification & QA* (`cross-validator`, `tester`, `issue-auditor`) verifying fixes with on-device tests and auto-ticketing follow-on edge cases.
2. **Conserved Token Envelopes (`internal/orchestration/ultracode_envelope.go`)**:
   The parent budget is partitioned deterministically among child allocations ($\sum \text{ChildTokens}_i \le \text{ParentBudget}$), guaranteeing fanout cannot trigger runaway token spend.
3. **Collision-Free Trunk Convergence (`dos arbitrate`)**:
   Before any leaf worker writes, its declared file tree is checked against active leases. Overlapping writes are refused with structured `COLLISION_RISK`, ensuring zero git lock conflicts or destroyed peer WIP.
4. **Independent Witnessing over Self-Report**:
   Completed tasks require non-forgeable proof artifacts verified by `dos_commit_audit` (author-neutral diff matching) and `dos_verify` (registry and git ship-stamps).

---

## 4. Empirical Performance Receipts

### A. Apple Silicon M3 Pro Benchmarks (`node-macos-a`, 36 GiB UMA, Metal 4)
*Source: `BENCHMARK-AUTHORITY.md`, `docs/notes/MAC-AGENTIC-4X-QWEN38-2026-09-05.md`, Issue #3809, Issue #2723*
*Workload: Qwen3.8-27B-Q4_K_M ($17.1\text{ GB}$ weights), $K=4$ concurrent agents, $H=20$ turns, $P=4,096$ prefix, $S=128$, $D=64$.*

| Axis / Metric | `fak`-Native In-Kernel | `llama.cpp` Multi-Slot Baseline | Empirical Gain |
| :--- | :--- | :--- | :--- |
| **Total Wall-Clock Time** | **393.8 s** | 732.2 s | **1.86× faster end-to-end** |
| **Effective Throughput** | **13.00 tok/s** | 6.99 tok/s | **1.86× aggregate throughput** |
| **Interactive TTFT (P50)** | **12.6 ms** | 126,030.8 ms | **10,002.4× faster TTFT** (flat vs queue blowout) |
| **Interactive TTFT (P95)** | **12.9 ms** | 252,061.5 ms | **>19,500× faster tail latency** |
| **Preamble Evaluations** | **1 time globally** | 4 times (1 per slot) | **4.0× less preamble compute** |
| **Prompt Token Reuse** | **469,504 tokens (97.0%)** | 0 tokens (0.0% cross-slot) | **Full Track-1 witnessed reuse** |
| **Peak Memory Footprint** | **21.69 GiB** | 24.69 GiB | **3,072 MB (3.0 GiB) saved** |
| **Scaling at 20,000 Prefix** | **566.0 s (9.05 tok/s)** | 1,421.3 s (3.60 tok/s) | **2.51× speedup; 15,000 MB saved** |

### B. NVIDIA A100 Datacenter GPU Benchmarks (`fak-qwen-serve`, us-central1-f)
*Source: `BENCHMARK-AUTHORITY.md`, `docs/_witnesses/issue-10944-nvidia-gcp-overnight/fak_native_qwen38_a100_bench_raw.json`*
*Hardware: NVIDIA A100-SXM4-40GB, CUDA 13.0. Model: Qwen3.8-27B-Q4_K_M.*

```
NVIDIA A100 Concurrency Sweep (Shared Prefix = 1,370 tokens)
========================================================================================
Concurrency (N)   P50 TTFT (ms)   P95 TTFT (ms)   Cluster Tok/s   Cache Hit Rate   Errors
----------------------------------------------------------------------------------------
N = 1               1,319.43        1,319.43        1,040.65       97.9% (1344 tok)  0/1
N = 5               4,206.88        6,699.04          985.08       97.9% (1344 tok)  0/5
N = 10              8,396.90       14,017.50          970.56       97.9% (1344 tok)  0/10
N = 25             17,542.80       23,733.90        1,048.94       97.9% (1344 tok)  0/25
N = 50             27,330.10       34,705.50          997.32       97.9% (1344 tok)  0/50
Aggregate: 91/91 successful invocations, 0 errors, 970-1,049 cluster tokens/sec.
```

- **Prefix Caching Ablation @ 2048 Context on A100**:
  - Cold prompt TTFT: **$15,399.32 \text{ ms}$** (Prefill: $4.43\text{ s} \implies 463.4 \text{ tok/s}$).
  - In-kernel Radix KV cache hit TTFT: **$3,183.57 \text{ ms}$** (**4.84× faster TTFT**; Prefill: $0.00\text{ s}$).
  - Dynamic nonce prompt TTFT: **$19,478.28 \text{ ms}$** (**6.12× speedup** over cold unshared dynamic context).

---

## 5. Architectural Safety Invariants & Guardrails

To prevent subagent proliferation from destabilizing the host or blowing token budgets, `fak` enforces five hard invariants below the model layer:

1. **Strict Depth-1 Leaf Boundary (#12028, #12029)**:
   Leaf workers execute directly within package boundaries (1–3 files). To prevent recursion depth exhaustion, subagent delegation tools (`task_spawn`, `task_wait`) are stripped from leaf schemas, and the gateway enforces a hard `SubagentDepthRule` refusal if invoked by a child.
2. **Conserved Token Allocation**:
   Child workers draw from a single shared parent envelope (`ultracode_envelope.go`). When the envelope is exhausted, all child sessions freeze closed.
3. **Tree-Disjoint File Leases**:
   Subagents cannot write to overlapping paths. Two workers may run concurrently if and only if `dos arbitrate` confirms disjointness.
4. **Author-Neutral Diff Witnessing**:
   Worker completion reports are not evidence. The coordinator accepts work only after `dos commit-audit` verifies the diff witnessed the claimed change, and unaffected tests pass green.
5. **Clean Process Teardown**:
   When a session concludes or aborts, `defer agent.DisarmTaskTools()` cleans in-process thread handles, reaps stationary KV slots, and releases lease locks.
