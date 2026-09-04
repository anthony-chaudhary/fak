---
title: "Hot-Swappable AI Inference Serving Architecture — Concept, Control Plane, and RSI Sweep Spine (2026-09-03)"
description: "Architecture for dynamic zero-downtime reconfiguration of AI inference serving parameters, shift-left validation, monotonic epochs, and transient sweep APIs for autonomous RSI agents."
---

# Hot-Swappable AI Inference Serving Architecture — Concept, Control Plane, and RSI Sweep Spine (2026-09-03)

## 1. Executive Summary & Root Cause Analysis

### 1.1 The Industry Pathology: Startup-Flag Immutability
All major AI inference engines in the current ecosystem (vLLM, SGLang, Dynamo, Mooncake, HiCache, TensorRT-LLM) are architected as immutable binaries with static startup parameters:

```text
"All of the software in the AI inference engine stack (vllm, sglang, dynamo, mooncake, hicache, etc.)
is miserable to work with. They're all built as immutable binaries. EVERY SINGLE KNOB IS A STARTUP FLAG!
Change completion deadline? REDEPLOY.
Alter speculative depth? REDEPLOY.
Mod queue size limit? REDEPLOY.
We've had to fork and move things into db based configs where possible so we can experiment.
Completely unrealistic to bounce the whole service every time you want to tweak something."
```

In production and research environments, this startup-flag rigidity causes severe system pathologies:
1. **Cold-Start Blackouts:** Rebounding an 8×H100/H200 node hosting a 70B+ or MoE model incurs 2 to 8 minutes of downtime while re-reading tens of gigabytes of weights from disk/network, establishing CUDA IPC across tensor-parallel ranks, allocating hundreds of gigabytes of GPU VRAM, and executing CUDA graph warmup captures.
2. **Prefix-Cache Annihilation:** In-flight prefix-cache trees (Radix trees in SGLang, hash-indexed block tables in vLLM/HiCache) are completely purged on process restart. Multi-turn agent sessions lose prompt cache locality, collapsing Time-To-First-Token (TTFT) across the cluster.
3. **Impossibility of Dynamic Adaptation:** Real-time adaptation to diurnal traffic shifts (e.g., dynamically trading throughput for latency via batch sizes, or contracting speculative depth $K$ under heavy prefill saturation) is blocked.
4. **Stifled Recursive Self-Improvement (RSI):** Autonomous benchmark agents, Bayesian optimization controllers, and RL search loops cannot perform parameter sweeps or Pareto frontier exploration over serving configurations without redeployment thrashing.

### 1.2 Root Cause Decomposition
Why did the inference stack converge on this flawed pattern?
1. **Scripted Tooling Inertia:** Engine entry points were originally designed as prototype Python CLI scripts relying on `argparse`/`click` passing static arguments to top-level constructors.
2. **Monolithic GPU Memory Allocation:** Memory managers (e.g., PagedAttention block allocators) assume static physical pools (`gpu_memory_utilization = 0.90`) established once at initialization via `cudaMalloc` to avoid runtime CUDA memory allocator locks.
3. **Tight Coupling of Configuration and Computation:** Worker threads, ray actors, and asynchronous scheduler loops close over configuration objects by value rather than through atomic pointer dereferences or versioned generation epochs.

---

## 2. Value Frame & Problem Checklist

### 2.1 Value Frame
- **Centrality:** Core. Inference serving adaptability directly dictates cluster utilization, tail latency SLAs, and autonomous agent experimentation velocity.
- **For:** Operators managing multi-tenant LLM inference clusters and autonomous RSI agents running continuous performance sweeps.
- **Problem:** Every operational adjustment (timeouts, queue sizes, speculative depths, preemption rules, memory budgets) requires terminating the inference process, wiping prefix caches, and incurring multi-minute cold starts.
- **Today:** Operators maintain external database wrappers or fork open-source engines to hack dynamic behavior; `fak` has isolated atomic reloads for policy and routing manifests but lacks a unified, wire-addressable dynamic configuration control plane across gateway, scheduler, and engine subsystems.
- **Better because:** Operators and agents can steer, shift, sweep, and hot-swap every serving knob over HTTP/gRPC/UDS with microsecond latency, zero request drops, preserved prefix caches, shift-left relational validation, and instant rollback on anomaly.
- **Witness:** A high-frequency parameter sweep (altering queue limits, completion deadlines, speculative depth $K$, and batch token ceilings at 50 Hz under 100% saturation) completes with zero dropped requests, zero CUDA errors, monotonic epoch progression, and preserved prefix cache hit rates.
- **Next-best alternative:** Process bounces wrapped in blue/green Kubernetes rolling updates. This doubles GPU hardware requirements, purges prefix caches, takes minutes per change, and cannot support fine-grained agentic sweeps.

### 2.2 P1–P4 Problem Checklist
| Check | Requirement |
|---|---|
| **P1 managed context** | Preserve prefix caches (Radix trees, hash-indexed KV blocks) and active session lineage across hot swaps. Never wipe state when adjusting operational parameters. |
| **P2 net-true efficiency** | Zero-copy atomic reads on the inference hot path ($<1$ ns CPU overhead); async drain and background rebalancing for resource shifts; eliminate cold-start downtime. |
| **P3 bounded adaptation** | Shift-left schema and relational invariant validation; monotonic configuration epochs; automatic watchdog rollback to Last-Known-Good (LKG) on performance anomaly. |
| **P4 integrated operations** | First-class introspection (`GET /v1/control/effective`), dry-run diff preview (`POST /v1/control/apply?dry_run=true`), and transient sweep API (`POST /v1/control/sweep`) with cryptographic execution receipts. |

---

## 3. Four-Tier Taxonomy of Hot-Swappable Knobs

AI inference serving knobs are classified into four distinct tiers based on state mutability, synchronization cost, memory impact, and lifecycle orchestration requirements:

```text
+-------------------------------------------------------------------------------+
| Tier 0: Pure Scalars & Soft Limits                                            |
| Primitive: Atomic values (atomic.Uint64, atomic.Value, atomic.Pointer)         |
| Latency Impact: < 1 ns (L1 cache hit; in-register dereference)                |
| Lifecycle: Instantaneous; zero engine disruption.                             |
+-------------------------------------------------------------------------------+
                                      |
+-------------------------------------------------------------------------------+
| Tier 1: Algorithmic & Scheduling Policies                                     |
| Primitive: RCU (Read-Copy-Update) / Atomic Pointer Swap                       |
| Latency Impact: ~1-5 ns read barrier at iteration boundary; zero pause       |
| Lifecycle: Atomic epoch transition; in-flight iterations finish under old     |
+-------------------------------------------------------------------------------+
                                      |
+-------------------------------------------------------------------------------+
| Tier 2: Resource Allocations & Memory Partitions                              |
| Primitive: Dynamic Credit/Quota Managers, Asymmetric Elastic Allocator       |
| Latency Impact: Async background rebalancing; zero worker stall                |
| Lifecycle: Expand is instant; Shrink requires active block/sequence drain     |
+-------------------------------------------------------------------------------+
                                      |
+-------------------------------------------------------------------------------+
| Tier 3: Subsystem & Module Hot-Swapping                                       |
| Primitive: Multi-Rank State Machine + CUDA Stream Staging Pipelines           |
| Latency Impact: Pipelined DMA transfer; atomic generation switch at quiesce   |
| Lifecycle: Prepare -> Load -> Warm -> Quiesce/Switch -> Drain Old             |
+-------------------------------------------------------------------------------+
```

### 3.1 Tier 0: Pure Scalars & Soft Limits
- **Scope:** Single-word primitives governing gate limits, thresholds, and scalars without structural memory side-effects.
- **Concrete Knobs:**
  - Request timeouts & completion deadlines (`completion_deadline_ms`, `stream_progress_timeout`).
  - Maximum wait queue depth (`max_waiting_seqs`).
  - Rate-limiting token buckets (burst capacity, refill rates).
  - Speculative draft depth $K$ clamped to static ceiling: $K \in [0, K_{\max\_preallocated}]$.
  - Speculative acceptance threshold (entropy/probability threshold for token acceptance).
  - Context compaction history budget & anchor head (`compact_history_budget`, `compact_anchor_head`).
  - Log levels, tracing sampling rate, metrics export intervals.
- **Mechanism:** `atomic.Uint64` / `atomic.Pointer[ScalarConfig]` loaded directly during request ingress or scheduler ticks. Zero locking overhead.

### 3.2 Tier 1: Algorithmic, Routing & Scheduling Policies
- **Scope:** Composite policy structs governing how requests are prioritized, batched, preempted, or routed.
- **Concrete Knobs:**
  - Priority queuing strategy (FCFS, Deadline-Earliest-First, Cost-Fairness).
  - Continuous batching token budgets (`max_batch_tokens`, `max_num_seqs`).
  - Preemption strategy (`NativePreemptRecompute` vs. `NativePreemptSwap`).
  - Prefix-cache eviction algorithm (LRU, SLRU, S3-FIFO).
  - Aspect-based model routing tables and ensemble aggregation rules.
- **Mechanism:** Read-Copy-Update (RCU). Readers load an `atomic.Pointer[SchedulingPolicy]` once at the start of each scheduler iteration tick (e.g., every 10–25ms). The iteration runs lock-free on an immutable policy snapshot.

### 3.3 Tier 2: Resource Allocations & Memory Partitions
- **Scope:** Slicing, budgeting, and reserving physical or logical memory pools.
- **Concrete Knobs:**
  - PagedAttention KV-cache block quotas (maximum active blocks per tenant or global cache budget).
  - GPU-to-Host swap space reservation pools.
  - Active sequence concurrency ceilings (`max_running_seqs`).
  - Context MMU memory pool quotas.
- **Mechanism:** Asymmetric Elastic Allocator:
  - **Expansion:** Instantaneous increment of the atomic block budget counter. Blocks move from unreserved pool to the active free list.
  - **Contraction (Cascading Impact Handling):** Asynchronous progressive drain state machine:
    1. Enter `DRAINING` state; pause new sequence admissions requiring expansion.
    2. Evict unpinned prefix-cache blocks (e.g., LRU leaves) down to target watermark.
    3. If active sequences occupy blocks, allow them to finish or trigger non-destructive preemption (recompute-checkpoint or swap-to-host).
    4. Once active allocations fall below target, lock in new budget and return excess memory.

### 3.4 Tier 3: Subsystem & Module Hot-Swapping
- **Scope:** Heavyweight replacement of execution modules, weights, or auxiliary models requiring CUDA device transfers or multi-rank synchronization.
- **Concrete Knobs:**
  - Speculative draft model swap (e.g., swapping a 0.5B draft model for a 1.5B draft model or an MLP head).
  - Policy filter rule-sets & guardrail classifier models.
  - Tokenizer updates (regex, special tokens).
  - Native inference execution backend (e.g., Triton vs. specialized CUDA GEMM kernels).
- **Mechanism:** 5-Stage Lifecycle State Machine (`Prepare` $\to$ `Load` $\to$ `Warm` $\to$ `Quiesce/Switch` $\to$ `Drain Old`) with all-rank barrier across Tensor Parallel workers.

---

## 4. Shift-Left Validation & Control Plane Architecture

Dynamic configuration in safety-critical, high-throughput systems cannot rely on optimistic writes. Updates must be verified before admission and reversible in microseconds.

```text
                    Config Update Request (HTTP/gRPC/UDS)
                                     |
                                     v
                        +-------------------------+
                        |    Shift-Left Parser    |
                        |  (Schema & Type Bounds) |
                        +-------------------------+
                                     |
                                     v
                        +-------------------------+
                        |   Relational Validator  |
                        |  Checks Cross-Field     |
                        |  Invariants & Hardware  |
                        +-------------------------+
                                     |
                      [dry_run=true] | [dry_run=false]
                      +--------------+--------------+
                      |                             |
                      v                             v
             +-----------------+          +--------------------+
             | Diff & Resource |          | Two-Phase Commit   |
             | Impact Preview  |          | Generation Staging |
             +-----------------+          +--------------------+
                                                    |
                                                    v
                                          +--------------------+
                                          | Atomic Epoch Swap  |
                                          | (Active -> Epoch N)|
                                          +--------------------+
                                                    |
                                                    +--------------------+
                                                    |                    |
                                                    v (Health Check OK)  v (Metric Anomaly / Failure)
                                           +----------------+   +-------------------+
                                           | Commit Settled |   | Instant Rollback  |
                                           | Retire Epoch   |   | to LKG (Epoch N-1)|
                                           | N-1            |   +-------------------+
                                           +----------------+
```

### 4.1 Shift-Left Validation Contracts
1. **Syntactic & Type Bounds Checking:** Validates scalar ranges (e.g., $K \in [0, 8]$, `timeout_ms > 0`).
2. **Relational Invariant Verification:** Cross-field semantic safety checks:
   - `max_batch_tokens >= max_model_len` (prevents deadlock where a single sequence cannot fit in an iteration).
   - `speculative_draft_depth <= max_preallocated_draft_slots` (prevents GPU out-of-bounds pointer writes during draft verification).
   - `target_block_count * block_size_bytes <= available_vram - model_weights_bytes - activation_headroom`.
3. **Dry-Run & Impact Preview Endpoint:**
   `POST /v1/control/apply?dry_run=true` evaluates the proposed mutation against live engine state without applying it. Returns:
   - Structural unified diff (`current` vs. `proposed`).
   - Estimated VRAM delta ($\pm \Delta \text{ MB}$).
   - Required drain duration (estimated based on current sequence queue).
   - Transition risk score (`LOW`, `MEDIUM`, `HIGH_DRAIN_REQUIRED`).

### 4.2 Monotonic Epochs, LKG Double-Buffering & Canary Watchdog
- **Monotonic Configuration Epochs:** Every applied state mutation increments a 64-bit unsigned integer: `ConfigEpoch`.
- **Last-Known-Good (LKG) State:** The engine maintains an in-memory double buffer:
  - `ActiveSlot`: Pointer to current configuration.
  - `LKGSlot`: Pointer to the most recent configuration that completed a defined stabilization window (e.g., 60 seconds with zero runtime panics, zero CUDA exceptions, and latency within SLA).
- **Canary Anomaly Watchdog:** A lightweight monitor thread inspects engine telemetry every 500ms.
  - **Automatic Rollback Triggers:**
    1. Speculative acceptance rate collapses by $> 50\%$ relative to baseline.
    2. TTFT $p99$ degrades beyond configured SLA threshold.
    3. Memory allocation raises an internal soft `ENOMEM`.
    4. HTTP 5xx error rate on generation endpoints exceeds $0.1\%$.
  - **Rollback Execution:** The monitor performs an atomic swap back to `LKGSlot` and emits an audit event (`SYSTEM_CONFIG_AUTOMATIC_ROLLBACK`).

---

## 5. Wire Protocols & Sweep API for Agentic RSI

### 5.1 Wire Protocol Surface
The control plane exposes three complementary access routes:
1. **HTTP/JSON REST (`/v1/control/*`)**: Default ingress for operators, CI pipelines, and agent orchestration.
2. **Unix Domain Socket (`control.sock`)**: High-speed, local-host IPC with credential passing for co-located sidecars and daemons.
3. **Memory-Mapped Control Page (`/dev/shm/fak_engine_control`)**: Mapped directly into C++/CUDA worker processes and TP ranks for sub-5ns lock-free reads during batch execution.

### 5.2 The Transient Sweep API (`POST /v1/control/sweep`)
Designed specifically for autonomous Recursive Self-Improvement (RSI) agents running multi-arm parameter exploration and Pareto frontier generation without redeployments.

**Request Payload:**
```json
{
  "sweep_id": "swp_spec_draft_k_2026_09",
  "lease_duration_ms": 30000,
  "quota_completions": 500,
  "overrides": {
    "speculative_draft_depth": 5,
    "max_batch_tokens": 16384,
    "speculative_acceptance_threshold": 0.85
  },
  "tag_routing": {
    "header": "X-Experiment-Group",
    "value": "canary-k5"
  },
  "revert_criteria": {
    "min_acceptance_rate": 0.70,
    "max_p99_ttft_ms": 120
  }
}
```

**Response Payload:**
```json
{
  "status": "APPLIED_TRANSIENT",
  "epoch": 104,
  "previous_epoch": 103,
  "active_until_ms": 1788448830000,
  "remaining_quota": 500,
  "diff": {
    "speculative_draft_depth": {"from": 3, "to": 5},
    "max_batch_tokens": {"from": 8192, "to": 16384}
  }
}
```

**Key Semantics for Autonomous Agents:**
- **Lease Expiration:** If the agent crashes or fails to renew the lease, the engine automatically reverts to `previous_epoch` when `lease_duration_ms` elapses.
- **Quota Clamping:** Once exactly 500 completions have processed under `epoch: 104`, the engine automatically steps back to baseline configuration.
- **Traceable Receipts:** Every completion generated during the sweep carries telemetry headers linking performance directly to the epoch:
  ```http
  X-Inference-Config-Epoch: 104
  X-Inference-Sweep-ID: swp_spec_draft_k_2026_09
  X-Inference-Draft-Accepted-Ratio: 0.82
  ```

---

## 6. Verification, Testing & Concurrency Hardening Methodology

Testing hot-swapping in high-performance runtimes without creating flaky race conditions requires separating **state propagation** from **inference verification**.

### 6.1 Avoiding Flaky Race Conditions via Epoch Stamping
- **The Flaw in Naive Testing:** Firing a config change via HTTP and immediately asserting on a downstream metric often flakes due to in-flight queue latency.
- **The Generation Epoch Contract:**
  1. Every inference request is stamped with the `ConfigEpoch` active at the moment it was scheduled.
  2. Test assertions never poll or sleep. Instead, tests inspect the `ConfigEpoch` embedded in the response headers.
  3. A verification test asserts: *“For all responses where `Response.Epoch >= TargetEpoch`, the new behavior invariant strictly holds.”*

### 6.2 Deterministic Concurrency Test Scenarios (100% Saturation)
1. **The Speculative Depth Sawtooth:** Drive continuous generation at 100% capacity while toggling speculative draft depth $K \in [0, K_{\max}]$ at 50Hz.
   - *Invariant:* Zero CUDA kernel launches with invalid tensor shapes; acceptance metrics instantly adjust; zero NaN outputs.
2. **The Elastic Memory Squeeze:** Artificially contract the PagedAttention token block pool to near-zero free headroom during a 64k-token prefill burst.
   - *Invariant:* Engine triggers preemption / recompute gracefully; no process crash; allocator recovers without leaking KV blocks.
3. **Subsystem Hot-Swap Under Load:** Trigger a Tier 3 draft model or policy rule swap while 128 streams are actively verifying tokens.
   - *Invariant:* Zero CUDA stream deadlocks; in-flight sequences finish cleanly on the old model; subsequent tokens immediately utilize new weights.

### 6.3 Unforgeable Runtime Witnesses
1. **Introspection Receipt (`GET /v1/control/effective`):** Returns the active generation epoch, SHA-256 hash of active configuration, memory allocation counters, and exact timestamp of last mutation.
2. **Telemetry Stream Witness:** Monotonic metrics counters verifying policy transitions:
   - `fak_config_epoch_transitions_total{from="103", to="104"} 1`
   - `fak_kv_blocks_budget_active 32768`
   - `fak_speculative_draft_depth_applied 5`
3. **Response Header Provenance:**
   ```http
   X-Fak-Config-Epoch: 104
   X-Fak-Policy-Hash: a7f8c92...
   X-Fak-Witness: verified-atomic-swap
   ```

---

## 7. Comparative Analysis: Legacy Inference Engines vs. Fak Hot-Swap Architecture

| Dimension | Legacy Ecosystem (vLLM, SGLang, etc.) | Fak Hot-Swap Architecture |
| :--- | :--- | :--- |
| **Knob Modification** | CLI startup flags passed to entry script. | Dynamic Control Plane (REST / gRPC / UDS / shm). |
| **Change Latency** | 2 to 8 minutes (Process kill $\to$ reload $\to$ init). | **$< 1\ \mu\text{s}$** (Tier 0) to **$\sim 100\text{ ms}$** (Tier 3). |
| **Prefix Cache Impact** | **Total Annihilation** (Cold restart purges cache). | **Zero Disruption** (Radix/hash trees survive). |
| **GPU Memory Allocation** | Static `cudaMalloc` chunk at startup. | Static hardware pool with dynamic elastic partitioning. |
| **Agentic Sweeps** | Blocked without external redeployment scripts. | Supported natively via `/v1/control/sweep`. |
| **Rollback Capability** | Redeploy previous Docker image / restart script. | **Microsecond atomic rollback** to Last-Known-Good. |

---

## 8. Implementation Roadmap & Ticket Map (Don't Boil the Ocean)

To deliver this capability reliably without regressions on the shared trunk, the work is decomposed into seven sequential, independently verifiable spines:

1. **Epic / Research Spine:** `epic(serving): hot-swappable AI inference serving control plane and zero-downtime runtime reconfiguration`
2. **Spine 1 (Tier 0 Control Plane & Ingress):** `feat(gateway): unified hot-swap control plane ingress and atomic scalar configuration table (Tier 0)`
3. **Spine 2 (Shift-Left Validation & LKG Rollback):** `feat(control): shift-left dry-run validation, relational invariants, and canary auto-rollback watchdog`
4. **Spine 3 (Tier 1 Scheduling & Batching RCU):** `feat(scheduler): RCU iteration-boundary policy hot-swapping for batch token budgets and preemption rules (Tier 1)`
5. **Spine 4 (Dynamic Speculative Decoding Tuning):** `feat(nativeperf): dynamic speculative draft depth clamping (K in [0, K_max]) and acceptance threshold hot-swap`
6. **Spine 5 (Tier 2 Elastic Memory & Cache Drain):** `feat(modelengine): elastic PagedAttention KV-cache block allocator with progressive drain state machine (Tier 2)`
7. **Spine 6 (Agentic RSI Sweep API & Telemetry Receipts):** `feat(rsi): transient sweep API (POST /v1/control/sweep) with leased epochs and provenance-stamped completion receipts`
8. **Spine 7 (Deterministic Concurrency Stress Harness):** `test(serving): deterministic concurrency stress test suite for live configuration mutation under 100% saturation`
