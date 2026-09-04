---
title: "O(1) Agent Lifecycle Invariants, Thread Resource Quotas, and Growth Bounds Specification"
description: "Formal specification of the O(1) Agent Invariant Contract: launch complexity bounds, thread priority hierarchy, fail-closed admission backpressure, and resource growth ceilings."
---

# O(1) Agent Lifecycle Invariants, Thread Resource Quotas, and Growth Bounds

> **Contract Authority:** This document specifies the invariant bounds governing autonomous agent
> execution, thread resource allocation, queueing backpressure, and storage retention. The machine-checked
> ABI constants and types are defined in [`internal/abi/agent_invariants.go`](../../internal/abi/agent_invariants.go)
> and verified by [`internal/abi/agent_invariants_test.go`](../../internal/abi/agent_invariants_test.go).

---

## 1. Overview & Problem Statement

Autonomous agent fleets and long-running agent workflows degrade over time if lifecycle operations depend
on cumulative historical state. In naive agent harnesses, the $N$-th agent launch often attempts to scan
past logs, read unbounded historical journals, or evaluate uncompacted execution traces. This introduces
$O(N)$ or $O(N^2)$ degradation in latency, active memory, and disk I/O, leading to:

1. **Launch Latency Inflation:** Historical transcript sweeps and unbounded journal replays stall agent startup.
2. **Resource Exhaustion Storms:** Unbounded worker goroutine/thread creation during parallel dispatch exhausts host file descriptors and thread handles.
3. **Cascading Failure Without Backpressure:** Inability to fail closed when queues saturate, causing silent drops or infinite hangs.
4. **Storage Leaks:** Uncapped growth of scratch workspaces, run logs, and session journals across multi-day runs.

To ensure deterministic reliability, this specification formalizes the **O(1) Agent Invariant Contract**.
Every agent launch, execution step, and retirement cycle must adhere to strict mathematical bounds.

---

## 2. O(1) Launch Complexity Invariant

### 2.1 Formal Invariant Definition

Let $N \in \mathbb{N}$ denote the cumulative count of historical agent executions, runs, or sessions recorded
in the workspace or host environment.

Let:
- $T(N)$ be the wall-clock latency from spawn invocation to execution-ready state.
- $M(N)$ be the active resident memory footprint required to initialize the agent execution context.
- $IO(N)$ be the cumulative disk bytes read or written during the spawn sequence.

**The Invariant:**
$$\forall N \ge 0: \quad T(N) = O(1), \quad M(N) = O(1), \quad IO(N) = O(1)$$

The $N$-th agent launch must require zero linear sweeps over previous executions.

### 2.2 Architectural Requirements

To uphold this invariant:
1. **Decoupled Active State:** The agent launch sequence must bind only to fixed-size descriptors:
   - Unique Run ID (`RID-...`)
   - Explicit lane lease lease handle
   - Immutable, pinned policy digest
   - Bounded carryover seed (e.g. `StreamCarryoverSeed` / $O(1)$ context view)
2. **No Unbounded Journal Scans:** Historical journals (`.dos/`, run registries) must not be scanned sequentially on spawn. State lookup must use indexed handles or content-addressed identifiers.
3. **No Unbounded Git Graph Traversal:** Repository tree validation must be restricted to target commit SHAs or shallow branch pointers, never unbounded `git log` sweeps.
4. **Demand-Paged History:** If an agent requires access to historical context, it must demand-page through the context planner (`ctxplan`) or index queries rather than hydrating full transcripts into memory.

---

## 3. Thread Resource Envelope & Priority Tiers

To prevent thread starvation and noisy-neighbor interference between interactive user commands and
unattended background loops, execution threads are partitioned into four strict priority tiers.

### 3.1 Priority Tier Hierarchy

| Priority Tier | ABI Constant | Rank | Allocation Ceiling ($K$) | Target SLA | Intended Workload |
|---|---|:---:|:---:|:---:|---|
| **P0: System/Control** | `ThreadPriorityP0System` | 0 | $K_{P0} = 4$ | Realtime (<5ms) | Kernel health checks, heartbeats, liveness probes, lane lease renewal, emergency cancel/reclaim. |
| **P1: Interactive** | `ThreadPriorityP1Interactive` | 1 | $K_{P1} = 16$ | Low Latency (<50ms) | User CLI queries, direct RPC handling, interactive tool calls, live operator turns. |
| **P2: Batch/Dispatch** | `ThreadPriorityP2Batch` | 2 | $K_{P2} = 32$ | High Throughput | Autonomous background workers, issue sweeps, bulk verification, super-loop turns. |
| **P3: Speculative** | `ThreadPriorityP3Speculative` | 3 | $K_{P3} = 8$ | Opportunistic | Speculative tool execution, branch precomputation, background cache warming. |

### 3.2 Concurrency Caps & Ceilings

The system enforces a global concurrency envelope across all tiers:
$$K_{total} = \sum_{i=0}^{3} K_{Pi} = 64$$

- `DefaultWorkerPoolP0System = 4`
- `DefaultWorkerPoolP1Interactive = 16`
- `DefaultWorkerPoolP2Batch = 32`
- `DefaultWorkerPoolP3Speculative = 8`
- `DefaultMaxTotalWorkers = 64`

### 3.3 Starvation Prevention & Preemption Policy

1. **P0 Guaranteed Reservation:** P0 slots cannot be acquired by P1, P2, or P3 workloads under any condition.
2. **P3 Strict Preemption:** If host resource pressure increases or higher-priority queues congest, active P3 speculative tasks must immediately yield or be cancelled.
3. **Admission Gate:** Worker routines are not spawned via unchecked `go func()` calls. All execution units must pass an admission gate that verifies capacity against the priority tier's active pool.

---

## 4. Fail-Closed Backpressure Contract

When incoming workload exceeds available processing capacity or queue envelopes, the kernel adheres to a
**fail-closed backpressure contract**. Work is rejected deterministically before allocating system resources.

### 4.1 Bounded Admission Queue

- **Hard Capacity:** `MaxQueueCapacity = 512`.
- Any admission attempt when the queue depth has reached `MaxQueueCapacity` fails closed immediately.
- The system never permits unbounded in-memory queues or unbounded channel buffers.

### 4.2 Structured Error Definitions

Admission failures emit typed, structured errors rather than generic string messages:

```go
type AdmissionError struct {
    Code         string `json:"code"`
    Message      string `json:"message"`
    RetryAfterMS int64  `json:"retry_after_ms"`
}
```

#### Standard Error Sentinels:

1. **`ErrQueueFull` (`AdmissionCodeQueueFull = "ERR_QUEUE_FULL"`):**
   - Emitted when the pending queue depth reaches 512.
   - Default retry delay: `DefaultRetryAfterQueueFullMS = 50` ms.
2. **`ErrResourceConstrained` (`AdmissionCodeResourceConstrained = "ERR_RESOURCE_CONSTRAINED"`):**
   - Emitted when thread pools, active memory envelopes, or storage bounds saturate.
   - Default retry delay: `DefaultRetryAfterResourceConstrainedMS = 100` ms.

### 4.3 Backpressure Protocol & Client Retries

When a client or dispatcher receives an `AdmissionError`:
- It **must** honor the `retry_after_ms` delay.
- Retries must apply exponential backoff with decorrelated full jitter.
- If repeated backpressure errors occur, batch and speculative tasks must pause or scale down concurrency rather than hammering the admission gate.

---

## 5. Retention & Growth Bounds

To prevent persistent storage bloat and memory leaks, hard byte ceilings and rolling retention policies are
enforced across all persistence layers.

### 5.1 Hard Storage & Memory Ceilings

| Target Layer | ABI Constant | Hard Ceiling | Rolling / Eviction Policy |
|---|---|:---:|---|
| **Event / Run Journals** | `MaxJournalSizeBytes` | 64 MiB | Segment rotation at threshold; segment compaction; tombstoning completed runs. |
| **Workspace Scratch** | `MaxScratchStorageBytes` | 512 MiB | Global ceiling across all scratch producers; LRU reaping of unleased directories. |
| **Per-Run Scratch** | `MaxPerRunScratchBytes` | 50 MiB | Hard allocation cap per agent session; auto-reaped on session completion. |
| **In-Memory Caches** | `MaxInMemoryCacheBytes` | 256 MiB | Strict SLRU/LRU cache eviction; zero unbounded maps or persistent global pools. |
| **Per-Run Output Logs** | `MaxPerRunLogBytes` | 10 MiB | Head/tail preservation with middle-truncation on oversized logs. |

### 5.2 Lifecycle Phase Invariants

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │    Spawn     │ ──> │    Admit     │ ──> │   Execute    │ ──> │  Terminate   │ ──> │   Reclaim    │
 └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
   O(1) Descriptors     Queue <= 512         Tier Quota (K)       Status Ledgered      Scratch Reaped
   Zero History Scan    Check Res Envel      Preemptible P3       Witness Emitted      Memory Freed
```

1. **Spawn Phase:** Resolves runtime parameters and descriptors in $O(1)$ time without reading historical run logs.
2. **Admit Phase:** Checks pool capacity ($K_i$) and queue limits (depth $\le 512$). Returns `AdmissionError` on saturation.
3. **Execute Phase:** Enforces thread quota and per-run storage caps (`MaxPerRunScratchBytes`, `MaxPerRunLogBytes`).
4. **Terminate Phase:** Writes final state to immutable ledger; flushes buffered output.
5. **Reclaim Phase:** Immediately releases worker pool token, reaps per-run scratch, and marks volatile state for collection.

---

## 6. Implementation & ABI Mapping

All constants, enumerations, error types, and sentinel definitions specified herein are implemented in
`internal/abi`:

- **Source:** [`internal/abi/agent_invariants.go`](../../internal/abi/agent_invariants.go)
- **Unit Verification:** [`internal/abi/agent_invariants_test.go`](../../internal/abi/agent_invariants_test.go)

Any modification to these bounds must preserve additive ABI compatibility and pass the full unit test suite.
