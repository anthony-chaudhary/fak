---
title: "Clean-Coordinator and Bounded-Microagent Execution Topology Specification"
description: "Formal specification of the clean-coordinator pattern, atomic S0/S1 leaf decomposition, compact receipt ingestion, context savings invariants, and structured ABSTAIN boundaries for autonomous agent fleets."
---

# Clean-Coordinator and Bounded-Microagent Execution Topology Specification

> **Contract Authority:** This document specifies the architectural invariants, context boundary contracts,
> atomic S0/S1 task decomposition rules, and structured refusal boundaries for autonomous coordinator-worker
> execution topologies. The machine-checked Go implementation resides in
> [`internal/microagent/coordinator.go`](../../internal/microagent/coordinator.go) and is verified by
> [`internal/microagent/coordinator_test.go`](../../internal/microagent/coordinator_test.go).

---

## 1. Overview & Problem Statement

Autonomous agent fleets processing high-volume backlogs (such as bulk issue triage, super-loop turns, and
multi-task refactorings) suffer severe degradation when the coordinating agent directly absorbs intermediate
worker transcripts. In naive orchestrator implementations:

1. **Context Window Inflation & Explosion:** Every tool execution, compiler output, test failure traceback,
   and terminal chatter is streamed directly into the parent context. A 10-task batch routinely consumes tens
   of thousands of tokens in noisy compiler logs, exhausting context limits and prompting expensive, lossy
   compaction.
2. **Context Pollution from Failures:** When a sub-task encounters compilation errors or panics, the entire
   raw traceback (50–500 lines of compiler diagnostics) pollutes the coordinator's reasoning space. This
   triggers error-looping, self-narrating confabulation, and hallucinations where subsequent task decisions
   are biased by un-isolated historical errors.
3. **Prompt Cache Invalidation:** Injecting dynamic, non-deterministic compiler traces and timestamps into the
   coordinator context breaks provider KV-cache prefix alignment across turns, driving up API latency and cost.
4. **Over-Scaffolding & Scope Creep:** Fast or smaller models (e.g. 7B/14B parameters, fast/flash tier)
   struggle with multi-file, cross-subsystem tasks, generating speculative abstractions, broken lock logic, or
   unsolicited architectural sprawl.

### The Clean-Coordinator Principle

The **Clean-Coordinator Pattern** decouples coordination from execution:

$$\text{CoordinatorContext}(T) = \text{CoordinatorContext}(T-1) + \text{CompactReceipt}(W) \quad (\text{NOT} \quad \text{RawTranscript}(W))$$

The coordinator context **never ingests raw worker command logs, compiler transcripts, or intermediate shell output**.
It ingests **only compact, author-neutral, typed receipts**.

---

## 2. Architecture & Topology

```
                                  ┌─────────────────────────────┐
                                  │      Clean Coordinator      │
                                  │   (Context Cap: O(1)/task)   │
                                  └──────────────┬──────────────┘
                                                 │
                      ┌──────────────────────────┴──────────────────────────┐
                      │ Dispatch S0/S1 Task Descriptors                     │ Ingest Compact Receipts
                      │ (1-3 files, 1 witness cmd)                          │ (Status, Files, SHA, ExitCode)
                      ▼                                                     ▲
          ┌───────────────────────┐                             ┌───────────┴───────────┐
          │   Task Admission      │                             │   Receipt Admission   │
          │ S0/S1 & Risk Checker  │                             │   Context Aggregator  │
          └───────────┬───────────┘                             └───────────▲───────────┘
                      │                                                     │
         ┌────────────┴────────────┐                                        │
         ▼                         ▼                                        │
  [Standard Task]          [High-Risk Boundary]                             │
         │                         │                                        │
         ▼                         ▼                                        │
┌──────────────────┐      ┌──────────────────┐                              │
│ Bounded Worker   │      │ Structured       │                              │
│ (Small/Fast/7B)  │      │ ABSTAIN Emitter  │                              │
│ Sub-process / VM │      │ (Typed Refusal)  │                              │
└────────┬─────────┘      └────────┬─────────┘                              │
         │                         │                                        │
         │ Executes Task           │ Emits Escalation                       │
         │ Generates Transcript    │ Reason: FROZEN_ABI, etc.               │
         ▼                         │                                        │
┌──────────────────┐               │                                        │
│ Isolated Context │               │                                        │
│ Raw logs & diffs │               │                                        │
└────────┬─────────┘               │                                        │
         │                         │                                        │
         └─────────────┬───────────┘                                        │
                       ▼                                                    │
             ┌───────────────────┐                                          │
             │ Compact Receipt   │──────────────────────────────────────────┘
             │ (~50 tokens)      │
             └───────────────────┘
```

### 2.1 The Compact `WorkerReceipt` Contract

Workers operate in isolated child processes or bounded scratch environments. Upon completion, failure, or
refusal, the worker encapsulates its outcome into a single typed `WorkerReceipt`:

| Field | Type | Description |
|---|---|---|
| `TaskID` | `string` | Unique identifier matching the admitted task. |
| `Status` | `ReceiptStatus` | Closed vocabulary: `COMPLETED`, `FAILED`, `ABSTAIN`. |
| `TouchedFiles` | `[]string` | Exact relative file paths modified (enforcing atomic bounds $\le 3$). |
| `WitnessCommand` | `string` | The single deterministic witness command executed. |
| `WitnessExitCode` | `int` | Process exit status (must be `0` for `COMPLETED`). |
| `GitSHA` | `string` | Immutable commit hash recording the work. |
| `AbstainRationale` | `string` | Structured rationale explaining why a boundary was not crossed. |
| `TokensUsed` | `int` | Resource accounting: tokens consumed in the worker's isolated environment. |
| `Summary` | `string` | Compact, 1–2 sentence author-neutral summary of the deliverable. |

### 2.2 Context Savings Invariant ($\ge 90\%$)

Let $K$ be the number of tasks executed in a batch. Let $T_{\text{raw}}(i)$ be the token count of the raw
command output, compiler diagnostics, and test transcript for task $i$. Let $T_{\text{receipt}}(i)$ be the
token count of the compact receipt ingested into the coordinator context.

$$\text{TotalRawTokens} = \sum_{i=1}^K T_{\text{raw}}(i)$$
$$\text{TotalReceiptTokens} = \sum_{i=1}^K T_{\text{receipt}}(i)$$
$$\text{ReductionRatio} = \frac{\text{TotalRawTokens} - \text{TotalReceiptTokens}}{\text{TotalRawTokens}} \ge 0.90$$

In typical workloads, $T_{\text{raw}} \approx 1,500 - 5,000$ tokens per task, whereas $T_{\text{receipt}} \approx 40 - 70$
tokens, consistently achieving $\ge 95\%$ token reduction.

### 2.3 Zero Context Pollution Guarantee

When a sub-task encounters an error, panic, or compilation failure:
1. **Raw Traceback Quarantine:** Zero bytes of raw compiler stderr or stack trace enter the coordinator context.
2. **Failure Isolation:** The coordinator records the failure within its audit ledger (`FailedTasks()`).
3. **Optional Context Exclusion:** With failure quarantine enabled (`QuarantineFailures: true`), failed tasks
   contribute zero messages and zero tokens to the coordinator's active conversational history:
   $$\text{ContextPollution}(\text{FailedTask}) = 0 \text{ tokens}$$

---

## 3. Atomic S0/S1 Leaf Unit Decomposition

To maintain execution reliability and prevent scope inflation, tasks must be subdivided into minimal atomic units.

### 3.1 Structural Invariants

Every task admitted by the coordinator must satisfy:
1. **Bounded File Scope:** The task write surface must target between 1 and 3 closely related files:
   $$1 \le |\text{TargetFiles}| \le 3$$
   Broad sweeps across multiple packages or directory hierarchies are rejected at registration time.
2. **Exactly One Witness Command:** The task must specify exactly one verification command (e.g.
   `go test -v -race ./internal/microagent -run TestCoordinator`).
   Chained commands (`&&`, `;`, `||`) and shell scripts are rejected.
3. **Single Observable Deliverable:** The task must state a concise, concrete outcome.
4. **Phase Ordering:** Multi-step work must be decomposed into sequential S0/S1 phases: reproduction test
   first, minimal implementation second, targeted package verification third.

---

## 4. Structured ABSTAIN Boundaries

Smaller models (local 7B/14B models) and fast/flash models (Gemini 3.8 Flash & peers) deliver high throughput
on focused implementation but exhibit critical reliability drops when reasoning about deep architectural
constraints.

### 4.1 High-Risk Subsystems

The coordinator defines typed risk boundaries where smaller models must fail-to-abstain:

| Risk Category | Token | Boundary Description |
|---|---|---|
| Concurrency Locks | `CONCURRENCY_LOCK_INVARIANTS` | Mutex hierarchies, channel deadlock hazards, lock ordering, race conditions. |
| Frozen ABI | `FROZEN_ABI` | Modification to frozen interface definitions (`internal/abi/`). |
| Low-Level Kernels | `LOW_LEVEL_KERNELS` | SIMD intrinsics, CUDA kernels, GPU memory management, assembly, vDSO seams. |
| Security Policy Gate | `SECURITY_POLICY_GATE` | Capability floors, tool execution sandboxes, policy verification rules. |
| Protocol Migration | `PROTOCOL_MIGRATION` | Cross-subsystem wire format, breaking serialization changes. |

### 4.2 The Fail-to-Abstain Protocol

When a task intersects a high-risk boundary and is assigned to a bounded or fast model:
1. The model must not guess, speculate, or generate unverified diffs.
2. The worker must return a structured receipt with:
   - `Status`: `ABSTAIN`
   - `AbstainRationale`: A typed description naming the boundary and justification.
3. The coordinator catches the `ABSTAIN` status, registers an `Escalation` record, and routes the task
   to a higher-capability model (e.g. Opus / deep reasoning tier) or human operator.
4. No speculative diffs or broken code pollute the repository trunk or the coordinator context.

---

## 5. Verification & Acceptance Criteria

1. **Token Savings Proof:** In a 10-task batch execution, the measured coordinator context reduction ratio
   must meet or exceed $90\%$.
2. **ABSTAIN Escalation Proof:** A worker encountering `internal/abi` or concurrency invariants must
   produce a valid `ABSTAIN` receipt which the coordinator maps to an `Escalation` with appropriate risk category.
3. **Zero Pollution Proof:** A task generating substantial stderr/panic output must leave zero trace of the
   raw stderr text in the coordinator's message history.
