---
title: "CONCEPT-STUDY: Microsoft Vidur LLM inference simulator architecture, multi-replica routing, execution time modeling, and capacity planning"
description: "Exhaustive, pinned study of Microsoft Research's Vidur inference system simulator (MLSys'24) across main and canary branches, reconciling execution time regression modeling, prefix caching, multi-replica routing, and adaptive capacity search against fak's gateway and runtime seams."
date: 2026-09-03
---

# CONCEPT-STUDY: Microsoft Vidur LLM inference simulator architecture, multi-replica routing, execution time modeling, and capacity planning (2026-09-03)

**Verdict:** Microsoft Research's Vidur (MLSys'24) is a high-fidelity discrete-event simulator and capacity planning toolkit for large-scale LLM inference. While Vidur itself is an offline Python simulator and fak is a production in-kernel Go agent runtime, Vidur's scheduling and optimization algorithms provide valuable, high-fidelity mechanisms directly transferable to fak's gateway, continuous batching scheduler, performance profiler, and benchmark tooling:

1. **Slack-tolerant sticky multi-replica session routing** (`TolerantStickyLOPUncachedGlobalScheduler`): Enforces conversational session affinity to preserve multi-turn KV cache locality, but monitors cluster load imbalance with a bounded `tolerance_factor` slack window (`sticky_load_imbalance <= tolerance_factor * min_load_imbalance`), shedding session turns to cooler replicas when load concentrates.
2. **Uncached prefill token volume balancing** (`LOPUncachedGlobalScheduler`): Routes requests based on *remaining uncomputed prefill token volume* ($N_{prefill} - N_{cached}$) across replicas rather than naive request counts or binary cache-hit flags, balancing actual GPU compute work.
3. **Adaptive binary search for max sustainable capacity under latency SLO** (`CapacitySearch.search`): Efficiently maps the Pareto frontier of sustainable QPS within tail latency constraints (P90/P99 TTFT, TPOT, scheduling delay) by dynamically expanding search bounds when latency is low ($< SLO / 8$) and contracting sharply upon divergence.
4. **Earliest Deadline First (EDF) request queue** (`EDFRequestQueue`): Queues requests by absolute deadline expiration ($T_{arrival} + D_{SLO}$) rather than static priority, maximizing SLO attainment under mixed interactive and background batch workloads.
5. **Component-wise execution time decomposition separating host CPU scheduling overheads** (`ExecutionTime`): Decomposes step latency into individual projection kernels, attention phases (prefill chunk, decode, RoPE, KV cache save), collectives (AllReduce vs Send/Recv), and explicitly isolates host CPU scheduling, tensor preparation, and sampling overhead from GPU compute time.

---

## 1. Scope, Provenance, and Durable Receipt

Observed and pinned on **2026-09-03**:

| Repository / Branch | Pinned Revision | License | Notes |
|---|---|---|---|
| [`microsoft/vidur` (main)](https://github.com/microsoft/vidur) | `abae7f63aa857300f5cdc6f5e0d27860cd24721b` | MIT | Core MLSys'24 simulator: execution time regression models, Sarathi/vLLM/Orca schedulers, CPU overhead profiler, capacity search. |
| [`microsoft/vidur` (canary)](https://github.com/microsoft/vidur/tree/canary) | `25e0082dbbfb206fb0477c3ebbededa7ead78949` | MIT | Canary development branch: prefix caching, multi-tier Disk/VRAM KV cache managers, multi-replica global schedulers, and EDF request queue. |

**Durable study receipt:** `study_722a05f7439411a3291e4a376f5e48b3da011f06448fbc4480d2e1cb635d9553` (persisted via `fak study add`).

**License boundary:** Microsoft Vidur is licensed under the permissive MIT License (Copyright (c) Microsoft Corporation). All borrowed mechanisms in this study are clean-room **INSPIRE** implementations in Go, integrating idiomatically into fak's kernel architecture.

---

## 2. Worldview Reconstruction: Who They Built It For & Tradeoffs

To understand Vidur's architecture without ego dismissal, we reconstruct the authors' design incentives and operational environment:

1. **Who Microsoft built Vidur for:**
   - **Infrastructure capacity planners:** Engineers sizing enterprise GPU clusters (Azure OpenAI Service, Copilot) who must evaluate hundreds of deployment configurations (TP1..8, PP1..4, chunk sizes 256..2048, batch limits) across hardware SKUs (GPU server vs Pairwise NVLink vs H100) before procuring or allocating physical hardware.
   - **Systems researchers:** Designers evaluating new scheduling algorithms (chunked prefill, disaggregated prefill/decode, speculative decoding) without renting hundreds of GPUs for each hypothesis.
   - **Multi-tenant cloud operators:** Services serving heterogeneous workloads (interactive Copilot chat turns, batch document summarization) under strict latency Service Level Objectives (SLOs).

2. **What Vidur optimizes:**
   - **Fidelity without physical hardware:** Once a model SKU is profiled on a single device/node, Vidur simulates arbitrary multi-node, multi-replica clusters with $<8\%$ error relative to physical vLLM/Sarathi deployments.
   - **Analytical component isolation:** Separates GEMM compute, attention memory bandwidth, collective communication network latency, and host CPU runtime overhead into distinct measurable terms.
   - **Combinatorial exploration:** Uses adaptive search and Pareto curve generators to identify optimal cost-per-token configurations.

3. **Tradeoffs vs. fak:**
   - *Simulation vs. Native Execution:* Vidur is a discrete-event simulator running on virtual time ($t \to t_{next}$); fak is an in-kernel production agent runtime executing real inference on real hardware with zero Python runtime tax.
   - *Overhead elimination:* Vidur measures and models Python/Ray CPU overheads (`schedule_time`, `ray_comm_time`) because vLLM and Sarathi suffer from them; fak replaces Python entirely with compiled Go kernels, eliminating the very overhead Vidur models.
   - *Policy applicability:* While fak does not need Vidur's simulator loop for execution, the *scheduling and placement policies* Vidur designed to manage multi-tenant clusters apply directly to fak's gateway and multi-replica routing layer.

---

## 3. Subsystem Analysis & Key Mechanisms

### A. Slack-Tolerant Sticky Session Routing (`TolerantStickyLOPUncachedGlobalScheduler`)
*Source:* `vidur/scheduler/global_scheduler/tolerant_sticky_lop_uncached_global_scheduler.py:46-88@25e0082dbbfb206fb0477c3ebbededa7ead78949`.

In multi-turn chat and agentic workflows, conversational history creates shared prefix KV caches. Routing each turn to the replica holding the session's prior turns avoids recomputing prefill. However, rigid session affinity leads to severe cluster load imbalances when certain sessions become unusually active or verbose.

Vidur introduces a **tolerance slack factor**:
```python
def _is_replica_within_imbalance_slack(self, load_imbalance_map, replica_id):
    min_load_imbalance = min(load_imbalance_map.values())
    sticky_load_imbalance = load_imbalance_map[replica_id]
    return sticky_load_imbalance <= self._tolerance_factor * min_load_imbalance
```
If the sticky replica's load imbalance remains within `tolerance_factor * min_imbalance`, stickiness is preserved. As soon as the sticky replica exceeds this bound, the session unpins and migrates to the least loaded replica.

### B. Uncached Prefill Token Volume Balancing (`LOPUncachedGlobalScheduler`)
*Source:* `vidur/scheduler/global_scheduler/lop_uncached_global_scheduler.py:21-72@25e0082dbbfb206fb0477c3ebbededa7ead78949`.

Standard load balancers count active HTTP requests (Least Outstanding Requests). In LLM serving, request counts correlate poorly with GPU compute demand: a 4,000-token prompt with 3,900 cached tokens requires almost zero compute, whereas a 4,000-token prompt with 0 cached tokens requires a massive prefill pass.

Vidur's `LOPUncachedGlobalScheduler` tracks remaining uncached tokens:
```python
def _get_num_prefill_tokens_uncached(self, request, replica_id):
    cached = self._replica_schedulers[replica_id].get_cached_prefill_length(request)
    return request.num_prefill_tokens - cached
```
It routes incoming requests to the replica that minimizes the resulting peak pending prefill token volume across the cluster.

### C. Adaptive Capacity Search (`CapacitySearch.search`)
*Source:* `vidur/config_optimizer/config_explorer/capacity_search.py:125-182@abae7f63aa857300f5cdc6f5e0d27860cd24721b`.

Vidur replaces manual grid searches with an adaptive binary search that finds the maximum sustainable QPS satisfying a specified latency quantile (e.g. P99 scheduling delay $\le 100\text{ms}$):
- If scheduling delay is well within the SLO ($< SLO / 8$), it expands the upper search bound aggressively (up to 4x).
- If scheduling delay exceeds the SLO by an order of magnitude ($> 500\text{ms}$), it contracts the upper bound by 50%–75%.
- Continues until the search interval $|left - right| < \text{granularity} \times QPS$.

### D. Earliest Deadline First Request Queue (`EDFRequestQueue`)
*Source:* `vidur/scheduler/request_queue/edf_request_queue.py:1-40@25e0082dbbfb206fb0477c3ebbededa7ead78949`.

When serving mixed workloads (interactive agent loops requiring sub-second TTFT vs asynchronous background jobs), static priorities fail: high-priority bursts starve batch jobs, while batch jobs admitted before an interactive burst cause the interactive request to blow its deadline. Vidur implements EDF queuing where requests are ordered in a min-heap by target completion deadline:
$$\text{deadline} = \text{arrival\_time} + \text{target\_latency}$$

### E. Component-Wise Execution Time & CPU Overhead Modeling (`ExecutionTime`)
*Source:* `vidur/entities/execution_time.py:4-95@abae7f63aa857300f5cdc6f5e0d27860cd24721b` and `vidur/execution_time_predictor/base_execution_time_predictor.py:32-69@abae7f63aa857300f5cdc6f5e0d27860cd24721b`.

Vidur decomposes step latency into fine-grained terms:
- Attention: QKV pre-projection, RoPE, KV cache save/update, decode attention, chunked prefill attention, O post-projection.
- MLP: Up projection, gated activation, down projection.
- Collectives: Tensor parallel AllReduce vs pipeline parallel Send/Recv.
- Host CPU overhead: `schedule_time`, `sampler_e2e_time`, `prepare_inputs_e2e_time`, `process_model_outputs_time`, and `ray_comm_time`.

---

## 4. Current fak Witness & Gap Matrix

| Vidur Mechanism | fak Equivalent | Current fak Witness | On-Axis Gap & Disposition |
|---|---|---|---|
| **Slack-tolerant sticky session routing** | `internal/gateway/kv_fleet_routing.go` (`FleetCacheRouter`) | `internal/gateway/kv_fleet_routing.go:135-147`, `replica_router.go:55` | **PARTIAL → DEFAULT**. Fak has boolean `holds(prefixKey)` routing; once warm, all traffic lands on that replica without shedding under heavy load imbalance. Adding a tolerance slack factor prevents hot-spot stragglers. |
| **Uncached prefill token volume balancing** | `internal/gateway/replica_router.go` (`PickPolicy`) | `internal/gateway/replica_router.go:25-27`, `kv_fleet_routing.go:135` | **ABSENT → DEFAULT**. Fak balances by in-flight request count; it does not calculate remaining uncomputed prompt tokens across candidate replicas when making routing decisions. |
| **Adaptive binary capacity search** | `internal/localbench`, `cmd/fak/bench_local_test.go` | `internal/localbench/localbench.go:40`, `bench_local_test.go:50` | **ABSENT → DEFAULT**. Fak benchmarks run fixed client/QPS injection rates. Adding adaptive binary search discovers maximum sustainable QPS under strict tail latency SLOs automatically. |
| **Earliest Deadline First (EDF) request queue** | `internal/gateway/admission.go` | `internal/gateway/admission.go:33-40,89-93` | **PARTIAL → DEFAULT**. Fak uses static integer priority with `AgingRounds` starvation guards; it cannot prioritize requests by explicit deadline timestamps under mixed interactive/batch traffic. |
| **Component execution time & CPU overhead separation** | `internal/nativeperf`, `internal/roofline` | `internal/nativeperf/profile.go:27-64`, `internal/roofline/roofline.go:71` | **ABSENT → DEFAULT**. Fak captures coarse post-hoc phase durations (`prefill`, `steady-decode`); it does not isolate host CPU scheduling/sampling time from device kernel projection durations in step metrics. |
| **Discrete-event simulation engine** | Umbrella issue #10841 | `docs/notes/GLM52-DGX-ROOFLINE-DASHBOARD.md` | **DIVERGENT (Honest Tradeoff)**. Vidur simulates serving because running Python clusters is slow and expensive. Fak executes directly in-kernel. Standalone simulation is tracked under #10841 for offline spec-decode study, but production fak serves real requests. |

---

## 5. Candidate Borrows & Decomposed Work Items

### Candidate 1: Slack-tolerant sticky session routing for cross-replica prefix cache locality
- **Technique:** Maintain session affinity while load imbalance $\le \text{tolerance} \times \text{min\_load}$; shed sessions to cooler replicas when imbalance exceeds the slack threshold.
- **Source anchor:** `vidur/scheduler/global_scheduler/tolerant_sticky_lop_uncached_global_scheduler.py:46-88@25e0082dbbfb206fb0477c3ebbededa7ead78949`
- **Fak seam:** `internal/gateway/kv_fleet_routing.go:135-147` & `internal/gateway/replica_router.go:55-62`
- **Axis:** Multi-turn conversational KV-cache locality vs cluster load balancing.
- **Why their users made them build it:** Rigid sticky routing creates extreme tail latency when conversational traffic concentrates on specific workers.
- **Witness on-axis:** `PARTIAL`.
- **Disposition:** `DEFAULT`.
- **Filed Issue:** [#10852](https://github.com/anthony-chaudhary/fak/issues/10852)
- **First checkable step:** Add `ToleranceFactor float64` to `FleetCacheRouter` in `internal/gateway/kv_fleet_routing.go` and write a unit test in `kv_fleet_routing_test.go` demonstrating graceful shedding under load imbalance.

### Candidate 2: Route multi-replica requests by uncached prefill token volume
- **Technique:** Query cached prefix lengths per replica, compute remaining uncached prompt tokens ($N_{prefill} - N_{cached}$), and dispatch to minimize peak pending prefill token volume.
- **Source anchor:** `vidur/scheduler/global_scheduler/lop_uncached_global_scheduler.py:21-72@25e0082dbbfb206fb0477c3ebbededa7ead78949`
- **Fak seam:** `internal/gateway/replica_router.go:25-27` & `internal/gateway/kv_fleet_routing.go:135`
- **Axis:** Routing decision fidelity based on remaining prefill compute work rather than coarse request counts.
- **Why their users made them build it:** A 4,000-token prompt with 95% cached KV imposes negligible compute, while a cold prompt stalls the queue. Treating both as 1 request causes severe head-of-line blocking.
- **Witness on-axis:** `ABSENT`.
- **Disposition:** `DEFAULT`.
- **Filed Issue:** [#10853](https://github.com/anthony-chaudhary/fak/issues/10853)
- **First checkable step:** Expose an `UncachedTokens(replica, promptTokens)` helper in `internal/gateway/kv_fleet_routing.go` and prove via unit test that cold requests route away from replicas with pending cold prefills.

### Candidate 3: Adaptive binary search for max sustainable throughput under latency SLO
- **Technique:** Perform adaptive binary search over QPS: aggressively expand bounds when latency is low ($< SLO / 8$), contract sharply on divergence ($> 500\text{ms}$), and terminate on relative convergence.
- **Source anchor:** `vidur/config_optimizer/config_explorer/capacity_search.py:125-182@abae7f63aa857300f5cdc6f5e0d27860cd24721b`
- **Fak seam:** `internal/localbench/localbench.go:40` & `cmd/fak/bench_local_test.go:50`
- **Axis:** Automated discovery of maximum certified QPS under strict tail latency bounds.
- **Why their users made them build it:** Sizing deployments manually via grid sweeps is time-prohibitive.
- **Witness on-axis:** `ABSENT`.
- **Disposition:** `DEFAULT`.
- **Filed Issue:** [#10854](https://github.com/anthony-chaudhary/fak/issues/10854)
- **First checkable step:** Implement `AdaptiveCapacitySearch` in `internal/localbench/capacity_search.go` with a synthetic test verifying convergence within 12 iterations.

### Candidate 4: Earliest Deadline First (EDF) request queue for mixed interactive and batch SLOs
- **Technique:** Order waiting queue requests by absolute target deadline ($\text{arrival} + \text{target\_latency}$) rather than static integer priority.
- **Source anchor:** `vidur/scheduler/request_queue/edf_request_queue.py:1-40@25e0082dbbfb206fb0477c3ebbededa7ead78949`
- **Fak seam:** `internal/gateway/admission.go:33-40,89-93`
- **Axis:** Maximizing SLO attainment rate under heterogeneous mixed interactive/batch arrival streams.
- **Why their users made them build it:** Static priority starves batch traffic or causes interactive requests arriving during batch bursts to miss deadlines.
- **Witness on-axis:** `PARTIAL`.
- **Disposition:** `DEFAULT`.
- **Filed Issue:** [#10856](https://github.com/anthony-chaudhary/fak/issues/10856)
- **First checkable step:** Add optional `Deadline time.Time` to `admissionRequest` in `internal/gateway/admission.go` with unit tests verifying deadline-ordered promotion.

### Candidate 5: Decompose step latency into kernel projections and host CPU scheduling overheads
- **Technique:** Separate GPU execution (QKV proj, attention core, MLP up/down) from host CPU overhead (`schedule_time`, `prepare_inputs_time`, `sampler_time`).
- **Source anchor:** `vidur/entities/execution_time.py:4-95@abae7f63aa857300f5cdc6f5e0d27860cd24721b` & `vidur/execution_time_predictor/base_execution_time_predictor.py:32-69@abae7f63aa857300f5cdc6f5e0d27860cd24721b`
- **Fak seam:** `internal/nativeperf/profile.go:27-64` & `internal/roofline/roofline.go:71`
- **Axis:** Profiling diagnostic fidelity distinguishing device kernel stalls from host scheduling bottlenecks.
- **Why their users made them build it:** Python serving engines waste substantial time in host scheduling and IPC; isolating host from device time is required to identify optimization opportunities.
- **Witness on-axis:** `ABSENT`.
- **Disposition:** `DEFAULT`.
- **Filed Issue:** [#10861](https://github.com/anthony-chaudhary/fak/issues/10861)
- **First checkable step:** Add a structured `StepOverhead` field to `ProfileBundle` in `internal/nativeperf/profile.go` and test separating host scheduling time from device compute time.

---

## 6. Registration and Companions

- **Durable Study Receipt:** `study_722a05f7439411a3291e4a376f5e48b3da011f06448fbc4480d2e1cb635d9553`
- **Monitored Repository Registry:** Added `microsoft/vidur` to `docs/research/monitored-repositories.json`.
- **Index:** Added entry in `INDEX.md` under `## Notes & research`.
- **Companions:**
  - #10841 (`bench(simulation): integrate trace-driven discrete-event LLM serving simulator (Vidur-style)`)
  - `docs/notes/GLM52-DGX-ROOFLINE-DASHBOARD.md` (theoretical roofline ceiling dashboard)
  - `docs/fleet-compute-nodes.md` (sanctioned hardware compute topology)
  - `docs/benchmark-authority.md` (authoritative benchmark measurements)
