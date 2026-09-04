---
title: "Study: NVIDIA TensorRT-LLM — witnessed borrows for fak"
date: 2026-07-18
status: FILED 2026-07-18 as epic #5256 + children #5257 (spec-gate), #5258 (no-evict admission), #5259 (KV retention), #5260 (block-hash cache events)
repo: "https://github.com/NVIDIA/TensorRT-LLM"
pinned_sha: f4c5c935aa891b0826f73936c4831236cb6ff836
license: Apache-2.0 (LICENSE read — INSPIRE default; no bytes vendored)
family_key: fak-tensorrt-llm-study
prior_art_epics: ["#2236", "#50", "#637", "#3983", "#3900", "#3366", "#4352", "#4207", "#5243", "#3365", "#3352", "#3569", "#3809", "#3080"]
Companions: ["field-borrow (per-capability witness+file)", "sota-check (kernel-adjacent autotune-cache route)", "epic #5256"]
---

# Study: NVIDIA TensorRT-LLM `@f4c5c935`

A `/study C:\...\TensorRT-LLM --deep` pass. Pinned HEAD `f4c5c935aa891b0826f73936c4831236cb6ff836`
("[None][test] Consolidate test coverage for helix (#16570)"). Shallow (`--depth 1`) checkout —
history/rationale-from-commits is unavailable (honest limit). LICENSE is Apache-2.0, so INTEGRATE
would be license-compatible, but every borrow below is **INSPIRE** (clean-room, cited).

## Worldview (reconstructed from defaults / non-goals / benchmarks)

TensorRT-LLM serves **NVIDIA-GPU production at maximum throughput/latency** via *compiled* TensorRT
engines plus hand-tuned CUDA kernels. Every scheduling/policy choice optimizes peak hardware
utilization of a **token-serving** fleet:

- The default capacity policy is `kGUARANTEED_NO_EVICT` (`executor.h:1023@f4c5c935`) — a started
  request is reserved to completion so it never stalls mid-decode; its foil `kMAX_UTILIZATION` buys
  utilization with LIFO pause/resume.
- The dynamic batch tuner reshapes the batch/token budget to the live ISL/OSL mix
  (`dynamicBatchTuner.cpp:71-111`), and speculation *self-disables* when it stops paying
  (`speculation_gate.py`).
- KV retention is a **client-declared** vocabulary (priority 0-100 + TTL windows), not just observed
  telemetry — because a multi-tenant engine's caller knows which prefix is worth pinning.

fak's world is different: an **agent-orchestration substrate** fronting (often third-party) engines,
optimizing *cross-agent* cache economics and *fleet* liveness on CPU/Mac/consumer-GPU. So the borrows
are the **CPU-witnessable POLICY/CONTROL ideas**; the kernel/engine-compilation stack is worldview-
divergent and routed to `sota-check`.

## Fan-out coverage (6 parallel sub-readers, all verified against the tree)

1. **KV cache manager** — kvCacheManager, evictionPolicy, blockKey, kvCacheEventManager, kvCacheConfig/RetentionConfig.
2. **Batch scheduling** — capacityScheduler, microBatchScheduler, pauseRequests, dynamicBatchTuner, scheduler.py / scheduler_v2.py.
3. **Executor API** — executor.h Request/Response/Result, OutputConfig, ExtendedRuntimePerfKnobConfig, orchestratorConfig, stats streams.
4. **Speculative decoding** — speculation_gate, drafter, ngram, suffix_automaton, mtp, eagle3, auto_heuristic, interface.
5. **Disaggregated serving (P/D)** — transceiver, cacheFormatter/Transceiver, contextPhaseParams, dataTransceiverState, native/nixl agents.
6. **Quant / autotuner / auto_deploy / mapping / watchdogs** — quantization/mode.py, autotuner.py, auto_deploy, mapping.py, alltoall_watchdog, hang_detector.

### Completeness-critic residue (named skips, justified)
- `cpp/tensorrt_llm/kernels/*.cu` (attention/GEMM/MoE), `cutlass_extensions`, `deep_gemm`, `flash_mla`,
  `deep_ep`, `triton_kernels`, `triton_backend` — **pure GPU kernels**; not CPU-witnessable policy borrows.
  Routed to the `sota-check` lane (the prior-art matrix), not this study.
- **The un-checked-out `cpp/.../cubins/*.cubin` binaries** — failed to check out on Windows (path-length
  limit). They are **compiled GPU binary artifacts**, irrelevant to a code study; deliberately skipped
  (not fetched/fixed), per the study framing.
- `nanobind/` bindings, `benchmarks/`, `docker/`, `jenkins/` — plumbing/build, no borrowable policy.

## Candidate table

Legend — Witness: PRESENT-on-axis (dropped) / PARTIAL / ABSENT / DIVERGENT / WORLDVIEW. All INSPIRE.

| Borrow (one line) | Source `path:line@f4c5c935` | Axis | Their-worldview reason | Witness (fak seam) | Filed |
|---|---|---|---|---|---|
| Rolling-window acceptance-rate gate that permanently disables spec | `_torch/speculative/speculation_gate.py:33-82` | disable speculation when windowed accept rate < threshold | a mispredicting draft makes decode net-negative; bound the downside automatically | **PARTIAL** — `polymodel/specdecode.go` measures `MeanAcceptanceLength`, never gates | **#5257** |
| Load gate + batch-size→draft-length schedule (K→0) | `drafter.py:41-65, 106-134` | speculate only at low load; shrink K as batch grows | spec is a low-load latency win, a high-load throughput loss | PARTIAL (folded into #5257) | #5257 |
| Guaranteed-no-evict admission: reserve worst-case blocks-to-completion | `batch_manager/capacityScheduler.cpp:397-427`; math `kvCacheManager.cpp:3550-3589` | admit only if worst-case KV fits; started streams never evict | mid-decode preemption stalls a tenant; default policy | **PARTIAL** — `kvbudget.go` sizes streams-that-fit, no admission verdict | **#5258** |
| Client-declared KV retention (priority 0-100 + TTL window) over token ranges | `executor.h:589-617`; `kvCacheRetentionConfig.cpp:125`; TTL demotion `evictionPolicy.cpp:246-270` | per-request priority + TTL feeding eviction (not telemetry-only) | caller knows which prefix to pin; multi-tenant | **PARTIAL** — `radixkv/eviction_strategy.go` victimKey is telemetry-only | **#5259** |
| Context-vs-decode priority split; leaf-only eviction preserves hot interior prefixes | `executor.h:655`; `kvCacheManager.cpp:1322-1337` | decode KV evicts before prompt KV; interior prefix survives leaf eviction | decode KV is rarely a reuse target | PARTIAL (folded into #5259) | #5259 |
| Per-block-hash event stream: created/stored/removed/**updated**(tier+priority diff) | `executor.h:1741-1848`; `kvCacheEventManager.cpp:83-144` | block-hash events with tier-move/hotness diffs for a KV-aware router | external LB must know which tier holds a block & how hot | **PARTIAL** — `engine/cacheevents.go` metrics + `residency_router.go` add/drop only; no UPDATE/tier-move | **#5260** |
| Persistent device+version-stamped, shape-bucketed autotune-decision cache; atomic+locked; `exclude_from_cache` JIT opt-out | `_torch/autotuner.py:997,478,548,658,127` | cache the winning kernel tactic keyed by (op+device-cap+bucketed-shape), crash/race-safe, warmup-aware | re-tuning every launch across ranks/restarts is expensive | **ABSENT** but kernel-adjacent → **sota-check** | note |
| Prefix-reuse-aware admission *deferral* (`beneficialToSkip`) | `capacityScheduler.cpp:80-116` | defer a request so it reuses a peer's about-to-commit prefix vs recompute | duplicate-prompt hot path; convert recompute→reuse | PARTIAL → fold into coalescing **#5243** | note |
| Typed prefill→decode handoff token (`ContextPhaseParams` + `disaggRequestId`) | `executor/contextPhaseParams.cpp:53`; `executor.h:717,907` | self-describing serializable handoff blob correlating one request across two engines | prefill/decode run in different processes/hosts | PARTIAL → disagg **#50** / session-migration **#3352** | note |
| Dynamic batch/token tuner from moving-average load, ISL/OSL-shaped | `dynamicBatchTuner.cpp:71-111` | reshape batch/token budget to the live workload mix | context-heavy vs decode-heavy traffic needs different budgets | DIVERGENT (fak batches agent turns) → dispatch **#3365** | note |
| TTFT reordering: float first-token requests ahead under pressure | `pyexecutor/scheduler/scheduler_v2.py:211-283` | latency-vs-throughput admission ordering | reduce TTFT on the disagg generation server | PARTIAL (fak `dispatchaging`/`dispatchorder`) → note | note |
| Scheduler deadlock detector: raise on no-progress | `scheduler_v2.py:421-442` | liveness guard that fails loud instead of hanging | turn an invisible hang into an actionable config error | PRESENT-ish (fak `superloop` spinning detection) | drop |
| Per-request QoS: priority, `allottedTimeMs`→`kTIMED_OUT`, cacheSalt | `executor.h:669-718`; `types.h:573` | wall-clock deadline auto-cancel + reuse-namespace on the request | multi-tenant SLOs without separate engines | PARTIAL (fak has aging/budgets; cacheSalt=PRESENT via `radixkv/namespace.go`) | note |
| In-band per-request perf metrics (TTFT/queue/KV-transfer breakdown) | `types.h:490-545`; `executor.h:929` | attribute latency to the individual request, returned with its Result | latency debugging must be per-request | PARTIAL (fak metrics are aggregate) → note | note |
| Orchestrator/leader-worker spawn + closed MPI message enum | `orchestratorConfig.cpp:26`; `orchestratorUtils.h:30` | typed orchestrator config + closed control-message vocabulary | predictable multi-node bring-up | PRESENT-ish (fak dispatch/servicelease + closed verb grammar) | drop |
| alltoall watchdog + hang-detector cross-rank kill-propagation | `_torch/alltoall_watchdog.py:493`; `pyexecutor/hang_detector.py:47` | detect hung collective / stalled loop, hard-kill and free peer GPUs | one hung rank silently wedges the whole job | PARTIAL (fak `watchdoghealth`, `ep_decode_coord`) → note | note |
| Relaxed (top-k + logprob-delta) acceptance scoped to the thinking span | `_torch/speculative/mtp.py:807-845` | trade exactness for accept length during CoT | exact-match is needlessly strict while "thinking" | **DIVERGENT** — fak sells *lossless byte-identical* spec decode; relaxing forfeits the property it markets | drop (tradeoff) |
| Cross-topology KV reshaping (prefill TP8 → decode TP2×DP4) | `cacheFormatter.cpp:495-530` (`TargetRanksInfo`) | re-shard KV across differing parallel degrees on transfer | prefill & decode tuned to different parallelism | DIVERGENT (distributed multi-GPU; fak fronts engines) | drop (tradeoff) |
| `auto_deploy` declarative staged transform pipeline (capture→deploy) | `_torch/auto_deploy/transform/interface.py:109-146` | ordered config-declared graph-transform pipeline | HF model → sharded/quantized/compiled engine is dozens of ordered passes | DIVERGENT — fak does not compile engines | drop (tradeoff) |
| Orthogonal `QuantMode` IntFlag vocabulary + `has_*` predicates | `quantization/mode.py:68,247,378` | composable per-layer quant bit-flags | dozens of recipes share overlapping properties | DIVERGENT (fak's family is GGUF Q4_K/etc, different) → note | note |
| Per-layer quant-by-glob exclusion (ancestor-walk + subtree wildcard) | `models/modeling_utils.py:245` | apply/except quant per module-name pattern | checkpoints exclude specific layers (`lm_head`, `*eh_proj`) | PARTIAL (fak `ggufload`) → note | note |
| SGLang-style pluggable victim registry / SLRU scan-resistance | (fak already borrowed it) | pluggable eviction strategy; scan-resistant SLRU | — | **PRESENT** — `radixkv/eviction_strategy.go` | drop |
| Per-namespace/cacheSalt reuse isolation | `blockKey.cpp:367` | tenant/adapter-namespaced prefix reuse | multi-tenant reuse leakage is a correctness hole | **PRESENT** — `radixkv/namespace.go` | drop |
| Cache-event → prefix-aware fleet routing skeleton | `residency_router` neighbor | route to the replica holding the prefix | shared-prefix traffic shouldn't herd one replica | **PRESENT** — `gateway/residency_router.go` | drop |
| Lossless greedy draft→verify→accept→rollback loop | (fak already ships it) | byte-identical-to-greedy spec decode | — | **PRESENT** — `polymodel/specdecode.go` | drop |
| Model-free n-gram / suffix-automaton drafting (no draft model) | `ngram.py:84-141`; `suffix_automaton.py:75` | draft from the running context via suffix→continuation lookup | repetitive code/JSON/agent-loop output repeats n-grams | PARTIAL (fak `PickDrafter` is model-based) → note (compose with #5257) | note |

## Verdict summary

- **Filed:** epic **#5256** + 4 ship-alone leaves (**#5257** spec-gate, **#5258** no-evict admission, **#5259** KV retention, **#5260** block-hash cache events). No monolith.
- **Note-only residue:** DIVERGENT items carry their tradeoff + their user world (relaxed acceptance, cross-topology reshaping, auto_deploy). Kernel-adjacent autotune-cache → `sota-check`. Several PARTIALs fold into existing epics (#5243, #50, #3352, #3365) rather than spawning new leaves — respecting "we don't do everything."
- **Every dismissal earned by ablation:** PRESENT-on-axis drops name the exact fak file read (`radixkv/eviction_strategy.go`, `radixkv/namespace.go`, `gateway/residency_router.go`, `polymodel/specdecode.go`).

## Honest limits
- The `mcp__fak__fak_feature_query` witness returns the whole index (~600-950K chars) rather than a ranked list — unusable inline; the on-axis witness leaned on raw `Grep` + targeted reads of the named fak seams (the stronger witness the skill pairs it with).
- Shallow clone → no commit-history rationale; the worldview is reconstructed from defaults/non-goals/config, not the authors' testimony (falsifiable via the cited defaults).
- The GPU-bound half of each borrow (the kernels the policies feed) is not CPU-witnessable from this host — but every *filed* borrow is a CPU-witnessable scheduling/policy/datastructure idea.
