---
title: "Incumbent inference architecture bottlenecks — 2026-08-28"
description: "Analysis of structural bottlenecks in vLLM, SGLang, Dynamo, and Modular MAX."
---

# Incumbent inference architecture bottlenecks — 2026-08-28

Issue: [#9894](https://github.com/anthony-chaudhary/fak/issues/9894)

## Verdict

**FAK-authored inference:** The main structural bottleneck in vLLM, SGLang,
NVIDIA Dynamo, and Modular MAX is not one missing kernel optimization. It is the
interaction cross-product among scheduling policy, heterogeneous reusable state,
model/backend/hardware specialization, graph and compilation lifecycle,
distributed ownership, compatibility surfaces, and observability. As these
systems add capabilities, those interactions tend to concentrate in schedulers,
model runners, cache managers, routers, and configuration matrices.

FAK should keep those interactions explicit rather than reproduce the incumbent
shape. The smallest useful architecture is a set of versioned typed contracts for:

1. capability and operating-envelope description;
2. validated composition;
3. resource identity, lifecycle, residency, and ownership;
4. scheduler decision and atomic admission;
5. execution epochs and fast-path eligibility;
6. distributed lease and handoff state;
7. replayable causal evidence.

FAK already has the first substrate in `internal/composition`,
`internal/modeldescriptor`, `internal/resourcelifecycle`,
`internal/causalreceipt`, and `internal/archfitness`. The next move is to extend
those primitives, not introduce a second plugin, orchestration, or scorecard
architecture.

## Scope, cutoff, and evidence limits

- **Source cutoff:** upstream and local evidence observed through **2026-08-28**.
- **Pinned upstream revisions:**
  - vLLM: `vllm-project/vllm@df14152ac6b09f76345dae05b99541c3e87f8d35`.
  - SGLang: `sgl-project/sglang@d12b313b93e1547d9b02c3a84426aa88519fc494`.
  - NVIDIA Dynamo: `ai-dynamo/dynamo@379ccca634a1cd2fdf05993e9725fa4f95a540af`.
  - Modular public monorepo: `modular/modular@1c9fd2e03331f77d3a1034127cb3700b7fa43c02`.
- **Release context observed upstream:** SGLang v0.5.18 and Dynamo v1.4.1 were
  released on 2026-08-22; MAX 26.5 was released on 2026-08-11.
- **Evidence labels used below:**
  - **Upstream-observed fact** means the behavior, design, file, release, or
    proposal is visible in a pinned upstream source.
  - **FAK-authored inference** means this document derives an architectural
    implication; it is not an upstream project's stated conclusion.
  - **Directional statistic** means a path-touch or size indicator useful for
    identifying activity, not a feature count, quality score, or causal measure.
  - **MAX public visibility limit** means the public Modular repository and
    release material do not expose enough proprietary runtime internals to make
    the same source-level claims possible for MAX as for vLLM or SGLang.
- This is an architecture study, not a matched throughput or quality benchmark.
  No performance ranking is claimed.
- Existing FAK research is reused rather than re-counted. In particular, this
  study builds on the composition architecture study and the pinned vLLM and
  SGLang studies listed in the source appendix.

## Fundamental versus accidental bottlenecks

### Fundamental constraints

These constraints remain even under an ideal implementation:

| Constraint | Why it persists |
|---|---|
| Finite accelerator memory and bandwidth | Weights, KV/state, activations, graph artifacts, and transfer buffers compete for bounded capacity. |
| Irregular request lengths and arrivals | Prefill and decode costs vary; agent workloads add pauses, resumes, tools, retries, and long-lived sessions. |
| Heterogeneous state semantics | Token KV, recurrent state, multimodal embeddings, adapters, and speculative state have different units, validity rules, and recomputation costs. |
| Synchronization and ownership | Multi-process and multi-device execution requires an authoritative answer to who may read, mutate, transfer, or reclaim state. |
| Dynamic graph eligibility | Batching, shapes, structured decoding, multimodal work, and recovery can invalidate a captured or compiled fast path. |
| Topology and latency tradeoffs | Placement must balance locality, queueing, transfer cost, memory pressure, and failure domains. |

### Accidental architecture bottlenecks

These are design choices that amplify the fundamental constraints:

| Accidental bottleneck | Failure mode |
|---|---|
| Scheduler owns policy and effects | Admission, allocation, preemption, transfer, execution, metrics, and failure handling converge in one change hotspot. |
| One cache abstraction owns every concern | Identity, allocation, residency, transfer, eviction, sharing, and ownership become coupled even when state kinds differ. |
| Central family/backend switches | Each model, device, quantization, and parallelism addition touches common control paths and expands regression scope. |
| CLI strings become architecture | Unvalidated flag combinations form a second compatibility system with hidden fallback and disable behavior. |
| One lifecycle entry point | Profiling, warmup, compilation, graph capture, empty-forward, and real execution acquire ambiguous ownership and ordering. |
| Transport completion implies handoff | A successful copy is mistaken for committed ownership, leaving stale readers, leaked leases, or double reclamation after failure. |
| Universal hardware abstraction | Semantically different devices are forced through one interface, obscuring where specialization is required. |
| Orchestration precedes local contracts | Distributed components multiply unclear scheduler, residency, and failure semantics instead of composing clean local primitives. |

## System matrix

| System | Upstream-observed structural pressure | FAK-authored inference | Disposition |
|---|---|---|---|
| vLLM | Scheduling, KV allocation/transfer, execution, preemption, speculation, and balancing converge around scheduler and runner paths. Model Runner V2 explicitly addresses lifecycle and state coupling accumulated in V1. Hybrid KV and NIXL lease designs expose state-type and ownership differences. | A scheduler should emit a typed, bounded transaction; it should not directly own every effect. Separate persistent semantic request state from step tensors, and split profile, compile, capture, warmup, and execute lifecycles. | Borrow continuous batching and explicit design documents; adapt lease/state mechanisms; avoid a new runner monolith. |
| SGLang | The scheduler and memory-pool paths integrate admission, pools, cache, workers, speculation, grammar, distributed roles, metrics, and failures. HiCache, PD/EPD, adaptive speculation, structured output, and breakable/piecewise graph work add distinct state and execution modes. | Backend interfaces alone do not contain the model × layout × device × topology cross-product. Execution epochs and semantic conformance fixtures must be first-class. | Borrow radix/prefix reuse and graph-recovery ideas; adapt distributed roles into explicit ownership; avoid central compatibility matrices. |
| NVIDIA Dynamo | Independent frontend, router, planner, and worker roles move pressure to routing, readiness, cache-state projection, placement, and failure reconciliation. Cache-aware routing and disaggregated prefill/decode require state observations from multiple processes. | Selection and admission should be atomic. Cache events need one authoritative projection, anticipated decode footprint should influence placement, and routing should price tier/transfer overlap. | Borrow evented control-plane mechanisms; defer making distributed orchestration FAK's core until local scheduler and residency contracts are clean. |
| Modular MAX | Public evidence shows a vertically integrated compiler/runtime/model-serving stack and active evolution across MAX, KGEN, Mojo, Bazel, compiler, runtime, packaging, and hardware-specific paths. MAX 26.5 moved GPU APIs out of the Mojo standard library into top-level `max` packages, added Apple M1 support and M5 hardware-MMA flash-attention prefill work, and exposed agent skills for serving, benchmarking, evaluation, and profiling. | Strong vertical ownership can enable specialization, but a large compiler/runtime surface can also make boundaries and generated artifacts harder for agents to navigate. FAK should copy explicit workflows and bounded specialization, not infer or reproduce hidden internals. | Borrow operator/agent workflows and provenance; adapt hardware leaves behind semantic tests; defer conclusions about proprietary scheduling and memory internals. |

### vLLM

**Upstream-observed facts:**

- The roadmap and Model Runner V2 design identify model-runner integration debt,
  persistent request-state coupling, and lifecycle ambiguity.
- The V1 `dummy_run` path accumulated profiling, compilation, graph capture,
  warmup, and distributed empty-forward responsibilities.
- The model-development RFC argues that uniform model/compiler abstractions made
  model code opaque and fragile across hardware, and notes that direct model code
  is easier for coding agents to modify.
- Hybrid KV designs must represent full, sliding-window, local, recurrent/Mamba,
  shared, and speculative state with different allocation and reuse semantics.
- NIXL cache leases require renewal, heartbeat, expiration, reclamation, and
  recovery behavior for stale or crashed participants.

**FAK-authored inference:** vLLM's scaling pressure is an integration problem.
FAK should preserve direct model/backend leaves and local compilation while
requiring them to satisfy shared semantic fixtures. Its scheduler should consume
validated capability and resource receipts, emit a decision, and let lifecycle
owners apply that decision.

### SGLang

**Upstream-observed facts:**

- At the pinned revision, `scheduler.py` is approximately 5,466 lines and
  `memory_pool.py` approximately 5,172 lines. These line counts are **directional
  statistics**, not quality judgments.
- Memory paths cover multiple attention and state layouts, including MHA, MLA,
  DSA, sparse, quantized, and FP4 cases.
- HiCache, prefill/decode and encode/prefill/decode disaggregation, adaptive
  speculative decoding, structured output, and breakable/piecewise CUDA graphs
  each introduce distinct admission, residency, or execution behavior.
- Deployment roles affect request state, ownership, retries, completion,
  recovery, and observability rather than remaining outside engine semantics.

**Directional statistics:** Path-touch counts from 2026-06-01 through
2026-08-28 were: diffusion 516, scheduler/managers 478, attention 402, cache 378,
speculation 289, MoE 213, disaggregation 212, distributed 77, multimodal 73, and
gateway 10. These counts indicate where change activity landed; they are not
feature counts, code ownership measures, or proof of maturity.

**FAK-authored inference:** FAK should model execution as explicit epochs rather
than one generic forward loop. Initial epochs should cover prefill, graphable
decode, constrained decode, multimodal encode, tool wait, resume/rebuild, and
recovery. Each epoch declares state inputs, side effects, graph eligibility, and
decline/fallback reasons.

### NVIDIA Dynamo

**Upstream-observed facts:**

- Dynamo decomposes serving into independently scalable workers and control-plane
  roles and supports disaggregated serving and cache-aware routing.
- Its architecture depends on readiness, discovery, routing, placement, and KV
  state observations crossing process boundaries.
- Multiple cache-event producers can describe a state that changes while routing
  and admission decisions are in flight.

**FAK-authored inference:** independent services remove a local code bottleneck
only by creating a distributed consistency bottleneck. FAK should define one
canonical cache-event projection with monotonic generations and make
selection-plus-admission one transaction. Placement should include expected
future decode residency, not only current prefix locality, and should weight
transfer against overlap rather than treating all tiers equally.

### Modular MAX

**Upstream-observed facts:**

- The public `modular/modular` repository spans Mojo, MAX-facing packages,
  compiler/runtime integration, model examples, packaging, and Bazel build
  infrastructure.
- MAX 26.5 release material describes GPU API movement to top-level `max`
  packages, Apple M1 enablement, M5 hardware-MMA flash-attention prefill work,
  and experimental eager-interpreter improvements.
- Public agent skills include `serve-model`, `benchmark-model`, `eval-model`, and
  `profile-model`, making common workflows explicit and reproducible.

**MAX public visibility limit:** the public repository does not expose enough of
MAX's proprietary runtime internals to substantiate source-level claims about
its production scheduler, memory allocator, graph cache, or distributed
ownership protocols. The matrix therefore describes visible packaging,
workflow, compiler, and hardware-specialization evidence only. Any claim that
MAX has the same internal choke points as vLLM or SGLang would be speculation.

**FAK-authored inference:** the useful lesson is not “build a larger compiler
stack.” It is to make high-value workflows discoverable, keep hardware-specific
optimization bounded, and carry provenance from generated or compiled artifacts
back to semantic source and evaluation evidence.

## Cross-system patterns

1. **Policy centralization becomes effect centralization.** Once a scheduler can
   inspect every subsystem, it tends to mutate every subsystem.
2. **The reusable-state problem is larger than token KV.** Recurrent, multimodal,
   speculative, adapter, compiled-graph, and agent-session state require
   different validity and lifetime rules.
3. **Fast paths are conditional capabilities.** Graph capture, compilation,
   speculation, and fused kernels need explicit eligibility and decline reasons;
   silent fallback hides architecture debt and invalidates performance evidence.
4. **Distribution creates ownership work.** Routing and transfer are not complete
   until authoritative ownership, generation, and reclamation state agree.
5. **Interfaces do not erase semantic coupling.** A uniform backend API can hide
   layout, topology, graph, precision, and failure differences rather than
   contain them.
6. **Compatibility surfaces grow faster than features.** Model × backend ×
   hardware × precision × parallelism × deployment-role combinations create a
   test and diagnosis architecture of their own.
7. **Observability must share the decision model.** Metrics that cannot answer
   why admission, placement, reuse, fallback, or reclamation happened are not
   enough to debug the composed system.

## Recent growth areas

**Upstream-observed facts and directional signals:**

- vLLM: Model Runner V2, hybrid and tiered state, disaggregated/elastic serving,
  multiple speculation families, MoE placement and balancing, multimodal/omni
  models, a compilation strategy pivot, and more nightly accuracy/numerics work.
- SGLang: diffusion, scheduler/managers, attention, cache, speculation, MoE,
  disaggregation, structured output, multimodal, and graph break/recovery paths.
- Dynamo: cache-aware routing, disaggregated prefill/decode, independent worker
  scaling, KV-event aggregation, and topology-aware placement.
- MAX: hardware-specific attention/compiler work, broader Apple hardware support,
  eager execution, packaging/API movement, and task-specific agent workflows.

**FAK-authored inference:** growth should enter FAK as new compositions and typed
resource/epoch variants, not as flags distributed through a central scheduler.
The first composition fixtures should cover Qwen3.8 native serving, mixed
attention/recurrent state, speculative decode, multimodal encode/decode,
disaggregated handoff, structured output, and MoE placement.

## Existing FAK primitives

| Existing primitive | Current contract | Architectural role |
|---|---|---|
| `internal/composition` | `fak.composition-snapshot/1` plus a validated composition receipt | Binds intent, model, execution, resource claims, and graph edges before execution. |
| `internal/modeldescriptor` | `fak.model-capability-descriptor/1` and `fak.model-onboarding-report/1` | Describes witnessed fak-native capability and enforces an onboarding coupling budget. |
| `internal/resourcelifecycle` | Claims, allocations, observations, ownership, placement, teardown, and receipts | Provides the seed for explicit resource authority and lifecycle accounting. |
| `internal/causalreceipt` | `fak.causal-receipt/1` | Joins IDs, decisions, phases, resources, metrics, and incident answers into replayable evidence. |
| `internal/archfitness` | `fak.architecture-fitness/1` and ratchet behavior | Measures dependency DAG, frozen seams, family switches, change amplification, descriptor coverage, ownership, fixtures, causal evidence, schema migration, hot-path scaling, privacy/cardinality, and stale exceptions. |

These are **repository-observed facts** at the pinned FAK checkout. They already
cover much of the required architecture language. Missing behavior should extend
these schemas or add narrowly owned leaves that compose with them.

## Target typed contracts

### 1. Capability descriptor and operating envelope

Declare supported model family, state kinds, precision/quality constraints,
backend/device requirements, graph/compile capabilities, topology assumptions,
and witnessed limits. Preserve the `fak-native` engine invariant and make every
fallback an explicit declined capability, never a silent engine switch.

### 2. Validated composition graph

Resolve intent, model descriptor, execution phases, resource claims, policy, and
backend leaves before admission. Invalid combinations fail at composition time,
not after partial allocation.

### 3. Resource algebra and lifecycle

Represent state identity, validity, residency, cost, and authority independently
of the mechanism that stores or transfers it. Lifecycle operations produce
receipts and use generation-checked transitions.

### 4. Scheduler decision and admission transaction

Use a pure or replayable planner input and output. The decision should include
selected work, resource reservations, preemption/reclamation intent, execution
epoch, fast-path selection, and explicit decline reasons. Admission commits only
when all required reservations are valid; otherwise it aborts without partial
ownership.

### 5. Cache event and state projection

Normalize allocation, materialization, hit, miss, transfer, invalidate, evict,
expire, and reclaim events into one authoritative projection. Producers may be
distributed, but identity and generation rules are shared.

### 6. Execution epoch and fast-path eligibility

An epoch declares semantic work and allowed execution mechanisms. Graph,
compiler, speculation, and kernel leaves return typed eligible, declined, or
failed outcomes. Profiling, warmup, capture, and execution remain distinct
lifecycle operations.

### 7. Distributed handoff and lease ownership

Model offer, reserve, transfer-start, transfer-complete, accept, commit, renew,
abort, expire, and reclaim. Transport success alone does not transfer authority.
One owner is authoritative for every generation.

### 8. Causal receipt

Join composition digest, scheduler decision, resource transitions, epoch,
fast-path eligibility, ownership handoffs, quality envelope, and metrics. The
receipt must answer both “what ran?” and “why did the preferred path not run?”

## Resource algebra

Every reusable resource should carry at least:

| Field | Meaning |
|---|---|
| `kind` / `layout` / `unit` | Semantic state type, physical representation, and accounting unit. |
| `producer` / `owner` | Who created the state and who currently has authority to mutate or release it. |
| `identity` / `reuse_key` | What content or request history makes reuse valid. |
| `generation` | Monotonic validity version used to reject stale observations and transfers. |
| `residency` | Device, host, remote tier, durable store, or absent. |
| `dependencies` / `recompute_cost` | Inputs required to rebuild and the estimated replacement cost. |
| `lifetime` / `lease` | Scope, expiry, renewal, and reclamation rules. |
| `transfer_protocol` | Offer, copy, verification, acceptance, commit, and abort semantics. |
| `precision` / `quality` | Quantization, loss, compatibility, and quality-envelope constraints. |

The algebra must cover token KV, recurrent/Mamba state, encoder and multimodal
embeddings, draft/speculative state, graph and compilation artifacts, adapters
or LoRA state, and durable agent-session state. Page/block allocation remains a
mechanism under this contract, not the universal semantic abstraction.

## Borrow, adapt, reject, and defer

### Borrow

- Continuous batching, chunked prefill, prefix/radix reuse, and bounded
  preemption mechanisms from vLLM and SGLang.
- Explicit design/RFC artifacts for runner, cache, graph, and disaggregation
  changes.
- Cache-event publication and topology-aware routing mechanisms from Dynamo.
- MAX's task-specific serve, benchmark, evaluate, and profile workflows.
- Direct, hardware-specialized model/kernel code when paired with semantic
  conformance tests and provenance.

### Adapt

- Convert scheduler choices into typed decisions and atomic admission receipts.
- Convert cache managers into resource-specific mechanisms under one event and
  ownership contract.
- Convert graph breaks and fallbacks into execution-epoch transitions with
  explicit reasons.
- Convert distributed roles into generation-checked leases and handoffs.
- Convert compiler/generated artifacts into deterministic outputs tied to source,
  descriptor revision, and evaluation evidence.

### Reject

- A universal callback-hub scheduler that owns all side effects.
- A single page abstraction for all reusable state.
- Dynamic hot-path plugin discovery for mechanisms known at build time.
- Central model/backend/family switches for ordinary onboarding.
- Silent llama.cpp, backend, graph, precision, or quality fallback.
- Internal microservices created only to look modular.
- Backend copies without shared semantic tests.

### Defer

- Orchestration-first decomposition of FAK into independent services.
- A universal distributed control plane before local ownership and admission
  semantics are replayable and tested.
- A broad compiler framework beyond the concrete needs of current fak-native
  kernels and Qwen3.8 serving.
- Conclusions about proprietary MAX runtime internals until public or directly
  witnessed evidence exists.

## Agentic-development workflow and coupling budgets

FAK should optimize not only runtime throughput but also the cost and reliability
of an agent making a change:

1. Add or update a witnessed capability descriptor.
2. Implement one bounded model, backend, kernel, resource, or policy leaf.
3. Add a minimal composition fixture that proves the capability resolves.
4. Run semantic conformance tests shared across specializations.
5. Capture a causal receipt for the success path and at least one decline or
   recovery path.
6. Measure onboarding amplification with the existing
   `modeldescriptor` report and `archfitness` ratchet.
7. Generate code only when regeneration is deterministic and provenance points
   to semantic source, generator version, and output digest.

The coupling budget should continue to count:

- core switches added;
- files changed outside the owning leaf;
- architecture/family branches;
- duplicated lifecycle code;
- duplicated metrics or causal projection;
- packages and composition fixtures intersected by the change.

**FAK-authored inference:** ordinary model or backend onboarding should require no
central dispatcher edit. If a capability cannot fit the budget, the change must
name the missing contract rather than normalize another exception. Planning
should remain pure or replayable; allocation, transfer, execution, and teardown
are effectful stages whose receipts can be independently witnessed.

## Priority sequence

1. **Freeze scheduler decision/admission semantics.** Define the planner input,
   typed decision, reservation set, commit/abort behavior, and causal projection.
2. **Strengthen resource algebra and cache-event projection.** Add identity,
   generation, residency, lease, transfer, and reclaim semantics across KV and
   non-KV state.
3. **Introduce execution epochs and typed fast-path eligibility.** Separate
   profile, compile, capture, warmup, execute, tool-wait, resume, and recovery.
4. **Prove local Qwen3.8 compositions.** Exercise mixed state, structured output,
   speculation, multimodal work, and MoE placement without central switches.
5. **Add distributed handoff only after local receipts replay cleanly.** Then
   test disaggregated prefill/decode and remote-tier ownership under failure.
6. **Ratchet architecture fitness.** Add findings to the existing scorecard for
   scheduler side effects, ambiguous generations, silent fallback, lifecycle
   conflation, and onboarding amplification.
7. **Optimize mechanisms inside the frozen contracts.** Tune kernels, allocator,
   graph capture, routing, and tier overlap without changing semantic ownership.

## Issue coverage and gaps

### Covered by existing work

- **#8395 — Qwen3.8 serving baseline:** primary operating-envelope and native
  composition target.
- **#8783 — architecture debt cohort:** venue for ratcheted central-switch,
  change-amplification, and lifecycle debt.
- **#9271 — Dynamo exhaustive research:** deeper upstream coverage for routing,
  disaggregation, and cache-state projection.
- **#9279–#9285 — architecture epic and leaves (closed):** delivered composition,
  capability descriptors and coupling budget, resource lifecycle, causal receipt,
  and architecture fitness substrate.
- **#9377 — Qwen3.8 native IR optimization:** compiler/IR specialization inside
  the native operating envelope.
- **#9378 — allocator fragmentation:** mechanism-level residency and allocation
  work that should consume the resource contract.
- **#9887 — learning-mesh upstream-study ingestion:** durable ingestion of
  upstream changes and provenance.
- **#9894 — this study:** cross-incumbent synthesis and target constraints.

### Remaining gaps to reconcile under #9894 before filing new issues

These are architecture gaps, not newly created tickets:

1. A concrete scheduler decision/admission transaction schema and replay fixture.
2. A canonical cache-event projection with generation and ownership invariants.
3. Execution-epoch and fast-path eligibility schemas with decline reasons.
4. Distributed handoff/readiness/failure semantics joined to resource receipts.
5. Architecture-fitness checks for scheduler side effects, lifecycle conflation,
   silent fallback, and generation ambiguity.
6. A witnessed onboarding change-amplification report across at least two model
   or backend specializations.
7. Composition fixtures for mixed state, speculative decode, multimodal work,
   MoE placement, structured output, and disaggregated handoff.

Before creating any follow-on, search the listed issues and their descendants;
attach the gap to an existing owner when its intended outcome and witness match.

## Source appendix

### Local FAK architecture and synthesis

- [`docs/research/agent-serving-composition-architecture-study-2026-08-26.md`](agent-serving-composition-architecture-study-2026-08-26.md)
- [`docs/research/vllm-fak-join-2026-08-27/README.md`](vllm-fak-join-2026-08-27/README.md)
- [`docs/research/sglang-2026-08-27/README.md`](sglang-2026-08-27/README.md)
- [`docs/notes/CONCEPT-STUDY-DYNAMO-2026-07-18.md`](../notes/CONCEPT-STUDY-DYNAMO-2026-07-18.md)
- [`docs/notes/CONCEPT-STUDY-MOJO-2026-08-21.md`](../notes/CONCEPT-STUDY-MOJO-2026-08-21.md)
- [`internal/composition/composition.go`](../../internal/composition/composition.go)
- [`internal/modeldescriptor/descriptor.go`](../../internal/modeldescriptor/descriptor.go)
- [`internal/resourcelifecycle/lifecycle.go`](../../internal/resourcelifecycle/lifecycle.go)
- [`internal/causalreceipt/receipt.go`](../../internal/causalreceipt/receipt.go)
- [`internal/archfitness/fitness.go`](../../internal/archfitness/fitness.go)

### vLLM upstream locators

- Pinned tree: <https://github.com/vllm-project/vllm/tree/df14152ac6b09f76345dae05b99541c3e87f8d35>
- Roadmap issue #39749: <https://github.com/vllm-project/vllm/issues/39749>
- Model-development RFC #42770: <https://github.com/vllm-project/vllm/issues/42770>
- Model Runner V2 design: <https://github.com/vllm-project/vllm/blob/df14152ac6b09f76345dae05b99541c3e87f8d35/docs/design/model_runner_v2.md>
- Hybrid KV cache manager: <https://github.com/vllm-project/vllm/blob/df14152ac6b09f76345dae05b99541c3e87f8d35/docs/design/hybrid_kv_cache_manager.md>
- NIXL KV cache lease: <https://github.com/vllm-project/vllm/blob/df14152ac6b09f76345dae05b99541c3e87f8d35/docs/design/nixl_kv_cache_lease.md>
- Disaggregated prefill: <https://github.com/vllm-project/vllm/blob/df14152ac6b09f76345dae05b99541c3e87f8d35/docs/features/disagg_prefill.md>

### SGLang upstream locators

- Pinned tree: <https://github.com/sgl-project/sglang/tree/d12b313b93e1547d9b02c3a84426aa88519fc494>
- Scheduler: <https://github.com/sgl-project/sglang/blob/d12b313b93e1547d9b02c3a84426aa88519fc494/python/sglang/srt/managers/scheduler.py>
- Memory pool: <https://github.com/sgl-project/sglang/blob/d12b313b93e1547d9b02c3a84426aa88519fc494/python/sglang/srt/mem_cache/memory_pool.py>
- Scheduler/admission example, PR #17026: <https://github.com/sgl-project/sglang/pull/17026>
- Radix/prefix reuse, issue #26618: <https://github.com/sgl-project/sglang/issues/26618>
- Chunked prefill, PR #35300: <https://github.com/sgl-project/sglang/pull/35300>
- Prefill/decode disaggregation, PR #35224: <https://github.com/sgl-project/sglang/pull/35224>
- Adaptive speculation, PR #21272: <https://github.com/sgl-project/sglang/pull/21272>
- Structured output, PR #28804: <https://github.com/sgl-project/sglang/pull/28804>
- Multimodal/MoE, PR #27602: <https://github.com/sgl-project/sglang/pull/27602>
- Distributed execution, issue #22084: <https://github.com/sgl-project/sglang/issues/22084>
- Observability, PR #23169: <https://github.com/sgl-project/sglang/pull/23169>
- Failure/reliability, PR #32118: <https://github.com/sgl-project/sglang/pull/32118>

### NVIDIA Dynamo upstream locators

- Pinned tree: <https://github.com/ai-dynamo/dynamo/tree/379ccca634a1cd2fdf05993e9725fa4f95a540af>
- Project documentation: <https://docs.nvidia.com/dynamo/latest/>
- Architecture overview: <https://docs.nvidia.com/dynamo/latest/architecture/overview.html>
- KV-aware routing: <https://docs.nvidia.com/dynamo/latest/components/router/router.html>
- Disaggregated serving: <https://docs.nvidia.com/dynamo/latest/architecture/disagg_serving.html>

### Modular MAX public locators

- Pinned public tree: <https://github.com/modular/modular/tree/1c9fd2e03331f77d3a1034127cb3700b7fa43c02>
- MAX documentation: <https://docs.modular.com/max/>
- Modular release notes: <https://docs.modular.com/max/changelog/>
- Public agent skills: <https://github.com/modular/modular/tree/1c9fd2e03331f77d3a1034127cb3700b7fa43c02/.agents/skills>

The MAX links are public-product and public-repository evidence. They do not
remove the proprietary-runtime visibility limit stated above.
