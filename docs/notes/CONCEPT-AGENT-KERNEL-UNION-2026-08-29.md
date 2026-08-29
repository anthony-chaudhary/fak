---
title: "The agent-kernel union: harness, runtime, memory, and the layers between"
description: "A many-dimensional architecture map for coordinating FAK's native harness, highly quantized runtime, semantic memory, cache, tools, trust, scheduling, and outcome learning."
status: working
last_updated: 2026-08-29
---

# The agent-kernel union: harness, runtime, memory, and the layers between

Date: 2026-08-29  
Issue: [#10177](https://github.com/anthony-chaudhary/fak/issues/10177)  
Study receipt: `study_a1d0889044a0abefbf535796220098d4aff6b7c17bd5fdbd4e8851d69e67579d`  
Companions: [#6042](https://github.com/anthony-chaudhary/fak/issues/6042), [#9279](https://github.com/anthony-chaudhary/fak/issues/9279), [#1570](https://github.com/anthony-chaudhary/fak/issues/1570), [#2387](https://github.com/anthony-chaudhary/fak/issues/2387), [#1911](https://github.com/anthony-chaudhary/fak/issues/1911)

## Verdict first

FAK's largest potential advantage is not three good modules that exchange messages. It is one kernel that can connect an agent object's **semantic life** to the cost and correctness of every **physical representation** of that object.

The harness knows why an object exists, which future steps and agents depend on it, its deadline, its authority, and when it becomes dead. Memory knows whether it is useful, fresh, contradicted, witnessed, tainted, durable, or expensive to retrieve. The native runtime knows whether its serialized bytes, tokens, KV, recurrent state, tool artifact, model weights, and device allocation are resident, compatible, costly, or under pressure. A terminal witness knows whether keeping or using it actually helped.

The important fusion is therefore **causal identity plus bounded feedback**, not a god object or one storage heap:

```text
semantic task graph
      |
      v
versioned execution envelope
      |
      +--> semantic memory record
      +--> serialized prompt/tool segment
      +--> token range
      +--> attention KV / recurrent snapshot
      +--> tool artifact / sandbox state
      +--> placement and quantization receipt
      |
      v
witnessed task outcome -> utility, invalidation, calibration, retry policy
```

Each representation keeps its real owner, authority, geometry, lifetime, and mutation rules. The union supplies lineage and coordination across them. In other words: **one identity spine, several typed lifecycles**.

This is a stronger and safer center than either extreme:

- A loose stack cannot see enough to make globally good decisions.
- A monolith erases the differences between semantic facts, prompt bytes, KV tensors, recurrent state, tool effects, and policy authority.

FAK already ships many of the nouns and several real arrows. The live union remains **PARTIAL** because its observations are richer than its online decisions.

## Value frame

- **For:** operators and harness authors running long-lived, tool-using agents through fak-native, highly quantized models.
- **Problem:** harness intent, memory value, prompt layout, cache state, model state, quantization, tool effects, and outcomes are governed by local policies that can work against one another.
- **Today:** FAK has typed coordination adapters plus strong context, cache, recall, policy, native-inference, trajectory, and harness primitives. Semantic memory still does not automatically influence live native turn planning, model routing, KV policy, or tool scheduling, and runtime outcomes do not close one bounded memory-policy loop.
- **Better because:** every layer contributes only the facts it owns to one versioned, content-free envelope; one constrained plan can then optimize accepted task completion instead of a proxy such as tokens/s, cache hit rate, or retrieval score.
- **Witness:** shadow the union on captured Qwen3.8 agent trajectories first. Graduate one policy at a time only when it beats the layer-local control inside a declared quality band after coordinator, retrieval, transfer, retry, tool, and verification cost.

Centrality: **Core**. P1 managed context, P2 net-true efficiency, P3 bounded adaptation, and P4 integrated operations all meet at this seam.

## What fusion means here

The word *fusion* covers four increasingly strong choices. FAK should support each deliberately rather than treating them as synonyms.

| Level | Meaning | Earliest useful outcome | Main risk |
|---|---|---|---|
| Contract composition | Modules exchange typed identities, observations, constraints, and receipts. | Explainability, portability, safer integration. | Schemas proliferate without changing the live path. |
| Control-plane fusion | One bounded planner coordinates memory, cache, route, schedule, tools, and harness pressure. | Avoid locally rational but globally expensive actions. | Oscillation, stale observations, or a policy god object. |
| Data-plane fusion | Compatible representations share arenas, handles, prefixes, weights, or transfers. | Zero-copy or one-copy reuse, lower footprint, faster continuation. | Trust leaks, incompatible layouts, or false identity. |
| Learning-plane fusion | Witnessed outcomes update retrieval, retention, routing, quantization, and tool policy. | The system improves from its own real workload. | Self-reinforcing errors, poisoning, or optimizing a proxy. |

The first two are the general architectural advantage. Data-plane fusion is selective: shared quantized weights and exact compatible prefixes are good candidates; merging semantic memory and KV into one mutable lifecycle is not. Learning-plane fusion is last because it needs the strongest provenance and rollback discipline.

## The shared execution envelope

The minimum common object is a content-free descriptor. Call it `AgentWorkingSetObject` for discussion; the name is not yet an API commitment.

```text
identity
  object_id, task_id, step_id, attempt_id, branch_ids, tenant/trust_domain

semantic role
  instruction | tool_schema | retrieved_fact | plan_state | tool_result |
  checkpoint | negative_experience | output_constraint

lineage and authority
  producer, consumers/refcount, source_digest, witness, taint, capability,
  observed_at, valid_from, valid_to, supersedes, invalidated_by

value and lifetime
  deadline, reuse_horizon, predicted_fanout, durability, quality_criticality,
  retrieval_score, outcome_utility, contradiction_state

representations
  canonical_bytes_digest, serializer/tokenizer identity, token span,
  prefix/KV IDs, recurrent-state snapshot, tool artifact, checkpoint handle

physical state
  model_revision, weight_quant, KV/state dtype, RoPE/layout regime,
  bytes by tier, location, warm/cold state, transfer/recompute cost, pressure

control
  hard pins, allowed loss/error budget, policy owner, advisory confidence,
  generation, valid_until, idempotency key, rollback/compensation contract

outcome
  used_by, attention/access evidence, tool/task result, retries, accepted quality,
  wall time, resource-time, net cost, independent witness
```

The envelope is not hidden session state. It is versioned, hashable, generation-bound, and explicit enough to replay. MCP's 2026-07-28 move from hidden protocol sessions toward self-describing requests and explicit application handles reinforces this direction; an application may remain stateful, but the state handle should be visible in the work graph ([MCP release](https://blog.modelcontextprotocol.io/posts/2026-07-28/)).

## Pairwise information flow

### Harness to runtime

The harness can tell the runtime things an ordinary inference request cannot:

- task DAG, dependencies, critical path, fan-out/fan-in, and remaining child count;
- expected next tool or model phase and likely continuation length;
- deadline, interactivity, cancellation state, aggregate budget, and value of finishing;
- stable versus volatile prompt segments and exact serializer/tool-schema versions;
- output grammar, tool-call schema, and whether a turn can be elided mechanically;
- tool purity, idempotence, side effects, expected duration, and compensation path;
- which branches share a prefix and which results have exactly one consumer;
- authority, tenant, workspace, sensitivity, and witness requirements.

The runtime can use this to batch siblings, clone exact prefixes, retain KV across short tool gaps, evict dead intermediates, prewarm the right model, reserve result-tail capacity, prioritize the critical path, or reduce fan-out under pressure.

### Runtime to harness

The runtime should return typed physical truth rather than generic "busy" or "slow":

- actual model and engine, weight quantization, KV/state dtype, and compatibility regime;
- admission delay, queue and batch headroom, prefill/decode capacity, and device pressure;
- prefix/KV affinity, first divergence, residency tier, and reuse confidence;
- predicted time to first token, post-tool resume delay, decode tail, and saturation;
- model-load or transfer cost, available warm alternatives, and failure domain;
- a typed pressure action: execute, reduce fan-out, delay, reroute, compact, or escalate.

The harness can then change the graph before it becomes expensive: defer low-value branches, run independent tools while decode is congested, route a schema-fragile step to a stronger envelope, or ask for a smaller retrieval view.

### Harness to memory

The harness supplies memory with purpose and lifetime:

- objective, subgoal, task class, user and workspace scope;
- which branch or future step is expected to consume a fact;
- expected reuse horizon, deadline, and promotion boundary;
- whether a result is ephemeral, checkpoint state, a reusable plan, a durable fact, or negative knowledge;
- required source authority and acceptable uncertainty;
- task completion, user correction, rollback, and explicit "this is no longer true" events.

This lets memory distinguish a one-turn tool blob from a durable invariant and avoid promoting raw trajectory text just because it was available.

### Memory to harness

Memory should return more than text and a similarity score:

- candidate identity, provenance, witness, reference time, observed time, and validity interval;
- contradiction and supersession state;
- confidence, coverage, omissions, and whether evidence is advisory or load-bearing;
- token cost, available lossless/lossy views, source reachability, and expected retrieval latency;
- prior witnessed outcomes, reusable plan templates, failed approaches, and unresolved work;
- scope and capability constraints for any remembered tool, credential, or action.

The harness can decide whether to inject, query further, ask the user, route to a stronger model, or keep the item out of control instructions.

### Memory to runtime

Memory can expose physical-policy inputs without exposing storage internals:

- exact cell and canonical-byte digests, serializer and tokenizer identity;
- relevance, outcome utility, freshness, witness strength, and invalidation dependencies;
- required precision and quality criticality;
- expected reuse horizon, fan-out, consumer count, and retrieval deadline;
- known prefix/KV handles or compatible cached representations;
- whether a compact view is allowed and what source must remain recoverable.

The runtime can map logical value to physical choices: prompt placement, KV priority, prefetch, offload, transfer, recomputation, or precision. TensorRT-LLM already demonstrates the lower-level actuator shape with per-token-range priority and duration plus separate decode retention; FAK's opportunity is to drive an equivalent native actuator from semantic value and trust ([TensorRT-LLM KV cache](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)).

### Runtime to memory

The runtime closes the loop with realized rather than predicted cost:

- whether the item was accessed, attended to, refetched, faulted, or unused;
- exact prefix/KV hits, first divergence, residence time, tier movement, and eviction cause;
- token, byte, queue, transfer, recompute, and energy/resource-time cost;
- actual model/quantization/state envelope and any quality or conformance failure;
- tool/world invalidation events and changed source digests;
- terminal outcome and witness binding.

This supports bounded promotion, decay, negative utility, retest triggers, and workload-specific quantization calibration. Runtime attention or hidden-state taps should contribute bounded derived evidence, not raw activations by default.

## The adjacent layers belong in the union

Harness, runtime, and memory are the memorable triangle, but six adjacent layers are load-bearing.

| Layer | Why it belongs | Information it owns |
|---|---|---|
| Prompt compiler / tokenizer | Semantic equality is insufficient for prefix reuse. | Exact serialized segment identity, token spans, cache breakpoints, first divergence, grammar. |
| Scheduler / placement | A good object can still be on the wrong device or behind the wrong queue. | DAG critical path, locality, batch slots, capacity, transfer/recompute, deadlines. |
| Tool-effect system | Agent loops spend much of their life outside model execution. | Purity, idempotence, side effects, ETA, result freshness, retry and compensation. |
| Trust / policy | A useful prediction is not authority. | Capability, tenant, taint, provenance, approval, revocation, redaction, retention. |
| Evaluator / witness | Self-reported success cannot train policy safely. | Accepted outcome, quality band, counterfactual, failure class, rollback, credit assignment. |
| Telemetry / operations | The union must be observable and reversible on the real path. | Stable causal IDs, generation, timings, bytes, state transitions, errors, health, replay. |

OpenTelemetry's GenAI vocabulary is useful as an export projection, but not as FAK's source of truth. The internal event should remain dependency-free, bounded-cardinality, and content-free by default.

## The actual agentic load profile

The correct optimization unit is a **job/program**, not a model request. Agentix reaches the same conclusion externally: request-only servers miss dependencies and cumulative program waiting; its scheduler enriches calls with program context ([Agentix, NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/luo)).

A representative native agent loop is:

```text
cold model load / promotion
  -> large stable system + policy + tools + repository prefix
  -> short reasoning / structured tool-call decode
  -> external tool or human wait
  -> bursty, potentially untrusted result admission
  -> small delta prefill + short continuation decode
  -> compaction / page-out / memory proposal / cache lifetime choice
  -> repeat, branch, join, retry, or suspend
```

The load is shaped by six asymmetries:

1. A large stable prefix and a small volatile tail.
2. Expensive prefill and serial, bandwidth-sensitive decode.
3. Active model phases alternating with long tool-idle gaps.
4. Tiny trusted control state beside large untrusted tool output.
5. Many sibling agents sharing setup before their branches diverge.
6. Long-lived logical sessions whose physical state may be idle for minutes or hours.

FAK's 2026-08-28 public-safe Qwen trajectory readback selected 150 canonical sessions with 108,393,944 fresh-input tokens and 2,618,842,368 cache-read tokens. Cache reads were about 96.0% of observed prompt tokens in that topical cohort. This is evidence that prefix reuse is central to the current Qwen work, not a claim about every user or every agent workload (`docs/notes/QWEN-TRAJECTORY-SNAPSHOT-DOGFOOD-2026-08-27.md:82-89`).

Continuum makes one agent-specific opportunity concrete: tools create pauses during which ordinary request schedulers evict KV, even though the same job may resume soon. Its proposed TTL considers reload/recompute and queue cost rather than pinning indefinitely ([Continuum v6](https://arxiv.org/abs/2511.02230)).

## Why highly quantized models change the design

Quantization is not merely a runtime implementation detail. It changes which parts of the union are scarce, reliable, and worth coordinating.

### 1. Weight residency becomes a high-value shared asset

A quantized model is smaller than its full-precision counterpart, but still expensive to load, move, and warm. The harness can expose future task/branch demand so the runtime keeps the right model resident across tool waits and co-hosts compatible work instead of churning weights.

The freed capacity is a portfolio choice, not automatically "more batch": it can buy more resident models, more concurrent seats, longer KV, a higher-precision critical state tier, tool sandboxes, or OS page-cache headroom. The working-set envelope makes that trade explicit.

### 2. Weight precision, KV precision, and recurrent-state precision are different axes

FAK's live `KVCache` still stores float32 rows; its 4-bit KV codec is explicitly not wired into decode (`internal/model/kvquant.go:20-24`). The radix reuse fence already binds model, KV dtype, quant mode, and RoPE identity (`internal/radixkv/regimefence.go:27-46`). This is the right correctness stance: exact prompt identity is not enough when the physical decode regime differs.

For hybrid attention/recurrent models, recurrent state cannot be treated as ordinary per-token KV. FAK's cache returns a typed unsupported verdict for middle eviction when Gated-DeltaNet recurrence would retain the removed information (`internal/model/kvcache.go:45-54`). Any union contract must carry state kind and geometry, not a generic `cache_bytes` field.

### 3. Agentic quality is the acceptance gate

Perplexity or a generic language benchmark is not enough. ACBench found that its tested 4-bit GPTQ/AWQ arms had only 1%-3% loss on workflow/tool tasks but 10%-15% loss on real-application accuracy. The result is bounded to its models, methods, and tasks, but it decisively rejects a global "4-bit is safe" assumption ([ACBench, ICML 2025](https://proceedings.mlr.press/v267/dong25k.html)).

The reusable experience key must therefore include at least:

```text
model revision + prompt template + weight quant method + KV/state precision
+ task/phase class + tool-schema version + output grammar + hardware/backend
```

Positive and negative memory should be scoped to that envelope. A plan that worked for full precision or one tool schema may be actively misleading for a different low-bit regime.

### 4. The harness can compensate, but must not hide quality loss

Highly quantized models may benefit disproportionately from:

- exact, stable tool schemas;
- concise witnessed retrieval rather than a raw memory dump;
- grammar-constrained JSON/tool output;
- deterministic mechanical turn avoidance;
- stronger-model escalation for fragile phases;
- explicit correlated-call and task-outcome checks.

These are system optimizations only if they preserve the declared task-quality band. They cannot turn a broken model result into a nominal pass.

### 5. Workload-specific calibration becomes possible

Owning the harness supplies the real calibration distribution: repository instructions, code, diffs, JSON/tool schemas, memory views, error messages, and long tool-result tails. Owning the runtime makes it possible to test quantization and kernel changes on this distribution. Owning the outcome witness prevents optimization toward reconstruction error alone.

### 6. Avoided work can dominate faster work

On a slow local quantized path, deleting one unnecessary model turn, retaining a reusable prefix, or avoiding a refetch can be worth more than a modest kernel speedup. The opportunities multiply:

```text
fewer model turns
  x less prefill per surviving turn
  x more compatible decode work per resident weight stream
  x more resident sessions per byte
  x fewer retries from witnessed memory and tool policy
```

This is an opportunity decomposition, not a 100x performance claim.

## Many-dimensional advantage map

| Dimension | Union advantage | First checkable policy |
|---|---|---|
| Task latency | Schedule the job's critical path, including tools and waits, not isolated requests. | Shadow program-aware priority against FCFS with full job completion time. |
| Prefill cost | Harness-defined stable segments become exact compiler/cache boundaries. | Compare first-divergence and fresh-prefill tokens with stable-prefix layout on/off. |
| Decode throughput | Fan-out siblings reveal compatible batch opportunities before requests arrive. | Batch sibling continuations over one exact prefix; measure throughput and tail latency. |
| Tool wait | Expected tool duration and continuation probability govern KV retain/offload/evict. | Tool-aware TTL versus LRU with queue and recompute cost charged. |
| Retrieval | Runtime price and capacity choose full evidence, compact view, top-k facts, or no recall. | Cache-aware retrieval budget A/B on accepted task outcome. |
| Memory quality | Only witnessed outcomes credit memories; contradictions create tombstones. | Recall on/off plus stale/contradiction and refetch counts. |
| Quantization quality | Results are scoped to exact model/quant/task/phase envelopes. | Tool/JSON/long-context/task suite across precision arms. |
| Resource use | Semantic lifetime informs physical retention; dead refcount-1 results can disappear. | Prove refcount and rollback, then compare resident bytes and replay equivalence. |
| Multi-agent sharing | Shared setup, tool catalogs, policies, plans, and memory blocks are named once. | Cross-agent prefix and memory reuse with explicit promotion and tenant isolation. |
| Safety | Trust domain, provenance, taint, and capability travel with every representation. | Same bytes under two trust domains must never produce a cross-domain hit or action. |
| Reliability | Checkpoint state, semantic memory, and physical caches remain distinct but correlated. | Crash/restart equivalence with stale-generation refusal. |
| Adaptation | Optional policies consume common events and emit closed actions. | Kill-switch every policy to layer-local behavior without corrupting work. |
| Observability | One causal receipt explains choice, cost, state movement, tool effect, and outcome. | Deterministic replay reproduces the decision without payload contents. |
| Privacy | Coordination uses digests, sizes, classes, and IDs by default. | Content-free trace test and cardinality/scrubbing gate. |
| Energy/cost | Retain, move, or recompute decisions can price accelerator, host, network, and storage. | Resource-time and energy-aware shadow choice with net cost. |
| Product UX | The agent can explain what it remembers, what is paged out, and why it paused. | Operator view derived from the same receipt, not a second state model. |

## Highest-leverage policies

### Default candidates after proof

1. **Stable-prefix compiler.** Use harness segment volatility and exact serializer identity to keep setup/tool catalogs stable and append volatile state late.
2. **Program-aware admission.** Optimize task completion and critical-path delay, with hard policy and budget constraints.
3. **Outcome-gated memory.** Promote or credit only through an external witness; retain failures as scoped counterevidence.
4. **Quantization receipt.** Every run and learned outcome records actual model, weight quantization, KV/state precision, backend, and template.
5. **Explicit state handles.** Checkpoint, semantic memory, tool artifact, and cache references are visible, typed, and separately governed.

### Optional modules

1. Tool-aware KV TTL and tiering.
2. Logical-memory value to token-range KV priority.
3. Cache-aware retrieval budgets.
4. Background consolidation during tool waits.
5. Plan-template memory for repeated workflows.
6. Shared memory attachments with explicit scope and revocation.
7. Phase-specific route/escalation under low-bit quality uncertainty.

### Watch

1. Adaptive KV precision.
2. Quantized recurrent-state tiers.
3. Hidden-state or attention-derived memory utility.
4. Speculative read-only tool execution learned from trajectories.
5. Learned transition policies over observe -> score -> predict -> act.

### Exclude

1. Raw trajectories automatically promoted to memory.
2. Semantic similarity used as proof of exact prompt/KV reuse.
3. One lifecycle for logical memory and physical KV.
4. Remote annotations or model predictions treated as authority.
5. Contradictions overwritten or erased instead of invalidated with history retained.
6. A global "4-bit safe" or "quantization bad" label detached from task and method.
7. A controller whose own cost, instability, or failure path is unmeasured.

## Negative memory is first-class

The highest-value memory may be "do not repeat this under these conditions." A negative record needs structure, not a prose lesson:

```text
subject/action
failure_or_contradiction
scope = model_revision + weight_quant + KV/state dtype + tool/schema + environment
preconditions
observed_at / valid_from / valid_to
source + external witness
retry, idempotency, rollback, and compensation status
confidence
expiry or retest trigger
superseded_by / disconfirmed_by
policy taint
```

High-value negatives include invalidated facts, stale schemas, unsafe-to-speculate tools, non-idempotent retries, plans that matched semantically but failed operationally, retrievals that distracted the model, quant-specific tool/JSON failures, prefetches with negative net value, and user-corrected assumptions. "Unknown" remains distinct from a negative fact.

Graphiti's episode model is useful prior art: episodes carry provenance and reference time; its bulk path explicitly omits edge invalidation and is therefore appropriate only when invalidation is unnecessary. That negative design fact supports explicit correctness modes for fast memory ingestion ([Graphiti episode ingestion](https://help.getzep.com/graphiti/core-concepts/adding-episodes)).

## Current FAK witness

### PRESENT

- `internal/coordination/harness_adapter.go:108-196` carries a harness-neutral work graph, fan-out, concurrency, budgets, cancellation, witness, capabilities, and context/cache/placement/serve requirements.
- `internal/coordination/context_adapter.go:42-220` carries generations, bytes/tokens, warm state, reuse horizon, transfer/rehydration cost, and closed pin/prefetch/transfer/compact/evict actions.
- `internal/coordination/serve_adapter.go:62-178` carries queue/batch and separate prefill/decode capacity, admission delay, cache affinity, cancellation, provenance, and bounded decisions.
- `internal/radixkv/regimefence.go:27-119` fails closed unless model, KV dtype, quant mode, and RoPE regime match.
- `internal/model/kvcache.go:3-27` makes attention KV kernel-owned and keeps pre-RoPE keys for coherent eviction.
- `internal/recall/utility.go:3-20` closes witnessed outcome utility back into clean recall ranking.
- `internal/ctxmmu` owns content admission, quarantine, paging, durability, and memory-write adjudication primitives.
- `internal/trajectory` and existing receipts provide a causal extension seam.

### PARTIAL

- The top-level `coordination.Build` input is still coarse: reusable-prefix bytes, scalar context pressure, compute availability/queue depth, and serve admission/backpressure (`internal/coordination/plan.go:23-96`). The richer adapters do not yet become one semantic-memory-aware live turn plan.
- Context planning can drive model-side KV elision, but structured tool continuations bypass the general planner and the native turn-tax seam records semantic query cost as a structural zero because that tier is unreachable.
- Refcount-1 tool-result fusion is sound and rollback-gated but explicitly default-off (`internal/agent/fusion.go:10-27`).
- Durable memory-write adjudication is explicitly a pure, unwired library (`internal/ctxmmu/memwrite.go:29-36`).
- KV quantization and quantized demotion mechanisms exist, but live decode remains float32 and the quality-gated promotion path is not established.
- Static harness composition and budgets exist, but live receipts do not consistently attach effective harness identity and memory decisions to every model/tool/cache effect.

### ABSENT

- One online snapshot joining harness/task, semantic memory, prompt layout, cache/KV/recurrent state, tool effects, actual quantization, placement, and evidence.
- Automatic semantic recall into native prompt layout, route choice, KV policy, or scheduler decisions.
- A runtime-to-memory loop that credits realized utility and physical cost under the exact execution envelope.
- A quality-constrained shadow and actuation benchmark over the complete union.

The canonical architecture states the same broad non-claim: existing leaves participate, but the complete `snapshot -> constrained plan -> typed action/effect` fold remains program work (`docs/architecture.md:93-130`).

## Self-query witness

Observed 2026-08-29 against installed `fak 0.45.0` (`build 29813d463db5`):

```text
fak capabilities "persistent memory runtime scheduling harness tool calls"
  -> tool-output compression; native receipt validation; capability floor

fak capabilities "trajectory outcomes memory retrieval cache residency"
  -> stable prompt/context reuse; cache/token attribution; native receipt validation

fak capabilities "quantized model routing agentic workload"
  -> model routing; turn avoidance; compression; native serving; quality evaluation
```

Verdict: **PARTIAL**. The cards surface strong adjacent capabilities, but not one live memory-aware coordination loop.

The `fak index docs|leaves|verbs|claims` invocations reported `DEV_COMMAND_MOVED`; the suggested `fak dev index ...` path did not return within the bounded observation window. The initial study query attempt did not return within its bounded window and was not treated as an empty result. After the disposition was complete, `fak study add` persisted `study_a1d0889044a0abefbf535796220098d4aff6b7c17bd5fdbd4e8851d69e67579d`; a fresh correctly ordered `fak study search --limit 5 "Harness-runtime-memory union for highly quantized agentic workloads"` rediscovered that same ID. Raw `rg` searches over `internal`, `cmd`, and `docs` supplied the code cross-check.

## Source ledger

Observed at `2026-08-29T14:49:49-07:00`. External mechanisms are evidence and inspiration; no source code was copied in this pass.

| Source | Source event/state | Fact and effect on this decision | Disposition / refresh |
|---|---|---|---|
| MCP specification `2026-07-28` | Shipped 2026-07-28; immutable revision | Stateless core, self-describing requests, explicit application handles, deterministic cacheable lists. Strengthens explicit execution handles and stable tool catalogs. | ADAPT concept; docs CC-BY-4.0/spec contributions Apache-2.0. Refresh next spec. |
| Agentix, NSDI 2026 | Peer-reviewed May 2026, ISBN 978-1-939133-54-0 | Request-only serving misses program dependencies and cumulative blocking. Makes program/job the scheduling unit. | INSPIRE-ONLY paper mechanism. Refresh artifact/reproduction. |
| Continuum, arXiv:2511.02230v6 | Experimental preprint revised 2026-05-25 | Tool pauses justify bounded KV TTL using reload/recompute and queue cost. Makes tool duration a cache-policy input. | WATCH / INSPIRE-ONLY. Refresh publication and independent run. |
| TensorRT-LLM KV cache docs | Shipped current docs observed 2026-08-29 | Token-range and decode retention priority/duration, offload, events, and hybrid-state distinctions show useful actuator seams. | OPTIONAL-MODULE adapter pattern; Apache-2.0 code only after exact-revision provenance check. |
| Graphiti v3 episode docs | Shipped mutable docs observed 2026-08-29 | Episodes bind provenance/reference time; bulk ingestion omits invalidation. Makes temporal tombstones and correctness modes first-class. | ADAPT schema spirit; Apache-2.0 source only after pinned code check. |
| ACBench, ICML 2025 / PMLR 267 | Peer-reviewed 2025 | Its tested 4-bit arms lost modestly on tool/workflow tasks but materially more on real applications. Forces task- and method-specific quantization envelopes. | DEFAULT evidence rule; no code used. Refresh newer models/methods. |
| FAK agent-serving composition study | Shipped decision record 2026-08-26 at `docs/research/agent-serving-composition-architecture-study-2026-08-26.md` | Already establishes typed composition, heterogeneous state, resource lifecycle, and causal evidence. This note extends it with pairwise feedback, memory, agentic load, and quantization. | EXTEND, not duplicate. Refresh when the live fold ships. |

Unavailable or deliberately bounded source classes: this pass did not perform exact-revision code/license audits for paper artifacts because it copies no implementation; issue/PR/history mining was bounded to the mechanisms that changed the FAK conclusion. Any direct port requires a new pinned license/provenance pass.

## Failure modes and falsifiers

| Failure mode | Required falsifier |
|---|---|
| Semantic sameness is mistaken for reusable KV. | Hash exact serialized/tokenized segments and report the actual first divergence and reused tokens. |
| Retrieval costs more than it saves. | Include index/embedding, transfer, serialization, prefill, retries, and tool refetches in recall on/off A/B. |
| Memory is relevant but stale or poisoning. | Measure terminal quality, contradiction/staleness violations, refetches, and user corrections. |
| Quantization preserves perplexity but breaks tools. | Test exact tool name, argument correlation, JSON grammar, long-context recall, and accepted task outcome. |
| KV or recurrent-state quantization accumulates harmful error. | Long sequential agent replay across precision arms with exact model/config and quality gate. |
| Quantization is not the bottleneck. | Per-phase roofline over weights, KV/recurrent state, dequant, launches, transfer, queue, and tool time. |
| Tool-aware pinning harms other jobs. | Compare job completion distribution, queueing, eviction regret, and fairness against LRU/offload controls. |
| Prefetch guesses waste more than they save. | Record precision/recall, useful prefetched bytes, pollution, and net latency/cost. |
| Cross-layer policies oscillate. | Bind plans to observation generations; count reversals; require hysteresis/dwell to beat local controls. |
| Fusion breaks replay or recovery. | Refcount proof, byte-identical rollback, and crash/resume equivalence. |
| Shared state crosses tenants or capabilities. | Adversarial same-content/different-authority test with no hit-shaped side channel or effect. |
| The controller costs more than the win. | Run the full union versus all-cross-layer-signals-disabled, charging CPU, allocation, serialization, delay, retries, and operator cost. |
| External results do not transfer. | Require fak-native, exact-Qwen3.8, matched hardware/workload/quality receipts before claiming benefit. |

## Smallest working spine

The next implementation should be shadow-only and reuse existing seams:

1. Add a versioned, content-free `TurnSnapshot` projection over existing coordination, memory, cache, native receipt, tool, and trajectory types.
2. Include harness/task identity, authority, tool-schema digest, memory candidates, context plan, cache/KV/recurrent regime, actual quantization, device/serve pressure, route candidates, budget, and witness requirements.
3. Emit a closed `CoordinationPlan`: recall cells, stabilize prefix, prune tools, retain/offload/evict state, prewarm/place model, route/escalate, and admit/quarantine result.
4. Bind every plan to observation generations and make every action advisory in the first rung.
5. Replay one captured Qwen3.8 agent trajectory and compare shadow decisions with actual outcomes and costs.
6. Promote only one policy at a time behind a kill switch after a quality-constrained A/B.

The first likely policy is tool-aware KV retention because it uses information the harness uniquely owns, a physical actuator the runtime owns, and an agent-specific idle-gap cost. The second is cache-aware retrieval budgeting because it exercises the full memory-to-runtime and runtime-to-memory loop without requiring tensor mutation.

## Completion state

This note completes the requested architecture study and creates the durable research tracker. It does **not** complete the live union. The codebase has enough primitives to implement a shadow spine without inventing another scheduler, memory store, or event ledger; the missing work is the bounded cross-layer projection, plan, receipt, and real Qwen3.8 counterfactual.
