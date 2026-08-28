---
title: "Agent-serving composition architecture field study"
description: "Status: decision record, not an implementation claim."
---

# Agent-serving composition architecture field study

Date: 2026-08-26  
Issue: [#9286](https://github.com/anthony-chaudhary/fak/issues/9286)  
Parent architecture: [#9279](https://github.com/anthony-chaudhary/fak/issues/9279)  
Status: decision record, not an implementation claim

> **TL;DR:** FAK composition is the frozen-ABI graph that joins agent work, model serving, and memory.

**Next action:** implement the smallest validated graph in #9280, using the ticket constraints below.

Quickly verify this decision record after editing it:

```bash
fak validate --mine docs/research/agent-serving-composition-architecture-study-2026-08-26.md
```

## What to borrow

Five field patterns fit FAK:

1. Model work has explicit phases and state transfers.
2. Each memory or state kind keeps its real geometry.
3. A resource request is separate from its allocation and observed outcome.
4. The kernel composes focused capabilities at its existing trust boundary.
5. Receipts share a vocabulary, while payload capture remains opt-in.

FAK should borrow these contract shapes without importing another project's control plane.
Execution and policy remain FAK-owned. So do placement, cache, and evidence.

One finding changes the original proposal materially: validation must cover feature interactions. A model can support hybrid state, and a backend can support transfer, while the pair remains invalid.

## Pinned sources

The study used exact source states rather than floating documentation.

| Source | Pinned state | Inspected evidence |
|---|---|---|
| vLLM | `v0.28.0` / `2cf0a6915ce544dc493a0990f2ea38d81601128a`, released 2026-08-26 | Disaggregated-prefill design, hybrid-cache design, connector API, cache interfaces, current allocator, and tests |
| Kubernetes | `v1.37.0` / `f54c212e3a2f75d674b717a9b29052b20b60aefc`, released 2026-08-26 | `staging/src/k8s.io/api/resource/v1/types.go` |
| Model Context Protocol | spec tag `2026-07-28` / `5f5440bb26a62e2cf3440b92da5a667efa03b267` | Architecture, changelog, roots, sampling, elicitation, resources, prompts, and tools |
| OpenTelemetry semantic conventions | `v1.44.0` / `e10a930844c6951757a43b849d364f7d056ac32b`, released 2026-08-04 | GenAI documents redirect to the dedicated repository |
| OpenTelemetry GenAI conventions | `56d6b11a02129319bf371083fa134b7ce989c976`, dated 2026-08-22 | Agent, model, event, and metric conventions |

The vLLM hybrid-cache design describes the older commit `458e74` and calls itself early-stage. The pinned v0.28.0 code now includes packed layouts and mixed-page-size tests. Present-tense conclusions below use both the historical design and current code.

## FAK baseline

FAK already has the right load-bearing boundary:

- `ARCHITECTURE.md` defines a layered dependency DAG.
- `internal/architest` enforces import direction and graph invariants.
- The frozen ABI includes the additive `abi.Verdict` union.
- Addressable payloads use `abi.Ref`.
- Synchronous calls retain typed `Submit` and `Reap` semantics.
- Provisional effects retain transaction, promote, and rollback semantics.
- Existing registry and engine-driver surfaces already provide composition machinery.
- Model and compute packages own execution.
- Context and cache packages own reuse and placement state.
- Gateway and receipt packages own serving evidence.

The gap is composition debt between those modules. The answer is not a parallel architecture API. The new graph must ride the frozen ABI and existing registries.

### Reproducible variation signal

At FAK `24b256ae5def3c7a846063578105eff2cd14bca3`, this source search found family or state terms across many model and compute files:

| Search | Files |
|---|---:|
| `qwen3|qwen4` | 150 |
| `glm` | 142 |
| `gdn|mamba|ssm` | 88 |
| `minimax` | 28 |
| `llama3` | 11 |

Reproduction command:

```bash
for term in 'qwen3|qwen4' 'glm' 'gdn|mamba|ssm' 'minimax' 'llama3'; do
  git grep -l -i -E "$term" -- \
    'internal/model/*.go' \
    'internal/modelengine/*.go' \
    'internal/compute/*.go' | wc -l
done
```

These coarse counts are not a quality score. They show why locally clean additions can still increase cross-plane branching.

The current code-quality `architecture` KPI finds large files and functions. Keep that useful measurement intact. The new score in #9285 should measure system composition instead.

## Decision matrix

### vLLM serving and state patterns

| Pattern | Decision | FAK translation |
|---|---|---|
| Prefill and decode run as separate phases, with explicit KV and result transfer | Adapt | Describe phase inputs, outputs, barriers, transfer, and recomputation. Start with a local single-process consumer. |
| Disaggregated prefill is experimental and does not claim higher throughput | Adopt as an evidence rule | Name the target metric. Charge transfer, queueing, recovery, and verification before reporting a gain. |
| Connector APIs separate scheduler decisions from worker execution | Adapt | Let composition select and validate movement. A typed runtime leaf performs it. |
| Full, sliding, local, and recurrent state use distinct cache specifications | Adopt | Give each state kind its own geometry and behavior. Do not flatten recurrent state into attention KV. |
| The older generic path used a common page size, while current code adds packed and mixed-size paths | Adapt the lesson | Keep allocation unit and alignment behind typed placement policy. Coalesce only when evidence supports it. |
| Hybrid combinations have explicit constraints and edge cases | Adopt | Validate the whole model, backend, layout, state, and feature combination. Independent booleans are insufficient. |

The phase contract is useful even when every phase runs in one process. Distributed scheduling is not part of the first spine.

### Kubernetes resource-claim patterns

| Pattern | Decision | FAK translation |
|---|---|---|
| `DeviceClass`, claim, request, constraint, configuration, and allocation result are distinct | Adapt | Keep desired resources separate from resolved placement and actual outcome. |
| Selectors and prioritized alternatives guide allocation | Adapt narrowly | Start with typed constraints and ordered alternatives. Do not add a general expression language. |
| Devices publish attributes and capacities | Adopt | Let planners consume typed snapshots instead of backend internals. |
| API objects, controllers, schedulers, and drivers reconcile cluster state | Reject for the spine | Use immutable in-process snapshots and typed allocators. Distribution can arrive behind the same contract later. |

This supports GPUs, memory tiers, object stores, and computer endpoints without turning FAK into Kubernetes.

### MCP host and primitive patterns

| Pattern | Decision | FAK translation |
|---|---|---|
| Resources, prompts, tools, sampling, and elicitation remain distinct | Adapt | Preserve active primitive identity. Do not turn every agent need into a generic tool call or memory blob. |
| The host owns context, permissions, authorization, and client lifecycle | Adapt | Keep the FAK kernel as the trust and composition boundary. Leaves advertise bounded capabilities. |
| The 2026-07-28 protocol is stateless and carries version and capabilities on each request | Adapt, do not copy | Resolve current declarations into a stable FAK snapshot before execution. Do not recreate the removed initialization handshake. |
| `server/discover` offers optional up-front discovery | Adapt | Discovery can populate the snapshot, but cannot mutate a running graph silently. |
| Roots is deprecated and was advisory rather than access control | Adopt only the negative rule | Scope hints may guide composition. They never authorize access, and Roots should not become a new FAK primitive. |
| Tool annotations from untrusted servers are untrusted | Adopt | Record provenance. Policy must not trust a remote safety label without an independent witness. |
| Clients isolate server security boundaries | Adapt | Give each external endpoint an explicit owner, session, and isolation domain. Shared reuse needs a compatibility key. |

MCP's primitive separation is useful. Its trust rules do not replace FAK policy or adjudication.

### OpenTelemetry evidence patterns

| Pattern | Decision | FAK translation |
|---|---|---|
| Agent, operation, conversation, model, tool, token, and error attributes share a vocabulary | Adapt | Define FAK-owned causal IDs and map them to receipts, metrics, and optional exporters. |
| Prompts, outputs, and tool payloads are opt-in because they may be sensitive | Adopt | Default evidence contains IDs, digests, sizes, decisions, timings, and outcomes. |
| Conventions focus mainly on agent and model operations | Adapt | Begin with work, graph, phase, resource, policy, and operation identity. External fields are projections. |
| A full telemetry SDK runs in the hot kernel | Reject | Keep dependency-free internal events and bounded adapters at existing output seams. |

The vocabulary can interoperate with OpenTelemetry without making it the source of truth.

## Required FAK contracts

### Capability snapshot

A snapshot is versioned, hashable data resolved from current declarations. It describes what may participate in one composition.

It needs these sections:

- Identity: model variation, engine, endpoint, and external primitive.
- Features: tool calling, multimodality, and quantization. State features distinguish attention from recurrent state.
- Requirements: kernels, layouts, and memory. They also cover locality, policy class, and protocol support.
- Envelope: context, batch, quality tier, and device constraints.
- Provenance: internal descriptor, witnessed external capability, or untrusted annotation.
- Compatibility: named rules over combinations rather than independent flags.

Executable behavior stays behind typed Go interfaces. Descriptors cannot become a stringly typed plugin language.

### Validated composition graph

The first graph should remain small. Its request side includes:

- the `Submit`-time work intent;
- policy identity; and
- provisional transaction context.

Its execution side includes:

- the model variation;
- the native engine selected through existing registry seams;
- execution phases;
- resource claims;
- addressable `Ref` identities where needed;
- active external primitives; and
- evidence IDs and schema versions.

Validation returns deterministic typed reasons before allocation. A resolved hot path uses direct handles and remains O(1) with respect to installed capabilities.

The graph has a stable digest. Availability and allocation results live in separate snapshots. New outcomes use registered `Verdict` kinds with fail-closed fallback.

The validator answers two questions:

```text
Does every node provide its required capability?
Is this exact combination supported and witnessed?
```

### Resource lifecycle

Use common metadata plus kind-specific geometry.

Common metadata has four groups.

Identity and ownership:

- identity;
- compatibility key;
- owner and isolation domain; and
- session or turn scope.

Lifecycle and sharing:

- requested lifetime;
- actual lifetime; and
- mutability and shareability.

Placement and cost:

- requested locality;
- actual locality;
- capacity and bytes;
- alignment and quality effect;
- transfer and recompute cost; and
- retention and eviction cost.

Safety and lineage:

- sensitivity and persistence policy; and
- dependencies and invalidation lineage.

Kind-specific payloads preserve real storage differences. Weights differ from attention KV. Recurrent state and adapters also retain their own geometry. The same rule applies to managed context, artifacts, computer sessions, and scratch.

Keep three records:

1. Claim: what the graph needs.
2. Allocation: what was resolved and reserved.
3. Observation: what happened during use and release.

### Causal evidence

One turn-level receipt should join four evidence groups.

Identity:

- work ID;
- composition graph ID;
- capability snapshot; and
- module versions.

Decisions:

- policy decision;
- validation decision;
- planned engine and backend;
- planned model variation;
- actual engine and backend; and
- actual model variation.

Runtime cost:

- phase time and queue time;
- resource claim and allocation;
- movement, reuse, eviction, and release; and
- token and cache accounting.

Outcome:

- tool or computer operation status;
- quality and operating-envelope evidence; and
- typed failure or degradation reasons.

Global metrics keep bounded dimensions. High-cardinality graph and resource IDs belong in receipts or traces. Useful default evidence does not require content capture.

## Shift-left architecture fitness

#9285 should measure properties that imports, builds, tests, and file size cannot cover.

The first baseline should report:

1. Family branches outside approved leaves.
2. Churn or bypasses around frozen ABI seams.
3. Onboarding change amplification for representative features.
4. Descriptor and interaction-fixture coverage.
5. Resource kinds with unclear ownership or invalidation.
6. Graph nodes missing causal evidence.
7. Schema changes missing version or migration fixtures.
8. Dynamic lookup left on the resolved hot path.
9. Default content capture or unbounded metric dimensions.
10. Exceptions without an issue, owner, expiry, and witnessed reason.

Start in report-only mode. Emit stable JSON with exact file and symbol evidence. Then ratchet no-new-hard-debt.

A headline score is optional. Every deduction must remain independently actionable.

## Questions a captured run must answer

- Why was this model and backend selected or rejected?
- Which interaction made the graph invalid?
- Which state loaded or reused?
- Which state moved, recomputed, or evicted?
- Who owned that state, and when was it released?
- Did the engine remain fak-native?
- Where did TTFT and inter-token latency go?
- Where did total latency, bytes, and energy go?
- Did a placement change improve its declared metric after overhead?
- Which policy decision allowed or denied the external operation?
- Can an operator answer these questions without payload content?

## Ticket refinements

### #9280 composition spine

Require the spine to resolve declarations into an immutable FAK snapshot.

It must use the existing ABI:

- ride `Submit` and `Reap`;
- use `Ref` and registered `Verdict` semantics; and
- preserve the provisional lifecycle.

It must also:

- validate feature interactions;
- separate claims, allocation, and observations;
- model phases and movement explicitly;
- keep advisory scope separate from authority; and
- record fak-native engine identity on an O(1) resolved path.

### #9282 model descriptors

Factor model variation across topology, state kind, and layout. Keep tokenizer and agent capabilities explicit. Do the same for backend needs and quality evidence. Test supported and forbidden combinations.

### #9283 resource lifecycle

Keep kind-specific geometry. Include claim, allocation, and observation records. Track compatibility and isolation. Also record lineage, actual locality, and net-true movement cost.

### #9284 causal receipt

Use a FAK-owned identity chain:

```text
work → graph → phase → resource → policy or operation → outcome
```

Use OpenTelemetry GenAI only as an export vocabulary. Check default content privacy and metric cardinality.

### #9285 architecture score

Score interaction fixtures and change amplification. Score lifecycle ownership, evidence projection, and schema migration separately. Include a false-green fixture where each capability exists but their combination is invalid.

## Adopt, adapt, reject, defer

Adopt:

- explicit phase and state contracts;
- heterogeneous state kinds;
- host-owned composition;
- advisory scope separate from policy;
- provenance-aware metadata; and
- opt-in payload telemetry.

Adapt:

- scheduler and executor separation;
- resource claims and allocation results;
- resource attributes and capacities;
- active MCP primitive separation;
- per-request declarations into a stable FAK snapshot; and
- causal GenAI vocabulary.

Reject:

- topology-only performance claims;
- one universal storage geometry;
- cluster reconciliation in the first spine;
- advisory metadata as authority;
- untrusted safety annotations; and
- a full telemetry SDK in the hot kernel.

Defer:

- distributed schedulers;
- remote phase placement;
- general constraint languages;
- mutation of a running graph; and
- automatic OTLP export.

The spine should leave typed seams for deferred work without implementing it.

## Source locator appendix

These anchors apply to the pinned trees above.

| Evidence | Pinned source locator |
|---|---|
| Disaggregated prefill status, goals, and throughput caveat | vLLM `docs/features/disagg_prefill.md:8-31@2cf0a691` |
| Scheduler and worker connector responsibilities | vLLM `vllm/distributed/kv_transfer/kv_connector/v1/base.py:54-260@2cf0a691` |
| Historical hybrid-cache assumptions | vLLM `docs/design/hybrid_kv_cache_manager.md:3-80@2cf0a691` |
| Packed layout and mixed-page-size evolution | vLLM `vllm/v1/core/kv_cache_utils.py:1270-1335@2cf0a691`; `tests/v1/core/test_kv_cache_utils.py:2145-2195@2cf0a691` |
| Claim, request, constraint, and allocation separation | Kubernetes `staging/src/k8s.io/api/resource/v1/types.go@f54c212e` |
| Stateless per-request capabilities and optional discovery | MCP `architecture/index.mdx:7-9,116-174@5f5440bb`; `changelog.mdx:10-17@5f5440bb` |
| Host, client, server, and primitive responsibilities | MCP `architecture/index.mdx:15-113@5f5440bb` |
| Roots deprecation and advisory scope | MCP `client/roots.mdx:1-80@5f5440bb` |
| Untrusted tool annotations | MCP `server/tools.mdx@5f5440bb`, security section |
| GenAI identity, operation, token, and error vocabulary | OTel GenAI agent, model, event, and metric documents at `56d6b11a` |
| Sensitive payload fields are opt-in | OTel GenAI agent and model span documents at `56d6b11a` |
| Frozen FAK ABI | `ARCHITECTURE.md:120-145@24b256ae`; `internal/abi/types.go@24b256ae`; `internal/abi/registry.go@24b256ae` |

## Completion witness for #9286

The research issue is complete when:

- this decision record is committed;
- the five implementation issues contain these constraints;
- repository documentation checks pass; and
- later contradictory evidence updates the relevant decision with a newer pinned source.
