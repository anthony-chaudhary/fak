---
title: "llm-d repository study: what FAK should borrow, compose, or leave upstream"
description: "Revision-pinned study of llm-d v0.9/main across architecture, guides, tests, releases, issues, PRs, provenance, and FAK's current serving seams."
date: 2026-08-18
status: studied
source: https://github.com/llm-d/llm-d
source_revision: 48fa8c0a76012038c76f30936fff9727c2e6909d
source_release: v0.9.0
---

# llm-d repository study

## Verdict

> **TL;DR:** Use llm-d as FAK's Kubernetes inference control plane. Borrow its typed routing and executable-guide discipline; keep worker placement, P/D topology, and autoscaling under one owner.

llm-d is a Kubernetes-native *distributed inference assembly*. FAK should interoperate with it, learn from its routing and operations contracts, and avoid duplicating its control plane. Its strongest idea is the decomposition. Each concern has a separate control-plane contract: routing; model-server deployment; prefill/decode placement; KV-state observation; flow control; autoscaling; and guide verification. FAK already has the lower-level kernel seams for
an llm-d backend, cache-aware worker choice, native P/D, and KV-transfer governance. The
useful remaining work is therefore narrow:

1. Update FAK's integration guide for llm-d v0.9 and assign one owner at each routing boundary.
2. Add a live compatibility witness. The existing OpenAI-shape smoke test proves protocol shape only. Cluster interoperability still needs a live run.
3. Watch llm-d's workload-variant autoscaler and exact KV-event routing. Copy neither until FAK has stronger fleet evidence.

This is an **Enabling** study.

- For: operators already running Kubernetes inference.
- Problem: FAK's integration page shows how to point at llm-d, but it does not explain llm-d's ownership boundary or the strength of each witness.
- Today: the systems compose, yet operators can easily misread who routes, queues, or scales a request.
- Better because: this study pins the upstream contracts and classifies each overlap.
- Witness: immutable source links, FAK code and tests, issue history, and the self-query transcript described below.

## Scope and evidence discipline

- Observed at: 2026-08-18.
- Pinned main: [`llm-d/llm-d@48fa8c0`](https://github.com/llm-d/llm-d/tree/48fa8c0a76012038c76f30936fff9727c2e6909d), committed 2026-08-18.
- Latest release observed: [`v0.9.0`](https://github.com/llm-d/llm-d/releases), published 2026-08-17; its peeled commit is `3291bca445be5bd309387fa78cc24f487f07003d`.
- Repository state observed: 4,051 stars and 686 forks. GitHub showed 202 open issues and a last push at `2026-08-18T18:05:57Z`. These numbers show adoption and activity. Quality requires separate evidence.
- License: root Apache-2.0 at
  [`LICENSE@48fa8c0`](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/LICENSE).
  No git submodules or root `NOTICE` were present. The tree does include two explicit patches
  over upstream NVSHMEM and SGLang sources, so code reuse still requires per-file provenance
  inspection. This note ports no source bytes. Every candidate mechanism is **INSPIRE** or INTEROPERATE.
- Evidence read: project and architecture docs, guide manifests, workflows, and nightly matrices. The history pass covered `v0.8.1..v0.9.0` (247 commits and 750 changed files). It also covered issue and PR states plus selected reviews. Release notes, licensing, and patch provenance completed the pass. GitHub Discussions are disabled, so proposals, issues, and PR reviews are the
  rationale record.
- Refresh trigger: a new llm-d minor release or a changed InferencePool/EPP or ModelService API. Also refresh when precise prefix routing becomes the default, WVA crosses its documented maturity boundary, or FAK lands a live-cluster witness.

## What llm-d actually is

The repository is predominantly an integration and operations distribution. Scheduler implementations live in the assembled components. The pinned tree has about 740 guide files and 136 architecture-doc files. It assembles versioned external components through Helm and Kustomize. Gateway API resources, model-server images, and scripts complete the deployment. The architecture page
makes the separation explicit:

- **Model servers** execute inference, usually vLLM, with other servers admitted through the
  same deployment contracts ([architecture](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/README.md)).
- **Inference Gateway / Endpoint Picker Provider (EPP)** observes request and serving state. It filters and scores candidates, then chooses an endpoint
  ([request handling](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/core/router/epp/request-handling.md),
  [scheduling](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/core/router/epp/scheduling.md)).
- **ModelService** is the higher-level deployment API that renders workload topology and
  routing resources; **InferencePool** is the Gateway API inference-extension grouping that
  connects endpoints to EPP
  ([InferencePool](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/core/inferencepool.md)).
- **Advanced modules** cover P/D disaggregation and KV-event indexing. They also cover autoscaling, batch serving, and payload processing. The OpenAI client contract stays stable.

That design serves platform teams running accelerator fleets on Kubernetes. They need portable APIs, replaceable components, observable SLOs, and reproducible workload guides. This deployment boundary differs from FAK's. Each project should stay focused on its own boundary.

## Load-bearing mechanisms

### 1. Routing is a filter/score plugin pipeline

EPP first filters infeasible endpoints, then scores the survivors with independently
configurable plugins. Prefix locality is one signal among load, queue state, accelerator
availability, and deployment topology; the scheduler can normalize and weight signals rather
than hard-code one universal policy. The data layer deliberately separates request-derived
state from endpoint/model-server state
([data layer](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/core/router/epp/datalayer.md)).

The transferable principle is typed signal composition with an observable reason for the winner. llm-d's current weights are context, not a template. FAK's current `CacheAwarePolicy` already implements
power-of-two choice over prefix overlap and load, and its later Dynamo-derived issues added
richer load/tier terms. The llm-d evidence strengthens the need to keep these terms modular
and explainable.

### 2. Approximate and precise prefix routing are distinct products

The ordinary prefix-aware path derives locality from request tokens and cached state. The
precise path enables model-server KV-event publication, aligns block hashing, tokenizes through
a render service, and feeds exact create/evict state to the router
([precise prefix routing](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/advanced/kv-management/prefix-cache-aware-routing.md)).

Merged PR [#2203](https://github.com/llm-d/llm-d/pull/2203) is particularly useful: it adds a
wide-EP/LWS precise-routing variant rather than silently replacing the approximate default.
Its manifests wire a `precise-prefix-cache-producer`, a render service, per-rank KV-event
publication, and load-gated affinity. That is a good bounded-superset pattern. Exactness adds several dependencies: state, tokenization, ports, and new failure modes. llm-d therefore keeps it as an explicit variant.

### 3. Flow control is an admission contract, not just load balancing

llm-d's flow-control design gives the router a request queue and concurrency bounds, supports
priority through request metadata, and exports queue/admission metrics
([flow control](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/core/router/epp/flow-control.md)).
Merged documentation PRs [#2213](https://github.com/llm-d/llm-d/pull/2213) and
[#2249](https://github.com/llm-d/llm-d/pull/2249) tightened verification and observability,
which matters more than merely exposing knobs: overload behavior has to be visible from the
same guide used to deploy it.

FAK has continuous-batching queue gates, priority/preemption leaves, and receiver-granted
KV-transfer credits, but those are several planes. The useful lesson is to describe which
queue is controlled and which metric proves shedding or admission; “flow control exists” is
too conflated to be useful.

### 4. P/D disaggregation is topology plus state transfer

llm-d describes prefill and decode as separately scalable workloads connected by KV transfer,
with routing and deployment resources selecting roles
([P/D architecture](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/advanced/disaggregation/README.md)).
The repository carries runnable P/D guides; role split alone does not count as proof.

FAK's `NativePDService` and transfer-governance seams are native kernel mechanisms. llm-d is a strong external orchestrator option. FAK should retain both behind the EngineDriver and
integration boundaries, not grow a second Kubernetes operator.

### 5. Workload-variant autoscaling is deliberately higher-order

The Workload Variant Autoscaler observes request characteristics and routes/scales among
variant deployments, and avoids forcing one replica count to serve every prompt/output mix
([autoscaling architecture](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/advanced/autoscaling/README.md)).
That idea is relevant to FAK's model/engine routing, but Kubernetes replica control is outside
the kernel's best-default boundary. Treat WVA as an optional platform module and a watch item
until FAK has a measured demand-to-placement contract worth exporting.

### 6. The guides are executable compatibility contracts

The repository's center of gravity is workload guides plus release/nightly matrices. The
v0.9 notes call out promoted guides, platform/accelerator variants, routing and observability
changes, and known limitations; the workflow tree runs guide-specific e2e/nightly lanes.
The frozen `v0.9.0-rc.1` rows in [`release/README.md@48fa8c0`](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/release/README.md) are mixed. Precise-prefix lanes pass on several platforms. P/D includes dry-run and guide-error cells; GKE flow control is `prereqs-error`; CKS WVA is failing. The live badges above those frozen rows can move after this observation,
so neither the guide's presence nor one green platform generalizes to every deployment.
Recent fixes such as [#2296](https://github.com/llm-d/llm-d/pull/2296) (P2P cache-sharing guide
and GLM results) and [#2307](https://github.com/llm-d/llm-d/pull/2307) (FMA launcher RBAC)
show why manifests and live verification are load-bearing code. A client-shape unit test covers only the protocol edge. Cluster interoperability needs a live witness.

## Candidate borrow matrix

The `fak_feature_query` MCP surface was dogfooded with five queries. The queries covered precise prefix routing, admission and priority, and heterogeneous autoscaling. Two more covered P/D transfer and adapter interop. It returned only lexical false-neighbor skill/tool cards for the first four queries
and did not discover the existing engine adapter for the fifth. Per the field-borrow rule,
that lexical score serves only as discovery evidence. The classifications below come from direct
code, tests, docs, and issue-state readback.

| Candidate | FAK state | Coverage decision | Evidence and action |
|---|---|---|---|
| OpenAI-compatible llm-d backend adapter | PRESENT | DEFAULT interop seam | `internal/engine/llmd.go`, its tests, `cmd/fak/llmd_smoke.go`, and `docs/integrations/llm-d.md` resolve llm-d URLs, preserve auth/header policy, and probe OpenAI endpoints. Keep it as the supported composition path. |
| Prefix-overlap plus load-aware worker routing | PRESENT structurally; fleet benchmark evidence is narrower | DEFAULT | `internal/gateway/residency_router.go` ingests add/drop events and scores overlap against load; issues #41, #2238, #5272, #5274, and #5275 document the evolution.<br>Do not clone EPP. Keep modular score terms and require routed-vs-blind witnesses for broad performance claims. |
| Exact KV-event-backed global index | PARTIAL | OPTIONAL-MODULE / WATCH | FAK ingests external KV events and has cache metadata, but llm-d's precise variant couples tokenizer/render and block-hash identity across pods.<br>Preserve the external event seam; do not make that operational bundle a FAK default without a live heterogeneous-fleet witness. Existing #5260/#2238 cover the concept; no duplicate issue. |
| Request admission, priority, and overload observability | **PARTIAL across planes** | DEFAULT kernel policy, **RECIPE** for llm-d | Native scheduler/preemption and gateway policies cover core control; llm-d owns Kubernetes gateway queues when it is the backend.<br>Document one owner per queue and propagate priority intentionally. Do not stack two opaque queues. |
| Native P/D role split and governed KV transfer | PRESENT | DEFAULT native; **RECIPE** external | `internal/modelengine/native_pd.go` and tests cover native roles; transport/credit packages govern transfer. When llm-d is selected, let llm-d own pod topology and use FAK as front gate/agent kernel. |
| Workload-variant replica autoscaling | ABSENT as a FAK-owned closed loop | OPTIONAL-MODULE / WATCH | FAK routes models and engines but does not need to own Kubernetes replica reconciliation. Interoperate with WVA; reconsider only when a FAK demand signal and measured placement objective justify an exported controller contract.<br>|
| ModelService/InferencePool lifecycle controller | **ABSENT by design** | EXCLUDE | This is llm-d/Kubernetes control-plane territory. FAK should consume a stable endpoint or emit deployment recipes. CRDs and controllers stay with llm-d. |
| Guide-as-tested-product release matrix | PARTIAL | DEFAULT for integrations | FAK has smoke/unit docs but no llm-d cluster lane. Borrow the practice: a named compatibility witness should deploy or target a pinned llm-d release, run `/v1/models` plus streaming chat through FAK, capture routing/health evidence, and record component versions.<br>|
| Pluginized request/endpoint signal pipeline | PARTIAL | DEFAULT principle | FAK has typed routing seams but some policies remain bespoke. New routing signals should name source, freshness, normalization, weight, fail-open/closed behavior, and explanation output.<br>This is a design constraint; FAK does not need an EPP clone. |

### Best-default frontier and bounded superset

- Best default for a FAK user with one endpoint: FAK owns policy/context/tool mediation;
  the selected engine, including llm-d, owns model execution.
- Best default for an llm-d platform: llm-d owns Kubernetes endpoint selection,
  ModelService/InferencePool lifecycle, replica topology, and autoscaling; FAK sits in front for
  agent-kernel policy and managed context.
- Optional superset: FAK-native worker routing and P/D remain available for non-Kubernetes
  or embedded serving. Exact cross-pod KV indexing and WVA remain modular external choices.
- Unsupported composition: two independent cache-aware routers both selecting workers for
  the same request without shared state. The outer router sees only an llm-d gateway endpoint;
  it must not claim the inner worker placement or cache hit as its own.

## Surviving work after deduplication

Two gaps survive. Existing issues cover five nearby areas. They include fleet KV routing and P/D. Richer scoring, event streams, and transport control are also covered. This study does not reopen them.

1. Refresh the llm-d integration contract for v0.9 ([#8016](https://github.com/anthony-chaudhary/fak/issues/8016)). The current page is useful but lacks
   a pinned version, precise ownership table, queue/double-routing warning, and clear distinction
   between unit smoke and live cluster proof. This is documentation/productization work.
2. Add a live llm-d compatibility witness ([#8017](https://github.com/anthony-chaudhary/fak/issues/8017)). The minimum witness uses a pinned llm-d deployment or sanctioned cluster. It configures FAK through `--engine llm-d`, calls `/v1/models`, and sends a streaming chat request. The artifact captures component versions plus auth, error, and streaming behavior. Routing performance needs a separate locality-vs-blind experiment. The compatibility witness cannot establish it.

The WVA and exact-prefix variants remain **WATCH**, not silently deferred FAK commitments:
they have explicit refresh triggers above and are intentionally assigned to llm-d unless
measured FAK-specific demand changes the boundary.

## Claims and limits

- This study verifies repository contracts and FAK source state. It did **not** run llm-d on a Kubernetes/GPU cluster. Therefore it makes no new performance or scale claim. Throughput, TTFT, and cache-hit effects remain unmeasured here.
- llm-d release-note benchmark numbers are upstream-observed results, not FAK-authored facts;
  none are repeated here as a FAK comparison.
- The precise-routing and WVA surfaces are moving quickly. Open issues and PRs are direction,
  not shipped behavior; only merged files at the pinned revision support the descriptions above.
- FAK's self-query miss shows a discoverability gap in the lexical capability catalog, not
  absence of all five mechanisms. Direct source readback overrides that weak negative.

## Source trail

- [llm-d pinned tree](https://github.com/llm-d/llm-d/tree/48fa8c0a76012038c76f30936fff9727c2e6909d)
- [v0.9.0 release](https://github.com/llm-d/llm-d/releases)
- [Architecture](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/README.md)
- [EPP scheduling](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/core/router/epp/scheduling.md)
- [EPP flow control](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/core/router/epp/flow-control.md)
- [Precise prefix routing](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/advanced/kv-management/prefix-cache-aware-routing.md)
- [P/D disaggregation](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/advanced/disaggregation/README.md)
- [Autoscaling](https://github.com/llm-d/llm-d/blob/48fa8c0a76012038c76f30936fff9727c2e6909d/docs/architecture/advanced/autoscaling/README.md)
- [Wide-EP precise routing PR #2203](https://github.com/llm-d/llm-d/pull/2203)
- [Flow-control verification PR #2213](https://github.com/llm-d/llm-d/pull/2213)
- [Flow-control configuration PR #2249](https://github.com/llm-d/llm-d/pull/2249)
- [P2P cache-sharing hardening PR #2296](https://github.com/llm-d/llm-d/pull/2296)
- [FMA RBAC fix PR #2307](https://github.com/llm-d/llm-d/pull/2307)
- [FAK llm-d integration](../integrations/llm-d.md)
- [FAK serving SOTA note](../serving/pd-disaggregation-kv-routing-sota.md)



## Exhaustive inventory refresh (2026-08-25)

Issue [#8988](https://github.com/anthony-chaudhary/fak/issues/8988) refreshes the
study denominator without replacing the mechanism analysis above. The exhaustive map
is [`inventory/llm-d-llm-d.json`](../research/inventory/llm-d-llm-d.json), generated from
`llm-d/llm-d@3243fcf1191348b55c7811267a98117f8b7a6910`. It indexes 1,129 files,
593 directories, 30,601,578 bytes, and 11 top-level subsystems. The revision is 17
commits after the prior checked revision; the intervening changes are predominantly
v0.9 guide, artifact, CI, template, and platform corrections, not a reversal of the
ownership decisions above.

### Complete source-class read-back

- **Tree:** README/docs, architecture/design, runtime/configuration, tests/fixtures,
  history/release machinery, roadmap markers, and `LICENSE` were walked at the pinned
  revision. `.git` is the only skipped control directory.
- **Forge:** GitHub GraphQL exact aggregates and paginated REST/CLI read-back covered
  all 533 issues (124 open), 1,841 pull requests (100 open), zero discussions, 11
  releases, 66 tags, 11 branches, 11 milestones, and one project as observed on
  2026-08-25. `v0.9.0` is the latest release; the `v0.10.0` and `v1.0.0` milestones
  are directional signals, not shipped contracts.
- **History and provenance:** the local clone supplied refs and the full revision
  delta. Both the pinned `LICENSE` and repository API identify Apache-2.0. FAK may
  adapt ideas with attribution, but copied implementation must retain applicable
  license and notice provenance.
- **FAK self-query:** three `fak capabilities` queries covered Kubernetes routing,
  KV-event locality, flow control, P/D disaggregation, autoscaling, Gateway API,
  and vLLM integration terms. They found FAK-native model choice, context reuse,
  live control, and evidence surfaces, but no Kubernetes inference control-plane
  equivalent. That confirms interoperation rather than duplication.

The pinned map's `non_tree_study` block records the commands, aggregate counts,
roadmap treatment, license decision, self-query results, and candidate matrix. This
keeps mutable forge observations separate from claims about the immutable tree.

### Refreshed FAK decisions and follow-ons

| Candidate | Decision | Durable owner |
|---|---|---|
| v0.9 ownership and double-routing contract | **Already shipped.** FAK owns policy/model/cache decisions; llm-d owns cluster endpoint selection and Kubernetes assembly. | [#8016](https://github.com/anthony-chaudhary/fak/issues/8016), `docs/integrations/llm-d.md` |
| Pinned live-cluster compatibility receipt | **Keep open.** Static inventory evidence cannot replace a live cluster witness. | [#8017](https://github.com/anthony-chaudhary/fak/issues/8017) |
| KV-event locality, speculative cache-warm metadata, and matched prefix-reuse measurement | **Borrow as bounded inspiration, preserving fak-native cache ownership.** | [#3888](https://github.com/anthony-chaudhary/fak/issues/3888), [#6082](https://github.com/anthony-chaudhary/fak/issues/6082) |
| Endpoint-scoring plugins, flow control, P/D topology, workload-variant autoscaling, Gateway API, and deployment guides | **Do not port.** These are llm-d control-plane responsibilities; FAK should expose signals and compatibility seams instead of building a second Kubernetes inference platform. | Interoperation boundary above |

Every surviving action is shipped or already tracked, so this refresh files no duplicate
follow-on. The row-specific completion witness is:

```text
fak study-monitor --registry docs/research/monitored-repositories.json --inventory-check --json
repository=llm-d/llm-d mode=exhaustive ready=true indexed_revision=3243fcf1191348b55c7811267a98117f8b7a6910
```
