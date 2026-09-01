---
title: "There Is No Address study: consumer-owned locality and the FAK ready-endpoint seam"
description: "Pinned source and NIXL audit, factual corrections, and a FAK mitigation that joins KV identity, consumer-owned routes, end-to-end pricing, and hop receipts."
---

# There Is No Address: consumer-owned locality and the FAK ready-endpoint seam

## Verdict

The essay is directionally right about the hard boundary: a network descriptor
can name a stable public rung such as DRAM or HBM, but it cannot make
consumer-private SMEM, TMEM, or PE SRAM remotely addressable, nor can it replace
the consumer compiler's local layout and schedule. Cross-engine disaggregation
therefore needs a handoff contract that ends at **ready to consume**, not merely
**bytes arrived**.

FAK can mitigate this without pretending to solve cross-vendor compiler
composition. Reuse the shipped `fabricmap` graph as the control-plane contract:
the consumer publishes a versioned, expiring route from its public landing rung
to an opaque private ready endpoint; a NIXL/RDMA descriptor is one directed edge;
the consumer's TMA/kernel/compiler stage is another; and `KVTransferOK` is
admitted only after every edge returns a matching receipt. Then price the entire
byte-sized ready-to-attend route against recomputation.

Two implementation leaves survive: [#8259](https://github.com/anthony-chaudhary/fak/issues/8259)
binds KV admission to the terminal consumer-ready receipt, and
[#8261](https://github.com/anthony-chaudhary/fak/issues/8261) makes
`fabricmap.Request.Bytes` drive an explicit estimated-ready-time objective.
Topology transport, device adapters, and live hardware proof remain in existing
[#3310](https://github.com/anthony-chaudhary/fak/issues/3310) and
[#6409](https://github.com/anthony-chaudhary/fak/issues/6409).

## Observation identity

| Field | Witnessed value |
|---|---|
| Observed at | `2026-08-20T12:41:45-07:00` |
| Essay | [Hiraditya, *There Is No Address*](https://hiraditya.github.io/posts/there-is-no-address/), published 2026-08-19 04:30 PDT |
| Essay source | [`hiraditya/hiraditya.github.io@ba4c8701`](https://github.com/hiraditya/hiraditya.github.io/blob/ba4c8701d6f9d48943e3a911c017204f3b0f27c2/_posts/2026-08-19-there-is-no-address.md), final post-changing commit, 2026-08-19 10:07 PDT |
| Site HEAD at check | [`2547c55f35391c210d274fa7ea36fcbc383b5d8a`](https://github.com/hiraditya/hiraditya.github.io/commit/2547c55f35391c210d274fa7ea36fcbc383b5d8a) |
| NIXL reconstruction at essay read date | [`ai-dynamo/nixl@341f87bf`](https://github.com/ai-dynamo/nixl/commit/341f87bf0f0427f2a4ca3a670a14d136aff2136d), last main commit on 2026-08-18 |
| NIXL deep-read snapshot | [`ai-dynamo/nixl@538dba36`](https://github.com/ai-dynamo/nixl/commit/538dba3682ba821acacf4630e4eed5aa42ba62d1), observed 2026-08-20 12:19 PDT |
| NIXL release state | [`v1.4.0`](https://github.com/ai-dynamo/nixl/releases/tag/v1.4.0), published 2026-08-14; release tag is not an ancestor of the deep-read main snapshot |
| FAK comparison state | `internal/cachemeta@r50+g2d7508ce8a`, `internal/cacheprice@r9+gbf3a9c2d06`, `internal/fabricmap@r8+gf3c4d0816d`, `internal/model@r447+g21aa6f1e4f` |

Refresh this note when the teased fifth essay appears, NIXL changes descriptor or
backend-selection semantics, FAK ships #8259/#8261, or a real heterogeneous
consumer adapter supplies ready-to-attend measurements.

## Problem centrality and value frame

**Enabling.** The work does not replace FAK's kernel checkpoint or move the data
itself. It makes the existing disaggregated-KV admission claim honest across a
split optimization domain.

- **For:** an operator moving KV or activations between independently owned
  inference engines.
- **Problem:** transport completion at a public memory rung does not prove the
  consumer can use the data; locality, conversion, and private staging remain.
- **Today:** FAK has strong KV identity, a generic directed fabric graph, and
  per-hop receipts, but they are unwired and route price ignores request size.
- **Better because:** the consumer owns the private realization plan, FAK owns
  its admission/expiry/receipts, and the full handoff competes with recomputation.
- **Witness:** a deterministic two-hop route refuses a one-hop success and emits
  `KVTransferOK` only after the consumer-ready receipt; small and large payloads
  can select different estimated-ready-time routes.

The P1-P4 checks all apply:

- **P1 managed context:** carry an opaque, content-addressed plan reference,
  generation, and expiry; never inline compiler schedules into a universal
  descriptor or agent context.
- **P2 net-true efficiency:** count network, registration, conversion, local
  redistribution, and queue time through the final usable rung; compare that
  total with recomputation.
- **P3 bounded adaptation:** accept only live consumer-authored capability
  snapshots and authorized directed links; missing, expired, or partial paths
  fail closed to recompute/replan.
- **P4 integrated operations:** join KV/materialization identity, route/plan
  identity, every hop receipt, terminal ready endpoint, and typed outcome.

## What the essay actually establishes

### One owner normally closes the hierarchy

The essay compares CPU cache control plus compiler locality, programmer/TMA-owned
GPU staging, and compiler-owned Cerebras placement/routing. The important
generalization is not that every machine has the same hierarchy; it is that the
normal optimization loop sees the whole hierarchy. That is an architectural
synthesis, not an experimentally proven law, but it accurately names the seam a
cross-vendor handoff opens.

### A remote transfer terminates at a public rung

At the NIXL snapshot corresponding to the essay's stated read date,
`nixl_mem_t` contains DRAM, VRAM, block, object, and file segments
([`nixl_types.h:37-41@341f87b`](https://github.com/ai-dynamo/nixl/blob/341f87bf0f0427f2a4ca3a670a14d136aff2136d/src/api/cpp/nixl_types.h#L37-L41)).
`nixlBasicDesc` is a contiguous `addr`, `len`, `devId` triple
([`nixl_descriptors.h:27-39@341f87b`](https://github.com/ai-dynamo/nixl/blob/341f87bf0f0427f2a4ca3a670a14d136aff2136d/src/api/cpp/nixl_descriptors.h#L27-L39)).
The backend guide makes the type dependence explicit: an address is a pointer
for DRAM/VRAM but an offset for file/object storage, while `devId` changes meaning
with the space
([`BackendGuide.md:112-125@538dba`](https://github.com/ai-dynamo/nixl/blob/538dba3682ba821acacf4630e4eed5aa42ba62d1/docs/BackendGuide.md#L112-L125)).

That is a compact heterogeneous transfer vocabulary, not an address-free or
capability-safe ontology. It describes the rung a backend can register and reach.
The consumer still owns any subsequent private-memory stage.

### Layout, locality, and scheduling costs cross together

The GPU example is the mild form: data can arrive in HBM while the consuming
kernel still stages/tiles it into shared or tensor memory. A distributed private
SRAM machine makes that second topology impossible to hide behind a global
pointer. The essay's LPX example then shows why small repeated activation
crossings are latency-sensitive while a large one-time P/D KV handoff is more
bandwidth-sensitive. These are useful distinctions, but the activation timing is
explicitly derived rather than measured, and no cited system was run.

### The composition choices are real, but not exhaustive

The essay offers bilateral agreement, a neutral format with conversion and
redistribution, or producer-authoritative compiler merging. Those are useful
failure modes, not the only possible contract. FAK's fourth shape is
**consumer-authoritative realization by reference**: the consumer publishes what
it can realize now, the producer chooses among offers, and neither side exports
its private compiler schedule. This avoids pairwise compiler merging while
retaining the neutral/public transport boundary.

## Corrections and evidentiary limits

The central thesis survives, but four caveats should travel with it:

1. **“The descriptor is the entire shared vocabulary” is too strong.** NIXL's
   `nixlBlobDesc` adds opaque registration metadata and `nixlRemoteDesc` adds
   remote-agent identity
   ([`nixl_descriptors.h:140-207@341f87b`](https://github.com/ai-dynamo/nixl/blob/341f87bf0f0427f2a4ca3a670a14d136aff2136d/src/api/cpp/nixl_descriptors.h#L140-L207)).
   That metadata can carry a FAK plan reference; it still cannot create a remote
   address for consumer-private memory or validate the local schedule.
2. **The 1.13 GB / 90 Gbps example is DistServe, not Splitwise.** The primary
   DistServe paper gives 1.13 GB for one 512-token OPT-66B KV cache and 90 Gbps at
   10 requests/s
   ([arXiv:2401.09670](https://arxiv.org/html/2401.09670)). The essay's reference
   is unlinked and misattributes the figure.
3. **The Blackwell capacities lack device/die scope.** NVIDIA's B200/GB200 tuning
   guide gives 256 KB combined L1/texture/shared, 228 KB shared capacity per B200
   SM, and 126 MB L2 for GB200; compute capability 12.0 variants have different
   limits
   ([NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)).
   The essay's 128 KB and roughly 64 MB values are not a safe generic Blackwell
   description.
4. **Several statements are hypotheses.** Exact WSE ingress/staging, model-level
   KV placement sizes, LPX deployment status/latency, and the single-owner
   generalization are not established by the cited experiments. The post itself
   discloses Claude Opus 5 assistance, derived activation arithmetic, and that no
   system was run.

## NIXL worldview and negative knowledge

NIXL assumes an external conductor owns allocation, orchestration, metadata
exchange, and semantic validity. It supplies registration, backend plugins,
remote metadata, and asynchronous transfer handles
([`docs/nixl.md:31-66@538dba`](https://github.com/ai-dynamo/nixl/blob/538dba3682ba821acacf4630e4eed5aa42ba62d1/docs/nixl.md#L31-L66)).
That separation is exactly why FAK should sit above it as governor rather than
try to absorb it as a universal memory model.

Two current details strengthen that choice:

- Transfer creation intersects registered backend sets and selects the first
  compatible match; preference/exhaustive search is left for later, while
  `estimateXferCost` operates after a backend/handle exists
  ([`nixl_agent.cpp:890-1077@538dba`](https://github.com/ai-dynamo/nixl/blob/538dba3682ba821acacf4630e4eed5aa42ba62d1/src/core/nixl_agent.cpp#L890-L1077)).
  FAK must not assume this path has optimized end-to-end readiness.
- Selected physical-path visibility remains an open need in
  [NIXL issue #1975](https://github.com/ai-dynamo/nixl/issues/1975). Registration
  lifetime, overlap, cancellation, and metadata replacement also have documented
  open edge cases; a FAK route must therefore bind generation/expiry and accept
  only witnessed hop completion, not a naked success assertion.

Stride-compressed prepared descriptors are worth watching as an implementation
technique: NIXL commit
[`8ddd260`](https://github.com/ai-dynamo/nixl/commit/8ddd260e547584b8dbd027d3bd9e759040aab30a)
reports about 44 MB becoming 7 KB for roughly 1.39 million descriptors. It is a
useful control-footprint optimization, not a solution to the split ownership
problem.

## FAK inward map

FAK self-query found no direct capability for consumer-relative addressability,
layout/staging plans, or conversion-authority negotiation. Adjacent capabilities
make the state **PARTIAL**, not absent wholesale:

| FAK seam | What is present | What is still missing |
|---|---|---|
| `internal/cachemeta@r50+g2d7508ce8a` | `KVManifest` and `MaterializationKey` bind source/model/tokenizer/serializer/position/policy identity; `KVTransfer` records backend, tiers, outcome, and bytes | Route/plan identity, terminal consumer-ready endpoint, and per-hop receipts |
| `internal/cachemeta/hardware.go:122-137` | Tier latency/bandwidth/capacity and `AttendableInPlace` | Addressability relative to a consumer execution unit; private local rungs |
| `internal/fabricmap@r8+gf3c4d0816d` | Arbitrary endpoints, directed multi-hop links, versioned/expiring provider snapshots, authorization, adapters, and strict hop receipts | KV binding; `Request.Bytes` is not used by route selection |
| `internal/cacheprice@r9+gbf3a9c2d06` | Local/remote/recompute choice using one recompute-equivalent transfer toll | Network + conversion + redistribution + queue breakdown from the whole route |
| `internal/model@r447+g21aa6f1e4f` | SHM/TCP implementations and fail-closed UCX-RDMA/NVMe-oF/object registry rows | Multi-hop consumer-ready selection instead of one coarse backend |

The load-bearing `fabricmap` details are already shipped:

- endpoints and directed links are open-ended, with no built-in hierarchy
  (`internal/fabricmap/fabricmap.go:11-64`);
- providers publish independent, generation-checked, expiring snapshots and
  conflicts fail closed (`internal/fabricmap/provider.go:24-119`);
- execution is complete only when every hop supplies matching identity, bytes,
  timing, and integrity evidence (`internal/fabricmap/execution.go:62-125`).

The narrow generic bug is also concrete: `Request.Bytes` exists at
`fabricmap.go:46-55`, but `Graph.Plan` at `:106-166` ranks static cost, latency,
hops, and IDs without reading it. `cacheprice.CheapestRoute` then consumes one
opaque toll, so it cannot repair a badly ranked multi-hop route.

## Mitigation contract

The smallest coherent FAK flow is:

1. **Identity:** `KVManifest` / `MaterializationKey` proves what the artifact is
   and whether the consumer may reuse it.
2. **Consumer offer:** the consumer's `fabricmap.Provider` publishes a
   generation-bound, expiring public landing endpoint, opaque realization ID,
   private ready endpoint, and directed local link(s). The reference may ride in
   NIXL blob metadata or a separate control message.
3. **Route:** FAK composes producer-public -> consumer-public transport with
   consumer-public -> consumer-ready local realization. It never invents the
   reverse edge or interprets the private schedule.
4. **Economics:** choose the explicit byte-sized ready-time objective and compare
   total time/cost with recomputation. Unknown link data fails closed or uses a
   declared compatibility policy; an estimate is labeled as an estimate.
5. **Admission:** policy authorizes every hop for tenant, operation, and data
   semantics. Missing/expired/conflicting offers refuse or replan.
6. **Witness:** the remote adapter and consumer-local adapter independently
   receipt their effects. Only the terminal ready receipt can become
   `KVTransferOK`; a public-rung-only receipt is partial.
7. **Calibration:** real adapters feed observed latency/bandwidth/path identity
   back into provider snapshots and the cache-price ledger. No performance claim
   ships from the synthetic spine.

## Candidate disposition

| Candidate | On-axis state | Portfolio route | Disposition |
|---|---|---|---|
| Consumer-owned public-to-private realization by opaque reference | PARTIAL substrate, integration absent | DEFAULT | **FILED #8259**; reuse `fabricmap`, do not create a parallel `KVTransferPlan` graph |
| Byte-sized end-to-end estimated ready time | SHIPPED after the original study | DEFAULT objective, compatibility-preserving | **SHIPPED #8261**; the refreshed inventory found no duplicate follow-on |
| NIXL/RDMA/TMA/CSL engine adapters and measured topology | PARTIAL / hardware-gated | OPTIONAL-MODULE | **DEDUP #3310/#6409** |
| Bilateral fixed producer/consumer layout capsule | ABSENT | RECIPE | Use only for a named pair with a captured witness; no generic issue yet |
| Universal neutral KV/layout interchange | ABSENT | WATCH | Revisit only after two incompatible engines and measured conversion economics |
| Producer-authoritative compiler merge | ABSENT / worldview mismatch | WATCH | Do not make FAK own vendor compiler composition |
| Remote direct address into SMEM/TMEM/PE-private SRAM | Physically unavailable on the described paths | EXCLUDE | Never manufacture a fake universal address |
| NIXL stride-compressed prepared descriptors | PRESENT upstream, absent locally | WATCH | Consider only if FAK's real route-control footprint measures as a bottleneck |

Existing issue searches also found #3316 (KV governance events), #3413 (real
vLLM/SGLang arena import), #5269 (peer KV fetch tier), and #2242 (P/D SLO pool
planning). None owns terminal consumer-private readiness or byte-sized
`fabricmap` route selection, so #8259 was not a duplicate; #8261 has since shipped and is closed.

## Licensing and provenance

NIXL core is Apache-2.0, except its documented DeepEP-derived example subtree;
some distributed Python wheels may include separately proprietary NVIDIA
components. This study copies no NIXL code. The two FAK leaves are clean-room Go
over existing FAK primitives.

The site's root MIT file belongs to the Chirpy theme lineage and does not clearly
grant MIT terms over every authored post, while the rendered site advertises CC
BY 4.0. Treat the essay as **INSPIRE-only**, paraphrase it, use only short
attributed quotations if needed, and do not vendor prose or figures.

## Exhaustive inventory refresh (2026-08-25)

Issue #8992 refreshed the denominator at the same pinned revision. The machine-generated map is
[`docs/research/inventory/hiraditya-hiraditya.github.io.json`](../research/inventory/hiraditya-hiraditya.github.io.json):
61 indexed files across 11 reported subsystems, with every local file walked and every required
non-tree source class recorded separately. GitHub read-back through the revision timestamp found
17 issue-endpoint records (two open issue-only post drafts and 15 closed pull requests), 15 pull
requests (one merged), zero discussions, zero releases, and zero tags. Reachable git history has
306 commits; four touch the target essay. No test/fixture suite, changelog, roadmap, or unfinished-work artifact
exists in the pinned tree; `tools/test.sh` is a site-build helper, not a test suite.

The FAK decision did not expand. #8259 remains the single open terminal-consumer-readiness leaf;
#8261 has shipped; adapter/topology measurement remains deduplicated to #3310/#6409; neutral
interchange stays watch-only; and a fictional universal private-memory address remains excluded.
The candidate matrix, FAK self-query witness, completeness critic, provenance caveat, and issue
tracking are pinned in the inventory map. No new follow-on survived this refresh.

## Coverage and honest limits

The pass covered the exact essay, all four commits that changed it, adjacent
Parts 1/2/4, source/reference health, site provenance, NIXL's public API,
descriptor serialization, registration/remote metadata, selection/lifecycle
code, representative UCX/GDS paths, tests, full history, release/tag state,
license/attributions, and open/closed issue/PR classes. FAK comparison used
self-query, indexed docs/claims/leaves/verbs, direct Go reads, module versions,
and issue deduplication.

No source system or GPU/WSE path was run. Remaining unknowns are exact WSE
ingress/staging behavior, model-specific KV placement sizes, LPX production
availability/latency, and measured end-to-end route economics. Those limits are
why hardware/adapter work stays in #3310/#6409 and why #8259/#8261 begin with
semantic and deterministic witnesses only.

## Companions

- `field-borrow` supplied the on-axis self-query, disposition, licensing, and
  issue-registration discipline.
- `study-repo` supplied the pinned NIXL docs/code/tests/history/issues/releases
  audit and completeness critic.
- Parent/spine links: #3259, #6377, #3310, #3316, #6409.

## Exhaustive NIXL refresh (2026-08-25)

Issue #8993 refreshes the NIXL denominator without changing the essay's core
conclusion. The pinned source is
[`ai-dynamo/nixl@b0cbb237354d72b83500d5214d2b4484f9866fa3`](https://github.com/ai-dynamo/nixl/tree/b0cbb237354d72b83500d5214d2b4484f9866fa3)
(commit time `2026-08-25T17:36:59Z`). The generated exhaustive map is
[`docs/research/inventory/ai-dynamo-nixl.json`](../research/inventory/ai-dynamo-nixl.json):
776 files, 163 directories, 14 immediate subsystems, 392 runtime files, 185 test
files, and 80 documentation files. The only skipped control directory was
`.git`.

The non-tree denominator is also pinned to that commit timestamp. GitHub REST
pagination found 250 issues (67 open, 183 closed) and 1,904 pull requests (198
open, 1,706 closed) created by the cutoff. GraphQL reported zero discussions.
The release/history pass covered 20 published releases and 28 reachable tags;
`v1.4.0` was latest. Repository-wide unfinished-work-marker, unsupported, and deprecation
searches supplied the roadmap surface because no dedicated roadmap file exists.
The provenance pass covered Apache-2.0 core licensing, attribution manifests,
the proprietary component notice, the separately licensed DeepEP example,
`SECURITY.md`, and `CONTRIBUTING.md`.

Representative source read-back corrected the generated classifier's narrow
architecture label: `docs/nixl.md`, `docs/BackendGuide.md`, the C++/Python/Rust
APIs, descriptors, agent/metadata lifecycle, backend plugins, telemetry and
tracing, registration tests, and benchmark harnesses were inspected directly.
The completeness critic therefore treats filenames as discovery aids, not proof
of semantic coverage.

### Refreshed borrow decisions and follow-ons

| Candidate | Pinned evidence | FAK decision | Follow-on |
|---|---|---|---|
| Consumer-ready admission above transfer completion | `src/api/cpp/nixl.h`; upstream PR [#2051](https://github.com/ai-dynamo/nixl/pull/2051) | **Borrow the boundary, not the subsystem.** NIXL owns transfer and remote-metadata lifetime; FAK owns terminal consumer-private readiness. | Existing #8259 |
| Byte-sized end-to-end ready-time pricing | `src/api/cpp/nixl_params.h`; `benchmark/nixlbench/` | **FAK-native.** Route price must include consumer realization, not only substrate transfer. | Existing #8261 |
| Standby-then-commit activation under live traffic | merged upstream PR [#2138](https://github.com/ai-dynamo/nixl/pull/2138) | **Borrow the lifecycle pattern.** Preserve active views while staging replacements, then independently receipt activation. | Evidence for #8259; no duplicate issue |
| Request-scoped transfer correlation | open upstream PR [#2137](https://github.com/ai-dynamo/nixl/pull/2137) | **Watch.** The observability shape is attractive but not settled at the cutoff. | Monitor upstream; file only after stable semantics and a FAK gap witness |
| Backend adapters and measured topology | `docs/BackendGuide.md`; `src/plugins/` | **Optional and hardware-gated.** Never make a convenience path silently replace fak-native execution. | Deduplicated to #3310/#6409 |

Candidate-specific `fak capabilities` self-queries found native routing and
capability-floor surfaces and no exact built-in NIXL adapter. That is not a
reason to manufacture parity: it reinforces the existing division in which
NIXL may serve as an explicitly selected transfer substrate while FAK retains
route admission, pricing, lifecycle governance, and receipts. No new issue was
filed because every surviving action is already owned or explicitly watched.
