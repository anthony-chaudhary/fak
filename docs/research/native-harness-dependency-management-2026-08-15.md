# Native harness dependency management: research frame

**Date:** 2026-08-15  
**Master:** [#6886](https://github.com/anthony-chaudhary/fak/issues/6886)  
**Related masters:** [#6887](https://github.com/anthony-chaudhary/fak/issues/6887), [#6889](https://github.com/anthony-chaudhary/fak/issues/6889)  
**Status:** research input, not a shipped compatibility claim

## Decision in one page

A fak-native harness needs a dedicated **composition resolver**, but it should not become a package installer or one global compatibility matrix. Its job is narrower and more useful:

1. accept a desired workload and a set of versioned component references;
2. ask existing domain owners for constraints and observations;
3. separate technical feasibility from fitness for purpose;
4. resolve the conditional dependency graph at a named lifecycle phase;
5. attach provenance, proof tier, and freshness to every decisive fact; and
6. emit either a launch-bound receipt or a small actionable conflict explanation.

The central modeling choice is to keep five kinds of truth distinct:

| Kind | Example | Authority | Can it be cached? |
|---|---|---|---|
| Identity | `backend=sglang@0.x`, artifact hash, harness profile revision | component/registry owner | yes, immutable identities indefinitely |
| Declared constraint | backend requires a CUDA range; tool requires network capability | component owner | yes, until declaration/version changes |
| Observed fact | node compute capability, free memory, installed driver | host/fleet probe | briefly, with observation time |
| Witnessed claim | this artifact decoded on this kernel/device tuple with this quality result | evidence producer + verifier | until an invalidating identity changes or policy expires it |
| Workload requirement | legal review requires citations and human approval; coding needs repository write/test loop | domain adapter/operator | for the workload-contract version |

Collapsing these kinds causes the current confusion. A package declaration cannot prove a live GPU is available. A successful import cannot prove a quantized kernel executed on-device. A technically runnable coding stack is not thereby suitable for legal review. A recommendation must not silently become a hard launch gate.

The smallest spine should resolve one harness + model artifact + tools + policy + backend + observed node tuple. It should accept one stack and reject one transitive conflict, with a receipt identifying the authority and evidence for each decisive edge. Children #6891 and #6892 own that spine and its independent proof.

## Problem classification and value frame

**Centrality: Enabling.** Dependency resolution makes native-harness portability and integrated operation reliable, but it is not itself the user outcome.

- **P1 managed context:** expose only the effective resolved stack, open assumptions, and conflict chain. Do not inject the full component catalog into the model context.
- **P2 net-true efficiency:** a resolver is valuable only if avoided launch failures and reduced operator search exceed declaration, evidence, and resolver maintenance costs. #6892 must compare against the competent current workflow, not against no validation.
- **P3 bounded adaptation:** version declarations, preserve authority, expire observations/evidence deliberately, and fail closed only for mandatory unknowns.
- **P4 integrated operations:** the resolution result must bind build, preflight, dispatch, launch, and later revalidation. A detached planning report is not enough.

**For:** operators and builders composing fak-native harnesses.  
**Problem:** independent components can each look supported while their exact combination, host, and workload is not.  
**Today:** domain-local checks and prose matrices answer slices of the question.  
**Better because:** one receipt explains what can stack, what baseline is mandatory or recommended, and what remains unproved.  
**Witness:** a real satisfiable stack and a seeded transitive incompatibility driven through the same resolver seam.

## What already exists, and who should retain authority

This proposal composes existing authorities; it does not flatten them.

| Current authority | What it knows | Resolver relationship | Known gap |
|---|---|---|---|
| `internal/harnessprofile` and `internal/harnessselect` | harness/profile selection, path/tag applicability, overlap behavior | supply selected harness identities and constraints | contextual selection is not whole-stack dependency resolution; `harnessselect` is currently peer WIP and must not be treated as shipped |
| `internal/harnessinit`, `docs/harness-kit-contract.md`, #6793, #6805 | harness construction/conformance and semantic upgrade compatibility | validate harness adapter contracts | do not generalize harness-kit ABI negotiation into model/hardware truth |
| `internal/portabilitycontract` and portability lab/adapter docs | operation/feature compatibility and acceptance evidence | expose typed adapter capabilities and receipts | feature support alone does not select artifacts, infrastructure, or workload fitness |
| `internal/quantmeta` and #6224 | quantized artifact/runtime metadata adjudication | own artifact-runtime allow/refuse result | its current contract intentionally covers a narrower tuple; support evidence and infrastructure baseline remain external inputs |
| `internal/supportmaturity` | staged support declarations and machine-readable validation | provide maturity/provenance vocabulary where applicable | maturity is not satisfiability and should not become a substitute for a witnessed edge |
| `docs/HARDWARE-MATRIX.md`, `docs/quantization-support.md`, proofs/notes | curated hardware and quantization support statements | seed migration research for #6889/#6895 | prose and generated views must not become duplicate mutable authorities |
| account/model/quota preflight in #6849 | spawn-time credentials, model compatibility, quota | remain a launch-phase validator | transient account/quota state is not a static build dependency |
| cross-layer intent work in #6881 | independently owned intent vocabularies and cross-layer contradictions | likely envelope/composition neighbor | intent vocabulary must not absorb artifact/runtime dependencies |
| policy engine | action authority and default-deny controls | validate workload/stack authority constraints | policy permission is not model quality or technical support |

The resolver should depend on narrow provider interfaces, not import every domain package into a central god package. Each provider returns normalized facts plus ownership/provenance; the resolver owns graph semantics, phase ordering, deterministic choice, and explanation.

## The dependency model

### Nodes

A node is a versioned identity, not a free-form name. Initial node classes:

- `workload-contract`
- `harness-profile`
- `adapter` or `tool-provider`
- `policy-profile`
- `model-artifact`
- `serving-backend`
- `kernel-path`
- `runtime` (driver/toolkit/OS/architecture facts when relevant)
- `infrastructure-baseline` (observed node/device/capacity tuple)
- `evidence-artifact`

Capabilities such as `repo.patch`, `citation.traceable`, `quant.awq.w4a16`, or `device.cuda.sm90` are **provided properties**, not necessarily installable nodes. This allows alternatives without inventing fake packages.

### Edge algebra

Start smaller than mature package solvers:

| Relation | Meaning | Failure behavior |
|---|---|---|
| `requires` | mandatory condition for this phase/workload | unsatisfied blocks; unknown blocks only when the requirement says it must be proven |
| `conflicts` | the identities/properties cannot be composed under the condition | blocks with both owners named |
| `provides` | component satisfies a named capability/property | participates in alternative satisfaction |
| `substitutes` | named alternatives can satisfy the same requirement, with explicit scope | resolver may choose deterministically and explain choice |
| `recommends` | improves a declared objective but is not mandatory | warning/ranking input; never silently blocks |
| `evidenced-by` | links a claim to proof tier, provenance, environment, and freshness | missing/expired evidence changes claim state, not component identity |

Avoid `optional` as an edge with ambiguous semantics. An optional component is simply an unrequested component; a recommendation is explicit. Avoid unconstrained numeric “compatibility scores” in the satisfiability core. Fitness ranking belongs after hard feasibility and must preserve dimensions and uncertainty (#6887).

### Conditions

Edges may be conditional on normalized facts, for example:

```text
model-artifact:foo-awq@sha256:…
  requires quant.awq.w4a16
  requires model.arch.foo

backend:bar@1.4 + kernel:baz@0.9
  provides quant.awq.w4a16
    when device.vendor=nvidia
     and device.compute_capability>=8.0
     and driver in validated-range
```

Conditions need a closed, typed operator set in the first spine (exact identity, set membership, ordered version range, numeric capacity floor). Arbitrary scripts make resolution non-reproducible and unsafe. Domain-specific complex validation should return a signed/hashed adjudication fact through an adapter rather than executable predicates in the manifest.

### Required versus recommended infrastructure

Every baseline answer should carry three layers:

1. **minimum required** — without it the declared workload cannot execute safely/correctly;
2. **validated baseline** — exact or bounded tuple with current evidence at the required proof tier;
3. **recommended baseline** — a non-blocking choice justified by a named objective such as latency, capacity margin, cost, or quality.

“Minimum” is not automatically “validated,” and “validated” is not automatically “recommended.” If only the minimum is known, fak should say so. If a slower fallback exists, its quality/performance/capacity consequences are part of the resolution receipt.

## Four resolution phases

A single timeless `compatible=true` is structurally wrong. Resolve and revalidate by phase:

| Phase | Stable inputs | Dynamic inputs | Appropriate proof |
|---|---|---|---|
| Build/assembly | identities, manifests, ABI/schema ranges, policy references | none or locked registry data | parser/schema/contract tests; deterministic graph fixture |
| Host preflight | build receipt, backend/kernel requirements | OS/arch/device/driver/memory/account/network observations | live probe plus domain adjudicators |
| Launch | preflight receipt, workload contract | quota, allocation, endpoint/tool readiness, current policy | launch-bound nonce/time/host receipt and fail-closed recheck |
| Runtime | launched stack identity | health, capacity pressure, revocation, evidence-invalidating drift | continuous checks and event receipts; bounded fallback policy |

A later phase may invalidate an earlier receipt but should not rewrite history. Receipts form a chain: each names parent receipt hashes, newly observed facts, decisions, unresolved recommendations, and expiry.

## Explanation and proof contract

A useful refusal is an **unsatisfied core**, not a dump of the graph:

```text
REFUSE launch
wanted: legal-review@r3 on node:n7
blocker: workload requires citation.traceable at proof>=evaluated
candidate harness ponytail@r8 does not provide it
available substitute: legal-review-harness@r4
authority: workload adapter legal-review@r3
missing witness: citation fixture suite v2
```

For a transitive hardware case:

```text
artifact A -> requires AWQ W4A16
backend B -> provides AWQ W4A16 only through kernel K
kernel K -> requires compute capability >= 8.0
observed node N -> compute capability 7.5
```

The resolver must identify the shortest stable causal chain it can justify, but “minimal” needs a declared meaning. For the spine, use inclusion-minimal decisive constraints with deterministic ordering; #6892 should compare against an independent small-graph oracle. Ranked remedies should distinguish:

- change a selected component;
- choose another observed baseline/node;
- relax a recommendation (never a requirement without operator authority);
- collect a missing witness;
- wait/retry for transient launch state.

Evidence must be matched to claim class:

| Claim | Insufficient evidence | Required representative evidence |
|---|---|---|
| Manifest parses | prose example | schema/parser test |
| Graph resolution is correct | one golden fixture | golden spine plus independent/property oracle over supported algebra |
| Backend imports artifact | file recognized | load/decode witness with exact hashes |
| Quant path runs on device | CPU fallback or compile | device execution/GEMM or full decode proof showing selected kernel/device |
| Quality retained | successful tokens | named evaluation against tuned baseline |
| Capacity/throughput suitable | one decode | workload-shaped capacity/performance witness |
| Workload fit | technical compatibility | domain-owned requirements plus relevant evaluation/review |
| Production support | one lab run | declared soak/operating-envelope evidence and freshness policy |

Unknown, unsupported, contradicted, and stale are separate states. “No evidence found” must never be rendered as “unsupported”; a vendor claim must never silently become a fak-witnessed claim.

## Motivating example 1: coding versus legal review

Suppose `ponytail` is a highly effective coding harness. It provides repository search, patch application, test execution, and code-review loops. All of its binaries and model adapters may be technically compatible with the operator's machine.

For `workload=coding-change`, those provisions can make it the preferred stack after hard policy and model/tool checks pass.

For `workload=legal-review`, the same stack may be technically runnable but not demonstrated to provide:

- jurisdiction/version-scoped authority sources;
- citation traceability from conclusions to source spans;
- confidentiality/data-residency controls;
- conflict/privilege handling;
- required human review and escalation.

The right output is not “ponytail is bad” or “legal capability score 42.” It is:

- feasibility: compatible;
- workload fitness: not established for this contract;
- mandatory control: absent or unproved;
- recommendation: select a domain adapter that provides the control, or collect the named evidence;
- authority: the legal workload adapter/operator, not the generic resolver.

This example proves why dependency management and workload selection are adjacent but distinct. #6886 resolves hard composition; #6887/#6893 define and validate purpose-specific requirements and recommendations.

## Motivating example 2: quantization and hardware

“AWQ supported” can hide at least these variables:

- model architecture and layer shapes;
- weight/activation/KV quantization distinction;
- bit width, group size, zero-point and packing layout;
- artifact dialect/converter version;
- serving backend and selected kernel implementation;
- accelerator vendor/architecture/compute capability;
- driver/runtime/library versions;
- fallback path and whether it silently runs on CPU or dequantizes;
- memory/capacity target;
- decode correctness, quality, throughput, and soak proof tiers.

Therefore support is a conditional edge over normalized identities, not a boolean attached to “AWQ.” #6889/#6895 own the support graph; #6896 ingests and expires lab evidence; #6897 uses it with #6886 during preflight. #6224 remains the artifact-runtime adjudicator rather than being replaced by the graph.

## Field borrowing

These fields solve different slices. The useful design is deliberately hybrid.

| Field | Borrow | Do not borrow blindly |
|---|---|---|
| Package dependency solvers (PubGrub/SAT family) | version ranges, incompatibility derivation, deterministic explanations, lock/receipt concept | packages are mostly declared static facts; fak also has observed hosts, evidence expiry, workload authority, and phased revalidation |
| npm peer dependencies | a component can require a capability/version supplied by its embedding composition rather than install it privately | peer-dependency warning history shows that weak semantics create late surprise; mandatory fak edges must be explicit and phase-bound |
| Build systems | explicit DAGs, content identities, hermetic inputs, cache keys, reproducible action receipts | hardware availability, quota, policy, and quality are not build actions and cannot all be hermeticized |
| Kubernetes scheduling | hard node affinity versus preferred affinity; taints/tolerations; observed resource placement | scheduling predicates prove placement eligibility, not model quality or kernel correctness; cluster labels are claims requiring authority |
| Kubernetes device plugins | vendor/domain owner advertises device resources through a narrow interface | resource count alone loses driver/kernel/artifact compatibility and proof tier |
| SBOM + VEX | preserve component identity/provenance while attaching contextual status statements; machine-readable downstream use | vulnerability status is not general compatibility, and a supplier assertion is not automatically a runtime witness |
| Software product-line feature models | requires/excludes constraints and valid-configuration reasoning | global feature models centralize vocabulary and struggle with independently owned, rapidly changing runtime facts |
| Hardware compatibility databases | tuple-shaped support records and certification levels | flat certification tables age; fak needs exact evidence links, unknown/stale states, and generated views |
| Policy engines | default-deny authority boundaries, explainable decisions, explicit inputs | policy authorization should not decide technical feasibility or rank quality |

### Borrowing decision

- **Adopt:** typed requires/conflicts/provides; hard versus preferred constraints; immutable identities; lock-style receipts; explicit provenance/status documents; deterministic explanation.
- **Adapt:** version solving becomes conditional graph resolution over phase-specific declared and observed facts; scheduler labels become authority-bearing observations; certification becomes evidence tier + freshness.
- **Reject:** one universal feature enum, arbitrary predicate scripts, a flat support boolean, recommendation-as-requirement, and a single aggregate fitness score.

## Minimal machine contract to test next

This is illustrative research syntax, not a committed schema:

```json
{
  "schema": "fak-stack-request/0",
  "workload": {"id": "coding-change", "revision": "r1"},
  "components": [
    {"kind": "harness", "id": "ponytail", "revision": "r8"},
    {"kind": "model-artifact", "id": "sha256:artifact"},
    {"kind": "backend", "id": "backend-b", "version": "1.4.2"},
    {"kind": "policy", "id": "repo-write-reviewed", "revision": "r3"}
  ],
  "target": {"node": "observed-at-preflight"},
  "proof_floor": {"model-execution": "device-decode"},
  "objectives": ["correctness", "latency", "cost"]
}
```

The receipt—not the request—contains expanded dependencies, chosen substitutes, exact observed facts, evidence links, warnings, unresolved unknowns, and expiry. Catalog facts should be independently versioned and referenced by hash/revision rather than copied into each request.

## Validation strategy

1. **Schema/contract:** malformed identities, unknown operators, ambiguous ownership, and cyclic authority fail before resolution.
2. **Golden end-to-end:** one representative native-harness stack resolves through the real CLI/library and emits a stable receipt.
3. **Negative end-to-end:** one transitive quantization/hardware contradiction emits an inclusion-minimal deterministic chain.
4. **Independent oracle:** enumerate small graphs and compare satisfiable/refused outcomes against a simple independent solver (#6892).
5. **Mutation:** delete/invert decisive edges and verify the witness fails; seed stale/conflicting evidence and recommendation/requirement confusion.
6. **Live preflight:** observe a sanctioned node and bind the host facts to the receipt; do not infer GPU support from the development workstation.
7. **Purpose fitness:** run coding and legal-review fixtures through domain-owned contracts and preserve reviewer disagreement (#6894).
8. **Net-true benchmark:** measure operator time, failed launches caught, false refusals, explanation usefulness, resolution latency, and catalog maintenance against current docs plus existing preflights.

Success is not “all tests green.” Success is that tests and live witnesses jointly cover the specific claim class, while receipts state what remains unknown.

## Risks and controls

| Risk | Control |
|---|---|
| Central god schema couples every subsystem | narrow provider interfaces; namespaced independently versioned facts; architecture tests |
| Catalog becomes another stale matrix | source facts with provenance/freshness; generate human views; invalidate on identity changes |
| Resolver overclaims professional suitability | separate fitness; domain authority and external review; explicit “not evaluated” |
| Dynamic state makes builds irreproducible | phase receipts; immutable assembly inputs; host/launch revalidation rather than hidden mutation |
| Recommendations accidentally block | distinct relation and output channel; tests proving recommendations are non-blocking |
| Solver complexity consumes the value | smallest algebra first; deterministic algorithm; net-true benchmark before optimization |
| Vendor claims appear witnessed | provenance labels and conflict policy; never upgrade proof tier without matching evidence |
| Fallback hides wrong kernel/device | receipt records selected path and cost; device witness asserts execution path |

## Issue program

### Masters

- #6886 — evidence-aware native-harness dependency resolution.
- #6887 — workload/risk/evidence fitness selection.
- #6889 — queryable hardware and quantization support evidence.

### Spawned contracts

- #6890 — inventory authorities and complete field-borrow research.
- #6891 — minimal native-harness resolver spine.
- #6892 — soundness, explanation, and net-true proof.
- #6893 — workload contract/domain adapter model.
- #6894 — informed-operator recommendation benchmark.
- #6895 — provenance-bearing hardware/quant support graph.
- #6896 — lab witness ingestion and stale-edge invalidation.
- #6897 — required/recommended infrastructure preflight.

The order is research (#6890) -> minimal working resolver (#6891) -> independent proof (#6892), while workload fitness and support evidence proceed through their separate owners and meet at preflight (#6897). This ordering preserves a working spine before exhaustive matrix expansion.

## Sources consulted

Accessed 2026-08-15. These sources motivate patterns; they are not evidence that fak implements them.

- Dart Pub, “Package Versioning / solver documentation,” especially incompatibility-driven resolution: <https://github.com/dart-lang/pub/blob/master/doc/solver.md>
- npm, `package.json` peer dependency semantics: <https://docs.npmjs.com/cli/v11/configuring-npm/package-json#peerdependencies>
- Kubernetes, node affinity hard and preferred scheduling constraints: <https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/>
- Kubernetes, device plugin framework and vendor-advertised resources: <https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/>
- SPDX, security use cases and VEX context: <https://spdx.dev/learn/areas-of-interest/security/>
- CycloneDX, VEX capability and contextual status exchange: <https://cyclonedx.org/capabilities/vex/>
- Repository authorities cited above: `docs/harness-kit-contract.md`, `docs/portability-adapter-sdk.md`, `docs/generation-abi-compatibility-policy.md`, `docs/HARDWARE-MATRIX.md`, `docs/quantization-support.md`, `internal/harnessprofile`, `internal/portabilitycontract`, `internal/quantmeta`, and `internal/supportmaturity` at the commit used by this note.
