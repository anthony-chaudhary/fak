---
title: "The disaggregable compute-claim ladder"
description: "The scale-free compute-claim taxonomy under epic #3259: a disaggregable ladder of compute claims and the profiles that make them scale-free."
---

# The disaggregable compute-claim ladder + scale-free profiles

Parent epic: [#3259](https://github.com/anthony-chaudhary/fak/issues/3259) — *the claim-space is the
universal compute-partition primitive*. Concept note:
[`docs/notes/CONCEPT-CLAIM-SPACE-IS-THE-DISAGGREGATION-PRIMITIVE-2026-07-08.md`](../notes/CONCEPT-CLAIM-SPACE-IS-THE-DISAGGREGATION-PRIMITIVE-2026-07-08.md).
This is **C3 (#3270)** — the design/enumeration deliverable. C1 (#3268) is the pricer, C2 (#3269) the
admission wiring; nothing here is a new inference kernel.

## Why this doc exists

The **control plane** already has a *declared* claim taxonomy: `dos.toml` `[lanes.trees]` maps ~200 lanes
to canonical file-tree globs, with an `exclusive` set and an `autopick` ladder, read by
[`laneadmit.ParseTaxonomy`](../../internal/laneadmit/laneadmit.go) and
[`regionadmit.LoadTaxonomy`](../../internal/regionadmit/regionadmit.go). Both refuse over the *same closed
vocabulary* — `laneadmit` and `regionadmit` each alias `dispatchorder.ReasonCollisionRisk` rather than
mint their own token.

The **compute plane** has no equivalent. Its claimable units — a prefill phase, an expert rank, a weight
shard, a KV span on a tier — live *implicitly* inside each placer, each with a bespoke partitioner and no
shared refusal vocabulary. Before C1 can price a compute-region collision and C2 can admit one, we have to
enumerate *what the claimable compute units even are*, at the grain the epic calls for ("pull EVERYTHING
apart … down to 100s of items").

Read [`dispatchorder.Candidate`](../../internal/dispatchorder/dispatchorder.go) with fresh eyes and it is
a **resource request**: `Lane` is a resource class, `Tree` is an address range, `Mode` is a lock
discipline, and `Plan` admits the disjoint set and serializes the rest. Nothing in that algebra is
specific to files. This doc names the compute-plane rows of the same table.

### The algebra (invariant across every row below)

A **claim** is `(class, range, mode)`. Two claims are **independent** iff they are of *different classes*,
OR their *ranges are disjoint under the class's disjointness test*, OR *both hold the range `shared`*. A
placer's job is exactly the control-plane job: **admit the maximal independent set, serialize the rest,
refuse the un-addressable** — over one closed refusal vocabulary. `collisionOf` in `dispatchorder` is that
independence test for file trees; C1 generalizes it per class.

---

## 1. The disaggregable-unit ladder (coarse → fine)

Each row: the **claimed unit**, its **address form** (the shape of `range`), its **disjointness test**,
the **existing fak mechanism** (`file:symbol`), and a status tag:

- **`[SHIPPED]`** — the unit is claimed *and* placed by a live mechanism today.
- **`[SEAM]`** — a real partitioner exists and computes the placement, but it does *not* yet speak the
  shared claim-space (its own ad-hoc disjointness test, no shared `Decide`, no shared refusal token).
- **`[GAP]`** — named as a *future* claimable unit; no partitioner exists yet.

| # | Claimed unit | Address form (`range`) | Disjointness test | fak mechanism (`file:symbol`) | Status |
|---|---|---|---|---|---|
| 0 | **File tree** (control-plane baseline) | repo-relative globs | tree-prefix overlap | `dispatchorder.go` `collisionOf` / `TreesOverlap`; `regionadmit.Decide`; `laneadmit.Decide` | `[SHIPPED]` |
| 1 | **Request → sub-query → reasoning-step** (routing grain) | an `Aspect` token (`request`/`tool_call`/…) | same aspect ⇒ same routed engine | `modelroute/modelroute.go` `Aspect`, `AspectRequest`, `AspectToolCall`; `Route`/`Combine` | `[SHIPPED]` |
| 2 | **Prefill phase / decode phase** | a phase role + resident worker key | same worker admits one phase stream | `modelengine/native_pd.go` `NativePDCluster` (`Admit`, `routeDecode`) | `[SEAM]` |
| 3 | **Model (co-residency)** | a `ModelID` in a fixed byte budget | same slot / drafter is exclusive | `polymodel/polymodel.go` `Pool.Admit`, `PickDrafter`, `CanShare` (`FAK_POLYMODEL`, default OFF) | `[SEAM]` |
| 4 | **Layer / layer-group (pipeline stage)** | a contiguous layer interval `[lo,hi)` | interval overlap on the layer axis | `model/pipeline.go` `PartitionPlan`, `NewPartitionPlan`, `StageSpec`; `modelengine/pipeline.go` `PipelineEngine` | `[SEAM]` |
| 5 | **MoE expert / expert-group** | an expert rank ∈ `[1,NumExperts]`, or a group ∈ `[0,NGroup)` | rank-interval / group-set overlap | `model/expert_parallel.go` `ExpertParallelPlan` (fails closed outside `[1,NumExperts]`); `model/moe.go` `glmMoeEPFFN`, `Config.NGroup`/`TopKGroup` | `[SEAM]` |
| 6 | **Attention head / head-group (TP shard)** | a shard rank ∈ `[0,ranks)` over a dim | shard-rank / dim-interval overlap | `model/tensor_parallel.go` `TPPlan`, `NewTPPlan`, `Validate` | `[SEAM]` |
| 7 | **KV span × tier** | `[from,len)` on a named tier (HBM/DRAM/CXL/disk) | same tier ∧ span overlap | `kvmmu/attention.go`, `cachemeta/cachemeta.go`; admission `compute/kvadmission.go` `DecideKVAdmission`, `PlanKVAdmission` | `[SEAM]` |
| 8 | **Collective rank (process group)** | a rank ∈ `[0,size)` in a group | one rank held by one participant | `model/dist_collective.go` `DistComm` (`Coordinate`/`Join`/`AllReduceSum`) | `[SEAM]` |
| 9 | **Draft / verify roles (speculative)** | a `(drafter, target)` pair over a prefix | same drafter serialized across targets | `polymodel/polymodel.go` `AcceptGreedy`, `PickDrafter`; `modelroute/polybridge.go` | `[SEAM]` |
| 10 | **GEMM tile / block** | a tile `(row-range, col-range)` of an op | 2-D tile-rectangle overlap | — | `[GAP]` |
| 11 | **Sampler / logit-processor, tokenizer, embedding, norm/rope stages** | a named per-token pipeline stage | stage identity (one owner per stage instance) | — | `[GAP]` |

**How the tags were assigned.** Rows 0–1 are `[SHIPPED]`: a live mechanism both *claims* and *places* over
a real disjointness test today (`dispatchorder`/`regionadmit` for trees; `modelroute` for aspects). Rows
2–9 are `[SEAM]`: each partitioner **exists and is exercised** (e.g. `ExpertParallelPlan` returns an error
for a rank outside `[1,NumExperts]` — that *is* `regionadmit` refusing an out-of-taxonomy lane; `routeDecode`
picking the resident decode worker *is* `dispatchorder` picking the freshest non-colliding candidate) but
none of them route through the shared `Decide` or carry the shared refusal token — that unification is
exactly C1+C2. Rows 10–11 are `[GAP]`: no partitioner exists; they are named so the ladder is honest about
where "hundreds of units" is aspiration, not shipped substrate.

### The disjointness tests reduce to three primitives

Every `[SEAM]` row's test is one of three shapes, which is why C1 can price them "behind the same
`collisionOf` interface":

- **integer-set / rank-interval overlap** — rows 5, 6, 8 (and device ids): `range` parses to a set of
  ranks (a `lo-hi` interval expands to its members); claims collide iff the sets intersect.
- **tier-qualified span overlap** — row 7: `range` is `TIER:[from,len)`; claims collide iff the tier
  qualifier matches *and* the half-open `[from, from+len)` intervals intersect.
- **token / identity overlap** — rows 1, 2, 3, 9, 11: `range` is an opaque identity token (an aspect, a
  `phase@device`, a `ModelID`, a `drafter→target`); claims collide iff the tokens are equal.

Row 4 (layer interval) and row 10 (GEMM tile) are interval / rectangle overlaps — the interval primitive
in 1-D and 2-D respectively. C1 (#3268) implements the integer-set and tier-qualified-span primitives
first (they cover rows 5–8, the epic's headline "expert ranks / TP shards / KV spans"); the token and 2-D
primitives are the additive follow-ons.

---

## 2. Scale-free profiles

The claim *vocabulary* above is invariant across substrate scale. What changes between a phone and a
hyperscaler is only **the unit count** and **the backend behind the collective seam** (`DistComm` today is
host-f32, not NCCL). The same `(class, range, mode)` algebra and the same closed refusal vocabulary hold at
every scale — including the degenerate one.

### A. Phone / IoT — 1 device
Claim-space **degenerates to one-exclusive-at-a-time**: every claim is `mode=exclusive` on the single
device, so the maximal independent set has size ≤ 1 and the placer is a mutex. The vocabulary is *identical*
— a decode claim and a prefill claim still collide (same device), they just can never both be admitted.
Rows in play: 1 (routing is trivial — one engine), 2 (prefill and decode time-share), 7 (KV lives in one
tier). Everything else degenerates to "the one device." No collective backend (`DistComm.Size() == 1`).

### B. Workstation — 1 host, a few devices / one big device
Poly-model **residency** turns on (row 3): several models co-resident in one byte budget
(`polymodel.Pool`), a shared prefill claim (row 2) served once and reused by `CanShare`/`PrefixDigest`,
speculative draft/verify (row 9) between same-`Family` models. **Tens of units.** Collective is local
(`LocalCollective`). This is the first scale where the independent set is meaningfully > 1 and the pricer
earns its keep (two co-targeted models want the same drafter → serialize).

### C. Small cluster — a few hosts over `DistComm`
Native prefill/decode disaggregation across workers (row 2, `NativePDCluster` over `DistComm`),
expert-parallel MoE (row 5, `ExpertParallelPlan` sharding experts across ranks), TP shards for attention
(row 6). **Hundreds of units** (experts × ranks × phases). Collective is `DistComm` `Coordinate`/`Join`
(host-f32 all-reduce/all-gather). Every placement is a claim admitted iff its rank-interval is disjoint
from the live set — the compute twin of the fleet admitting disjoint file trees.

### D. Hyperscaler — a real device mesh, `[GAP]`
DP × TP × PP × EP over a physical mesh: **thousands of claimable units** (data-parallel replicas × tensor
shards × pipeline stages × experts). The *algebra is unchanged* — a claim is still `(class, range, mode)`,
independence is still range-disjointness — but it is `[GAP]` on the **backend**: `DistComm` is host-f32,
not NCCL/RCCL, and there is no device-mesh topology object. The taxonomy is ready; the collective substrate
behind the seam is the unbuilt part. Naming this honestly is the point: the claim-space does not need a
rewrite to reach this scale, only a real collective backend behind `DistComm`.

**Invariant, stated once:** the unit count grows A → D by orders of magnitude and the collective backend
changes, but the claim tuple, the disjointness tests (§1), the maximal-independent-set admission, and the
closed refusal vocabulary are byte-for-byte the same code path. That is the epic's thesis, made checkable.

---

## 3. Proposed `[compute.claims]` declaration

The compute analogue of `dos.toml` `[lanes.trees]`: a declared table naming each claimable class, its
address form, and its disjointness primitive, that C1/C2 read the way `laneadmit.ParseTaxonomy` reads
`[lanes]`. Kept honest — a row is **declarable** only if its `[SEAM]` mechanism exists; `[GAP]` rows are
listed commented-out so the file never claims a unit the substrate cannot place.

```toml
# dos.toml (proposed) — the compute-plane twin of [lanes.trees].
# form: how `range` is spelled.  test: which §1 disjointness primitive prices a collision.
# status mirrors the ladder; C2's admission refuses any claim whose class is absent here.

[compute.claims.decode]      # row 2   [SEAM]
  form   = "phase@worker"    # e.g. "decode@w0"
  test   = "token"           # identity overlap
  mode   = "exclusive"       # one decode stream per worker

[compute.claims.prefill]     # row 2   [SEAM]
  form   = "phase@worker"
  test   = "token"
  mode   = "shared"          # a prefill result is shareable (PrefixDigest reuse)

[compute.claims.model]       # row 3   [SEAM] — gated on FAK_POLYMODEL
  form   = "model-id"
  test   = "token"
  mode   = "exclusive"

[compute.claims.stage]       # row 4   [SEAM]
  form   = "layers:[lo,hi)"
  test   = "interval"
  mode   = "exclusive"

[compute.claims.expert]      # row 5   [SEAM]
  form   = "rank:lo-hi"      # e.g. "4-7"; group via "g:0-1"
  test   = "rank-set"        # integer-set overlap
  mode   = "exclusive"

[compute.claims.tp_shard]    # row 6   [SEAM]
  form   = "rank:i/n"
  test   = "rank-set"
  mode   = "exclusive"

[compute.claims.kv]          # row 7   [SEAM]
  form   = "TIER:[from,len)" # e.g. "HBM:[0,128)"
  test   = "span"            # tier-qualified span overlap
  mode   = "exclusive"

[compute.claims.collective]  # row 8   [SEAM]
  form   = "rank:i/size"
  test   = "rank-set"
  mode   = "exclusive"

[compute.claims.spec]        # row 9   [SEAM]
  form   = "drafter->target"
  test   = "token"           # same drafter serialized across targets
  mode   = "exclusive"

# --- [GAP] rows: named, NOT declarable until the mechanism ships -------------
# [compute.claims.gemm_tile]   # row 10  form="rows:R,cols:C" test="rect"  — no partitioner yet
# [compute.claims.stage_op]    # row 11  form="op@token"      test="token" — sampler/tokenizer/embed/norm
```

C2's shared `Decide` refuses a claim whose `class` is not declared here with the same closed token an
out-of-taxonomy lane gets — the compute twin of `regionadmit` refusing an unknown lane. A `[GAP]` class is
therefore *automatically* refused (fail-closed) until its row is uncommented, which happens only when the
`[SEAM]` mechanism behind it lands. The taxonomy can never over-promise a unit the substrate cannot place.

---

## Fences (what this doc is not)

- **Design/enumeration only.** No code, no benchmark number is asserted here. C1 (#3268) prices; C2 (#3269)
  admits and wires the `Submit` seam; C4 (#3271) expresses polymodel fusion as a claim.
- **`[GAP]` rows are future units**, not shipped ones. GEMM-tile (row 10) and per-stage sampler/tokenizer/
  embedding/norm (row 11) have *no* partitioner today and are named only to show where the ladder continues.
- **Hyperscaler scale (profile D) is `[GAP]` on the backend.** The claim-space reaches it without an algebra
  change; the unbuilt part is a real collective (NCCL/RCCL) and a device-mesh topology behind `DistComm`.
- The `[compute.claims]` TOML above is a *proposed* schema for C1/C2 to read, not a shipped parser.
