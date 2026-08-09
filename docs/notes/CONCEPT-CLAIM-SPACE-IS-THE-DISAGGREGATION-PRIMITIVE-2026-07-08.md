---
title: "The claim-space IS the disaggregation primitive — one partition"
description: "fak's control-plane claim-space (agent/file partitioning) and its compute-plane disaggregation are one partition algebra at two granularities."
---

# The claim-space IS the disaggregation primitive — one partition algebra, two planes

Date: 2026-07-08
Status: synthesis / concept note **plus the first two unification leaves shipped** — the
note's §6 gap list is no longer wholly open: `dispatchorder`'s compute-resource axis (#3268)
and the shared compute admission kernel + `Submit`-seam contention pricing
(`internal/computeadmit`, #3269) both landed, and §6 now records which rungs closed and which
stayed open. Draws a through-line between fak's **control-plane
claim-space** (the `dos arbitrate` lane-lease + `dispatchorder` fan-out pricer that decides
which agent may touch which files) and its **compute-plane disaggregation** work (prefill/decode
split, KV tiers, MoE expert-parallel, tensor-parallel). Asserts **no benchmark number**. Tags every
load-bearing claim `[SHIPPED]` / `[SEAM-ONLY]` / `[GAP]`. Where it borrows a distributed-systems
term it disclaims the scope, the same discipline `modelroute.ReduceAllReduce` holds ("NOT a tensor
all-reduce … the scope is scalars").

Related: [`THROUGHPUT-TRUST-SHARED-SPINE-2026-06-24.md`](THROUGHPUT-TRUST-SHARED-SPINE-2026-06-24.md)
(shared serving spine), [`dual-track-serving-plan.md`](../serving/dual-track-serving-plan.md)
(epic #50), [`pd-disaggregation-kv-routing-sota.md`](../serving/pd-disaggregation-kv-routing-sota.md)
(ride-vs-own matrix), epics #637 (throughput+trust one spine), #639 (MPI-shaped fleet comm),
#1911 (agentic-first scheduling), #748 (agent-OS process model).

---

## 1. The crux

fak has, independently, built two things that are the **same algebra** wearing different names:

- **A control plane that partitions *work* across *agents*.** Given N candidate work-units and the
  leases already held, which one may a worker take, which collide, which serialize? This is
  `dos arbitrate` / `laneadmit.Decide` / `regionadmit.Decide` / `dispatchorder.Plan`. Its unit of
  claim is a **file tree**; its independence test is **tree-disjointness**; its output is
  admit / serialize / refuse over a **closed refusal vocabulary**. `[SHIPPED]`

- **A compute plane that partitions *inference* across *devices*.** Prefill on one worker, decode on
  N others; this expert on rank r, that shard on rank s; this KV span in HBM, that one demoted to
  DRAM/CXL/disk. This is `NativePDCluster`, `ExpertParallelPlan`, `TPPlan`, `kvmmu`/`cachemeta`
  tiering. Its unit of claim is a **compute region** (a phase, an expert, a shard, a span); its
  independence test is **data-disjointness**; its output is place / route / fail-closed. `[SHIPPED]`
  substrate, mostly single-box / CPU-ref (§4).

The thesis of this note: **these are one claim-space at two granularities.** "Disaggregate the
inference engine" is not a different project from "partition the agent fleet" — it is the *same
partition primitive applied at a finer grain*. The control plane claims directories; the compute
plane claims prefill-phases and expert-ranks; a fully general system claims **hundreds of units of
either kind through one admission kernel**. Today the two planes are built separately and do not
share that kernel — which is precisely where the leverage (and the open work) is.

## 2. What the arbiter claim-space already is (the algebra) — `[SHIPPED]`

The control plane is a layered, pure-kernel admission system. Four claim grains, coarse→fine, each
a decision function (state in, verdict out, no I/O; the impure half lives in the `cmd/fak` shell):

| Grain | Where | Claims | Stops |
|---|---|---|---|
| File-tree lock lease | `internal/leaseref` (`refs/fak/locks/*`) | a file tree | two agents editing the same FILES |
| Intent lease | `internal/leaseref/intent.go` (`intent-<key>`) | a target (issue/signature) | two agents fixing the same ISSUE in different files |
| Session descriptor | `internal/leaseref/session.go` | a run identity | — |
| Per-act adjudication | `internal/adjudicator/decide.go` | one tool call | an unsafe act (finest grain) |

The admission rule itself is expressed twice as pure twins — `laneadmit.Decide` and
`regionadmit.Decide` — applying one contract strongest-rule-first: an **exclusive lane runs alone**;
a **named lane serializes even on disjoint trees**; otherwise **tree geometry decides**
(`dispatchorder.TreesOverlap`: an empty tree is unknown blast-radius and collides conservatively;
`**`/`**/*` overlap everything; else prefix-containment). The taxonomy those twins read —
~200 lanes, each mapped to a canonical tree glob (`internal/<lane>/**`), with an `exclusive` set and
an `autopick` ladder — is declared once in `dos.toml` and read by two dependency-free TOML scanners.

The fan-out pricer `dispatchorder.Plan` is the load-bearing generalization already present in-tree.
Its `Candidate` carries exactly three claim fields — **`Lane`, `Tree`, `Mode`** (exclusive/shared) —
plus recency/priority/generation/blocked-by bookkeeping. `Plan` collapses same-key duplicates to the
freshest, prices the pre-launch collision graph (`collisionOf`: shared/shared may overlap, any
exclusive participant must be tree-disjoint, same non-empty lane always collides), computes the
`maxSafeSet` (exact DFS ≤24, greedy above), and emits `RepartitionAdvice` telling a held worker to
**narrow its tree or declare its scope** before it can re-enter a wave. Every layer refuses with the
same closed tokens — `COLLISION_RISK`, `INTENT_COLLISION`, `LEASE_CONTENDED`, `POLICY_BLOCK`,
`SELF_MODIFY`, `EGRESS_BLOCK`, `DEFAULT_DENY`.

Read `dispatchorder.Candidate` with fresh eyes and the shape is unmistakable: it is a **resource
request**. `Lane` is a resource class, `Tree` is the address range, `Mode` is the lock discipline,
and `Plan` is a scheduler that admits the disjoint set and serializes the rest. Nothing about that
algebra is specific to files.

## 3. Disaggregation is that algebra at compute granularity — `[SHIPPED]` parts, `[SEAM-ONLY]` unification

Every disaggregation axis fak has built is, structurally, a claim over a compute region with a
disjointness test — but each one currently ships its **own ad-hoc partitioner** instead of speaking
the claim-space:

| Disaggregation axis | fak mechanism | Claimed unit | Disjointness test | Speaks claim-space today? |
|---|---|---|---|---|
| Prefill / decode split | `modelengine/native_pd.go` `NativePDCluster` | a phase role (1 prefill, N decode) | role identity + prefix residency (`routeDecode`) | No — bespoke router `[SEAM]` |
| KV-span placement / tiering | `kvmmu` + `cachemeta` (HBM/DRAM/CXL/disk) | a `[from,len)` KV span on a tier | materialization tuple + tier capacity | No — bespoke `[SEAM]` |
| MoE expert parallel | `model/moe.go` `glmMoeEPFFN`, `expert_parallel.go` `ExpertParallelPlan` | an expert rank ∈ [1,N] | rank-local expert set | No — bespoke plan `[SEAM]` |
| Tensor parallel | `model/tensor_parallel.go` `TPPlan` | a weight shard on a rank | rank-ordered tiling | No — bespoke plan `[SEAM]` |
| Cross-process collective | `model/dist_collective.go` `DistComm` | a rank in a star process group | coordinator-rooted membership | No — bespoke `[SEAM]`, host-f32 not NCCL |
| Model residency / poly-model | `polymodel/polymodel.go` (`FAK_POLYMODEL`, default OFF) | a model resident in one kernel | `Family` + `PrefixDigest` share test | No — bespoke `[SEAM]` |
| Per-aspect model routing | `modelroute/modelroute.go` `Route`/`Combine` | a model per request/tool-call/sub-query | first-match rule envelope | Partially — a *routing* algebra, not an *admission* one `[SHIPPED]` |

The pattern is exact: each row **claims a region, tests disjointness, and places or fails closed** —
the identical shape to `laneadmit.Decide` claiming a file tree. `ExpertParallelPlan` "fails closed on
ranks outside [1,NumExperts]" is `regionadmit` refusing an out-of-taxonomy lane. `routeDecode`
picking the best resident decode worker is cache-aware power-of-two routing — the compute twin of
`dispatchorder` picking the freshest non-colliding candidate. What was missing was never the
mechanisms; it was that **they did not share the admission kernel, the closed refusal vocabulary, or
the collision pricer.**

The **kernel half of that is now shipped** and the table's last column is what remains open. Since
#3268 and #3269, an expert-rank collision *is* priced as a `COLLISION_RISK` and a compute-region
overflow *does* emit `RepartitionAdvice` — via `dispatchorder.ComputeClaim` /
`ComputeClaimsContend` and the shared `computeadmit.Decide(Request, []Lease, Taxonomy)`, which
refuses an out-of-taxonomy rank band `POLICY_BLOCK`/`out_of_taxonomy` and a contended region
`COLLISION_RISK` + advice. What is still open is **adoption, not algebra**: each placer keeps its
own mechanism byte-identical (deliberately — that was the fence), so `routeDecode`,
`ExpertParallelPlan`, `TPPlan`, and the KV-tier placer do not yet *call* the shared `Decide` from
their own paths. `computeadmit` ships the request constructors that state each placer's claim
(`ExpertRanksRequest`/`ExpertTaxonomy`, `DecodeWorkerRequest`/`DecodeTaxonomy`); the call sites are
the `[SEAM-ONLY]` remainder.

## 4. The granularity axis: one primitive from a phone to a hyperscaler

Because the primitive is "claim a region, test disjointness, admit the safe set," it is **scale-free**.
The same `Plan`/`Decide` kernel describes:

- **A phone / IoT node** — 1 model, 1 device, a handful of lanes. The claim-space degenerates to
  "one exclusive claim at a time"; the pricer is a no-op but the *vocabulary and refusal discipline
  are identical*. `[SHIPPED]` (this is just today's single-box default).
- **A workstation** — poly-model residency, prefill shared across a few co-resident models, decode
  one at a time (`polymodel-prefill-share-plan.md`). Tens of claimable units. `[SEAM-ONLY]`.
- **A small cluster** — native P/D over `DistComm`, EP-for-MoE sharded across ranks. Hundreds of
  claimable units (phases × experts × spans). `[SEAM-ONLY]`, host-f32 today.
- **A hyperscaler** — DP × TP × PP × EP over a real device mesh + NCCL/RDMA. Thousands of claimable
  units. `[GAP]` — the device backend behind the collective seam is greenfield (`THROUGHPUT-TRUST`
  §4: no `ncclAllReduce` in-tree, `cudaSetDevice(0)` hardcoded).

The point is not that fak runs at hyperscale today (it does not). It is that **the partition algebra
does not change across those four rows** — only the number of claimable units and the backend behind
the collective seam do. That invariance is the asset. A control plane that already prices file-tree
collisions for an agent fleet is the *same* control plane that should price expert-rank collisions
for a device mesh; building it twice is the waste this note names.

## 5. The split point: tool-call-as-syscall + route-before-adjudicate — `[SHIPPED]` seam

Where does a claim acquire its compute placement? At the **syscall seam**. `modelroute` establishes a
load-bearing ordering contract — **route before adjudicate**: choose the model/engine and write it to
`abi.ToolCall.Engine` *before* `Kernel.Submit` adjudicates and enqueues, and an ensemble expands to N
independently-adjudicated calls (`modelroute.Combine` over first/vote/best-of/concat). `abi`'s frozen
`Submit`/`Reap` async spine already carries `SubmissionHandle{Seq, Queue, Opaque}` — "which completion
queue (multi-engine / multi-queue)" — and `Ref{Taint, Scope}` with `ShareScope = Agent|Fleet|Tenant`.

That is the seam where the two planes meet: a tool call is a **claim** (it wants a model, an engine, a
queue, a share-scope), and `Submit` is where an admission kernel prices that claim against the
live compute leases exactly as `dispatchorder` prices a work-unit against live file leases. As of
#3269 that wire exists: `computeadmit.SubmitAdmitter` is an `abi.Adjudicator` that resolves a call's
compute claim (explicit `compute_*` `Meta` keys, else the bound `Engine`→region route) and refuses a
contending one `COLLISION_RISK` inside the `Submit` fold, paired with the `abi.ResultAdmitter` half
that frees the region when the holder's call **reaps** — so two co-targeted ensemble members
serialize and hand off with no scheduler and no queue. `[SHIPPED]` as a decision, exercised against a
real `kernel.New`/`Submit`/`Reap`; `[SEAM-ONLY]` as exposure, because both are driver-blind registry
seams (`internal/kernel` may never import the leaf) and **no production wiring layer registers the
gate yet** — installing it is one `RegisterAdjudicator` + `RegisterResultAdmitter` pair (§6).

## 6. The gap, and which rungs of it have since closed (honestly marked)

The mechanisms exist on both planes; the **unification** was the gap. Two of its four rungs have since
landed. Retagged against the tree:

- **`[SHIPPED]` A compute-resource axis on the claim** (#3268, closed). `dispatchorder.Candidate` no
  longer claims `Lane`/`Tree`/`Mode` only: `ComputeClaim{Class, Range, Mode}` states "prefill role on
  device 0" or "expert ranks 4–7", and `ComputeClaimsContend`/`RangeWithin` price its collision through
  the same `collisionOf`/`maxSafeSet`/`RepartitionAdvice` machinery a file tree gets.
- **`[SHIPPED]` kernel / `[SEAM-ONLY]` adoption — one admission kernel over both** (#3269).
  `internal/computeadmit.Decide(Request, []Lease, Taxonomy)` is the compute-plane twin of
  `laneadmit`/`regionadmit.Decide`: pure, strongest-rule-first, refusing `POLICY_BLOCK`/`out_of_taxonomy`
  for a claim outside its class's declared address space (`ExpertParallelPlan`'s "ranks outside [1,N]"
  fail-closed in shared form) and `COLLISION_RISK` + `RepartitionAdvice` for a contended region. The
  *`[SEAM-ONLY]`* remainder is adoption: by design the placers' mechanisms stay byte-identical, so
  `routeDecode`, `ExpertParallelPlan`, `TPPlan`, and the KV-tier placer still decide privately and do
  not yet call `Decide` from their own paths.
- **`[SHIPPED]` decision / `[SEAM-ONLY]` install — contention pricing at `Submit`** (#3269).
  `computeadmit.SubmitAdmitter` consults live compute leases in the `Kernel.Submit` adjudication fold
  and frees the region in the `Kernel.Reap` result-admission fold, so two ensemble members targeting the
  same over-subscribed device serialize and hand off exactly as two agents on one tree do. Decision
  only — no scheduler, no preemption, no queue. Claim-less calls defer untouched and the release half is
  no-opinion, which is what makes installing it additive. Not yet installed by any production wiring
  layer (§5).
- **`[SEAM-ONLY]` The device backend under the collective seam** (unchanged from `THROUGHPUT-TRUST` §4):
  `DistComm` is real cross-process host-f32; NCCL/RDMA is greenfield and hardware-gated. Any hyperscaler
  row is gated on this regardless of the unification.

Closing these did **not** require new inference kernels — the P/D, EP, TP, KV-tier mechanisms were all
already present. It required teaching the *existing arbiter* to claim a compute region (#3268, done) and
giving the *existing compute partitioners* a shared answer to reach for (#3269, done). What remains is
the last mile of both: the placers calling that answer, and a wiring layer installing the `Submit` gate.

## 7. Honest fences

- **Asserts no benchmark number.** Nothing here claims tokens/sec, goodput, or a parity figure. The
  parity-must-be-measured rule (#44 / `THROUGHPUT-TRUST` §4) stands: no throughput claim without an S2
  same-hardware run artifact.
- **"Claim-space", "resource request", "scheduler" are borrowed OS/distributed terms.** The scope here
  is fak's admission *decision* algebra, not a kernel scheduler with preemption/quanta. Same
  borrow-the-term / disclaim-the-scope discipline as `ReduceAllReduce`.
- **The unification began as a *design judgment this note recommended*; two rungs of it are now a fact
  the tree states.** `internal/computeadmit` asserts exactly "one admission kernel over files and
  compute" — but only as a **decision algebra**, and only where something asks it. The placers still
  decide privately, nothing installs the `Submit` gate in a shipped process, and §3's "same algebra" for
  the rows that have not been retagged remains an observation about shape.
- **Single-box today.** Every compute-plane row above `[SHIPPED]` is CPU-ref / host-f32 / single-process
  unless a real device mesh exists; the multi-node and hyperscaler rows are `[SEAM-ONLY]`/`[GAP]`.

## 8. What this reframes (relation to existing epics)

This note is connective tissue, not a new program. It re-reads the open serving/orchestration epics
through one lens:

- **#50 (dual-track serving)** builds the compute-plane disaggregation mechanisms (P/D, paged KV,
  scheduler, TP/EP). This note says: as those land, make them *claim-space citizens* rather than private
  routers, so the control plane already shipped for agents governs them for free.
- **#639 (MPI-shaped fleet comm)** already names the coordination boundaries (`abi.ShareScope`,
  `Kernel.Submit`/`Reap`, `modelroute.Combine`) as the group/collective vocabulary. The claim-space
  unification is the *admission* dual of that *communication* vocabulary — who may hold a region, not
  just who may message whom.
- **#1911 (agentic-first scheduling)** argues the sharing gains come from scheduling *structure*. The
  claim-space is that structure made explicit: a schedulable unit is a priced claim.
- **#748 (agent-OS process model)** is the OS framing; this note supplies its memory-management dual —
  the claim-space is the address-space allocator to #748's process table.
- **#637 / `THROUGHPUT-TRUST`** established that throughput and trust share one *serving* spine. This
  note extends the same "one spine, two products" logic up into the *control* plane: the agent
  arbiter and the compute scheduler are one admission spine, two granularities.

## 9. Non-goals / non-claims

- Not proposing to rewrite `NativePDCluster`, `ExpertParallelPlan`, or `TPPlan` — proposing they answer
  through a shared admission kernel, additively.
- Not claiming fak disaggregates at hyperscale, or beats any engine, today. The device backend is
  greenfield (§6) and no number is asserted.
- Not a decision doc. It records a through-line and the gap; the concrete "add a compute axis to the
  claim" work is tracked under **epic #3259** — children #3268 the pricer axis (**landed**), #3269 the
  shared admission kernel + `Submit`-seam pricing (**landed**, `internal/computeadmit`), #3270 the
  compute-claim taxonomy, #3271 the polymodel fusion split.
