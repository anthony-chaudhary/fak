# Throughput parity and the trust/L3 moat share one spine — connecting tissue

Date: 2026-06-24
Status: synthesis note. Connects [`dual-track-serving-plan.md`](../serving/dual-track-serving-plan.md)
(the throughput / parity track) to [`L3-DISAGGREGATED-CACHE-REIMAGINED.md`](L3-DISAGGREGATED-CACHE-REIMAGINED.md)
(the trust / disaggregation track). Asserts no benchmark number. Tags every claim
`[SHIPPED]` / `[SEAM-ONLY]` / `[GAP]`. Where the adversarial verification pass did not
fully confirm a load-bearing claim, this note carries the correction inline rather
than the original framing.

> **Non-goal reconciliation (epic #637, acceptance item 1).** This epic **supersedes**
> #50's Track-B non-goal *"not chasing raw single-GPU throughput parity vs vLLM"*
> (`dual-track-serving-plan.md` §9). The non-goal is **retired in scope**: native
> multi-node throughput parity with vLLM/SGLang/TensorRT-LLM is now an explicit goal of
> this epic's throughput branch (S6b), sequenced *behind* the shared spine (S1–S5) so the
> parity spend never strands the trust option. This is the in-tree form of the acceptance
> item ("supersede the non-goal here with a cross-link"); the #50 GitHub body retains its
> Track-B wording until an operator edits it directly. Nothing here asserts a measured
> parity number — see §4 (parity stays UNMEASURED until S2 runs).

---

## 1. The crux

The two roadmaps look like two products — raw tokens/sec parity with vLLM/SGLang/TRT-LLM
on one side, provable-forgetting / governed-KV on the other — but the bottom half of each
plan is **the same code**. The reinforcement is not a slogan; it is concentrated in a
small set of seams that each path independently needs:

- A request **lifecycle seam** (admit → per-step decode → stream → reclaim). Throughput
  needs it for goodput; trust needs per-token streaming so adjudication can act *as a span
  materializes* instead of after the turn lands.
- A **KV-span byte form** and a **byte mover**. Throughput needs it to disaggregate prefill
  from decode (the Track-B native P/D data plane); trust needs the identical blob so an
  exact-span eviction can survive off-box on an L3 tier.
- A **paged KV allocator**. Throughput needs it for the VRAM/batch-size win; trust needs it
  to be the moment the bit-exact eviction moat either survives paging or is dropped.
- A **continuous-batching scheduler**. Throughput needs it as the single biggest single-node
  multiplier; trust needs it as the *only place governed two-class admission can live*.

And there is exactly **one genuine, load-bearing tension** under all of it:

> The bit-exact middle-span eviction moat needs fak to OWN pre-RoPE `Kraw` f32 rows in its
> own address space ([`internal/model/kvcache.go:14`](../../internal/model/kvcache.go),
> re-RoPE at [`:128`](../../internal/model/kvcache.go)). The *fastest* path to a throughput
> parity number is to RIDE vLLM/SGLang (Track A), which own the KV and expose only a coarse
> whole-prefix reset (`SupportsExactSpan=false`,
> [`internal/enginecache/enginecache.go:239`](../../internal/enginecache/enginecache.go)).
> So the move that buys throughput parity quickest is the move that degrades the trust moat
> to a whole-prefix flush on that same span. `[CONFIRMED]` by verification.

The escape hatch is real and is the whole reason the L3 epic exists: the moat **survives
in full over a fak-governed L3 tier holding fak-serialized spans** (trust-M5/M8), because
there fak still owns the `Kraw` rows even though the span is off-box. So the resolution is
not "pick one" — it is "ride engines for the compute-parity number, govern an L3 tier for
the moat, and never let a ridden-engine parity number be reported as a moat claim." Keeping
those two claims separable is itself a deliverable (see §4, honesty discipline).

---

## 2. The shared-spine investment thesis

**Build the KV span-serializer and the byte mover ONCE.** It is the highest-leverage
shared dollar in the portfolio, and the repo already routes both tracks through the same
metadata grammar above it — what is missing is the bytes, on both sides, identically.

### 2a. `KVCache → bytes` span serializer — `[GAP]`, one artifact both tracks need

Today a span has no portable byte form. `Evict` re-derives survivors from **owned** `Kraw`
pre-RoPE rows ([`kvcache.go:128`](../../internal/model/kvcache.go)) and `Clone` splices
in-process ([`kvcache.go:173`](../../internal/model/kvcache.go)). To move a span off-box you
must serialize `K`/`Kraw`/`V`/`pos` for a `[from,len)` range into a self-describing blob,
keyed by the materialization tuple (model / tokenizer / serializer / position-regime,
[`internal/cachemeta/materialization.go:62`](../../internal/cachemeta/materialization.go))
so a deserialized span **fails closed under a wrong regime**.

- Throughput unlock: this blob **is** the Track-B native P/D data-plane payload (#29,
  [`dual-track-serving-plan.md:199`](../serving/dual-track-serving-plan.md)).
- Trust unlock: this is the **same** serialization an L3 tier holds so exact-span Evict
  survives off-box (trust-M2). Same artifact, keyed the same way.

### 2b. `StageTransport` byte mover (TCP first; RDMA/UCX behind the same seam) — `[GAP]`

This is the universal gate. `cachemeta` names WHERE a span lives and HOW it shares zero-copy
(`ShareKind` rdma/cxl_hdm/dmabuf/mmap,
[`internal/cachemeta/hardware.go:50`](../../internal/cachemeta/hardware.go)) but emits no
bytes — `BytesMoved` is a *reported counter*, no path copies KV
([`internal/cachemeta/kvtransfer.go:54`](../../internal/cachemeta/kvtransfer.go)).
`xenginekv`'s `AttachArena` takes a shared-memory/CUDA-IPC buffer
([`arena.go:94`](../../internal/xenginekv/arena.go)) but the engine-specific transport that
maps a real KV region into the Arena is **explicitly the unbuilt remainder**
([`arena.go:35`](../../internal/xenginekv/arena.go)). `[CONFIRMED]` by verification: the
arena is a Go `[]byte` in-process stand-in; nothing moves KV over a wire anywhere in-tree.

- Throughput unlock: native prefill/decode disaggregation — the wire a P→D transfer rides.
- Trust unlock: **every** L3 governance milestone (trust M5–M9) is control-path-only *on top
  of this one mover*. Built once, both consume it.

> Honest correction the verification forced (claim `kv-transport-reinforces`, PARTIAL): the
> repo does **not** today assert "one transport, both paths." #29 (the KV→bytes serializer)
> appears in exactly one place; the L3 child-B `L3RegionBackend` instead cites the
> *network transport* under the poisoned tracker id #493, which both docs map to the
> **pipeline-parallel stage handoff** (#85/#30), a different transport than #29's KV-byte
> mover. So "one mover serves both" is a defensible *engineering judgment this note
> recommends*, not a fact the tree currently states. To make it true, child-B's dependency
> must be re-pointed from the PP-stage transport to the KV-byte serializer. Treat it as a
> decision (§5), not a done deal.

### 2c. The seams above the bytes are already shared and shipped — `[SHIPPED]` / `[SEAM-ONLY]`

- The `abi.KVBackend` factory seam ([`internal/abi/registry.go:633`](../../internal/abi/registry.go))
  is the per-session contract `kvmmu` enforces through with zero concrete-model dependency
  ([`internal/kvmmu/kvmmu.go:131`](../../internal/kvmmu/kvmmu.go)) — a remote tier attaches
  the *same way* the region/page-out backends do. `[SHIPPED]` (the seam), `[GAP]` (a remote
  impl).
- The payload-free disaggregation grammar — `ExactSpanTargets`
  ([`internal/cachemeta/external_invalidation.go:93`](../../internal/cachemeta/external_invalidation.go))
  — names a precise remote eviction without ever touching the data path. `[SHIPPED]`.
- The portable signed `DeletionCertificate`
  ([`internal/deletioncert/deletioncert.go:133`](../../internal/deletioncert/deletioncert.go))
  binds a bit-exact eviction to a hash-chained journal row. `[SHIPPED]`, self-signed only
  until the external anchor is populated ([`:90`](../../internal/deletioncert/deletioncert.go)).

---

## 3. Worst-regret-minimizing sequence

Front-load the entire shared spine (S1–S5); every dollar through S5 is double-counted. Then
fork into a **hardware-ungated trust branch** (S6a, keeps the moat option fully alive cheaply)
running in parallel with a **hardware-gated throughput branch** (S6b, calendar-dominated by
GPU server access, the most expensive narrowly-single-track work) sequenced last.

| Step | What | Path | Size | Owning child(ren) | Anchors |
|---|---|---|---|---|---|
| **S1** | Extend the `EngineDriver` seam from one-shot `Complete` to admit/step/stream/cancel, **reviewed against the native-scheduler shape, not just the adapter shape**. Thread `ctx` into the decode loop + incremental detokenizer so tokens flush live. | **both** | L | #46, #47, #48 | `EngineDriver` is `Complete`+`Caps` only at [`registry.go:590`](../../internal/abi/registry.go); the sole impl one-shots `sess.Generate` with `ctx` never consulted in decode ([`internal/modelengine/modelengine.go:139`](../../internal/modelengine/modelengine.go)); doc names this the #1 risk ([`dual-track-serving-plan.md:90`](../serving/dual-track-serving-plan.md)) |
| **S2** | Parity bench harness: drive identical Poisson load against fak AND a real vLLM/SGLang/TRT-LLM endpoint on the same hardware; emit TTFT/TPOT/goodput/queue-depth/KV-util. Net-new — `cmd/paritybench` measures capability/safety/cost A/B, a different axis. | **both** | L | #44, #43 | [`cmd/paritybench/main.go`](../../cmd/paritybench/main.go); [`dual-track-serving-plan.md:85`](../serving/dual-track-serving-plan.md) |
| **S2½** | Wire the fleet router into the gateway hot path: N-upstream dispatch + residency index + membership/health, so throughput dispatch and trust admission attach at the same layer. | **both** | L | #45, #41, #42 | Static N-upstream dispatch is now wired through `ReplicaRouter` + `fak serve --replica-base-url` ([`replica_router.go`](../../internal/gateway/replica_router.go)); the remaining gap is the residency index + health/drain/failover layer, so placement is still round-robin rather than KV/load-aware. |
| **S3** | `KVCache → bytes` span serializer keyed by the materialization tuple, fails closed under a wrong regime. | **both** | L | #29 | [`kvcache.go:128`](../../internal/model/kvcache.go), [`:173`](../../internal/model/kvcache.go); [`materialization.go:62`](../../internal/cachemeta/materialization.go) |
| **S4** | `StageTransport` byte mover — **TCP path first** (no special hardware, exercisable before GPU server), RDMA/UCX as a backend swap behind the same seam. | **both** | XL | #29 (KV bytes), #85/#30 (PP stage handoff) | [`arena.go:35`](../../internal/xenginekv/arena.go); [`hardware.go:50`](../../internal/cachemeta/hardware.go); [`kvtransfer.go:54`](../../internal/cachemeta/kvtransfer.go) |
| **S5** | Continuous-batching scheduler over `StepBatchActive` (admit/retire/rebuild-running-batch + admission/priority) **and** paged/block KV allocator that **proves the bit-exact `Evict` survives paging** (#33 gate). Sequenced together — the scheduler admits/preempts against the allocator. | **both** | XL | #36, #35, #34, #33 | [`batch_step.go:75`](../../internal/model/batch_step.go) (StepBatchActive, per-step GEMM), [`:135`](../../internal/model/batch_step.go) (GenerateBatch is STATIC fixed-B, no loop); [`kvcache.go:92`](../../internal/model/kvcache.go) (memmove-compacting Evict) |
| **S6a** | **Trust branch, hardware-UNGATED.** Asyncify `KVBackend.Prefill` (add residency-transfer methods, keep the local dense path byte-identical) → referee sidecar return-digest verify + durability-gated admission → `L3RegionBackend` behind the frozen Resolver/KVBackend seam → durability-tiered promotion → per-span `DeletionCertificate` over the pool. In-process L3 stub before a real external L3 cache. | **trust** | L (on top of spine) | #638 (unblock), #53 (+#55-#58) | [`registry.go:633`](../../internal/abi/registry.go); [`kvmmu.go:131`](../../internal/kvmmu/kvmmu.go); [`external_invalidation.go:93`](../../internal/cachemeta/external_invalidation.go); [`deletioncert.go:133`](../../internal/deletioncert/deletioncert.go) |
| **S6b** | **Throughput branch, hardware-GATED.** Real 2-GPU device `CollectiveBackend` (lift hardcoded `cudaSetDevice(0)`) → multi-process TP serving (`ForwardTP` per-rank) → network PP serve loop → native P/D split → **last**: GLM-5.2 MLA-aware TP + MoE expert-parallel (the decompositions `ForwardTP` fails closed on). | **throughput** | XL | #28, #30, #85, #274 | [`compute.go:350`](../../internal/compute/compute.go) (cpu-ref collectives); [`cuda_kernels.cu:52`](../../internal/compute/cuda_kernels.cu) (`cudaSetDevice(0)`); [`internal/model/tensor_parallel_forward.go:76`](../../internal/model/tensor_parallel_forward.go) (fails closed on MLA/DSA/MoE/quant/ParallelResidual/linear-attn) |

S2½ may land opportunistically once S1 lands, but **S1–S5 are the shared-spine gate before
any track-specific parity claim**. A native fak parity claim belongs after S6b only if S2 has
also produced a committed same-hardware run artifact; until then, §4 marks parity
UNMEASURED.

The logic of the order: every dollar through S5 is spent on code both tracks need, so under
unknown popularity nothing through S5 is regretted. S6a is cheap and hardware-ungated, so the
trust option stays fully alive without waiting on GPU server. S6b is the most expensive, most
narrowly-single-track, and most hardware-blocked work, so it is deferred until popularity and
hardware justify it.

---

## 4. Honest fences

**Greenfield, build-from-scratch:**

- **No device communicator.** No real NCCL/RCCL device API anywhere (`git grep
  ncclAllReduce|ncclComm|nccl.h` → nothing); `cudaSetDevice(0)` is hardcoded and is the only
  `cudaSetDevice` in `internal/` ([`cuda_kernels.cu:52`](../../internal/compute/cuda_kernels.cu)).
  Every multi-GPU rung (S6b) is gated on building this AND on real multi-GPU hardware to test it.
  - *Correction the verification forced (claim `no-nccl`, PARTIAL):* "genuinely no
    world-size / device-mesh / per-rank binding — true greenfield" is **overstated**. The
    tree already has compiled per-rank collective *contracts*
    ([`internal/compute/collective.go`](../../internal/compute/collective.go) cpu-ref
    AllReduce/AllGather/ReduceScatter/AllToAll), rank-ordered TP sharding
    ([`internal/model/tensor_parallel.go`](../../internal/model/tensor_parallel.go)
    LocalCollective/TPPlan), a `DistComm` process-group
    ([`internal/model/dist_collective.go`](../../internal/model/dist_collective.go), host-float32
    cross-process, explicitly NOT NCCL/multi-GPU), and a `StageTransport` seam — all deliberate
    single-box CPU-ref implementations built as the swap point. What is **truly greenfield is
    the NCCL/RDMA *device backend* and the cross-process *wire transport* behind those seams**,
    not the collective/TP/pipeline architecture, which exists and is bit-exact-tested on one box.

**Seam-only (the algebra exists, the system does not):**

- **No continuous-batching scheduler.** `StepBatch`/`StepBatchActive` are bit-exact per-step
  GEMM primitives; `GenerateBatch` is STATIC fixed-B with EOS re-feed
  ([`batch_step.go:135`](../../internal/model/batch_step.go)). The scheduler the comment says
  "*would* call" the primitive does not exist.
  - *Correction (claim `spine-shared-enginedriver-stepbatch`, PARTIAL):* the framing that the
    admit/step/stream/cancel lifecycle is *already* shared spine is **aspirational**.
    `EngineDriver` today is one-shot `Complete`+`Caps` ([`registry.go:590`](../../internal/abi/registry.go))
    with no step/stream/cancel; the unified lifecycle is future work (#46), and the
    `Admit`/`Evict` symbols in `registry.go` are on *other* interfaces. The seam and the
    primitive exist and are reused by static batching; the *lifecycle* is what S1/S5 build.
- **No paged KV.** Flat `[]float32` + memmove-compacting `Evict`
  ([`kvcache.go:11`,`:92`](../../internal/model/kvcache.go)). No BlockManager / block table /
  COW prefix share. The #33 gate (Evict survives paging) is unproven.
- **Streaming is synthesized post-adjudication** — TTFT == whole-turn until S1.
- **`KVBackend.Prefill` is synchronous-dense `[]float32`**
  ([`registry.go:635`](../../internal/abi/registry.go), inherited from
  [`internal/model/kv.go:386`](../../internal/model/kv.go)). A remote/async L3 backend cannot
  satisfy it as-is; S6a must add residency-transfer methods. `[CONFIRMED]`.
- **The external L3 cache is out-of-tree** (a separate workspace), not imported or built, and
  the repo guard refuses out-of-workspace writes — so child-A's "prove against a real external
  store" cannot be demonstrated from this workspace alone; the in-process stub is the honest
  first proof.
- **Recurrent/hybrid caches reject mid-span evict entirely** —
  `RecurrentEvictUnsupportedError` ([`kvcache.go` Evict path](../../internal/model/kvcache.go),
  Gated-DeltaNet linear-attention). The quarantine + deletion-cert moat **structurally does not
  exist** for those families. The flagship (753B GLM-5.2, MoE+MLA+DSA+quant) is partly exactly
  the architecture the moat cannot cover and the throughput TP path fails closed on
  ([`tensor_parallel_forward.go:76`](../../internal/model/tensor_parallel_forward.go)) — the two
  tracks value the same model for opposite reasons.

**The parity-must-be-measured rule (#44):** a throughput-parity claim is **MEASURED only by the
S2 harness**, never asserted from the algebra. Parity is the goodput-vs-latency *curve* within a
stated margin of the SOTA engine on the *exact* same hardware+model+quant, with the run artifact
committed. Until S2 exists and a run is committed, **every tokens/sec statement stays explicitly
UNMEASURED**, and three claims must never collapse into one: "we hit parity by orchestrating
SGLang" ≠ "native fak hit parity" ≠ "fak governs this engine's KV."

**Exactness oracle scoping:** row-parallel/MoE reductions reassociate (documented ~1e-6 drift),
so multi-node native output is bit-exact only vs a *shard-grouped* reference, not vs single-box.
Any "identical output" claim must name the right oracle per path or it overclaims against the
moat's `max|Δ|=0` premise.

---

## 5. Operator forks — 1 decided (epic #637, acceptance item 5), 4 still open

1. **One transport or two? — DECIDED: one KV-byte mover, both paths.** L3 child-B's
   *remote-KV* dependency should point at the KV-byte serializer (#29), so "one mover,
   both paths" is now the recorded epic decision, not just the §2b recommendation. The
   poisoned pre-migration id #493 resolves to the **PP stage-handoff** children #85/#30
   (per [`dual-track-serving-plan.md`](../serving/dual-track-serving-plan.md) §10 de-dup
   map) — those stay the pipeline-parallel hidden-state transport on their own seam; what
   child-B should consume is #29, the KV→bytes mover the throughput P/D path and the
   trust/L3 path both consume. The #53 child-B GitHub body keeps its inline #493 wording
   until an operator re-points it directly.

2. **RDMA vs UCX vs gRPC/TCP for the production byte mover.** S4 builds TCP first regardless
   (hardware-ungated, exercisable before GPU server). The fork is which *production* backend to invest
   the systems effort in behind the same `StageTransport` seam — RDMA (lowest latency, hardware-
   coupled), UCX (portable abstraction over RDMA/TCP), or gRPC (simplest, operationally cheapest,
   slowest). Decision can be deferred until after TCP proves the seam.

3. **Ride an external L3 cache vs `xenginekv`-in-tree for the first L3 proof.** The external L3
   cache is unreachable from this workspace (out-of-tree, guard-refused). The first credible L3
   governance demo can be an in-process `xenginekv` stub (fully under our control, no external
   store) or a real external-L3-cache bridge (proves "against a real external store" but needs
   cross-workspace access). Recommend stub-first to keep S6a hardware- and dependency-ungated.

4. **How hard to resource the NCCL/RDMA device substrate (S6b) given it is pure greenfield and
   hardware-blocked.** This is the most expensive, most narrowly-throughput-only, most calendar-
   dominated work. The worst-regret sequence defers it behind the entire shared spine and runs it
   in parallel with the hardware-ungated trust branch. The fork is the *pace*: a standing GPU server/GCP
   budget to push it continuously, vs. a demand-triggered build once a parity number is actually
   contested. Recommend demand-triggered — nothing in S6b strands the trust option, and S2/S5
   make the parity gap *measurable* before the spend.

5. **The load-bearing strategic fork: which claim is the product?** Riding SGLang gets a parity
   *number* fastest but degrades the moat on those spans; governing an L3 tier preserves the moat
   but is greenfield above S5. The honest answer is to keep both alive through S5 (cheap) and let
   the S2 harness + early-adopter demand pick — but the operator should decide *now* whether a
   ridden-engine parity number is allowed to be marketed before the native/governed path exists,
   because the ride path tempts collapsing three separable claims into one.
