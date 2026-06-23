# Poly-model serving — host many, share the prefill, decode one

> **Design decision doc** for the multi-model axis of in-kernel serving: hosting
> **tens of models in one kernel**, sharing/amortizing the **prefill**, and
> serializing **decode to a single lane** — plus the cache-led, next-generation
> **multi-token-prediction (MTP)** that the co-resident models unlock. This is the
> *orthogonal* axis to the disaggregated-serving epic (#50): that plan scales **one
> model across many nodes**; this one packs **many models onto one kernel**.
>
> **Scope:** design + one shipped deterministic core (`internal/polymodel`). The
> GPU/engine wiring (real multi-model residency on a backend, the verify pass, the
> bit-exact rollback call, cross-model prefill share) is explicitly sequenced, not
> claimed here.
>
> **Provenance:** every `[SHIPPED]/[PARTIAL]/[SEAM-ONLY]/[GAP]` mark carries a
> `file:line` pointer verified against the working tree on 2026-06-22. Line numbers
> drift; re-anchor with the `rg` snippets in [§9](#9-how-to-re-verify). **No
> benchmark number is asserted** — the speedup model in [§5](#5-next-gen-mtp--the-co-resident-models-are-the-speculation-ensemble)
> is closed-form arithmetic, gated on a measured run before any tokens/sec claim.

## 1. TL;DR — the decision

1. **Host many, decode one.** Keep tens of models *prefill-warm* in one kernel, but
   let only **one model decode at a time**. This is not a compromise — it falls
   straight out of the hardware: prefill and decode have opposite bottlenecks.
2. **The asymmetry is the whole design.** Prefill is **compute-bound** (one weight
   stream amortized over every prompt token — a batched GEMM), so prefilling many
   models is parallelizable and interleavable. Decode is **HBM-bandwidth-bound**
   (each token streams the model's *entire* weight set from memory), so two models
   decoding at once need 2× memory bandwidth against a fixed bus. Residency is
   cheap (capacity); decode is scarce (bandwidth). So you host many and decode one.
3. **The cache is the lever — twice.** (a) fak's prefix cache already eliminates
   redundant prefill *for one model*; the poly-model extension shares prefill
   *across* models (the radix tree is already model-agnostic in structure). (b) The
   bit-exact KV `Clone`/`Evict` make speculative MTP *correct and cheap*: clone is
   the fork, `Evict` is the exact rollback of rejected draft tokens.
4. **Next-gen MTP = the idle models become the speculation ensemble.** When model
   X decodes, the *other* warm models (or X's own cheaper quant) draft tokens that
   X **verifies in one prefill-shaped pass** — turning the spare residency into
   decode-lane throughput instead of dead weight.
5. **What lands now:** `internal/polymodel`, the deterministic, GPU-free core —
   residency pool, serial decode-lane scheduler, and the speculative-accept
   arithmetic — proven by a witness suite. Everything that touches a real backend
   is sequenced in [§7](#7-the-child-map--sequencing).

## 2. The asymmetry, in the code (not in the abstract)

The two regimes are already distinct execution paths in `internal/model`:

| Regime | Bottleneck | How it runs today | Anchor |
|---|---|---|---|
| **Prefill** | compute (FLOPs) | one batched GEMM sweeps **all P prompt tokens** through each weight row once — `matMulBatch`, `prefillBatched` | `internal/model/parallel.go:339`; `internal/model/prefill_batch.go:21` |
| **Decode** | HBM bandwidth | one token at a time; each step streams the whole weight set — `Session.Step` → `token`. The `Q4`/`Q4K` flags exist *only* to cut decode **bytes/token** | `internal/model/kv.go:1076`; `internal/model/kv.go:258` |

The decode path is *defined* by bandwidth: the entire reason int4 decode exists
(`internal/model/kv.go:258`, "int4 streams ~1.8× fewer bytes/token") is that decode
is memory-bound, not compute-bound. That single fact forces "decode one": you cannot
serialize fewer than one model's weight stream, and you cannot afford to run several
at once. Prefill, being compute-bound, is the half you *can* fan out and share.

There is even multi-user **static** batching already — `StepBatch` runs B users'
decode tokens in lockstep, reusing one weight stream B-fold (`internal/model/batch.go:1122`),
and its own comment says it "is exactly what a continuous-batching scheduler would
call" (`batch.go:1147`). But that is **one model, many users**. The poly-model lane is
**many models, time-shared** — a caller-side scheduler over the same primitive.

## 3. Host many, decode one — the architecture

```
   prefill-warm residency (capacity-bound: HOST MANY)
   ┌───────────────────────────────────────────────────────────┐
   │  model A   model B   model C   …   model N   (weights warm) │
   │   KV/prefix cache per model, shared where prefixes match    │
   └───────────────┬───────────────────────────────────────────┘
                   │  exactly one at a time
                   ▼
        ┌──────────────────────┐
        │   THE DECODE LANE     │  bandwidth-bound: DECODE ONE
        │  (one weight stream)  │  round-robin / priority / FCFS
        └──────────────────────┘
```

- **Residency** is bounded by a weight-byte **budget**, not by an architectural cap.
  Admit past the budget → evict the coldest **unpinned** model (LRU). A hot working
  set of a few models stays warm; a long tail pages in/out. (`polymodel.Pool`.)
- **The decode lane** is a single serial resource. A scheduler picks which resident
  model decodes next (priority, then FCFS), and round-robins a quantum of decode
  tokens so no model starves. The load-bearing invariant — *at most one model
  decodes per step* — is guaranteed by construction and asserted by the witness.
  (`polymodel.Schedule` / `polymodel.NextDecoder`.)
- **Why it pays:** `polymodel.DecodeBandwidthBytes` makes the cost explicit — decode
  traffic is `Σ tokens × WeightBytes` over the decoding model only. N models decoding
  concurrently would multiply that by N against a fixed bus; holding them *warm*
  costs zero bandwidth until they decode.

## 4. The cache is the lever — prefill sharing across models

fak already does intra-model prefill elimination: a computed prefix's KV is cloned
into the next session bit-identically, skipping its prefill (`internal/model/kv.go:123`
`Clone`, `:221` `SessionFromPrefix`). The poly-model question is **can two *different*
models share prefill?** The honest taxonomy:

| Sharing mode | What's shared | Status | Note |
|---|---|---|---|
| **Same model, same prefix** | the whole KV prefix | **[SHIPPED]** | `Clone`/`SessionFromPrefix`; RadixAttention reuse (`internal/radixkv/radixkv.go:72`) |
| **Cross-model, same prefix, structurally** | the radix *tree* can already index one prefix node regardless of model | **[SEAM-ONLY]** | the tree keys on **token-ids only** (`radixkv.go:51,96,173`); model identity lives in the *verdict* layer, not the structure |
| **Cross-model reuse, served** | actually serving model B from model A's KV | **[GAP]** | `cachemeta.CheckResidentClaim` requires an **exact ModelID match** before a HIT (`internal/cachemeta/manifest.go:107`) — a deliberate barrier, not a structural one |
| **Shared lower-layer bands** (adapter / distilled siblings) | the prefill of layers two models genuinely share | **[GAP]** | the per-layer `KVCache.K[][]float32` (`kv.go:20`) makes layer-banded sharing *expressible*; correctness needs the models to share those layers' weights + RoPE convention |

The key, non-obvious finding: **the prefix cache is model-agnostic at the data-structure
level and single-model only at the policy level.** Cross-model prefill share is a
*verdict-layer decision* (lift the exact-ModelID barrier for a declared-compatible
family), not a cache rewrite. That is the cheapest unlock on this axis and the first
real-backend rung ([§7](#7-the-child-map--sequencing)).

## 5. Next-gen MTP — the co-resident models are the speculation ensemble

Classic MTP / speculative decoding: a cheap **draft** proposes K tokens; the expensive
**target verifies all K in ONE forward pass**. That verify is **prefill-shaped**
(parallel over K tokens, compute-bound) — so the target pays *one* compute-bound pass
instead of *K* bandwidth-bound decode steps. On the single serial decode lane, that is
the whole speedup.

The poly-model twist: **the spare warm models are the drafters.** When X decodes, a
cheaper same-family resident (or X's own Q4 twin) drafts; X verifies. The spare
residency stops being idle capacity and becomes decode throughput.

Two things make this *correct and cheap* in fak specifically, and they are fak's
existing moat:

- **The fork is a `Clone`; the rollback is bit-exact.** Accepted draft tokens are
  kept; rejected ones are removed with `KVCache.Evict` (`internal/model/kv.go:60`) —
  byte-identical to never having drafted them (the `Kraw` single-rotation re-RoPE,
  proven by `TestKVQuarantineEqualsNeverSaw`, `internal/model/evict_test.go:86`).
  Page-shared engines (vLLM/SGLang) can only flush whole prefixes; they cannot evict
  a rejected speculative span exactly. The precision-policy path already does exactly
  this shape — speculate in Q8, inspect, **roll the KV back** to recompute in f32
  (`internal/model/kv.go:272`) — so the rollback substrate is shipped and exercised.
- **The ABI seam for it is frozen.** `abi.SpeculationContext` (`internal/abi/types.go:112`),
  `abi.TxnID` (`:126`), `abi.Outcome` with `OutcomeSquashed`/`OutcomeRolledBack`
  (`:128`), and the `abi.ProvisionalSink{Promote,Rollback}` interface (`:144`) ride
  every `ToolCall.Spec` (`:164`); `abi.RegisterProvisionalSink` (`internal/abi/registry.go:552`)
  and the reserved `OpsSpec` range (`registry.go:51`) are pre-allocated. **No
  implementation exists** (no `internal/spec`) — `ARCHITECTURE.md`'s bake-in
  walkthrough lists it as a *future* leaf, not a shipped one. `polymodel` is the
  policy/accounting brain a `ProvisionalSink` implementation will consult.

**The honest speedup model** (`polymodel.EffectiveTokensPerVerify`): with draft length
K and per-token acceptance probability `a`, one verify advances
`E = Σ_{i=0}^{K} a^i = (1 − a^{K+1})/(1 − a)` real tokens (Leviathan et al.). So the
lane yields ~E tokens per bandwidth-bound pass instead of 1 — *before* subtracting
draft cost. This is arithmetic, not a measurement; a real number needs the bench
harness (#44).

## 6. Capability honesty table

Legend: **[SHIPPED]** real & proven · **[PARTIAL]** real but incomplete ·
**[SEAM-ONLY]** interface/seam exists, no production impl · **[GAP]** absent.

| Capability | Status | Anchor |
|---|---|---|
| Bit-exact KV `Clone` (the speculative fork) | **[SHIPPED]** | `internal/model/kv.go:123` |
| Bit-exact middle-span `Evict` (the speculative rollback) | **[SHIPPED]** | `internal/model/kv.go:60`; proof `internal/model/evict_test.go:86` |
| Speculate→inspect→KV-rollback path (precision policy) | **[SHIPPED]** | `internal/model/kv.go:272`; `internal/model/dynamic_precision.go` |
| Prefill batched / decode-per-token asymmetry | **[SHIPPED]** | `internal/model/parallel.go:339`; `internal/model/kv.go:1076` |
| Multi-user static decode batch (`StepBatch`) | **[SHIPPED]** | `internal/model/batch.go:1122` |
| RadixAttention prefix reuse (model-agnostic tree) | **[SHIPPED]** | `internal/radixkv/radixkv.go:72,51,96` |
| Speculation/MTP ABI envelope (frozen) | **[SEAM-ONLY]** | `internal/abi/types.go:112,126,128,144`; `internal/abi/registry.go:552` |
| Multi-model residency pool + serial decode lane + accept core | **[SHIPPED]** | `internal/polymodel/` (this work) |
| Continuous-batching admit/evict loop | **[SEAM-ONLY]** | `internal/model/batch.go:1147` (comment); no scheduler |
| Multiple models hosted in one process | **[GAP]** | single `modelengine.Default`, one `*model.Model` (`internal/modelengine/modelengine.go:54,226`); gateway binds one planner (`internal/gateway/gateway.go:253`) |
| Multi-model weight residency / whole-model eviction on a backend | **[GAP]** | per-weight budget is single-model only (`internal/compute/vulkan.go:164`); `gpulease` is process-wide mutex (`internal/gpulease/lease.go:36`) |
| Cross-model prefill share (served) | **[GAP]** | exact-ModelID barrier (`internal/cachemeta/manifest.go:107`) |
| `ProvisionalSink` implementation / `internal/spec` | **[GAP]** | reserved range, no registrant |

## 7. The child map / sequencing

Ordered so each rung stands on a shipped one. None of these land in this doc.

1. **`internal/polymodel` — the deterministic core.** ✅ *this work.* Residency pool,
   serial decode lane, speculative-accept arithmetic, ensemble drafter selection.
   GPU-free, witness-proven. **Runnable witness:** `cmd/polymodelbench -selfcheck`
   hosts 10 synthetic models under a budget, drives the serial decode lane over real
   `model.Session` decode, and proves greedy speculative decode is **token-identical to
   plain greedy** even when an adversarial draft forces a rollback every round — the
   end-to-end proof that the rejected-draft `KVCache.Evict` rollback is bit-exact.
2. **Caller-side decode-lane scheduler over `StepBatch`/`Session`.** Wire `polymodel`
   to actually drive a multi-model decode loop (single backend) — the
   continuous-batching seam the `batch.go:1147` comment invites.
3. **Multi-model residency on a backend.** Lift the single-`Default` assumption
   (`modelengine.go`): a pool of `*model.Model`, weight-byte budget + LRU page-out,
   reusing `polymodel.Pool` as the policy.
4. **A `ProvisionalSink` + `internal/spec` implementation.** Implement the frozen ABI
   seam: drive `Promote`/`Rollback` against `KVCache.Evict`, with `polymodel.AcceptGreedy`
   as the accept decision.
5. **Cross-model prefill share (verdict-layer).** Lift the exact-ModelID barrier for a
   declared-compatible family in `cachemeta` — the cheap structural unlock from [§4].
6. **Bench harness numbers.** Gate every speedup claim on a measured run (#44): E vs
   draft cost, decode-lane utilization, residency hit-rate.

## 8. Non-goals

- **Not chasing raw single-GPU tokens/sec vs vLLM/SGLang.** This is a *capacity +
  governance* axis (host many cheaply, decode correctly), not a throughput contest.
- **No benchmark number is produced here.** [§5]'s `E` is closed-form; a tokens/sec
  claim needs the harness.
- **No kernel/core edit.** `polymodel` is a foundation leaf (imports nothing
  internal); the integration rungs plug into existing seams (`EngineDriver`,
  `ProvisionalSink`, `cachemeta`), never the frozen ABI.
- **Cross-architecture KV sharing stays out.** Two genuinely different model families
  cannot share KV bytes (different weights/RoPE/layout); §4's share is within a
  declared-compatible family only.

## 9. How to re-verify

```bash
rg -n 'func matMulBatch'                       internal/model/parallel.go      # prefill: compute-bound batched GEMM
rg -n 'func \(s \*Session\) Step|Q4 streams'   internal/model/kv.go            # decode: per-token bandwidth path
rg -n 'func \(c \*KVCache\) (Clone|Evict)'     internal/model/kv.go            # the fork + the bit-exact rollback
rg -n 'is exactly what a continuous-batching'  internal/model/batch.go         # the scheduler seam (SEAM-ONLY)
rg -n 'SpeculationContext|ProvisionalSink|TxnID' internal/abi/types.go         # frozen MTP envelope
rg -n 'func CheckResidentClaim'                internal/cachemeta/manifest.go  # the exact-ModelID barrier (cross-model GAP)
rg -n 'node.key|func \(t \*Tree\) walk'        internal/radixkv/radixkv.go     # model-agnostic prefix tree
go build ./internal/polymodel/ && go vet ./internal/polymodel/                 # the shipped core
```

---

*Sibling axis:* `docs/serving/dual-track-serving-plan.md` (one model, many nodes). ·
*Shipped core:* `internal/polymodel`.
