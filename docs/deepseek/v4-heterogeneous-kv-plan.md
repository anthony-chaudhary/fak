---
title: "DeepSeek V4 heterogeneous KV + on-disk prefix-reuse — prototype/design plan"
description: >
  Design plan (issue #3017, epic #3006) for an internal KV layout that represents
  V4's four concurrent KV objects — compressed CSA blocks, compressed HCA blocks,
  uncompressed tail state, and an SWA state cache — plus a disk-backed prefix-reuse
  tier with tail recompute. Maps each requirement onto a verified fak seam
  (path:line), specifies the three SWA disk-cache policies, the content-address /
  invalidation rules, and a weight-free synthetic benchmark fixture. Every estimate
  is labelled MODELED or host-gated; no number here is WITNESSED.
---

# DeepSeek V4 heterogeneous KV + on-disk prefix-reuse — prototype/design plan

> **Issue #3017** under parent epic **#3006** (native DeepSeek-V4 kernel track).
> This is a **design + fixture plan only.** No production KV path is rewritten by this
> note, no 1M-token forward pass is run, and no provider prompt-cache counter is
> laundered into a fak-owned KV-reuse number. Every current-state claim is witnessed
> against the exact `path:line` cited (read 2026-07-09 on `main`); every forward-looking
> quantity is labelled **MODELED** or **host-gated**, never WITNESSED.
>
> **Sibling plan notes** (same epic, resolving adjacent facets — read together):
> - `docs/notes/DEEPSEEK-V4-KV-LAYOUT-PLAN-2026-07-08.md` — the seam-map companion to this doc (#3017).
> - `docs/notes/DEEPSEEK-V4-ATTENTION-SEAM-MAP-2026-07-08.md` — the sparse-attention kernel seam map (#3016).
> - `docs/notes/DEEPSEEK-V4-FP4-QUANT-PLAN-2026-07-08.md` — weight precision (FP4/FP8).
> - `docs/notes/DEEPSEEK-V4-MOE-DISPATCH-BASELINE-2026-07-08.md` — MoE dispatch baseline.
> - `internal/gateway/deepseek_budget.go` — the deterministic long-context memory/compute budget (#3015).

## 1. Thesis — V4's cache is a heterogeneous plane, not an append-only buffer

fak's cache path today treats KV as a **dense, per-layer, append-only** buffer under the
standard PagedAttention assumption (one uniform block kind, positionally contiguous). The V4
technical report states its **hybrid attention violates the assumptions behind PagedAttention**
and requires **co-designing the KV layout with the sparse-attention kernels** rather than
bolting a cache onto a fixed kernel. Concretely, one V4 forward pass materialises **four
distinct KV objects at once**:

1. a **classical cache** holding the compressed **CSA** (Compressed Sparse Attention) KV blocks,
2. a **classical cache** holding the compressed **HCA** (Highly-Compressed Attention) KV blocks,
3. an **uncompressed tail** of the most-recent tokens (full-fidelity K/V), and
4. an **SWA** (Sliding-Window Attention) **state cache** for the window side branch.

The design target is therefore a **heterogeneous KV plane**: one owner over four typed
sub-caches, a **content-address discipline** so each compression tier is a distinct addressable
entry, a **disk tier** for compressed prefix reuse, and an explicit policy for how the SWA
branch is (or is not) persisted. The plane is assembled from seams fak **already has**, plus one
new plane object and one new compression layout — the gaps are named distinctly in §3.

## 2. V4 KV facts that drive the layout (issue grounding)

Sourced from the DeepSeek V4 technical report (arxiv 2606.19348v1, per the #3017 grounding) and
the V4 architecture axes carried in the #3016 attention seam map. Architecture constants (CSA
rate 4, HCA rate 128, SWA window 128, layer count) are **SOURCE_DOCUMENTED** per the report; all
derived storage/compute quantities below are **MODELED**.

- **Classical cache** stores the compressed CSA (rate-4) and HCA (rate-128) KV blocks.
- **State cache** stores the SWA (window-128) side state plus the uncompressed most-recent tail.
- **On-disk prefix reuse** stores the compressed CSA/HCA KV entries by content address; the SWA
  branch is governed by one of **three disk policies** (§4): *full caching*, *periodic
  checkpointing*, or *zero SWA caching* (recompute the tail on a hit). These trade disk storage
  and write amplification against tail-recompute compute.

## 3. Layout spec — V4 KV object → fak seam (verified `path:line`) or named gap

| V4 KV object / requirement | Nearest verified fak seam (`path:line`) | Fit / named gap |
|---|---|---|
| **Pluggable per-family KV layout** | `internal/model/kvlayout.go:28` `kvLayout` interface; `:53` `standardKVLayout`; `:98` `mlaKVLayout` | **Extension point.** CSA ≈ today's `mlaKVLayout` at rate 4; HCA is a **second impl** (gap). |
| **Compressed CSA KV block** (rate 4) | `internal/model/kvlayout.go:104` `mlaKVLayout.cacheStride`, `:114` `reconstructKV`; write side `:151` `mlaProject` | **Fit.** MLA already stores a low-rank latent row (`cacheStride = KVLatentDim + RopeDim`, `kvlayout.go:104`) and reconstructs per-head K/V; CSA is this discipline at a rate-4 compression ratio. |
| **Compressed HCA KV block** (rate 128) | *No seam yet* — proposed new `kvLayout` sibling `hcaKVLayout` | **GAP #1.** An aggressive rate-128 compression layout does not exist. Fence it; do not claim reuse of MLA numbers for it. |
| **Uncompressed tail state** | `internal/model/kvcache.go:11` `KVCache` (dense `K`/`V`; pre-RoPE `Kraw` at `:14`) | **Fit.** The existing dense cache *is* the tail. `Kraw` (pre-RoPE, `kvcache.go:14`) lets a repositioned span re-RoPE in a single rotation — the property the disk tier depends on. |
| **SWA state cache** (window 128) | `internal/model/swa.go:25` `MaxWindow`, `:48` `TrimToWindow`; per-layer bound `internal/model/config.go:827` `windowForLayer` | **PARTIAL / GAP #2.** `TrimToWindow` is bit-exact vs the full-window decode, but today it trims the *whole* cache to one window. V4 needs SWA as a *concurrent side branch* that coexists with the classical CSA/HCA caches, not a whole-cache trim. |
| **Mid-span eviction of a compressed cache** (bit-exact) | `internal/model/kvcache.go:94` `Evict`, `:158` `evictGLMDsa` | **Fit.** fak already evicts a span from the *middle* of a compressed GLM-DSA cache and re-RoPEs survivors from `Kraw` (`kvcache.go:130-150`); this is the primitive a compressed-prefix invalidation needs. |
| **Content-address / reuse key** | `internal/cachemeta/materialization.go:62` `MaterializationKey`, `:82` `Matches`, `:118` `MaterializationKeyOf`; token digest `internal/cachemeta/cachemeta.go:828` `DigestTokenIDs`; convention `cachemeta.go:102` `PositionPrefixAligned` | **Fit (extend).** The addressable-cache key (model + tokenizer + position mode + prefix digest) is the reuse identity; it must gain a **compression-tier axis** so a CSA span and an HCA span over the *same* tokens are distinct entries (§5). |
| **DSA index binding / reuse** | `internal/cachemeta/attention_index.go:16` `AttentionIndex`, `:58` `AttentionIndexBindingDigest`, `:143` `AttentionIndexLookup`, `:194` `AttentionIndexReferences` | **Fit.** The sparse-index artifact is already content-addressed and depends on a `ParentKV` span; a compressed-block reuse inherits the same causal/prefix-exact lookup discipline. |
| **Exact-span invalidation** | `internal/cachemeta/external_invalidation.go:33` `PlanExternalInvalidations`, `:116` `ExactSpanTargets`, `:164` `AttestInvalidations` | **Fit.** Poisoning a compressed span already lowers to an `ExactSpanTarget` (`external_invalidation.go:116`) and cascades to any dependent DSA `attention_index` (`:71`). |
| **On-disk prefix reuse (L3)** | `internal/l3kv/l3kv.go:52` `SpanStager`, `:98` `StageSpan`, `:124` `RestoreSpan`; byte source `internal/model/spanserialize.go:29` `SerializeSpan` | **Strong fit.** A durable, crash-safe, **digest-keyed** disk tier already exists with a fail-closed digest re-verify at restore. Compressed CSA/HCA disk entries stage through it. **Note the current fence:** `SerializeSpan` refuses a sidecar/hybrid cache (`spanserialize.go:33`), so the plane's serializer is **GAP #3** (below). |
| **Amplification / savings report** | `internal/cachevaluereport/audit.go` `FoldAudit` (savings/fidelity folding); markdown `internal/cachevaluereport/markdown.go` | **Fit.** The reporting harness exists; the §6 fixture feeds it storage/write/read-amplification rows. |
| **Longest-prefix hit/miss engine** | `internal/radixkv/binding.go` `BoundTree` / `Lookup`; `internal/radixkv/radixkv.go` | **Fit.** Token longest-prefix match under a binding key is the hit/miss engine the fixture models against. |

### Named gaps (do not paper over)

- **GAP #1 — HCA (rate-128) `kvLayout`.** No aggressive-compression layout exists. Proposed as a
  second `kvLayout` impl beside `mlaKVLayout`; shared with #3016.
- **GAP #2 — SWA as a concurrent side branch.** `swa.go` trims the whole cache; the plane needs
  SWA state to live *alongside* CSA/HCA, not replace them.
- **GAP #3 — heterogeneous-plane serializer.** `SerializeSpan` (`spanserialize.go:29`) supports
  only a plain softmax-KV cache and returns a typed error for sidecar/hybrid state
  (`spanserialize.go:33`). The plane needs a serializer that stages each typed sub-cache (and
  refuses fail-closed for a sub-cache it cannot serialize, mirroring the existing fence).
- **GAP #4 — `HeterogeneousKVPlane` object.** No object owns the four sub-caches together today;
  it is a proposed composition of the seams above.

### Proposed data-layout sketch (design, not code)

```
HeterogeneousKVPlane                      // proposed owner (GAP #4)
├── csa   : mlaKVLayout      (rate-4)     // internal/model/kvlayout.go:98  (fit)
├── hca   : hcaKVLayout      (rate-128)   // proposed second layout        (GAP #1)
├── swa   : window side-state (window-128)// internal/model/swa.go, side branch (GAP #2)
├── tail  : KVCache (dense K/V + Kraw)    // internal/model/kvcache.go:11   (fit)
└── addr  : MaterializationKey + tier-axis// internal/cachemeta/materialization.go:62 (extend)
```

Each sub-cache is addressed by a page-table tag naming its **kind** (`csa`/`hca`/`swa`/`tail`).
The report's warning is exactly that a flat page table cannot address heterogeneous block kinds
— the tag is what makes the plane legible to the sparse kernels the report co-designs against.

## 4. The three SWA disk-cache policies

Modeled as a pure decision over the disk tier (`internal/l3kv`), each with **distinct**
amplification math the synthetic fixture (§6) must keep separate — conflating them hides the very
tradeoff the ticket asks for. `W` = SWA window (128, SOURCE_DOCUMENTED), `N` = checkpoint stride
(tunable), `L` = layer count, `T` = context length. All cost columns are **MODELED**.

| Policy | On-disk write cost (MODELED) | On a prefix hit | Amplification character (MODELED) |
|---|---|---|---|
| **Full SWA caching** | persist SWA window state at every checkpoint | zero tail recompute | highest write + storage; lowest recompute compute |
| **Periodic checkpointing** | persist SWA state every `N` tokens | recompute tail from the nearest checkpoint (≤ `N` tokens × `L`) | balanced — the tunable middle; write ∝ `1/N`, recompute ∝ `N` |
| **Zero SWA caching** | store nothing for SWA | recompute the full window-`W` tail (`W` × `L`) | lowest storage/write; highest, bounded recompute (`W` is fixed at 128) |

**Tail-recompute math (MODELED, weight-free).** On a compressed-prefix hit, the classical
CSA/HCA blocks restore directly from disk (no recompute), but the SWA side branch and the
uncompressed tail may need reconstruction:

- *Full caching*: `recompute_tokens = 0`.
- *Periodic checkpointing*: `recompute_tokens = (hit_pos − nearest_checkpoint(hit_pos)) ≤ N`,
  bounded by `N`; expected `≈ N/2` under a uniform hit position.
- *Zero caching*: `recompute_tokens = min(W, hit_pos)` — bounded by the window `W=128` regardless
  of context length, which is the property that makes zero-caching viable at 1M context.

Because `W` is fixed at 128, zero-caching's worst-case recompute is **constant in context
length** — the fixture reports this as the headline tradeoff row, not a per-length blowup.

## 5. Content-address + invalidation rules (fak addressable-cache discipline)

The rules extend, not replace, fak's existing addressable-cache discipline.

**Content-address (reuse-key) rules.**

1. A compressed-block entry's identity is `MaterializationKey` (`materialization.go:62`) — model
   id + tokenizer id + position mode (`PositionPrefixAligned`, `cachemeta.go:102`) + prefix
   digest (`DigestTokenIDs`, `cachemeta.go:828`) — **plus a new `compression_tier` axis**
   (`csa_rate4` / `hca_rate128` / `tail_dense`). A CSA span and an HCA span over the *same*
   tokens are therefore **distinct addressable entries**; a hit never crosses tiers.
2. The disk tier keys bytes by **span digest** (`l3kv` `StageSpan(digest, …)`,
   `l3kv.go:98`), with a **fail-closed digest re-verify at restore** in the store — a corrupt or
   mismatched span is a typed FAULT, never a silent wrong hit (`l3kv.go:124-133`).
3. Prefix exactness is causal: reuse follows the `AttentionIndexLookup` rule that a non-causal
   candidate **faults by default** (`attention_index.go:147`), because prefix exactness depends
   on the index/compression being determined only by tokens at positions ≤ the query position.
4. Position discipline: compressed rows store their pre-RoPE form so a span restored at a
   different absolute position re-rotates in a **single** rotation (the `Kraw` invariant,
   `kvcache.go:14`, `spanserialize.go:20-22`) — composing two rotations drifts ~1e-6 and can flip
   a greedy token, so double-rotation is forbidden.

**Invalidation rules.**

1. Poisoning or refuting a compressed KV span lowers to an **exact-span** eviction directive
   (`PlanExternalInvalidations`, `external_invalidation.go:33`) projected to an `ExactSpanTarget`
   (`:116`); a directive with no named span yields an empty target set and the caller must
   **refuse** rather than "precisely evict nothing" (`external_invalidation.go:116-133`).
2. **Cascade to dependents:** any DSA `attention_index` whose `ParentKV` references the poisoned
   span is invalidated with it (`AttentionIndexReferences`, `attention_index.go:194`; wired in
   `PlanExternalInvalidations` at `external_invalidation.go:71`).
3. **Cascade across tiers (new rule):** invalidating a CSA/HCA span at one compression tier must
   also invalidate the *same-token* entries at the other compression tiers **and** any SWA
   checkpoint or disk-staged tail derived from that prefix — because all are derivations of the
   same poisoned tokens. This is the one invalidation edge the current graph does not yet encode
   (it is content-address-adjacent but tier-crossing); named here as **GAP #5** for the follow-on.
4. Degradation must be **named, not silent:** an engine without an exact-span API falls back to
   `whole_prefix_cache` scope only as an attested `KVEvictionAttestation.Degraded` receipt
   (`external_invalidation.go:135-173`), never a quiet coarse reset.

## 6. Synthetic benchmark fixture (the witness)

A **pure, weight-free** Go test — **no model load, no 1M-token serve, no GPU** — that exercises
the prefix hit/miss decision and the tail-recompute math directly. It is the ticket's acceptance
witness and belongs to a follow-on implementation ticket, not landed by this note.

**What it computes (all MODELED, deterministic, no `time`/`rand`):**

1. **Block accounting.** For CSA (rate-4) and HCA (rate-128), compute bytes/token/layer from the
   documented compression ratios and validate the compressed-vs-dense ratio **fails closed** on a
   mismatch (mirroring the field-lock discipline in `deepseekbench.RequiredFields`).
2. **Prefix hit/miss.** Drive a longest-prefix lookup (against the `radixkv` engine shape) over a
   synthetic token stream: assert that (a) a matching-tier prefix HITs, (b) a cross-tier request
   over the same tokens MISSes (rule §5.1), and (c) a non-causal candidate FAULTs (rule §5.3).
3. **Tail-recompute math.** For each of the three SWA policies (§4), compute `recompute_tokens`
   for a set of hit positions and assert the bounds: `0` (full), `≤N` (periodic), `min(W,hit_pos)`
   (zero) — the zero-policy row proving recompute is **constant in context length**.
4. **Amplification report.** Emit **storage / write / read** amplification rows for **128K, 512K,
   and 1M** contexts, one row per SWA policy, folded through
   `internal/cachevaluereport` (`audit.go` `FoldAudit`) so the tradeoff is a table, not prose.
   Every row carries a `MODELED` provenance label (the closed-vocabulary discipline of
   `internal/gateway/deepseek_budget.go`: `SOURCE_DOCUMENTED` / `PAPER_CLAIMED` / `MODELED` /
   `WITNESSED`), and the fixture emits **no** `WITNESSED` row — a witnessed row can only come from
   the #3013/#3014 serving telemetry.

**Package placement (open, flagged for operator input — not silently chosen).** Candidate homes:
an extension of `internal/l3kv` + `internal/cachemeta`; or a **new leaf** `internal/deepseekv4kv`
owning the plane + fixture and importing the seams above. Recommendation: the new leaf, so the
plane composition is testable in isolation immune to concurrent shared-tree WIP — but recorded as
a recommendation for the follow-on ticket, not a decision this design note makes.

## 7. Honest fences (what is NOT decided or built)

- **No `HeterogeneousKVPlane` object exists** (GAP #4) — proposed composition of real seams.
- **No HCA (rate-128) layout exists** (GAP #1) — proposed second `kvLayout`.
- **SWA is whole-cache trim today** (GAP #2), not a concurrent side branch.
- **No heterogeneous-plane serializer exists** (GAP #3); `SerializeSpan` refuses sidecar/hybrid
  state fail-closed (`spanserialize.go:33`).
- **The cross-tier invalidation edge is unencoded** (GAP #5).
- **No V4 disk fixture / amplification report is landed** — §6 is its specification, not its code.
- **No provider prompt-cache counter is reused** as fak-owned KV reuse (explicit out-of-scope).
- **Every quantity in §2/§4/§6 is MODELED**; anything needing a GPU or a tuned serving engine is
  **host-gated** and yields no witnessed number here.

## 8. Next rungs

1. Land the **weight-free block-accounting + amplification fixture** (§6) under the agreed package.
2. Propose the **HCA `kvLayout`** leaf (GAP #1; shared with #3016).
3. Extend `MaterializationKey` with the **`compression_tier`** axis (§5.1).
4. Encode the **cross-tier invalidation edge** (GAP #5).
5. Add the **heterogeneous-plane serializer** over `l3kv.SpanStager` (GAP #3), fail-closed per sub-cache.
6. Pick an SWA disk policy per deployment and record *why* in the follow-on implementation ticket.

## Acceptance mapping

The issue's acceptance is: *layout spec + the 3 SWA policies + content-address/invalidation rules
+ synthetic fixture description; mark any number MODELED not WITNESSED.*

| Acceptance / witness criterion (issue #3017) | Where satisfied |
|---|---|
| **Layout spec** representing compressed CSA, compressed HCA, uncompressed tail, SWA state cache, disk-backed prefix hits w/ tail recompute | §1 (four-object thesis), §3 (object→seam table + gaps + data-layout sketch) |
| V4 hybrid attention **violates PagedAttention** / co-design KV layout with sparse kernels | §1 (thesis) and §3 (page-table kind-tag rationale; `kvLayout` extension point `kvlayout.go:28`) |
| KV layout = **classical cache** (CSA/HCA) + **state cache** (SWA + uncompressed tail) | §1, §2, §3 (classical rows: CSA `kvlayout.go:114`, HCA gap; state rows: SWA `swa.go`, tail `kvcache.go:11`) |
| **Three SWA disk-cache policies**: full caching / periodic checkpointing / zero SWA caching | §4 (policy table + tail-recompute math) |
| **Content-address rules** compatible with fak's addressable-cache discipline | §5 content-address rules (`MaterializationKey` `materialization.go:62`, `DigestTokenIDs` `cachemeta.go:828`, digest-keyed `l3kv`) |
| **Invalidation rules** | §5 invalidation rules (`PlanExternalInvalidations` `external_invalidation.go:33`, `ExactSpanTargets` `:116`, `AttentionIndexReferences` `:194`, named GAP #5) |
| **Synthetic benchmark fixture** exercising prefix hit/miss + tail-recompute math **without loading the model** | §6 (weight-free fixture: block accounting, hit/miss, recompute bounds, amplification report; no model load/GPU) |
| **Mark any number MODELED not WITNESSED** | §2/§4/§6 label every quantity MODELED; §7 fence states no WITNESSED row is emitted; provenance vocabulary reused from `deepseek_budget.go` |
| Cross-link **parent epic #3006** and **sibling docs/deepseek plan notes** | Header banner (epic #3006 + five sibling notes) |

## Sources (researched July 2026)

- DeepSeek V4 technical report — `https://arxiv.org/html/2606.19348v1` (hybrid attention violates
  PagedAttention assumptions; CSA/HCA/SWA structure; co-design of KV layout with sparse kernels).
  Also recorded as `DeepSeekBudgetSourceV4Paper` in `internal/gateway/deepseek_budget.go:95`.
- DeepSeek V4 Pro model card — `https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro`
  (parameter counts, 1M context ceiling, FP4/FP8 mixed precision). Recorded as
  `DeepSeekBudgetSourceV4ProCard` in `internal/gateway/deepseek_budget.go:96`.
- SGLang RadixAttention / prefix-cache prior art — `https://github.com/sgl-project/sglang`
  (longest-prefix radix reuse; the hit/miss discipline `internal/radixkv` mirrors).
- vLLM PagedAttention — `https://github.com/vllm-project/vllm` (the flat-page assumption V4's
  hybrid attention is reported to violate; the seam `internal/model/pagedkv.go` implements).
