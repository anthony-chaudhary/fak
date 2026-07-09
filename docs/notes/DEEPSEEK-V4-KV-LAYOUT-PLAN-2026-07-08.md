# DeepSeek V4 heterogeneous KV + on-disk prefix reuse — prototype plan

**2026-07-08.** Issue **#3017**, parent epic **#3006** (native DeepSeek-V4 kernel track).
**Design + fixture plan only** — no production KV rewrite lands here, no 1M-token serve is
run, and no provider cache counter is reused as fak-owned KV reuse. Current-state claims are
witnessed against the exact `path:line` cited (read 2026-07-08 on `main`). Companion to the
attention seam map (#3016) and the model-progress caching taxonomy
(`docs/notes/CONCEPT-MODEL-PROGRESS-CACHING-TAXONOMY-2026-07-07.md`).

## Thesis — V4's cache is a heterogeneous plane, not an append-only buffer

fak's cache story treats KV as a **dense, per-layer, append-only** buffer under normal
PagedAttention assumptions. The V4 report says its hybrid attention **violates the
assumptions behind PagedAttention** and requires **co-designing KV layout with the sparse
kernels**. Concretely, a single V4 forward pass produces *four* different KV objects at once:
a classical cache for **CSA/HCA**, a **state cache** for **SWA**, and an **uncompressed tail**
of the most recent tokens. So the design target is a **heterogeneous KV plane** with a
content-address discipline and a disk tier — assembled from seams fak already has, plus one
new plane object.

## The V4 KV facts that drive the layout (from the issue grounding)

Source: DeepSeek V4 technical report, https://arxiv.org/html/2606.19348v1 (per #3017 Grounding),
and the V4 Pro architecture numbers carried in #3016 (CSA rate 4, HCA rate 128, SWA window 128,
61 layers).

- **Classical cache** holds the compressed CSA (rate-4) and HCA (rate-128) KV blocks.
- **State cache** holds the SWA (window-128) side state plus uncompressed tail tokens.
- **On-disk prefix reuse** stores the compressed CSA/HCA KV entries; SWA is handled by one of
  **three disk policies** — *full caching*, *periodic checkpointing*, or *zero SWA caching*
  (recompute the tail on a hit). These trade storage for read/write/recompute amplification.

## Seam map — V4 KV requirement → fak seam (`path:line`) or proposed

| V4 requirement | Nearest fak seam (verified `path:line`) | Fit / gap |
|---|---|---|
| **Pluggable KV layout** (per model family) | `internal/model/kvlayout.go:28` `kvLayout` interface; `:53` `standardKVLayout`; `:98` `mlaKVLayout` | **The extension point.** CSA (rate-4) ≈ today's `mlaKVLayout`; HCA (rate-128) is a **second** impl (proposed). |
| **Compressed CSA KV block** | `internal/model/kvlayout.go:104` `mlaKVLayout.cacheStride`, `:114` `reconstructKV` | **Fit** — MLA already stores a compressed latent row and reconstructs K/V; CSA is this at rate 4. |
| **Compressed HCA KV block** (rate 128) | *No seam yet* — proposed new `kvLayout` sibling | **Gap.** Aggressive compression tier does not exist; fence it. |
| **Uncompressed tail state** | `internal/model/kvcache.go:11` `KVCache` (dense `K`/`V` + pre-RoPE `Kraw:14`) | **Fit** — the existing dense cache *is* the tail; the plane keeps it as one of four objects. |
| **SWA state cache** (window 128) | `internal/model/swa.go:25` `MaxWindow`, `:48` `TrimToWindow` | **Partial** — trim-to-window is bit-exact vs full-window, but today it trims the *whole* cache; V4 needs it as a *side* branch that coexists with the classical cache. |
| **Paged block allocation + COW share** | `internal/model/pagedkv.go:44` `PagedKVPool`, `:57` `NewPagedKVPool` | **Fit for the allocator**, but the report says V4 *violates* the flat-page assumption — the page table must address *heterogeneous* block kinds (compressed vs tail vs SWA), which is new. |
| **Mid-span eviction of compressed KV** (bit-exact) | `internal/model/kvcache.go:94` `Evict`, `:158` `evictGLMDsa` (compressed-KV compaction) | **Fit** — fak already evicts a span from the *middle* of a GLM-DSA compressed cache and keeps the rest correct; this is the primitive V4 prefix invalidation needs. |
| **Content-address / reuse key** | `internal/cachemeta/materialization.go:62` `MaterializationKey`, `:82` `Matches` | **Fit** — the addressable-cache discipline (model+tokenizer+prefix digest) is the reuse key; extend its axes to name the compression tier. |
| **Invalidation rules** | `internal/cachemeta/external_invalidation.go:33` `PlanExternalInvalidations`, `:116` `ExactSpanTargets` | **Fit** — exact-span invalidation directives already exist; a poisoned compressed span maps onto an `ExactSpanTarget`. |
| **Radix prefix hit/miss** | `internal/radixkv/binding.go:32` `BoundTree`, `:69` `Lookup` | **Fit** — longest-prefix token match with a binding key is the hit/miss engine. |
| **On-disk prefix reuse (L3)** | `internal/l3kv/l3kv.go:52` `SpanStager` (`StageSpan`/`RestoreSpan` **by span digest**, fail-closed digest re-verify) | **Strong fit** — a durable, crash-safe, digest-keyed disk tier already exists; V4's compressed-CSA/HCA disk entries stage through it. The **three SWA disk policies** are the new decision layered on top. |
| **Amplification / savings report** | `internal/cachevaluereport/audit.go:235` `FoldAudit` (savings/fidelity folding) | **Fit** — the reporting harness exists; a V4 fixture feeds it storage/write/read amplification rows. |

## The three SWA disk-cache policies (the tradeoff the ticket asks for)

Modeled as a pure decision over the disk tier (`l3kv`), each with distinct amplification math
the synthetic fixture must separate — conflating them would hide the tradeoff:

| Policy | On-disk cost | On a prefix hit | Amplification character |
|---|---|---|---|
| **Full SWA caching** | store SWA window every checkpoint | zero recompute | highest write/storage, lowest compute |
| **Periodic checkpointing** | store SWA state every N tokens | recompute tail from nearest checkpoint | balanced — the tunable middle |
| **Zero SWA caching** | store nothing for SWA | recompute the full window-128 tail | lowest storage, highest recompute |

## Interface + data-layout sketch (design, not code)

- **`HeterogeneousKVPlane`** (proposed) — one object owning four typed sub-caches: `csa`
  (compressed rate-4, via `mlaKVLayout`), `hca` (compressed rate-128, **new** layout),
  `swa` (window side state, via `swa.go`), and `tail` (dense, via `KVCache`). The page table
  (`pagedkv.go`) addresses blocks tagged by kind.
- **Reuse key** = `MaterializationKey` extended with a *compression-tier* axis so a CSA span
  and an HCA span of the same tokens are distinct addressable entries.
- **Disk staging** = `l3kv.SpanStager` keyed by span digest; SWA staging governed by the
  chosen policy above.

## Fixture plan (the witness)

A **pure, weight-free** Go test that:
1. Computes block accounting for CSA rate-4 and HCA rate-128 compression (bytes/token/layer)
   and validates it against the V4 numbers — fails closed on a mismatch.
2. Models SWA tail recompute for each of the three policies.
3. Emits a **storage / write / read amplification** report for **128K, 512K, and 1M**
   contexts, folded through `cachevaluereport` — no model load, no 1M serve.

Acceptance gate is still **open**: the issue asks whether this fixture lives under
`internal/kvcache` (does not exist as a package today), an extension of `internal/l3kv` /
`internal/cachemeta`, or a **new** `internal/deepseekv4kv`. Recommendation: a new leaf
`internal/deepseekv4kv` for the plane + fixture, importing the seams above — flagged for
operator input rather than silently chosen.

## Honest fences (what is NOT decided or built)

- **No `HeterogeneousKVPlane` object exists** — it is a proposed composition of real seams.
- **No HCA (rate-128) layout exists** — proposed second `kvLayout`.
- **SWA is whole-cache trim today**, not a concurrent side branch.
- **No V4 disk fixture / amplification report yet** — that is this ticket's witness, named
  above, not yet landed in this design note.
- **No provider cache counter is reused** as fak-owned KV reuse (explicit out-of-scope).

## Next rungs

1. Land the **weight-free block-accounting + amplification fixture** (the witness) under the
   agreed package.
2. Propose the **HCA `kvLayout`** leaf (shared with #3016).
3. Extend `MaterializationKey` with a compression-tier axis.
4. Pick an SWA disk policy per deployment and record *why* in the follow-on implementation
   ticket (per the issue's done-condition).
