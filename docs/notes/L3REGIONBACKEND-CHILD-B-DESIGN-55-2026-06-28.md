# `L3RegionBackend` — child B of epic #79, designed behind the frozen `Ref`/`Resolver`/`RegionBackend` seam

> **Design artifact for [#55](https://github.com/anthony-chaudhary/fak/issues/55) (Option B — unlocks G2 + G3 at the L3 tier).**
> This is a `design(` deliverable: an interface design, a token-for-token resolve/page-in/page-out/evict
> contract, the G2 and G3 folds, and an honest dependency + staged-plan analysis. **No `L3RegionBackend`
> implementation, no tests, no wiring land under this issue** (see Non-goals). Every "fak owns this"
> claim points at a primitive that already exists in this repo with a `file:line` citation; every
> *tier-projection* of those primitives onto the external L3 store is still **[GAP] unbuilt**.
>
> Parent positioning study: [`docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md`](L3-DISAGGREGATED-CACHE-REIMAGINED.md)
> (epic #79). This note designs the row that study labelled "child B (`L3RegionBackend`)".

## 0. The one-paragraph thesis

The external L3 KV-cache tier (the cama-complete reference: a Go cache server + vLLM/SGLang connectors
over a remote DRAM pool, content-addressed by `SHA256(prefix)`, slab-allocated, W-TinyLFU-evicted) is a
world-class router/addresser/fusion engine that is **semantics-free by design** — it places opaque cells
by a client-supplied hash and never verifies, evicts-by-content, or attests. A control-path-only
integration (Option A, the referee sidecar) rides in *front* of the L3 `get`/`set` and fills G1, G4, G5,
G6, G8, but it structurally **cannot** do **G2** (evict a poisoned span from the *middle* of pages shared
across sequences) or native **G3** (attest that the *L3 pages*, not just the local resident copy, were
destroyed). Both require fak to own — or faithfully proxy — the **span → L3-page-key mapping**. The only
place that mapping exists is *under* the frozen `Ref`/`Resolver`/`RegionBackend` seam, where a
`Ref.Digest` resolves to an L3 page-key set and paging a region in/out **is** an external-store
`mget`/`mset`. This note specifies that backend. It is the L3-backed dual of the already-shipped
in-address-space `RegionBackend`, `internal/xenginekv/arena.go` (the cross-engine zero-copy arena,
opt-in via `FAK_XENGINE_KV`): same ABI boundary, a remote page pool in place of a local `[]byte`.

## 1. The seam is reused UNCHANGED — no ABI edit (acceptance: frozen-seam reuse)

`L3RegionBackend` implements the **frozen** `abi.RegionBackend` interface (`internal/abi/registry.go:609`)
verbatim — it is **a backend swap, not a new seam, and edits no file under `internal/abi`** (that tree is
the one no worker may lease; the golden conformance test `testdata/abi_v0.1.golden` fails any breaking
change — `internal/abi/types.go:1`). The three frozen surfaces it rides, unchanged:

| Frozen surface | Signature (today) | `file:line` |
|---|---|---|
| `abi.RegionBackend` | `Resolver() Resolver` · `Caps() []Capability` | `internal/abi/registry.go:609` |
| `abi.Resolver` | `Resolve(ctx, Ref) ([]byte, error)` · `Put(ctx, []byte) (Ref, error)` | `internal/abi/types.go:105` |
| `abi.PageOutBackend` | `PageOut(ctx, Ref) (Ref, error)` · `PageIn(ctx, Ref) (Ref, error)` | `internal/abi/registry.go:618` |
| `abi.KVBackend` residency pair | `StageSpan(ctx, digest, from, n) (KVResidency, error)` · `RestoreSpan(ctx, digest) (KVResidency, error)` | `internal/abi/registry.go:642` |

The `Ref` it issues is `RefRegion` (`internal/abi/types.go:81`), `Handle` opaque (an L3 page-key-set
ordinal), `Digest` the content hash that licenses G1 (`internal/abi/types.go:64`), `Taint`/`Scope`
defaulted fail-closed (`TaintTainted`, `ScopeAgent`). Registration is the same opt-in, last-wins
`RegisterRegionBackend` (`internal/abi/registry.go:527`) the `xenginekv` backend already uses
(`internal/xenginekv/register.go:42`) — default builds keep the copy-CAS blob store live; an
`FAK_L3KV`-style opt-in swaps it in. The KV residency transfer rides the **already-frozen**
`KVBackend.StageSpan`/`RestoreSpan` pair, whose own doc comment states it exists to "unblock a remote L3
KV backend without shipping its transport" (`internal/abi/registry.go:634-641`). **Nothing below is a
proposed ABI addition.**

## 2. The `L3RegionBackend` interface — each method against its external-store counterpart (acceptance: interface + mget/mset/evict naming)

`L3RegionBackend` is a struct that satisfies `abi.RegionBackend` + `abi.PageOutBackend` and supplies a
`KVBackend` whose residency pair drives the L3 tier. Its `Resolver()` returns an `*L3Arena` — the remote
dual of `xenginekv.Arena`. The methods, named against the external store's verbs:

```
// L3RegionBackend: the L3-backed dual of xenginekv.backend. Reuses abi.RegionBackend UNCHANGED.
type L3RegionBackend struct {
    store  L3Store          // the external L3 client: Mget / Mset / Evict / Ack (CALLED, never rebuilt)
    pager  SpanPager        // span <-> ordered page-key set (the contract in §3)
    verify bool             // OPT-IN per-page digest-verify at page-in (§4, §6.5); default true, fail-closed
}

// L3Store is the thin client surface onto the external tier. fak CALLS it; it does
// not vendor or re-implement the slab allocator / swiss-table / W-TinyLFU / RDMA-ODP.
type L3Store interface {
    Mget(ctx context.Context, keys []PageKey) ([][]byte, error)   // external `get` (RDMA/TCP mget)
    Mset(ctx context.Context, keys []PageKey, pages [][]byte) error // external `set` (mset)
    Evict(ctx context.Context, keys []PageKey) (EvictAck, error)   // external `evict` + a store ack
}
```

| `RegionBackend`-family method | What it does | External-store counterpart |
|---|---|---|
| `Resolver().Resolve(ctx, ref)` | page-in the page-key set behind `ref`, digest-verify each, return bytes (a view, when co-resident) | **`mget`** |
| `Resolver().Put(ctx, bytes)` | place a fak-owned page (re-keyed survivor, tool arg/result) into the tier | **`mset`** |
| `PageOut(ctx, ref)` | hand the page-key-set HANDLE across without moving bytes (region stays resident) | handle relabel (no wire op) |
| `PageIn(ctx, handle)` | resolve the handle's page-key set back (`mget` + verify) | **`mget`** |
| `KVBackend.StageSpan(ctx, digest, from, n)` | `mset` the `[from,from+n)` page subset off-box; typed `KVResidency` | **`mset`** |
| `KVBackend.RestoreSpan(ctx, digest)` | `mget` a previously staged span by digest; typed OK/MISS/FAULT | **`mget`** |
| `evict(from, n)` (the G2 path, §5) | map `[from,from+n)` → page-key subset, drive `KVCache.Evict`, **`evict`** the old + stale keys, `mset` re-keyed survivors | **`evict` + `mset`** |

`Caps()` advertises `"zerocopy"` only where the page bytes are co-resident in fak's address space (the
co-residence handshake of §7b); over a pure remote `mget` it advertises nothing extra (the resolved
`[]byte` is a fetched copy, not a live view) — the honest capability boundary the `xenginekv` arena
already draws (`internal/xenginekv/arena.go:53-56`).

## 3. The span → page-key contract (acceptance: span → ordered page-key set + Ref.Digest derivation)

A logical KV span is a half-open range of sequence positions `[lo, hi)`. The L3 tier tiles a prefix into
fixed-granularity pages of `b` positions each (the connector's page size). The span maps to the **ordered**
set of page keys that tile it:

```
pages(lo, hi) = [ page(j) : j in floor(lo/b) .. ceil(hi/b) - 1 ]
PageKey(j)    = SHA256( prefix_tokens[0 : (j+1)*b] )   // the external tier's content-address: SHA256(prefix)
```

The key for page `j` is the prefix hash **up to that page's end** — exactly the `SHA256(prefix)` scheme
the external L3 tier content-addresses on (`docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md:16-20`). This
is the load-bearing detail: because a page key folds *all preceding tokens*, **evicting the middle of a
span re-keys every survivor after it** — the property §5 turns into G2.

`Ref.Digest` (G1, `internal/abi/types.go:64`) binds fak's identity to the store's key. For a page paged in
at key `K(j)`, fak recomputes `digest(bytes) = hex(SHA256(bytes))` and the page's `Ref.Digest` is that
value; the contract requires `Ref.Digest` to **derive deterministically from, and validate against**, the
client hash the store keyed on. Two admissible derivations, named so the implementation picks one and
documents it:

- **(a) content digest** — `Ref.Digest = SHA256(page_bytes)`; the store's `SHA256(prefix)` key is carried
  in `Ref.Handle`'s page-key-set entry and verified to be the key fak requested. Self-contained; the
  derivation fak already uses (`internal/xenginekv/arena.go:98-101`).
- **(b) prefix-bound digest** — `Ref.Digest = K(j) = SHA256(prefix_tokens[0:(j+1)*b])`; identical to the
  store key, so verify-on-page-in is "the bytes hash to the key the store filed them under" only if the
  store also content-addresses the *bytes* (it content-addresses the *prefix*, not the page bytes, so (b)
  alone does not catch a corrupted page — **(a) is the G1-complete choice**, (b) is an addressing alias).

The contract: **the implementation MUST use derivation (a)** so a paged-in page is verified against its
own bytes, and additionally assert the returned key equals the requested key. (b) is documented as the
addressing index, not the integrity check.

## 4. resolve / page-in / page-out / evict lifecycle (acceptance: token-for-token contract; fail-closed page-in verify)

```
resolve(ref) -> bytes:
    keys      := pager.PageKeys(ref)              // ordered page-key set behind the Ref (§3)
    pages, e  := store.Mget(ctx, keys)            // external `mget`
    if e != nil: return FAULT(e)                  // a remote stall/cancel is a typed fault, never a hang
    for i, p := range pages:
        if SHA256(p) != ref.PageDigest(i):        // G1 per-page verify (derivation (a), §3)
            return REFUSED(keys[i])               // *** FAIL-CLOSED: a page whose digest != its Ref is
                                                  //     REFUSED and NEVER returned to the engine ***
    return concat(pages)

pageIn(handle)  = mget(handle's page-key set) + the SAME per-page verify  // == resolve for a region Ref
pageOut(ref)    = relabel: a co-resident page-key set stays resident; the HANDLE is the paged-out pointer
                  (zero byte movement — the property a pinned KV region needs; cf. xenginekv PageOut,
                   internal/xenginekv/arena.go:206-216). A non-resident Ref's bytes are mset first.
evict(from, n)  = the G2 path, §5.
```

The **fail-closed page-in digest-verify** is the heart of "verify, don't trust" (G1): the external tier
assumes the connector's hash is the source of truth and does **no re-validation**
(`docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md:18-20`); fak holds `Ref.Digest` at the syscall boundary
and turns that silent assumption into a checked admission step. A digest mismatch yields a structured
refusal (the closed `ReasonCode` vocabulary, `internal/abi/reasons.go`; G8) — the poisoned/wrong page is
**never** handed to the engine. This gate is **opt-in and lives only at the page-in boundary** — see §6.5
for why it is not an inline per-RDMA-read byte-scan.

`StageSpan`/`RestoreSpan` return the typed `abi.KVResidency{Outcome, Digest, Positions, BytesMoved,
Reason}` (`internal/abi/kvbackend.go:49`): `KVResidencyOK` on a hit/move, `KVResidencyMiss` when the tier
no longer holds the span (the caller recomputes — **but is told**, never silently), `KVResidencyFault` on
a transport error or `ctx` deadline (`internal/abi/kvbackend.go:20-27`). The in-process default already
returns a typed `MISS` rather than silently recomputing (`internal/abi/registry.go:656-661`); the L3
backend is the first backend for which these outcomes are *not* a local no-op.

## 5. G2 — middle-eviction over the shared L3 span (acceptance: evict→KVCache.Evict→L3 delete; survivor invariant gate)

`evict(from, n)` is the operation a control-path-only integration cannot express. The fold:

```
1. subset := pager.PageKeys span ∩ [from, from+n)        // the page keys whose positions intersect the cut
2. removed := KVCache.Evict(from, n)                      // fak-OWNED resident re-rotation (internal/model/kvcache.go)
   //   survivors after the cut had K rotated by RoPE at their ORIGINAL absolute position; after compaction
   //   they sit lower, so each survivor's K is RE-ROTATED from the pre-RoPE Kraw at its NEW position —
   //   bit-exact to a prefill that never saw the evicted span. (V is not rotated.) internal/model/kvcache.go.
3. stale  := pager.PageKeys for every survivor whose position SHIFTED  // §3: a middle cut re-keys all later pages
4. store.Evict(ctx, subset ∪ stale)                      // external `evict`: delete the cut pages AND the now-stale survivor keys
5. for each shifted survivor page: store.Mset(ctx, newKey, reRotatedBytes)  // external `mset`: re-file survivors under their NEW keys
```

Steps 3–5 are precisely why this needs a backend *under* the seam: a sidecar in front of `get`/`set`
cannot recompute the re-rotated survivor bytes, so it cannot compute their new page keys, so it cannot
re-file them — it would leave the tier holding stale, mis-keyed survivor pages. fak owns
`KVCache.Evict`'s Kraw re-rotation, so it is the only layer that can produce the new survivor bytes and
hence the new keys.

**Survivor invariant — the checkable gate.** After `evict(from, n)`, each survivor's resident K is
**bit-exact** to a fresh prefill of the surviving tokens that never saw the evicted span:
`max|Δ| = 0`. This is the property `internal/model/evict_test.go` `TestEvictEqualsNeverSaw` already
proves for the resident cache; the G2 gate is that **the bytes `mset` back under the new keys are the
bytes that pass that test** — i.e. the L3 re-file carries the same `max|Δ|=0` survivors, not a recompute
that drifts. A stage gate (§8) asserts this against a mock/local L3 by reading the re-keyed pages back and
comparing to a never-saw-the-span prefill.

## 6. G3 — fold the L3 page-key eviction into the deletion certificate (acceptance: Mint extended; cert wire-independent)

`deletioncert.Mint(priv, Certificate)` is a **pure fold**: it runs no eviction, reads no journal, reads no
clock — the caller supplies every fact and it signs the canonical pre-image (`internal/deletioncert/deletioncert.go:133`).
The cert today attests a **box-level** KV-cache eviction: `Span`, `EvictedCount` (a self-report),
`Equivalence` (the `max|Δ|=0` claim), and an `Anchor` — a hash-chained journal row — plus `JournalHead`
(`internal/deletioncert/deletioncert.go:100-132`). The G3 **tier fold** projects it onto the L3 page-key
set **without editing the signed-field set's meaning** — it populates the existing fields with L3 facts and
records the page-key chain + store ack in the `Anchor`/`Scope` surface the schema already carries:

- `Span` / `EvictedCount` = the `[from,from+n)` cut and positions removed (from step 2 of §5).
- `Scope` = an L3-tier scope string, e.g. `"l3-working-set"` — the cert's own `Scope` field already
  exists to name the surface a receipt covers (`internal/deletioncert/deletioncert.go:106`); the parent
  study fences this exact scoping discipline (`docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md:113-117`):
  the cert proves the **L3 working set** is gone, **not** that bytes never leaked into weights, backups,
  or replicas.
- `Anchor` (the hash-chained row) folds **(i)** the ordered page-key set evicted (`subset ∪ stale` from
  §5, hash-chained so a dropped key is detectable) and **(ii)** the external store's **`EvictAck`** — the
  store's acknowledgement that those keys were destroyed. The page-key hash-chain + the ack are what lift
  the attestation from "fak's *local* copy is gone" to "the *L3 pages* are gone".
- `Equivalence` = the §5 survivor invariant (the survivors are bit-exact to never-saw).

**Wire-independence (the acceptance bar).** Because `Mint`/`Verify` are a pure fold over canonical bytes
and an Ed25519 signature (`internal/deletioncert/deletioncert.go:133-160`), the resulting certificate is
**byte-identical-verifiable with no access to the L3 wire** — a verifier holding only the certificate
re-checks the signature, the page-key chain, and the `Subject == Anchor.ResultDigest` binding offline.
The store ack is *folded in as a recorded fact*, not a live call at verify time. Honest fence, carried
verbatim from the shipped primitive: v1 is self-signed; `EvictedCount` and the store ack are
**self-reports** until an `ExternalAnchor` carries a third-party attestation
(`internal/deletioncert/deletioncert.go:122-132`) — the tier fold does not widen that claim.

## 6.5 Control-path constraint — fak admits, it does not ride the data path (acceptance: §6.5 verbatim)

The single sharpest constraint of the whole epic (`docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md:69-76`):
**keep fak on the control / admission path, never the data path.** The external tier's entire value is the
~**1–5 µs L3 RDMA read**; per-op digest verification of a multi-megabyte page at line rate would destroy
that budget. Therefore, in this design:

- The per-page digest-verify of §4 is an **OPT-IN admission gate at page-in boundaries** (`verify` flag,
  default-on but scoped), **NOT** an inline byte-scan of every L3 RDMA read on the data path. It runs when
  a page is *admitted* (first page-in / a cross-tenant share / a quarantine re-check), out of band, on the
  control path — **not** charged per-op at line rate against the 1–5 µs budget.
- The hot zero-copy / prefix-caching fast path is **untouched** — no inline per-RDMA-read byte-scanner
  (Non-goals). The data path stays the external tier's `mget` at its native latency.
- The staged-plan gates (§8) run against a **LOCAL or MOCK** L3 backend; talking to the **real external
  store over its wire protocol is OPTIONAL / follow-on**, never a hard gate dependency. fak's own guard
  refuses writes outside the repo and the external store lives in a separate, un-imported tree
  (`docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md:178-182`), so no CI gate may depend on reaching it.

## 7. Dependency analysis — explicit and honest (acceptance: HARD precondition named; [GAP], ~120h-class)

For `L3RegionBackend` to run at all, **one** of these must be true — a **HARD precondition, [GAP]
unbuilt, ~120h-class**:

- **(a) the model executes in fak's address space** — the v0.2 in-kernel engine. Then fak owns the
  resident KV outright; `KVCache.Evict` (§5) operates on cache fak holds, and page bytes are fak's to
  `mset`. This is the clean path and the one the shipped `xenginekv` arena assumes for its local form.
- **(b) a co-residence handshake with the external serving engine's arena** — fak addresses the *same*
  page bytes vLLM/SGLang holds (a shared-memory / CUDA-IPC-imported buffer). This is exactly the
  `xenginekv.AttachArena(buf []byte)` direction (`internal/xenginekv/arena.go:91-96`): the SEAM ships, the
  engine-specific transport that maps a real engine KV region into the arena does **not**. Without (a) or
  (b), the seam "below the Ref presumes the model runs in fak's address space" and G2 cannot reach the
  bytes — the gap [#55] names.

**Multi-node composition (#493).** When the L3 tier and the compute are on different hosts, the off-box
page move composes with the network `StageTransport` seam (`internal/model/pipeline.go:224`): `StageSpan`'s
`mset` and `RestoreSpan`'s `mget` carry over `StageTransport` instead of a local call, with no change to
the `RegionBackend`/`KVBackend` ABI. `StageTransport` is frozen; its network implementation is the #493
deliverable, not this one.

**Gating — Option A first (acceptance: gated behind the sidecar).** This backend is **gated behind Option
A (the referee sidecar)** proving the seam is worth deepening before any of this is built
(`docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md:95-101`; epic #79). G6 (what the L3 tier *admits*) is
owned upstream by #496 (S7 write-time durability) and is **referenced, not duplicated** here.

## 8. Staged plan — each stage with a fak-side gate (acceptance: mock → local → multi-node)

| Stage | What | Touches data path? | fak-side gate (the witness) |
|---|---|---|---|
| **S0 — mock L3** | `L3Store` = an in-process `map[PageKey][]byte` honoring the §3 page-key contract — the remote dual of `xenginekv.NewArena` (`internal/xenginekv/arena.go:84`). | No | `go test ./internal/...` (GPU-free): page-in digest-verify **REFUSES** a corrupted page; `evict(from,n)` → `KVCache.Evict` survivors are `max|Δ|=0` vs never-saw (§5); `Mint`/`Verify` round-trips the L3 page-key + ack fold offline (§6). |
| **S1 — local single-node store** | a real external L3 cache instance on `localhost`. | No | an integration test behind a build tag / `FAK_L3KV_LOCAL` opt-in — **OPTIONAL, not a hard CI gate** (§6.5). |
| **S2 — multi-node (#493)** | L3 tier off-host; `mget`/`mset` over `StageTransport`. | No (control path only) | per #493's witness; `dos verify` against the child that lands the network transport. |

The spine is ordered so progress never stalls on hardware or on a reachable external store: **S0 ships
from any box, including the GPU-less dev box and CI**; S1/S2 are gated, optional, follow-on.

## 9. Acceptance map (this artifact ⇄ the issue checklist)

- ✅ **`L3RegionBackend` interface, methods named against `mget`/`mset`/evict** — §2.
- ✅ **resolve/page-in/page-out/evict contract token-for-token; fail-closed per-page digest-verify** — §4.
- ✅ **G2 mapping `evict→page-key subset→KVCache.Evict→L3 delete`; survivor invariant as a checkable gate** — §5.
- ✅ **G3 fold: `deletioncert.Mint` attests the L3 page-key set; cert wire-independent** — §6.
- ✅ **dependency list explicit + honest: in-kernel (v0.2) OR co-residence handshake as HARD precondition, [GAP], ~120h-class** — §7.
- ✅ **frozen `Ref`/`Resolver`/`RegionBackend` reused UNCHANGED, no ABI edit, stated explicitly** — §1.
- ✅ **control-path constraint §6.5: opt-in page-in admission gate, not an inline RDMA byte-scan; 1–5 µs budget named; gates run on LOCAL/MOCK, real wire optional** — §6.5.
- ✅ **gated behind Option A (the sidecar) before anything is built** — §7.

## 10. Non-goals (carried from the issue, verbatim in intent)

- **Not built here.** Interface design + dependency analysis + staged plan only; no `L3RegionBackend`
  implementation, no tests, no wiring lands under #55.
- Does **not** fork or rewrite fak's generation / decode loops — the backend sits under the existing
  `Resolver`/`RegionBackend` seam the engines already call.
- Does **not** rebuild, re-implement, or vendor the external L3 store — it **CALLS** it
  (`mget`/`mset`/evict) as the integration target and exemplar.
- Does **not** touch the data path: no inline per-RDMA-read byte-scanner, no change to the L3 zero-copy /
  prefix-caching fast path.
- Does **not** re-file G6 (truth-duration / durability) — owned upstream by #496; referenced, not duplicated.

## Related, in this repo

- [`docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md`](L3-DISAGGREGATED-CACHE-REIMAGINED.md) — the epic #79 positioning study this designs child B of.
- `internal/abi/registry.go:609` — the frozen `RegionBackend` seam reused unchanged; `:642` the `KVBackend` residency pair; `:527` the opt-in `RegisterRegionBackend` swap.
- `internal/abi/types.go:64` — `Ref.Digest` (G1); `:81` `RefRegion`; `:105` `Resolver`.
- `internal/abi/kvbackend.go:49` — the typed `KVResidency` outcome (OK / MISS / FAULT).
- `internal/xenginekv/arena.go` — the shipped in-address-space `RegionBackend` this is the L3 dual of.
- `internal/model/kvcache.go` / `internal/model/evict_test.go` — `KVCache.Evict` (G2) and the `max|Δ|=0` survivor witness.
- `internal/deletioncert/deletioncert.go:133` — `deletioncert.Mint` (G3), projected onto the tier here.
- `internal/model/pipeline.go:224` — `StageTransport` (#493), the multi-node composition.

Related epics: #79 (parent), #493 (network `StageTransport`), #496 (S7 durability / G6). External source
system: an external disaggregated L3 KV-cache (out-of-tree exemplar, not imported by this module).
