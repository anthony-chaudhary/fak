---
title: "Evict-on-paged KV: proving the bit-exact middle-span Evict survives a block-paged layout (#33)"
description: "Design + prototype proof that fak's bit-exact, kernel-owned middle-span KV Evict survives the move from a contiguous-append cache to a block-paged allocator. Resolves the Evict-on-paged tension: the single-rotation re-RoPE invariant is physical-layout-independent, so it survives paging given a per-block Kraw plane and a copy-on-write split on re-RoPE. Go recommendation; unblocks #34."
---

# Evict-on-paged KV — does the bit-exact middle-span Evict survive paging? (#33)

_Design + proof gate for [#33](https://github.com/anthony-chaudhary/fak/issues/33).
Part of the dual-track disaggregated-serving epic (#50/#536). This must close before the
native paged allocator [#34](https://github.com/anthony-chaudhary/fak/issues/34) is built and
before the L3 KV-governance referee [#27](https://github.com/anthony-chaudhary/fak/issues/27)
can claim native exact-span eviction. Prototype + proof: `internal/model/paged_evict.go`,
`internal/model/paged_evict_test.go`._

## Verdict: GO

The bit-exact middle-span `Evict` **survives a block-paged KV layout**. The invariant the
value-add rests on — re-deriving each shifted survivor's post-RoPE K in a *single* rotation
from its pre-RoPE `Kraw` at its **new logical position** — depends only on the survivor's
`Kraw` and its new logical index, **not** on physical contiguity. Paging changes where a
token's bytes physically live; it does not change the position-to-angle arithmetic. So exact-
span eviction can move onto the paged allocator with two concrete requirements, both proven
here on real float32 KV with no GPU:

1. **The paged layout must carry a per-token `Kraw` plane** (pre-RoPE K) that travels with the
   token wherever it physically lands. The shipped allocator (`pagedkv.go`, #277) stored only
   K and V; `NewPagedKVPoolWithRaw` adds the third plane.
2. **A survivor re-RoPE must copy-on-write-split** any physical block it shares with a sequence
   that did not evict, because the re-RoPE mutates K in place.

Result: a mid-span evict on a deliberately non-contiguous paged layout is **byte-for-byte
identical** to the contiguous `KVCache.Evict` — both the cache (K, `Kraw`, V) and the next-token
logits — at `max|Δ| = 0` on amd64. This unblocks #34. The honest cost paging adds is named in
§5; it is real but small, and it is a cost vLLM/SGLang never pay only because they cannot do
middle-span eviction at all.

## 1. What the value-add actually depends on

The contiguous cache (`internal/model/kvcache.go`) stores each layer's K/V as one flat
`[]float32` indexed by `position * stride` (`stride = NumKVHeads*HeadDim`), with a parallel
`Kraw` (the same K **before** RoPE) and a `pos[]` of absolute positions. `Evict(from, n)`:

- splices the span out of every layer's K/`Kraw`/V (a contiguous compaction), then
- for every survivor whose new index `i` differs from its original `pos[i]`, re-derives
  `K[i]` from `Kraw[i]` in **one** RoPE rotation at the new position `i`
  (`copy(dst, Kraw[i]); applyRopeRow(dst, ropeRowForLayer(cfg, l, i))`). V is never rotated.

The single-rotation design is load-bearing for bit-exactness. Composing two rotations (old
angle then a delta) is mathematically equal but drifts ~1e-6 in float32 — enough to flip a
greedy token. `applyRopeRow` even pins its products to f32 to block the compiler's optional FMA
fusion, so the rotation is identical at every call site and architecture. The proof that this
equals a run that never saw the span is `evict_test.go` `TestKVQuarantineEqualsNeverSaw`.

Two facts are often described as "contiguity-dependent", but only one truly is:

- **Compaction** (the slice splice) *is* contiguity-dependent. On a paged layout there is no
  single slice to splice — survivors live in non-contiguous physical blocks.
- **The re-RoPE** is **not** contiguity-dependent. `ropeRowForLayer(cfg, l, i)` is a pure
  function of the new *logical* position `i`; it never reads a neighbour. Given the survivor's
  `Kraw` and its new logical index, the bit-exact K follows regardless of where the bytes sit.

That asymmetry is the whole resolution: paging only forces us to re-implement the *compaction*
(a re-index over a page table instead of a splice). The arithmetic that makes the result
*correct* is untouched.

## 2. Logical span → physical blocks; survivor re-index

A paged sequence (`PagedKV`) holds a page table mapping logical block → physical block id, with
fixed-size blocks (`blockTokens` positions each) drawn from a shared pool. Logical position
`p` lives at physical block `table[p / blockTokens]`, slot `p % blockTokens`.

A middle-span eviction `[from, from+n)` is, in general, **not block-aligned**: the span starts
and ends mid-block, and survivors after the span must each shift down to a new contiguous
logical position. This is exactly where the layouts diverge:

- **Whole-prefix reset** (what vLLM `reset_prefix_cache` / SGLang `flush_cache` expose) is a
  block-granular *truncation from the tail* — drop the trailing blocks, update nothing. It is a
  pure page-table edit because nothing is re-indexed.
- **Middle-span eviction with re-index** cannot be a pure pointer update. Removing a non-aligned
  middle span renumbers every following survivor, so within the blocks straddling the span and
  every block after it, survivors change slot. There is no page-table permutation that expresses
  "shift logical positions 5,6,… down by 3" without moving bytes, because block boundaries no
  longer align to the new logical numbering.

So exact-span eviction on a paged layout requires an **intra-block compaction pass** over the
survivors after the span. This prototype performs it the simplest correct way — gather the
survivors out (read-only) and rebuild the sequence into fresh blocks — which also gives the
copy-on-write property of §4 for free. A production allocator (#34) can compact in place with a
forward-copy that respects block ownership; the numerics are the same.

## 3. Survivor re-RoPE across block boundaries; where Kraw lives

For each survivor at new logical index `i` with original position `op`:

- **V** moves verbatim (never rotated).
- **`Kraw`** moves verbatim — it is pre-RoPE and therefore position-independent.
- **K**: if `op == i` (an unmoved survivor, i.e. one before the span) keep its stored post-RoPE
  K; otherwise re-derive it from `Kraw` in a single rotation at the new position `i`
  (`reropeRowFromRaw`, which reuses the exact `ropeRowForLayer` / `applyRopeRow` the contiguous
  path uses, so the result is bit-identical, not merely close). Alibi layers carry no RoPE, so
  the pre-RoPE row *is* the K and is copied verbatim, matching the contiguous path.

Because the rotation reads only `Kraw` and `i`, it works identically whether the survivor's
bytes are in physical block 0 or block 47. **There is no special "across a block boundary"
case for the re-RoPE** — that is the key finding. The only thing that crosses block boundaries
is the re-index bookkeeping (which new logical position maps to which physical slot), and that
is plain integer arithmetic.

**`Kraw` is stored per block, in lock-step with K and V.** It must be, because the re-RoPE of a
survivor needs that survivor's pre-RoPE K wherever the survivor physically lands; a centralized
`Kraw` slice would reintroduce exactly the contiguity assumption paging removes. The prototype
adds `Kraw` as a third plane in each physical block (`planeKraw`), so a block holds K, V, and
`Kraw` for each of its token-slots and each layer.

## 4. The one new constraint paging adds: copy-on-write on re-RoPE

The paged allocator's headline win is copy-on-write prefix sharing: `Fork()` shares every block
by reference count; a write copies just the one block touched. The re-RoPE during eviction *is*
a write — it mutates K in place. If a survivor's block is shared (ref > 1) with a sequence that
did **not** evict, re-RoPEing in place would corrupt that sibling's K (it would see a survivor
rotated to the *evicting* sequence's new numbering, which is wrong for a sequence that still has
the full prefix).

Therefore: **a survivor re-RoPE must trigger a copy-on-write split of any shared block before
mutating it.** The prototype satisfies this by rebuilding the evicting sequence into fresh
private blocks and releasing its references to the old (shared) ones; the forked sibling keeps
its references and its bytes untouched. `TestPagedEvictCOWLeavesForkedParentUnchanged` proves
the sibling is byte-for-byte unchanged after the fork evicts, while the evicting fork still
equals the contiguous `Evict`. A production allocator that compacts in place must call its
`ensureOwned`-equivalent on each block it writes during the re-RoPE pass.

## 5. The honest cost

Middle-span eviction on a paged layout costs an **O(survivors-after-the-span) intra-block
compaction** plus the re-RoPE (which the contiguous path also pays). This is more than a
whole-prefix reset, which is O(blocks dropped) with no byte movement. But the comparison is not
apples-to-apples: vLLM/SGLang pay *zero* for middle-span eviction only because they **cannot do
it** — their public API resets the whole prefix or radix. fak's exact-span quarantine is the
differentiator; paging keeps it, at a cost bounded by the suffix length, which is the same order
as the contiguous compaction it replaces. There is no new asymptotic cost, only the loss of the
"it was a contiguous slice we could splice" shortcut.

Memory: the `Kraw` plane makes a 3-plane block 1.5× the bytes of a K/V-only block. That is the
price of exact-span eviction on paged KV. A sequence that will never be quarantined can use the
2-plane pool and skip it. See §6 for the quantized-KV interaction.

## 6. FP8/INT8-KV interaction

The bit-exact re-RoPE depends on an **f32 `Kraw`**. A quantized-KV path (FP8/INT8 K) would store
a lossy `Kraw`; re-rotating from it is no longer bit-exact, and "compose two rotations" is even
worse. So exact-span eviction and quantized K are in tension. The decision:

- **Keep `Kraw` in f32** as the invariant carrier. V may be quantized freely (it is never
  rotated). K may be quantized for the attention read while a parallel **f32 `Kraw` shadow** is
  retained for any sequence that must support exact-span eviction. This is the same choice the
  KV-precision-tiers work already made (`#1047`: q8 keeps the pre-RoPE K in f32 so the cache
  stays ~2× denser while exact-span survives).
- A sequence that quantizes K **without** an f32 `Kraw` shadow **cannot** offer bit-exact
  exact-span eviction; it must degrade to whole-prefix flush, and that degradation must be
  surfaced honestly (the same honesty the ridden-engine Track-A path carries via
  `SupportsExactSpan=false`). Quantized-K-with-exact-span is therefore **deferred** to a future
  issue that either proves a quantization-aware re-derivation (it cannot be bit-exact by the
  definition of lossy quant) or formalizes the f32-`Kraw`-shadow cost. This design does not
  build it; it carries the constraint.

## 7. GLM-DSA path

The GLM-MoE-DSA cache evicts through `evictGLMDsa` (`kvcache.go`), which compacts the DSA
index/state and re-rotates survivors via `glm.rerotateSurvivor` using the **same new-logical-
index re-derivation**. The invariant is identical: re-derivation is a function of the new
logical position, not physical layout, so the §1–§4 argument transfers verbatim.

The GLM-DSA path is now covered by a separate paged-row witness rather than forced through the
dense `PagedKVPool`: its attention K, pre-RoPE Kraw, V, and learned-indexer K/Kraw rows have
different strides and element widths, so `paged_glmdsa.go` pages each row family under its own
fixed-size block table. `TestPagedGLMDsaEvictBitIdenticalToContiguous` snapshots a contiguous
GLM-DSA cache into that paged representation, evicts a middle span, materializes back to
`KVCache`, and proves the result plus the next decode step are bit-identical to
`KVCache.Evict`.

## 8. Prototype + proof (what landed)

- `internal/model/pagedkv.go` — generalized the block layout from a fixed 2 planes (K, V) to a
  pool-configurable plane count, backward-compatible (default 2 → existing #277 behavior and
  tests byte-identical), and refactored the tail write into one shared `appendPlanes` so the K/V
  and K/V/`Kraw` writes share the block-allocation / copy-on-write boundary.
- `internal/model/paged_evict.go` — `NewPagedKVPoolWithRaw` (the 3-plane pool), `AppendRaw`,
  `GatherKraw`, and `PagedKV.Evict` (the paged middle-span eviction: gather → re-index → single-
  rotation re-RoPE from the `Kraw` plane → rebuild into private blocks for the COW split).
- `internal/model/paged_evict_test.go`:
  - `TestPagedEvictBitIdenticalToContiguous` — on a churned, non-contiguous page table
    (asserted non-monotonic) with a non-block-aligned span, the paged Evict's K/`Kraw`/V **and**
    the next-token logits are byte-for-byte identical to `KVCache.Evict`.
  - `TestPagedEvictCOWLeavesForkedParentUnchanged` — a forked sibling that shares blocks COW is
    left byte-for-byte unchanged when the other fork evicts; the evicting fork still equals the
    contiguous Evict.
- `internal/model/paged_glmdsa.go` / `_test.go` — the GLM-DSA row-geometry variant:
  attention K/Kraw/V and learned-indexer K/Kraw are paged in separate row planes, then a
  middle-span evict is proven bit-identical to the contiguous GLM-DSA cache and next decode
  step.

Run: `wsl -e bash -lc 'cd /mnt/c/work/fak && go test ./internal/model -run "TestPagedEvict|TestPagedGLMDsaEvict"'`
(or `.\test.ps1` for the full Windows-host gate). Green at `max|Δ| = 0`.

## 9. Recommendation to #34 and #27

- **#34 (paged allocator carrying Evict): GO.** Build the real allocator with the `Kraw` plane
  from the start; carry `Evict` via the re-index + single-rotation-from-`Kraw` + COW-on-write
  pattern proven here. Keep the contiguous path behind a flag until the paged path is proven
  bit-identical on the live decode path (this gate proves the eviction math; #34 still owes the
  paged-attention gather and the serve-path wiring).
- **#27 (L3 KV-governance referee): native exact-span eviction is achievable** on the paged
  tier — the referee can attest a real bit-exact eviction on Track B, and degrade honestly to
  whole-prefix flush on Track A, exactly as its design states.
