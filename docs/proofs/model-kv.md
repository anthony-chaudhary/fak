# N6 · model/kv

The `model/kv` sub-module is the kernel-owned attention state — `KVCache` plus the per-position decoder-block math in `Session` that fills it. Each layer holds K and V as flat row-major `[pos·(NumKVHeads·HeadDim)]` slices, a parallel pre-RoPE `Kraw`, and a shared `pos[]` recording every entry's absolute RoPE position. "Correct" for this module (regime **N — numerical**) means four bit-level properties: (a) a position's K/V is written to and read back from the right `(layer, pos, head)` slot with no aliasing; (b) `Evict(from,n)` produces a cache *byte-identical* to one that never saw the span, by re-rotating each survivor's K from its stored pre-RoPE form to its new absolute position in a single rotation (not composed); (c) the sliding-window read mask drops *exactly* the out-of-window positions, keyed off `pos[]` so it survives eviction renumbering, and is a true no-op (max|Δ|=0) when no window is set; and (d) `SessionFromPrefix` (clone + suffix-prefill) is bit/token-identical to a full recompute. Witnesses are run natively on this macOS node (go1.26 darwin/arm64) against the present `.cache/smollm2-135m` oracle export.

> **Honesty note on the prompt's named witness.** The obligation named `TestHiddenStateRoundTripBitExact` for theorem (1). That test (`pipeline_test.go:404`) round-trips the **wire codec** (`MarshalHidden`/`UnmarshalHidden`) and passes, but it does **not** assert KV-slot placement. The binding witnesses for (1) are `TestStandardLayoutNoOp` and `TestSWAWindowUnsetIsNoOp`'s `assertKVCacheBitsEqual`. Recorded as fact, not promoted.

---

## Theorem T1 — KV append lands at the correct (layer,pos,head) slot; reload is bit-exact

**THEOREM.** For every layer `l`, cached index `j`, kv-head `kvh`, lane `d`, the K/V written by `blockStep` occupies slot `K[l][j·w + kvh·hd + d]` (`w = NumKVHeads·HeadDim`) with no overwrite or cross-layer aliasing, and reading it back through `standardKVLayout.reconstructKV` (or the inline GQA loop) reproduces that exact tensor bit-for-bit. The cache built by the cached prefill+decode path is byte-identical (K, Kraw, V, pos[]) to the legacy path: max|Δ| = 0.

**REGIME.** N.

**PROOF.** `blockStep` appends this position's post-RoPE K and V to the per-**layer** flat row at `fak/internal/model/kv.go:752-753` (`s.Cache.K[l] = append(...)`, V likewise), recording the position once (shared across layers) at `kv.go:577`. The slot index is the deterministic affine map `j·w + kvh·hd + d` (stride `kvStride`, `kv.go:47`) — no aliasing. The read side `standardKVLayout.reconstructKV` (`kvlayout.go:63-66`) returns `row[:w], row[w:2w]` verbatim — the identity reconstruction, which is what keeps the standard path bit-identical to the inline loop. `TestStandardLayoutNoOp` prefills a 7-token prompt, then per layer scores a query two ways — the verbatim inline blockStep GQA loop vs `attendOne` over rows rebuilt from the *same* cache through `standardKVLayout` — and asserts max|Δ|=0. `TestSWAWindowUnsetIsNoOp/prefill+decode` builds the cache via the current and legacy paths and runs `assertKVCacheBitsEqual` (`refactor_test.go:267`), an `math.Float32bits`-level equality of K, Kraw, V, pos[] per layer, after prefill and each of 4 decode steps.

**WITNESS.** `go test -run 'TestStandardLayoutNoOp|TestSWAWindowUnsetIsNoOp' ./internal/model/ -count=1 -timeout 180s -v`

**VERDICT.** **PROVEN** (2026-06-20). `TestStandardLayoutNoOp` PASS: `max|Δ|=0.000e+00`. `TestSWAWindowUnsetIsNoOp` PASS (all 3 subtests incl. the per-layer `assertKVCacheBitsEqual` over prefill + 4 decode steps).

**DOS.** bound at ship.

---

## Theorem T2 — span-exact eviction == a context that never saw the span (max|Δ|=0)

**THEOREM.** `KVCache.Evict(from,n)` removes a contiguous position span from every layer and re-rotates each survivor's post-RoPE K to its new absolute index, so the resulting cache and greedy continuation are byte/token-identical to a run that never saw the span. Every survivor's K equals a *single* RoPE rotation of its pre-RoPE `Kraw` at the new index (max|Δ|=0); and the equivalence is span-exact, not retroactive — a span evicted *after* downstream tokens attended cannot be un-seen.

**REGIME.** N (with a deletion-equivalence flavour: `max|Δ|=0`, "proven == never-saw-it").

**PROOF.** `Evict` (`kv.go:60-103`) splices the span out of `K[l]/Kraw[l]/V[l]` for every layer (`kv.go:77-79`) and out of the shared `pos[]` (`kv.go:84`), then for each survivor whose `pos[i] != i` re-derives `K[l][i]` from the **pre-RoPE** `Kraw[i]` in one rotation at the new index (`kv.go:92-97`). The pre-RoPE store is laid down at write time by `ropeRowQK` (`kv.go:807`) *before* the in-place rotation — this is what lets one rotation at the new position equal a fresh prefill rather than composing two (the design note at `kv.go:30` warns the composed form drifts ~1e-6 and flips a greedy token). `applyRopeRow` (`kv.go:485`) pins each product to f32 with explicit `float32()` casts to block opportunistic FMA fusion, so the rotation is bit-identical across the prefill site and the evict-reposition site on every architecture. `TestKVQuarantineEqualsNeverSaw` drives this on the real SmolLM2-135M oracle: write-time evict → greedy continuation equals the HF `NeverGreedy` reference token-for-token; middle-span evict → every survivor K == RoPE(Kraw,newpos) with max|Δ|=0; and the two negative controls (middle-span-too-late ≠ never; un-evicted == HF-poisoned ≠ never) make the equivalence non-vacuous. `TestEvictRepositionsWithLayerSpecificRopeTheta` re-checks the reposition invariant under per-layer RoPE theta + SandwichNorm with strict `assertFloat32BitsEqual`.

**WITNESS.** `go test -run 'TestKVQuarantineEqualsNeverSaw|TestEvictRepositionsWithLayerSpecificRopeTheta' ./internal/model/ -count=1 -timeout 180s -v`

**VERDICT.** **PROVEN** (2026-06-20). `TestKVQuarantineEqualsNeverSaw` PASS (1.76s): write-time `go == NEVER (HF ref)`; `reposition invariant K==RoPE(Kraw,newpos): max|Δ|=0.000e+00`; middle-span ≠ never (✓); poison perturbs (✓). `TestEvictRepositionsWithLayerSpecificRopeTheta` PASS. *(Both rungs SKIP without the gitignored `.cache/smollm2-135m` export and under `-short`; the cache is present on this node, so they ran weight-backed.)*

**DOS.** bound at ship.

---

## Theorem T3 — SWA masks exactly the out-of-window positions; prefix splice == recompute

**THEOREM.** With window `W`, the query at absolute position `p` attends *only* keys whose absolute pos ≥ `p-W+1`, keyed off `pos[]` so it survives Evict renumbering. With no window the path is bit-identical to full causal (max|Δ|=0); a window ≥ seq length reduces to full causal; a real window genuinely changes the output. Separately, `SessionFromPrefix` (clone then prefill suffix) yields logits and greedy continuation byte/token-identical to a full recompute of prefix+suffix (max|Δ|=0).

**REGIME.** N.

**PROOF.** The read loop in `blockStep` sets `lo = windowLoStep(pos, nPos, qpos, windowForLayer(l))` (`kv.go:760`) and scores/accumulates V only over `[lo, nPos)`. `windowLo` (`weights.go:720`) scans `pos[]` for the first index whose **absolute** position ≥ `qpos-W+1`, returning 0 when `W<0` (full causal — no keys dropped). Because `pos[]` is monotonic, the visible keys are always a contiguous suffix; a window only drops the oldest keys and never reorders survivors, so the masked in-order softmax+V sum is the same arithmetic restricted to a sub-range. `TestSWAWindowMasksOldKeys` checks the helper boundaries exactly (`[p-W+1,p]` table incl. `W=1` self-only and `W=-1` full causal), the eviction-renumber case (`pos=[0,1,2,7,8]`, `W=3`, `p=8` → `lo=3`, proving the bound follows `pos[]` not the index and avoids the slice-overrun trap), and end-to-end that windowed `Forward` equals an independent `-inf`-pre-softmax masked reference (max|Δ|=0), that a wide window equals full causal bit-identically, and that `W=3` differs from full causal at the last position (non-vacuous). `TestSWAWindowUnsetIsNoOp` pins the `W<0` default to max|Δ|=0 vs the pre-SWA reference across cacheless/prefill+decode/batched. **Prefix splice:** `SessionFromPrefix` (`kv.go:221`) calls `Clone` (`kv.go:123`), a byte-exact deep copy of K/Kraw/V/pos, then prefills only the suffix; `TestKVPrefixReuseMatchesRecompute` asserts reuse last-logit, argmax, 8-token greedy, and cache length all equal a full recompute — last-logit max|Δ|=0.

**WITNESS.** `go test -run 'TestSWAWindowMasksOldKeys|TestSWAWindowUnsetIsNoOp|TestKVPrefixReuseMatchesRecompute' ./internal/model/ -count=1 -timeout 180s -v`

**VERDICT.** **PROVEN** (2026-06-20). `TestSWAWindowMasksOldKeys` PASS (4 subtests). `TestSWAWindowUnsetIsNoOp` PASS. `TestKVPrefixReuseMatchesRecompute` PASS (0.53s): `last-logit max|Δ|=0.000e+00 (prefix prefill SKIPPED: 5 positions)`, reuse greedy == full greedy. *(The prefix-reuse rung is weight-backed — SKIP without the oracle cache / under `-short`; cache present here.)*

**DOS.** bound at ship.

---

### Reproduce all three at once

```bash
go test -run 'TestStandardLayoutNoOp|TestMLANaiveMatchesReference|TestSWAWindowUnsetIsNoOp|TestSWAWindowMasksOldKeys|TestKVQuarantineEqualsNeverSaw|TestKVPrefixReuseMatchesRecompute|TestEvictRepositionsWithLayerSpecificRopeTheta' ./internal/model/ -count=1 -timeout 180s -v
```

Observed 2026-06-20 on this node: `ok github.com/anthony-chaudhary/fak/internal/model 2.546s` — every listed rung PASS, every reported delta `max|Δ|=0.000e+00`.
