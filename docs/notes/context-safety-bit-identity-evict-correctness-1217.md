# Context-safety bit-identity evict-correctness signal (#1220, C3 of epic #1217)

_Research / design only. This is the **C3 value-decomposition spec** for the
context-safety epic [#1217](https://github.com/anthony-chaudhary/fak/issues/1217).
It specs guarantee **G1** — the bit-identity (`max|Δ| = 0`) evict-correctness
signal — as a **per-event WITNESSED datum** that feeds primitive (4), the
evict-correctness heatmap. The guarantee **already exists and is already tested**;
the job here is to **surface and cross-check it visually**, not to invent a new
compute claim. No code ships here — the deliverable is this committed spec: the
per-evict datum schema, the test/verdict it binds, and the doctrine-D
re-derivation that makes its `0` self-proving. It deepens guarantee **G1** of the
C1 doctrine note
([#1218](https://github.com/anthony-chaudhary/fak/issues/1218),
[`context-safety-visuals-tracking-1217.md`](context-safety-visuals-tracking-1217.md)),
populates the heatmap primitive schema'd in C6
([#1223](https://github.com/anthony-chaudhary/fak/issues/1223),
[`context-safety-visual-primitive-schemas-1217.md`](context-safety-visual-primitive-schemas-1217.md)),
renders the C2 failure-catalog's evict-correctness (F1-adjacent) class
([#1219](https://github.com/anthony-chaudhary/fak/issues/1219),
[`context-safety-failure-mode-catalog-1217.md`](context-safety-failure-mode-catalog-1217.md)),
and is re-derived by the C9 checker
([#1227](https://github.com/anthony-chaudhary/fak/issues/1227),
[`context-safety-rederivation-checker-1217.md`](context-safety-rederivation-checker-1217.md))._

---

## Why G1 is the strongest guarantee — the number proves its own correctness

> G2 (reuse bit) needs a second axis because a count is blind to correctness.
> G3 (S/N) needs a paired Faults axis because a ratio can be gamed by starving
> the turn. G1 needs **neither** crutch: `max|Δ| = 0` *is* its own proof. Either
> the surviving cache is byte-for-byte identical to one that never saw the
> evicted span, or it isn't — there is no "close enough," no denominator to
> inflate, no axis to game. This is why G1 is the strongest of the four
> guarantees, and why surfacing it is the lowest-risk child: there is no new
> compute claim to defend, only an existing, tested `0` to make visible and
> cross-checkable.

The honest fence for G1 is therefore *not* "is the metric gameable" (it isn't);
it is **"is the visible `0` re-derivable from the test/verdict that proved it, or
is it a trust-me `0` typed into a dashboard."** That is doctrine **D**
(re-derivation) and doctrine **C** (provenance), and that — not a new
correctness argument — is the whole content of this spec.

---

## The guarantee already exists and is already tested

The eviction primitive and its bit-identity correctness are already in the tree.
This spec invents none of it; it cites it so the visual can bind it.

### The typed boundary — `internal/model/kvcache.go`

`KVCache.Evict(from, n int)` (`kvcache.go:94`) "removes a contiguous span
`[from, from+n)` of cached positions from EVERY layer and compacts the survivors
so the cache is byte-for-byte what it would be if the span had NEVER been seen"
(the doc comment at `kvcache.go:78-80`). The subtlety the code calls out
(`kvcache.go:82-88`): a survivor that came *after* the evicted span was RoPE'd at
its original absolute position and must be **re-rotated** by the position delta
after compaction — "without this, end-span eviction passes by luck but
MIDDLE-span eviction is silently wrong." That re-rotation is exactly the seam a
correctness witness has to guard, because it is where a plausible-looking evict
goes silently wrong.

The boundary is **witnessable**: `CanEvict()` (`kvcache.go:71`) returns the typed
`RecurrentEvictUnsupportedError` (`kvcache.go:44`) for a hybrid Gated-DeltaNet
cache whose accumulated recurrent state has no per-token row to compact — the
verdict **fails closed** ("the cache is left byte-for-byte unchanged",
`kvcache.go:49-50`) rather than half-evicting. `TryEvict()` (`kvcache.go:105`)
surfaces that verdict as a typed error instead of a panic; `Evict()` is the
convenience form that panics the typed error for an unchecked caller
(`kvcache.go:94-100`). So the boundary already yields three distinguishable
per-event outcomes — **supported-and-correct**, **unsupported-refused-fail-closed**,
**panic** — which is precisely the cell vocabulary the heatmap needs (below).

### The correctness proof — two tests, `max|Δ| = 0`

The bit-identity is proven, not asserted, by two existing tests:

- `internal/model/paged_evict_test.go:84`
  `TestPagedEvictBitIdenticalToContiguous` — a **mid-span** evict (positions
  `[2,5)`, deliberately *not* block-aligned, `blockTokens=4`, so survivors cross
  block boundaries and must be re-RoPE'd; `paged_evict_test.go:88-89`) on a
  **non-contiguous** paged page table produces a cache (`K`, `Kraw`, `V`) **and
  next-token logits** byte-for-byte identical to the contiguous `KVCache.Evict`
  on the same data. The setup even asserts the page table is *physically
  non-contiguous* first (`paged_evict_test.go:99-101`) so the proof can't pass on
  a degenerate contiguous layout.
- `internal/model/longrope_test.go:179`
  `TestLongropeEvictEqualsNeverSaw` — the stronger end-to-end claim: a session
  that prefilled `[prefix ++ poison ++ query]`, evicted the poison, then
  continued is logits-identical to a session that prefilled `[prefix ++ query]`
  and **never saw the poison** (`longrope_test.go:186-198`), under longrope
  re-rotation.

Both prove `max|Δ| = 0` through the same helper, `assertFloat32BitsEqual`
(`internal/model/rope_test.go:68`), which compares `math.Float32bits(want[i])` to
`math.Float32bits(got[i])` (`rope_test.go:74`) — a **bit-level** equality, not a
tolerance. That is what makes the visible number a literal `0` and not an epsilon:
the test fails on a single differing bit. **This is the tamper-evident source the
G1 datum re-derives against** (doctrine D): the `0` is green only because this
bit-exact test exists and passes.

**Provenance: WITNESSED** — fak's own kernel, fak's own bit-exact assertion. The
G1 lane is never OBSERVED (no provider relays it) and never MODELED (there is
nothing projected — it is run).

---

## The per-evict datum (re-derivable schema, binds the existing test/verdict)

The job of C3 is one datum per eviction *event*, carrying the bit-identity verdict
as a WITNESSED point the heatmap renders and the C9 checker re-derives. The schema
adds **no compute** — every field is read from an existing source named in its
comment.

```
EvictCorrectness {
  EventSeq      uint64  // the journal CAP_EVICT Row.Seq this datum binds (journal.go:61, kind set journal.go:391)
  CapName       string  // journal Row.CapName — which span/capability was evicted (journal.go:87)
  From          int     // KVCache.Evict(from, n) span start (kvcache.go:94)
  N             int     // KVCache.Evict(from, n) span width
  MaxAbsDelta   float64 // 0 == bit-identical survivor+logits; >0 == drift. WITNESSED via assertFloat32BitsEqual (rope_test.go:68)
  Outcome       string  // "bit_clean" | "refused_unsupported" | "panic"  (the CanEvict/TryEvict verdict, kvcache.go:71/105)
  RefusedLayers []int   // RecurrentEvictUnsupportedError.Layers when Outcome=="refused_unsupported" (kvcache.go:56)
  WitnessTest   string  // the binding test that proves MaxAbsDelta: "TestPagedEvictBitIdenticalToContiguous" | "TestLongropeEvictEqualsNeverSaw"
  Provenance    string  // always "WITNESSED" for G1 (fence b)
  Rederived     bool    // doctrine D: true ONLY when MaxAbsDelta re-derives green from WitnessTest's last run; false => RED cell
}
```

Field bindings — every one grounded, none computed fresh:

- **`EventSeq` / `CapName`** bind to the tamper-evident event source: the
  hash-chained journal `Row` (`internal/journal/journal.go:60-91`), whose `Kind`
  enumerates `CAP_EVICT` (`journal.go:63`, set at `journal.go:391`) with the
  monotonic `Seq` (`journal.go:61`) and the capability fields populated for
  capability lifecycle events (`journal.go:83-87`). A `CAP_EVICT` row that left no
  journal entry is, by the C1 doctrine, *invisible* — and invisible is the worst
  class; the datum's existence is anchored to that row.
- **`From` / `N`** are the `KVCache.Evict(from, n)` span coordinates
  (`kvcache.go:94`) — the *what was removed* half of the event.
- **`MaxAbsDelta`** is the load-bearing number. It is **`0` by construction when
  `Outcome == "bit_clean"`**, because that outcome is *defined* by the bit-exact
  test passing (`assertFloat32BitsEqual`, `rope_test.go:74`, fails on one
  differing bit). A non-zero value is only renderable if the test *regressed* — in
  which case the cell is RED and `Rederived == false`.
- **`Outcome` / `RefusedLayers`** carry the typed verdict so the heatmap has three
  distinguishable cells, not a binary: `bit_clean` (supported, `max|Δ|=0`),
  `refused_unsupported` (the fail-closed `RecurrentEvictUnsupportedError`,
  `kvcache.go:44-56` — cache left unchanged, naming the recurrent
  `RefusedLayers`), and `panic` (an unchecked caller hit `Evict` on an unsupported
  cache, `kvcache.go:94-99`). A fail-closed refusal is **not** a value loss and is
  **not** a green `0` — it is its own visible cell, the honest "we declined to
  evict in place; rebuild from a clean prefix" verdict (`kvcache.go:52-53`).
- **`Rederived`** is doctrine **D**: green only when the visible `MaxAbsDelta`
  *re-derives* from the binding `WitnessTest`'s last run (the C9 checker reruns /
  reads the test verdict and reds on mismatch). A `0` with no passing test behind
  it is **not** green — it is the unearned-OK the whole epic forbids.

---

## Cross-check — how a reader falsifies the G1 cell from the visual itself

Per the C1 self-checking-visual doctrine, the heatmap cell for one evict is honest
only if a reader can falsify it without a second tool. G1 carries **C + D** (the
two C6 schemas the heatmap is specced under, `schemas-1217.md:134-137`):

- **C — provenance lane.** Every G1 cell is rendered in the **WITNESSED** channel
  (color/hatch, reusing the `tools/check_provenance_labels.py` vocabulary). G1 has
  no OBSERVED or MODELED variant — a G1 cell rendered in any other lane is itself
  a defect the lane exposes. This keeps fence (b) visible: the bit-identity `0` is
  never summed with a provider `cache_read` (OBSERVED) or a projected page-down
  value (MODELED).
- **D — the `0` proves its own correctness, re-derived.** The cell is green only
  when `MaxAbsDelta == 0` **and** `Rederived == true` — i.e. the number
  re-derives from the bit-exact test (`assertFloat32BitsEqual`, `rope_test.go:68`)
  that proved it. This is the same "rebuild and prove equal" discipline the tree
  already uses for the index (`ctxplan/image.go` proves
  `RestoreIndex(ix.Image()) == ix`) and the journal (`Verify` / `VerifyRows`,
  `journal.go:577` / `journal.go:619` recompute the chain) — here aimed at the
  evict number. A `MaxAbsDelta` that cannot be re-derived from a passing
  `WitnessTest` reds the cell; it does **not** render the typed-in `0` as OK.

The heatmap cell therefore answers, falsifiably, three things at once: *which span
was evicted* (`From`/`N`/`CapName`), *what the survivor delta was* (`MaxAbsDelta`,
green only at `0`), and *whether that `0` is backed by a re-run proof* (`Rederived`
+ `WitnessTest`). No single trust-me number.

---

## Honest `not yet` gaps (per fence c)

Per fence (c), the pieces with **no closing witness in the tree today** are named
here, not silently assumed green:

- **No per-event G1 datum is emitted today — only the tests exist.** The
  bit-identity is proven inside `go test` (`paged_evict_test.go:84`,
  `longrope_test.go:179`); it is **not** lowered into a `CAP_EVICT` journal row's
  surface as an `EvictCorrectness` datum. The `EvictCorrectness` schema above is
  the spec for that lowering; until the build epic emits it, a live `CAP_EVICT`
  row carries no bound `max|Δ|` and the heatmap has no per-event point to render —
  only the standing test pass/fail. This is the exact surface C3 specs and the
  build epic closes.
- **`Rederived` has no checker yet (doctrine D is a design target).** Re-deriving
  the visible `MaxAbsDelta` from the binding `WitnessTest`'s last run is child C9
  (#1227); no checker re-derives a *rendered* G1 cell from its source and reds on
  mismatch yet — the journal `Verify` and `ctxplan/image.go` round-trip are the
  in-tree *precedents*, not a G1 checker. Until C9 lands, `Rederived` defaults
  conservatively to `false` (the cell is not credited green) rather than trusting
  a typed `0`.
- **`refused_unsupported` for the hybrid recurrent cache is fail-closed but not
  yet surfaced as a heatmap cell.** `CanEvict` / `TryEvict` already return the
  typed `RecurrentEvictUnsupportedError` (`kvcache.go:44-56`, `:71`, `:105`) and
  leave the cache unchanged, so the *verdict* is witnessed — but no visual renders
  that refusal as its own (non-green, non-loss) cell distinct from a `bit_clean`
  `0`. The witness is closed; the surface is `not yet`.

These are doctrine content (named gaps), not green results — a silent gap here is
the invisible-value-destruction the epic exists to kill.

---

## Acceptance check (against #1220)

The issue's acceptance: *spec guarantee G1 — the bit-identity (`max|Δ|=0`)
evict-correctness signal — as a per-event WITNESSED datum feeding primitive (4),
binding the existing test/verdict with a re-derivable schema; no new compute
claim.*

- **G1 surfaced, not invented.** The note cites the existing primitive
  (`kvcache.go:94` `Evict`, `:71` `CanEvict`, `:105` `TryEvict`, `:44` the typed
  `RecurrentEvictUnsupportedError`) and the existing **passing** proofs
  (`paged_evict_test.go:84` `TestPagedEvictBitIdenticalToContiguous`,
  `longrope_test.go:179` `TestLongropeEvictEqualsNeverSaw`, via the bit-exact
  `assertFloat32BitsEqual` at `rope_test.go:68`). **No new compute claim** is made.
- **A re-derivable per-event datum** is specced: `EvictCorrectness{EventSeq,
  MaxAbsDelta, Outcome, WitnessTest, Provenance, Rederived, ...}`, each field bound
  to a real source (journal `CAP_EVICT` `Row` `journal.go:60-91`/`:391`; the
  `Evict` span coordinates; the bit-exact test as `WitnessTest`).
- **`0` proves its own correctness (D) + provenance (C).** The cell is green only
  when `MaxAbsDelta == 0` **and** `Rederived == true` (re-derived from the binding
  test), rendered in the **WITNESSED** lane, never summed with OBSERVED/MODELED
  (fence b).
- **The honest gaps** — no per-event datum emitted yet, no `Rederived` checker
  yet (C9), the fail-closed refusal not yet a heatmap cell — are named as `not
  yet`, not rendered as results.

---

_Filed as research / planning only under epic #1217. Parent: #1217. Surfaces
guarantee G1 of the C1 doctrine (#1218); feeds primitive (4), the
evict-correctness heatmap schema'd in C6 (#1223); renders the C2 failure-catalog's
evict-correctness (F1-adjacent) class (#1219); re-derived by the C9 checker
(#1227). Kept distinct from the security floor and from the #1147
self-tax/mediation-overhead plane — cross-linked, never blended. Design-only — no
implementation ships under #1217 until the notes are reviewed._
