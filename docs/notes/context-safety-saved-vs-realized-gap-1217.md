# Context-safety saved-vs-realized gap (#1221, C4 of epic #1217)

_Research / design only. This is the **C4 value-decomposition spec** for the
context-safety epic [#1217](https://github.com/anthony-chaudhary/fak/issues/1217).
It specs the one value-gap that matters for the MVP slice — **claimed-saved
tokens at page-out time vs later-realized reuse** — over the two WITNESSED taps
that already exist, provenance-labeled, **never blended** with the OBSERVED
provider `cache_read`. No code ships here — the deliverable is this committed
spec: the schema of the gap datum, the two derivations it is built from, the
cross-checks it carries, and the structured refusal it emits when a derivation
can't be earned. It binds guarantee **G2** of the C1 doctrine note
([#1218](https://github.com/anthony-chaudhary/fak/issues/1218),
[`context-safety-visuals-tracking-1217.md`](context-safety-visuals-tracking-1217.md))
and renders failure **F3** of the C2 catalog
([#1219](https://github.com/anthony-chaudhary/fak/issues/1219),
[`context-safety-failure-mode-catalog-1217.md`](context-safety-failure-mode-catalog-1217.md))._

---

## The gap, in one line

> A page-out **claims** it saved N tokens at event time. The saved bytes are
> only worth N if they are **reused** on a later turn. The claim is a forecast;
> the realized reuse is the witness. **The gap between them is the error bar** —
> and it is the value half of a context-removal event, the half a single
> "tokens saved" counter structurally cannot show.

This is the saved-vs-realized gap chart (#1217 visual primitive 2). It is the
**MVP value-gap** because it is the cheapest honest cross-check fak can make:
both halves are already WITNESSED taps in the tree, so the gap is a subtraction
of two real numbers, not a new measurement.

---

## The two witnessed derivations

The gap is built from exactly two taps, **both WITNESSED** (fak authored and
controls them — fak's own RadixAttention prefix match, not a provider relay):

### D1 — claimed-saved at event time (the forecast)

At a page-out / evict / compact event, fak records how many tokens it *expects*
to save — the span it is shedding that it forecasts will be reused. This is the
event-time half, and it lives on the context-event source of truth named by the
C1 doctrine:

- The hash-chained journal `internal/journal/journal.go` `CAP_EVICT` row carries
  the event and its claimed token count, `Seq`/`PrevHash`/`Hash` making it
  tamper-evident.
- The execution path `internal/engine/cacheevents.go` `CacheEventRecorder`
  (`cacheevents.go:304-317`) normalizes each demote/spill/evict transition into a
  `cachemeta.Entry` and emits the `fak_engine_cache_*` metric family — the seam
  between a placement *decision* and the *executed* event.

D1 is a **claim**: it is what fak forecast at event time, and it is honest only
when it is later checked against what actually bit.

### D2 — realized reuse on a later turn (the witness)

On a later turn, fak's RadixAttention prefix match either reuses the saved span
or it does not. The realized half is the WITNESSED `kv_prefix` family:

- `internal/cachewitness` folds the family into one provenance-labeled record:
  `KVPrefixWitness.ReuseRatio()` (`cachewitness.go:155`) over the WITNESSED
  metric `fak_gateway_kv_prefix_reused_tokens_total`
  (`internal/gateway/metrics.go:1531`), with the ratio surfaced at
  `metrics.go:1538`.
- The live tap is `cacheobs.Default.Snapshot()` — the in-kernel reuse counter,
  not a number relayed from outside.

D2 is the **witness**: it is what actually happened, recorded from fak's own
cache, not forecast.

### The gap datum

```
SavedRealizedGap {
  EventSeq        uint64   // journal Seq of the CAP_EVICT/page-out row (D1 anchor)
  EventKind       string   // evict | page_out | compact | reset
  ClaimedSaved    int      // D1: tokens fak forecast it would save at event time
  RealizedReused  int      // D2: tokens actually reused on a later turn (WITNESSED)
  Gap             int      // ClaimedSaved - RealizedReused (the error bar; >0 = over-claim)
  ReuseRatio      float64  // D2 / ClaimedSaved, in [0,1]; the realized fraction
  Provenance      string   // always "WITNESSED" for both halves; see fence below
  Reconciled      bool     // does D2's count reconcile with the coherence witness? (C14)
  Refusal         string   // "" | VALUE_UNRECONCILED | SAFETY_UNWITNESSED (C13)
}
```

`Gap = ClaimedSaved - RealizedReused`. A positive gap is an **over-claim**: fak
said it saved value it never realized. The chart plots `ClaimedSaved` and
`RealizedReused` as the two paired axes and renders `Gap` as the error bar
between them — primitive 2's defining shape.

---

## The cross-checks this datum carries

Per the C1 self-checking-visual doctrine, every datum names the mechanisms that
let a reader falsify it from the visual itself. The gap carries **A + D**:

- **A — paired-axis invariant.** The two axes *are* `ClaimedSaved` and
  `RealizedReused`. Showing only "tokens saved" (D1) is the lie this primitive
  exists to defeat: a forecast with no realized backing renders as a tall claim
  bar with a flat realized bar — the gap is visible, not hidden.
- **D — re-derivation.** Both halves re-derive from a tamper-evident source: D1
  from the journal `CAP_EVICT` row (replayable via
  `internal/journal/journal.go` `Verify`/`VerifyRows`), D2 from the
  `fak_gateway_kv_prefix_reused_tokens_total` counter and
  `cacheobs.Default.Snapshot()`. The panel is RED unless `ClaimedSaved`
  re-derives from the journal **and** `RealizedReused` re-derives from the
  metric tap. The checker that performs this re-derivation is child C9
  ([#1227](https://github.com/anthony-chaudhary/fak/issues/1227)); this spec
  defines *what* it re-derives, C9 defines *how*.

The second independent derivation D2 must reconcile against (the **correctness**
of the reused prefix, not just its count) is the subject of child C14
([#1232](https://github.com/anthony-chaudhary/fak/issues/1232),
`internal/cachemeta/prefix_coherence.go:51` `EvaluatePrefixCoherence`): a high
reuse *count* over a *broken* prefix is value that looks real and isn't. When
the count and the coherence witness disagree, `Reconciled = false` and the datum
fires `VALUE_UNRECONCILED` (C13, #1231). This spec carries the field and the
refusal; C14 owns the coherence derivation it reconciles against.

---

## The provenance fence — never blend with `cache_read`

The single load-bearing honesty fence on this primitive (#1217 fence b):

> The saved-vs-realized gap is built from **two WITNESSED taps**. The provider's
> `cache_read` token count is **OBSERVED** — relayed from an external party. It
> is **never summed into, never substituted for, and never plotted on the same
> axis as** the realized-reuse witness.

This is not a stylistic preference. The C1 doctrine proved *why* with fak's own
247-session corpus: provider cache-hit % (`cache_read/(cache_read+fresh)`) is a
**denominator artifact** that climbs to ~1.0 mechanically with session length
regardless of quality (0.88 under 50 turns → 0.99 at 150–200). If the
saved-vs-realized gap quietly used `cache_read` as its realized half, the gap
would close to zero on every long session *by construction* — the chart would
render "all saved value realized" precisely when the window is most bloated.
`RealizedReused` must come from fak's own RadixAttention prefix match
(`cachewitness` / `cacheobs`), which is invariant to that artifact because it
counts *fak's* reuse, not the provider's cache accounting.

The provider `cache_read` MAY appear on the same panel **only** as a separate,
visibly-labeled OBSERVED lane (doctrine C, the provenance channel) — never
merged with the WITNESSED axes.

---

## Honest `not yet` gaps (per fence c)

- **D1's per-event claimed-saved count is not yet a first-class journal field.**
  The `CAP_EVICT` row records the event and is tamper-evident, but the
  *forecast* token count (what fak expected to save) is today carried in the
  `fak_engine_cache_*` metric stream, not pinned to the row as a re-derivable
  field. C9's re-derivation checker needs D1 anchored to the journal `Seq` so
  the gap is reproducible from the chain alone. Binding `ClaimedSaved` to the
  `CAP_EVICT` row is the one schema change this primitive asks of the event
  source — named here, not assumed.
- **The reconciliation against coherence (C14) is `not yet`.** Until
  `EvaluatePrefixCoherence` is wired as D2's second derivation, `Reconciled`
  defaults to *unknown* and the datum may not render green on the strength of
  the count alone — it carries `SAFETY_UNWITNESSED` rather than an unearned OK.

---

## Acceptance check (against #1221)

The issue's acceptance: *the gap is computed from two witnessed taps,
provenance-labeled, never blended with the OBSERVED provider `cache_read`.*

- **Two witnessed taps.** D1 (claimed-saved, journal `CAP_EVICT` +
  `cacheevents.go` `CacheEventRecorder`) and D2 (realized reuse,
  `cachewitness.go:155` `ReuseRatio()` over `metrics.go:1531`
  `fak_gateway_kv_prefix_reused_tokens_total`, fed by
  `cacheobs.Default.Snapshot()`) — both WITNESSED, both cited to a real
  `file:line`.
- **Provenance-labeled.** The `Provenance` field and the doctrine-C lane keep
  every number tagged; the gap datum is WITNESSED on both halves.
- **Never blended with `cache_read`.** The provenance fence is stated with its
  grounding (the cache-hit denominator artifact) and the rule that `cache_read`
  may only ever be a separate OBSERVED lane, never an axis of the gap.
- **Cross-checks named.** A (paired-axis saved-vs-realized) + D (re-derive both
  from tamper-evident sources, the C9 checker), with the C14 coherence
  reconciliation field and the C13 refusal token carried for the
  count-vs-correctness disagreement.

---

_Filed as research / planning only under epic #1217. Parent: #1217. Binds
guarantee G2 of the C1 doctrine (#1218); renders failure F3 of the C2 catalog
(#1219); reconciled against the reuse-correctness witness C14 (#1232);
re-derived by the C9 checker (#1227); refuses green via the C13 DOS tokens
(#1231). Design-only — no implementation ships under #1217 until the notes are
reviewed._
