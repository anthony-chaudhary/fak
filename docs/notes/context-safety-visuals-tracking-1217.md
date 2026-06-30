# Context-safety doctrine (#1218, C1 of epic #1217)

_Research / design only. This is the **doctrine note** — the design note for the
context-safety epic [#1217](https://github.com/anthony-chaudhary/fak/issues/1217).
It states the **property** context safety is, the **four value guarantees** that
decompose "real retained value," the **self-checking-visual doctrine** every
visual under this epic must obey, and the **honesty fences** that keep the
property honest. No code ships here — the deliverable is this committed note,
reviewable as a design before any child is scoped. Its sibling is the C2
failure-mode catalog
([#1219](https://github.com/anthony-chaudhary/fak/issues/1219),
[`context-safety-failure-mode-catalog-1217.md`](context-safety-failure-mode-catalog-1217.md)),
which catalogs every failure this doctrine's plane must surface; read the two
together — this note names the guarantees, that one names what breaks them._

---

## The property — context safety is the value-preservation dual of the security floor

fak already proves a **security floor**: a *bad* call cannot ENTER the model's
context. That floor is result-side — the gate, quarantine, addressable evict —
and it is checkable: a denied call leaves a tamper-evident `DENY` / `QUARANTINE`
row in the hash-chained journal.

fak has **no equivalent floor on the other side of the wall.** When fak *removes*
something from context — evict, page-out, compact, shed, reset — it makes no
standing guarantee that it did **not silently destroy value**, and it gives the
operator no way to **see and cross-check** that it didn't.

> The security floor proves a *bad* call can't get *in*. The context-safety floor
> proves a *good* call's value doesn't silently get *lost* when fak sheds context
> — and shows it as a **cross-checkable picture**, not a number you have to trust.

That is the property this epic designs. It is the **dual** of the security floor:
same honesty DNA (witnessed, not self-reported; refuse-green rather than render
an unearned OK), opposite polarity (preserve value vs deny entry), and it is
**kept strictly distinct** from the mediation-overhead plane (#1147, the self-tax
plane) — that plane proves fak's mediation isn't *slower* than budget; this one
proves fak's shedding doesn't *lose value*. Same DNA, orthogonal axis,
cross-linked, never blended (fence a).

Why a single number can't carry this property — and we already proved it. fak
ships `internal/ctxplan/signalnoise.go` *because* provider cache-hit %
(`cache_read/(cache_read+fresh)`) is a denominator artifact: it climbs to ~1.0
**mechanically** with session length regardless of quality (measured on fak's own
247-session corpus: 0.88 under 50 turns → 0.99 at 150–200; two sessions both
at ~99% OBSERVED provider cache-hit differ 10× in density). High cache-hit on a bloated window is
*efficiently caching garbage*. The lesson generalizes: **every context-safety
claim needs an invariant metric AND a second orthogonal axis that catches the
first being gamed.** A visual showing one axis is a lie by construction. The four
guarantees and the self-checking doctrine below are the structural answer to
that.

---

## The context-event source of truth

A context-safety visual has two halves: the **event** (context X was removed) and
the **value guarantee** (which of G1–G4 held). The event half has a single
canonical, tamper-evident source.

**Primary — the hash-chained journal** (`internal/journal/journal.go`). The `Row`
schema (`journal.go:59-90`) already records the capability/context lifecycle
events `CAP_FAULT | CAP_EVICT | CAP_VERSION_BIND` with `Seq` (monotonic order),
`PrevHash` / `Hash` (the chain — tamper-evident), and a `Witness` field (the
bounded-disclosure claim the verdict surfaced). The chain is **replayable and
re-verifiable**: `Verify(path)` (`journal.go:577`) and `VerifyRows(rows)`
(`journal.go:619`) recompute the hash chain from the rows themselves. The event
ribbon (primitive 1) and the re-derivation checker (C9) both read from this — the
ribbon renders the rows, the checker re-derives every number against them and
reds on a mismatch. **This is the anchor: a context-removal event that left no
journal row is, by this doctrine, invisible — and invisible is the worst class.**

**Secondary event sources** (lowered into the same observability stream, not a
second chain):

- `internal/engine/capacity_adapter.go` — `CacheEventRecorder` is the control
  path that EXECUTES a `cachemeta` demote/spill/evict against the kernel-owned KV
  tier and emits the transition as `fak_engine_cache_*` metrics. It is the seam
  between a placement *decision* and the *executed* event.
- `internal/cachemeta/kvtransfer.go` — `KVTransfer{Direction, Tokens, Outcome}`
  with `Direction ∈ {offload, restore, route, migrate}` and the all-important
  `Outcome ∈ {ok, missed, fault}` (`kvtransfer.go:13-27`) plus `FaultReason`.
  This is the **tiered page-down event with a visible outcome** — it ties the
  multilevel-cache epic (#985) to an observed ok/missed/fault instead of a
  trusted "it's on disk."

The journal is the tamper-evident anchor; the two secondary sources carry the
finer-grained tier transitions. A visual must name which source each point came
from (the provenance lane, doctrine C below).

---

## The four value guarantees — what "real retained value" decomposes into

"Value" is not one thing. The operator must know BOTH that context X was saved
AND a metric that **guarantees its real retained value** — and that guarantee
decomposes into four, strongest first, **each already backed by an in-repo
witness seam**. A context-safety visual is honest only when it shows the *event*
and *which of G1–G4 held* for that event.

### G1 — Bit-identity of the survivor (`max|Δ| = 0`)

**What it proves.** Evicting a middle span left the cache **bit-identical to one
that never saw the evicted span** — `max|Δ| = 0` on the surviving cache *and* the
next-token logits. Not "close enough"; identical. This is the strongest
guarantee because the number proves its own correctness: 0 or it isn't.

**Witness seam (already tested).** The typed boundary is
`internal/model/kvcache.go:44-105` — `RecurrentEvictUnsupportedError` /
`CanEvict()` (`:71`) / `Evict()` (`:94`) / `TryEvict()` (`:105`). The correctness
is proven by `internal/model/paged_evict_test.go:84`
`TestPagedEvictBitIdenticalToContiguous` (a mid-span evict on a deliberately
non-contiguous page table is byte-for-byte identical to the contiguous `Evict` —
cache and logits, `max|Δ|=0`) and `internal/model/longrope_test.go:179`
`TestLongropeEvictEqualsNeverSaw` (the survivor equals a cache that never saw the
evicted span, under longrope re-rotation). **The guarantee already exists and is
tested** — the job (child C3, #1220) is to surface it as a per-event witnessed
datum and cross-check it visually (primitive 4, the evict-correctness heatmap),
not to invent it.

**Provenance:** WITNESSED (fak's own kernel, fak's own assertion).

### G2 — Reuse actually bit, and was *correct*

**What it proves.** Two parts. (a) The saved bytes were **reused** on a later
turn — a *count*, the realized half of a save claim. (b) The reused prefix was
**coherent** — a *correctness* witness, because a high reuse count over a *broken*
prefix is value that looks real and isn't.

**Witness seam.** Count: `internal/cachewitness` folds the WITNESSED `kv_prefix`
family — `KVPrefixWitness.ReuseRatio()` (`cachewitness.go:155`) over
`fak_gateway_kv_prefix_reused_tokens_total` (`internal/gateway/metrics.go:1531`),
fed by `cacheobs.Default.Snapshot()` — fak's OWN RadixAttention prefix match, not
a provider number. Correctness: `internal/cachemeta/prefix_coherence.go:51`
`EvaluatePrefixCoherence` (fail-closed) + `prefix_stability.go` `StabilityReport`.
The two must **reconcile** — disagreement (high count, broken prefix) is the
failure, and is the second independent derivation child C14 (#1232) specs against
C4's saved-vs-realized number. This guarantee drives primitive 2, the
saved-vs-realized gap chart — *the gap is the error bar*.

**Provenance:** WITNESSED for fak's RadixAttention reuse and the coherence
verdict; the provider `cache_read` stays OBSERVED and is **never blended** with
it (fence b).

### G3 — Signal retained, not noise

**What it proves.** The shed dropped *noise* and did **not** induce *faults*
(spans that were elided but then needed and demand-paged back). A compaction that
raises signal-to-noise by *starving the turn* is value destruction wearing a
cleanup costume — and a single-axis "S/N improved" visual hides exactly that.

**Witness seam.** The two orthogonal axes already exist as a pure function of
`ctxplan.Outcome`: `internal/ctxplan/signalnoise.go:69` `Ratio()` (the
caching/length-invariant signal axis — the answer to the cache-hit denominator
artifact), `:81` `FaultRatio()` (the orthogonal under-resident axis), graded by
`:249` `Grade()` (lean / ok / bloated / **starving**). `UnaccountedTokens`
(`:48,51`) is kept in the denominator so the ratio can never over-claim — the same
conservation discipline doctrine B below requires. This drives primitive 3, the
S/N + Faults paired strip; the `starving` grade is the named failure verdict.

**Provenance:** WITNESSED.

### G4 — Integrity of the cell

**What it proves.** A paged-out / recalled memory cell did **not silently rot** —
the bytes on disk still match the digest they were written under. The page reads
back, but it must be *the page that was saved*, not a corrupted one that merely
deserializes.

**Witness seam.** `internal/recall` `PageSyndrome` (`page_syndrome.go:101-122`):
the five-axis ECC-style per-page integrity roll-up — **digest** miss,
**quarantine**, write-time **durability** class, **witness** revocation,
**trust-epoch** staleness. `PageSyndromeFor` (`:122`) **recomputes** the syndrome
against the page's CAS body (a re-derivation, doctrine D); `Reusable()` (`:239`) /
`FailedEvidence()` (`:251`) name which axis failed; the repairable-vs-erasure
verdict is `fault_syndrome.go` (`SyndromeClass`: a digest miss → ERASURE). This
extends primitive 4 (the evict-correctness heatmap) to per-cell integrity.

**Provenance:** WITNESSED (the digest re-derivation is fak's own). Honesty note:
a REPAIRABLE syndrome is MODELED-recoverable, an ERASURE is a hard WITNESSED loss
— distinct lanes, never blended.

### Guarantee → witness → primitive at a glance

| # | Guarantee | What it proves | Existing witness seam | Primitive | Provenance |
|---|---|---|---|---|---|
| **G1** | Bit-identity of the survivor (`max\|Δ\|=0`) | evict left the cache bit-identical to one that never saw the span | `kvcache.go:44-105`; `paged_evict_test.go:84`, `longrope_test.go:179` | (4) evict-correctness heatmap | WITNESSED |
| **G2** | Reuse bit, and was correct | saved bytes reused later (count) AND prefix coherent (correctness) | `cachewitness.go:155` + `metrics.go:1531`; `prefix_coherence.go:51` | (2) saved-vs-realized gap | WITNESSED (provider `cache_read` OBSERVED, never summed) |
| **G3** | Signal retained, not noise | shed dropped noise, induced no faults | `signalnoise.go:69` `Ratio()` / `:81` `FaultRatio()` / `:249` `Grade()` | (3) S/N + Faults paired strip | WITNESSED |
| **G4** | Integrity of the cell | paged-out cell didn't silently rot | `recall/page_syndrome.go:101-122`, `fault_syndrome.go` | (4) heatmap (extended per-cell) | WITNESSED (REPAIRABLE = MODELED, never blended) |

---

## The self-checking-visual doctrine — a visual is honest only if the reader can falsify it from the visual itself

The operator's strongest requirement: **not a single trust-me number.** A visual
is *cross-checkable* only if a reader can **falsify it from the visual itself**.
Four mechanisms, increasing strength — every visual under this epic uses at least
**A + B**; the strongest use **D**.

- **A — Paired-axis invariant.** Show the metric *and* the axis that catches it
  being gamed: S/N **with** Faults; saved **with** realized; evicted **with**
  `max|Δ|`. One axis alone is a lie — this is the structural answer to the
  cache-hit denominator artifact.

- **B — Conservation roll-up.** A roll-up whose segments are witnessed sub-counts
  that **must sum to the total** (resident = signal + noise + unaccounted;
  evicted = bit-clean + reset + refused). If they don't sum, the visual is
  *visibly* broken — the eye does the check, no second tool needed. `ctxplan`
  already keeps `unaccounted` in the denominator so it never over-claims — same
  discipline (`signalnoise.go:48`).

- **C — Provenance lane.** Every point carries WITNESSED / OBSERVED / MODELED as
  a **visible channel** (color / hatch), reusing the
  `tools/check_provenance_labels.py` vocabulary. A modeled point rendered
  identically to a witnessed one is exactly the conflation the repo's whole
  scorecard family fights. No primitive may render a modeled point identically to
  a witnessed one.

- **D — Re-derivation / two-derivation reconciliation.** The strongest, and the
  one genuinely missing piece (see `not yet` below). A number is shown green only
  when (i) it **re-derives** from a tamper-evident source (the guard journal /
  git / disk / `/metrics`) — RED unless they match — and, for the strongest
  panels, (ii) **two independent derivations reconcile** (realized-reuse *count*
  G2(a) must reconcile with the *correctness* witness G2(b); a high count over a
  broken prefix is value that looks real and isn't). When a derivation *can't* be
  earned, the panel **refuses green with a structured DOS reason**
  (`SAFETY_UNRECOVERED` / `VALUE_UNRECONCILED` / `SAFETY_UNWITNESSED`, child C13
  #1231) rather than rendering an unearned OK. **The visual *is* a witness, not
  decoration** — it ships *with* its re-derivation checker.

  **In-repo precedents to generalize.** `ctxplan/image.go` already proves the
  rebuild-and-prove-equal discipline for the *index* — the loader never trusts
  derived state, it rebuilds and proves `RestoreIndex(ix.Image()) == ix`
  (`image.go:61,71`, with `image_test.go` the round-trip witness). The journal is
  replayable and `Verify` / `VerifyRows` recompute the chain. `check_memory.py`
  re-derives caps from disk; `dispatch_status.py`'s `closure_rate` is
  strict-diff-witnessed; `writeVCacheMetrics` (`metrics.go:1989,2016`) carries a
  `proven` break-even gate that refuses to claim value until break-even. C9
  (#1227) extends the `RestoreIndex(ix.Image()) == ix` pattern from the index to
  the **rendered visual** — the one analog the tree does not yet have.

A context-safety visual is honest only when it shows the **event** AND the
**value guarantee** (which G held), as a roll-up or time-series (never a single
point), with a visible provenance lane and at least one cross-check that reds
when the number contradicts its own parts.

---

## The honesty fences

These are the constraints that keep the property from drifting into the
comfortable-but-false. They are the same fences the sibling catalog (#1219)
carries — stated here as the doctrine that governs them.

**(a) Value-preservation is kept distinct from the security floor — and from the
mediation-overhead plane.** The security floor (deny entry) and the
context-safety floor (preserve value on exit) are *duals*, cross-linked, never
merged into one "fak is safe" number. Likewise the self-tax plane #1147
(mediation isn't *slower* than budget) is an orthogonal axis: cross-linked, never
blended. Three planes, one honesty DNA, distinct axes.

**(b) WITNESSED / OBSERVED / MODELED are three lanes, never summed.** `max|Δ|=0`
(G1), fak's RadixAttention reuse-bit and the coherence verdict (G2), the S/N and
Faults axes (G3), and the digest re-derivation (G4) are **WITNESSED** — fak
authored and controls them. The provider `cache_read` stays **OBSERVED** —
relayed from an external party. Any projected page-down-tier value, and a
REPAIRABLE cell-syndrome's recoverability, stay **MODELED** until measured
on-device. A roll-up that adds a WITNESSED number to an OBSERVED or MODELED one is
the conflation this whole epic exists to prevent.

**(c) An un-witnessed guarantee is a `not yet`, not a result.** A roll-up that
can't re-derive from source is a `not yet` gap, named — not a green panel. A
context-safety claim with no closing witness is recorded as a gap (below), never
rendered as an OK. *A named gap is doctrine content; a silent gap is the
invisible-value-destruction this epic exists to kill.*

**(d) This epic is design-only.** Each child produces a spec / design note (like
this one), not shipped code. Nothing under #1217 ships an implementation until the
notes are reviewed.

---

## Honest `not yet` gaps (un-witnessed today)

Per fence (c), the guarantees and mechanisms that have **no current closing
witness in the tree** are named here, not silently assumed:

- **The re-derivation witness for a rendered visual (doctrine D, the strongest
  cross-check) — `not yet`.** The journal IS replayable (`Verify` / `VerifyRows`)
  and `ctxplan/image.go` proves `RestoreIndex(ix.Image()) == ix` for the *index*,
  but **no checker re-derives a rendered visual's numbers from source and reds
  when they disagree.** Visuals today are rendered downstream (Grafana over
  `/metrics`, Jekyll over scorecard JSON) and never re-validated after render.
  This is the genuinely missing piece; children C9 (#1227, the checker) and C10
  (#1228, the CI gate that reds `make ci`) close it. Until then, doctrine D is a
  *design target*, not a shipped guarantee.

- **G4 cell-integrity surfaced as a visual — partial.** `recall.PageSyndrome`
  computes the five-axis integrity syndrome and `PageSyndromeFor` re-derives the
  digest against the CAS body (the witness exists), but there is **no visual** that
  renders per-cell integrity as a roll-up. The witness is closed; the surface is
  `not yet`.

- **The structured-refusal reason tokens (`SAFETY_UNRECOVERED` /
  `VALUE_UNRECONCILED` / `SAFETY_UNWITNESSED`) — `not yet`.** They are specified
  by child C13 (#1231) as new `dos.toml [reasons]`; they do not exist yet, so a
  panel cannot today refuse-green with a verifiable reason. The doctrine requires
  them; the tokens are unfiled work.

The two failure modes with no closing witness — **rendered-visual-drift** and
**harness-rewrite-defeats-the-shed** — are catalogued in the sibling note (#1219,
gaps G-A and G-B). They are real context-safety failures named as gaps so the
plane does not render a "value preserved" that a second manager silently reversed.

---

## Acceptance check (against #1218)

The issue's acceptance: *the note states the property, the four guarantees with
their existing witness seams, the four cross-check mechanisms, and the fences —
reviewable as a design before any code is scoped.*

- **The property** is stated as the value-preservation **dual of the security
  floor** (preserve value on exit ↔ deny a bad call on entry), with the
  single-number-is-a-lie argument grounded in the measured `signalnoise.go`
  cache-hit denominator artifact.
- **The context-event source of truth** is identified: the hash-chained journal
  `internal/journal/journal.go:59-90` (`CAP_FAULT | CAP_EVICT | CAP_VERSION_BIND`
  rows, `Seq` / `PrevHash` / `Hash` / `Witness`, replayable via `Verify` /
  `VerifyRows`), with the two secondary sources mapped (`capacity_adapter.go`
  `CacheEventRecorder`; `cachemeta/kvtransfer.go` `KVTransfer` ok/missed/fault).
- **The four guarantees** (G1 bit-identity `max|Δ|=0` evict-correctness; G2
  saved-vs-realized reuse-bit + correctness; G3 S/N + Faults paired; G4
  cell-integrity) are each bound to a real, cited `file:line` witness seam and the
  primitive that renders it.
- **The four cross-check mechanisms** (A paired-axis, B conservation roll-up, C
  provenance lane, D re-derivation / two-derivation reconciliation) are stated as
  the self-checking-visual doctrine, with their in-repo precedents
  (`RestoreIndex(ix.Image()) == ix`, journal `Verify`, the `proven` vCache gate).
- **The fences** (a value-preservation distinct from the security floor and #1147;
  b WITNESSED/OBSERVED/MODELED never summed; c un-witnessed = `not yet`; d
  design-only) are stated, and the un-witnessed pieces are named as honest gaps,
  not results.

---

_Filed as research / planning only under epic #1217. Parent: #1217. Sibling C2
failure-mode catalog: #1219
([`context-safety-failure-mode-catalog-1217.md`](context-safety-failure-mode-catalog-1217.md)).
Downstream: the value-decomposition specs (C3 #1220 / C4 #1221 / C5 #1222 / C14
#1232) bind the G1–G4 seams; the visual-primitive specs (C6 #1223 / C7 #1224 / C8
#1225) schema the doctrine A–D; the re-derivation witness (C9 #1227) + CI gate
(C10 #1228) + DOS refusal tokens (C13 #1231) close doctrine D._
