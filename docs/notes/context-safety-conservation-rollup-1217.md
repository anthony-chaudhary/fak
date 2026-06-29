# Context-safety conservation roll-up contract (#1224, C7 of epic #1217)

_Research / design only. This is the **C7 conservation roll-up contract** for the
context-safety epic [#1217](https://github.com/anthony-chaudhary/fak/issues/1217).
It specs **doctrine mechanism B** — the conservation roll-up — that feeds visual
**primitive (5)**, the residency conservation bar. The mechanism: a roll-up whose
segments are **WITNESSED sub-counts that MUST sum to the total**, so an over-claim
is structurally impossible and a reader **falsifies the visual with their own
eye** — no second tool needed. No code ships here — the deliverable is this
committed contract. It builds on the doctrine note C1
([#1218](https://github.com/anthony-chaudhary/fak/issues/1218),
[`context-safety-visuals-tracking-1217.md`](context-safety-visuals-tracking-1217.md),
mechanism B), schemas the primitive whose envelope is owned by C6
([#1223](https://github.com/anthony-chaudhary/fak/issues/1223),
[`context-safety-visual-primitive-schemas-1217.md`](context-safety-visual-primitive-schemas-1217.md),
primitive 5), and renders the failure cataloged as F2 by C2
([#1219](https://github.com/anthony-chaudhary/fak/issues/1219),
[`context-safety-failure-mode-catalog-1217.md`](context-safety-failure-mode-catalog-1217.md)).
The re-derivation that turns the sum green/RED is owned by C9
([#1227](https://github.com/anthony-chaudhary/fak/issues/1227),
[`context-safety-rederivation-checker-1217.md`](context-safety-rederivation-checker-1217.md))._

---

## The mechanism — why conservation is a self-check, not a chart style

The doctrine's strongest requirement is **no single trust-me number** (C1). A
conservation roll-up answers it without a second tool: its segments are witnessed
sub-counts, and the **invariant** is that they sum to a total drawn from the same
witnessed ledger. If `sum(segments) != total`, the bar is **visibly broken** — a
gap or an overflow the eye catches at a glance. The check is *in the geometry*,
not in a tooltip you must trust.

This is exactly the conservation discipline `ctxplan` already ships and tests.
`SignalNoise` keeps **`UnaccountedTokens`** as a first-class segment
(`internal/ctxplan/signalnoise.go:48-51`) — "resident cost neither hit, wasted,
nor pinned … an honest unknown" — and `ResidentTokens` is *defined* as their sum
(`signalnoise.go:57-59`: `SignalTokens + NoiseTokens + UnaccountedTokens`).
Because the unknown stays **in the denominator**, `Ratio()` (`signalnoise.go:69`)
"never over-claims signal" (`signalnoise.go:65-66`). The C7 contract generalizes
that one property — *the residual lives in the denominator so the whole can't
over-claim* — from the S/N ratio to every context-safety roll-up.

> The conservation contract: a roll-up is honest only when its WITNESSED segments
> **provably sum to a witnessed total**, and a **residual/unaccounted** segment
> absorbs the difference so no segment can borrow against the unknown. Render the
> sum, and the operator falsifies the picture from the picture.

---

## The two named conservations

C1 mechanism B names two conservations. Each decomposes one half of the
context-safety question — *what is still in the window* and *what left it* — into
witnessed sub-counts.

### Conservation 1 — `resident = signal + noise + unaccounted`

The standing "where did the window go" roll-up. The segments and their witnessed
source already exist as the `SignalNoise` token breakdown:

- **signal** = `SignalTokens` (`signalnoise.go:44`) — resident cost the turn
  referenced (`Outcome.Hits`) plus pinned spans.
- **noise** = `NoiseTokens` (`signalnoise.go:45-47`) — resident cost the turn
  never touched (`Outcome.Wasted`), unpinned.
- **unaccounted** = `UnaccountedTokens` (`signalnoise.go:48-51`) — resident cost
  the Outcome did not label; the honest-unknown **residual**.
- **total** = `ResidentTokens` (`signalnoise.go:57-59`), defined as the sum.

The invariant `signal + noise + unaccounted == resident` is not asserted by this
contract — it is **structurally true** because `ResidentTokens` is *computed* as
that sum in `ComputeSignalNoise` (`signalnoise.go:89` onward). The C7 datum
re-exposes the three as drawable segments so the bar carries the same identity the
struct does.

The orthogonal **fault** axis (`FaultTokens`, `signalnoise.go:52-56`) is the
under-resident pressure and is **deliberately NOT a segment of this sum** — it is
cost that is *not resident* (a span the plan elided). Folding it in would let
"raise S/N by dropping needed spans" hide as a better roll-up. It rides primitive
3 (the S/N + Faults paired strip, C6) on its own axis, cross-linked, never summed
into residency.

### Conservation 2 — `evicted = bit-clean + reset + refused`

The exit-side roll-up: of everything fak removed, how much left **cleanly**, how
much was **reset away**, and how much an evict was **refused** (so the promised
realized value went to zero). The segments tie to distinct witnessed seams:

- **bit-clean** — an evict that left the survivor bit-identical (`max|Δ|=0`): the
  G1 witness `internal/model/kvcache.go` (`CanEvict`/`Evict`/`TryEvict`,
  `kvcache.go:44-105`), recorded as a `CAP_EVICT` journal row.
- **reset** — context shed by the budget auto-reset, the
  `serve --reset-on-budget` path (`cmd/fak/serve.go:110`,
  `resetOnBudgetHook` `cmd/fak/serve.go:1121,1127`) that "re-arms the continuation
  trace with a carryover seed" instead of returning 409.
- **refused** — an evict that `TryEvict` declined with
  `RecurrentEvictUnsupportedError` (`internal/model/kvcache.go:71,105`): the
  refusal is *correct*, but realized value is **zero**. This is C2 failure **F2**
  (evict-on-recurrent refused): the ribbon shows an evict was attempted while the
  value axis goes to zero. The `refused` segment being **non-zero** is exactly how
  primitive 5 surfaces F2 — and because it is a segment of the `evicted` total, a
  refused evict that *claimed* realized value would break the sum (C2 §F2:
  cross-check B).

The token-count source for these segments is the residency snapshot, below.

---

## Grounding the residency segments — `internal/ctxresidency`

The witnessed sub-counts for the residency roll-up come from the residency
snapshot, which is itself a reconciled read over the kernel's own ledgers (not a
self-report). The relevant seams:

- **`State`** (`internal/ctxresidency/ctxresidency.go:16-31`) — the per-span
  residency class `resident` / `evictable` / `held`. `held` is the
  quarantined-and-evicted class (`ctxresidency.go:28-31`: "K/V span evicted …
  bytes sealed out of context") — the exit-side population the `evicted`
  conservation rolls up.
- **`EvictBlastRadius BlastRadius{Tokens, DependentEntries}`**
  (`ctxresidency.go:39-42`) — the **cost-of-evicting** axis: the K/V positions an
  eviction would drop and the live cachemeta entries it would invalidate. It is
  the read-only projection of "the SAME invalidation walk kvmmu.evict performs"
  (`ctxresidency.go:36-38`) — so a `bit-clean` segment can carry the real blast
  radius it reclaimed, not an estimate.
- **`Snapshot`** (`ctxresidency.go:73-89`) — the roll-up itself:
  `ResidentTokens` (`:79`, reconciles with `kvmmu.Context.CacheLen`), `HeldSpans`
  (`:80`, reconciles with `kvmmu.Context.Evicted`), `ByteHeld` (`:86`),
  `ByteCleared` (`:87`), `PollutionRate` (`:88`). The doc on `:76-81` states a
  **witness test asserts both [`ResidentTokens` and `HeldSpans`] reconcile** with
  the kernel's ledger "so the query can never miscount vs the kernel's own
  ledger." That reconciliation is what makes the segments **WITNESSED** rather
  than asserted — the conservation bar inherits it.

The `Snapshot` is a pure READ (`ctxresidency.go:91-98`): it "touches no gate
state and mutates nothing." A conservation bar built from it can be re-derived at
will (doctrine D, C9) without perturbing the thing it measures.

---

## The `ConservationRollup` datum (proposed shape)

The datum the residency conservation bar (primitive 5) consumes. It rides the C6
shared envelope (`CtxSafetyPanel` / `PanelPoint`,
[`context-safety-visual-primitive-schemas-1217.md`](context-safety-visual-primitive-schemas-1217.md));
this is the per-roll-up payload its `Points` carry. The schema-version tag mirrors
the `covmatrix` precedent (`internal/covmatrix/covmatrix.go:33`
`const Schema = "fak-coverage-matrix/1"`).

```
ConservationRollup {
  Schema     string            // "fak-ctxsafety-conservation/1"  (covmatrix precedent)
  Name       string            // "resident" | "evicted"  (the two named conservations)
  Total      int               // the witnessed total (e.g. Snapshot.ResidentTokens)
  Segments   []Segment         // the witnessed sub-counts; one is the residual
  Residual   string            // the label of the residual/unaccounted segment ("unaccounted")
  At         int64             // journal Seq or timestamp (the time axis; never a single point)
}

Segment {
  Label      string            // "signal" | "noise" | "unaccounted" | "bit-clean" | "reset" | "refused"
  Tokens     int               // the witnessed sub-count (token-weighted, the planner's Cost unit)
  Lane       string            // WITNESSED | OBSERVED | MODELED  (C6 PanelPoint.Lane; never summed across lanes)
  IsResidual bool              // true for the one segment that absorbs the unknown
}

// The invariant — the whole contract. Asserted at build AND re-derived at render.
//   assert sum(Segments[i].Tokens for i) == Total
```

**The invariant assertion** `sum(segments) == total` is the contract. Two rules
make it a *self-check* rather than a label:

1. **RED-on-mismatch.** If `sum(Segments) != Total`, the panel renders **RED** and
   refuses the green/OK state — it does **not** silently normalize the segments to
   fit the total (which would hide the very drift the bar exists to catch). The
   visible gap/overflow is the falsification. Where a structured refusal is
   wanted, the panel reds with the C13 reason `VALUE_UNRECONCILED`
   ([#1231](https://github.com/anthony-chaudhary/fak/issues/1231),
   [`context-safety-dos-refusal-tokens-1217.md`](context-safety-dos-refusal-tokens-1217.md))
   rather than an unearned OK.

2. **Residual in the denominator.** Exactly one segment is the residual
   (`IsResidual = true`, label `unaccounted`). It absorbs `Total - sum(named
   segments)` so a missing/unlabeled token lands in the honest-unknown bucket
   **inside the total**, never silently dropped. This is the
   `UnaccountedTokens`-in-the-denominator discipline (`signalnoise.go:48-66`)
   lifted to the roll-up: an over-claim is **structurally impossible** because the
   one place an over-claim could hide — "tokens we can't account for" — is itself a
   visible segment of the sum, not subtracted out.

**Lane purity (fence b).** `Lane` is per-segment and segments of **different
lanes are never summed into one total.** A `resident` conservation built from
WITNESSED `Snapshot` counts is one bar; an OBSERVED provider `cache_read` figure
or a MODELED projected page-down value is a **separate** bar (or a separate,
hatched sub-track), never a segment added into a WITNESSED total. A roll-up whose
`Total` is WITNESSED but one of whose `Segments` is OBSERVED/MODELED is itself a
RED — the conservation sum is only meaningful within one lane.

---

## How the bar falsifies itself — the reader's eye is the second tool

The residency conservation bar (primitive 5) renders each `ConservationRollup` as
a stacked bar of width `Total`, segments laid end to end:

- **Sums correctly →** the segments tile the bar exactly; green is *earned* by the
  geometry, not asserted by a number.
- **`sum < Total` →** a visible **gap** at the end of the bar — value left the
  window that no segment claims. The bar is RED.
- **`sum > Total` →** a visible **overflow** past the bar's width — a segment is
  double-counting. RED.
- **`refused` non-zero (Conservation 2) →** the `refused` segment is visibly
  present — F2 is surfaced as a colored band, not buried in a "realized value"
  scalar (C2 §F2: cross-check B).

The check needs no second tool because the **invariant is the picture**. This is
the conservation half of the doctrine's "falsify it from the visual itself" (C1
mechanism B). It pairs with C9's re-derivation (doctrine D): C7 makes the sum
*visible*; C9 makes the segments *re-derive from the tamper-evident source* (the
journal `CAP_EVICT` rows, the `ctxresidency.Snapshot` reconciliation) and reds
when render and source disagree. C7 is the eye's check; C9 is the machine's.

---

## Honest `not yet` gaps (per fence c)

- **No `ConservationRollup` datum or bar renderer exists in the tree — `not
  yet`.** The witnessed sub-counts exist (`SignalNoise` segments
  `signalnoise.go:44-59`; `ctxresidency.Snapshot` `ctxresidency.go:73-89`), and
  the `resident = signal + noise + unaccounted` identity is structurally true in
  `ComputeSignalNoise`. But **no payload assembles the `evicted = bit-clean +
  reset + refused` conservation**, and **no bar renders either sum**. The datum
  and the RED-on-mismatch renderer are the build follow-on (homed by C11,
  [#1229](https://github.com/anthony-chaudhary/fak/issues/1229)); only the
  `scoreboard/render.go` text fallback (C6) can carry the segments today.

- **The `evicted` total has no single witnessed ledger yet — `not yet`.** The
  three `evicted` segments come from three seams (`kvcache.go` evict, `serve.go`
  reset, `kvcache.go:71` refused), and the residency `Snapshot` carries `HeldSpans`
  / `ByteCleared` but not a unified `evicted-token` total decomposed into
  bit-clean/reset/refused. Until a single witnessed `evicted` total is folded
  (reconciled the way `ResidentTokens`/`HeldSpans` already are,
  `ctxresidency.go:76-81`), Conservation 2's sum cannot be re-derived — so it is a
  design target, not a shipped guarantee.

- **`EvictBlastRadius` as a rendered cost-of-evicting axis is `not yet`.** The
  `Snapshot` carries `BlastRadius{Tokens, DependentEntries}`
  (`ctxresidency.go:39-42`) but no panel renders it; the conservation bar is the
  first proposed consumer (the same gap C6 names for primitive 5).

- **The `VALUE_UNRECONCILED` refusal token does not exist yet — `not yet`.** The
  RED-on-mismatch rule wants to refuse-green with a structured reason, but the
  token is specified by C13 ([#1231](https://github.com/anthony-chaudhary/fak/issues/1231))
  as a new `dos.toml [reasons]` entry and is unfiled work; until it lands the
  panel can render RED but cannot emit a verifiable refusal reason.

---

## Acceptance check (against #1224)

The issue's acceptance: *spec doctrine mechanism B — the conservation roll-up
contract feeding primitive 5 — with the two named conservations, the
`ConservationRollup` datum (total, segments, a residual/unaccounted segment, the
`sum(segments)==total` invariant), and the RED-on-mismatch rule; over-claim
structurally impossible because the unaccounted residual is kept in the
denominator.*

- **The mechanism** is stated as the conservation self-check (witnessed segments
  that must sum; the eye does the check), grounded in the
  `UnaccountedTokens`-in-the-denominator discipline `SignalNoise` already ships
  and tests (`signalnoise.go:48-66`, `ResidentTokens` defined as the sum at
  `:57-59`).
- **The two named conservations** — `resident = signal + noise + unaccounted` (the
  `SignalNoise` token breakdown, with the orthogonal `FaultTokens` axis explicitly
  *excluded* from the sum) and `evicted = bit-clean + reset + refused` (G1
  `kvcache.go:44-105` / `serve --reset-on-budget` `serve.go:110` / refused
  `kvcache.go:71`, tied to C2 F2) — are each bound to real `file:line` seams.
- **The `ConservationRollup` datum** is specced (`Total`, `Segments[]` with
  per-segment `Lane`, a `Residual`/`IsResidual` unaccounted segment, the
  `sum(segments)==total` invariant), riding the C6 envelope and the `covmatrix`
  schema-tag precedent (`covmatrix.go:33`).
- **The RED-on-mismatch rule** is stated (gap/overflow renders RED, never
  silently normalized; structured refusal via C13 `VALUE_UNRECONCILED`), and the
  over-claim-is-impossible property is grounded in the residual-in-the-denominator
  rule.
- **The residency seams** (`ctxresidency` `State` `:16-31`, `BlastRadius`
  `:39-42`, `Snapshot` `:73-89` with its kernel-ledger reconciliation) are cited
  as the witnessed source, and the un-built/un-witnessed pieces are named as
  honest `not yet` gaps, not results.

---

_Filed as research / planning only under epic #1217. Parent: #1217. Specs
doctrine mechanism B of the C1 note (#1218); feeds visual primitive 5 whose
envelope is owned by C6 (#1223); renders C2 failure F2 (#1219); re-derived and
RED-gated by C9 (#1227); refuses-green via the C13 `VALUE_UNRECONCILED` token
(#1231). Kept distinct from the security floor and the #1147 self-tax plane
(fence a), with WITNESSED/OBSERVED/MODELED segments never summed across lanes
(fence b). Design-only — no implementation ships under #1217 until the notes are
reviewed._
