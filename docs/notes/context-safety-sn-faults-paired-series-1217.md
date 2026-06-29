# Context-safety S/N + Faults paired series (#1222, C5 of epic #1217)

_Research / design only. This is the **C5 value-decomposition spec** for the
context-safety epic [#1217](https://github.com/anthony-chaudhary/fak/issues/1217).
It specs guarantee **G3** — the signal-to-noise **and** Faults **paired
time-series** (not the current single-point read) that feeds visual primitive
**(3)**, the S/N + Faults paired strip. The whole point: a shed that *raises*
S/N by *starving the turn* — eliding spans, then demand-paging them back as
faults — is value destruction wearing a cleanup costume, and a single-axis "S/N
improved" visual is a lie by construction. So the two orthogonal axes must be
charted **together over time**. No code ships here — the deliverable is this
committed spec: the series datum, the two axes it pairs, the conservation it
keeps, and the named failure verdict it surfaces. It binds guarantee **G3** of
the C1 doctrine note
([#1218](https://github.com/anthony-chaudhary/fak/issues/1218),
[`context-safety-visuals-tracking-1217.md`](context-safety-visuals-tracking-1217.md))
and renders failure **F4** of the C2 catalog
([#1219](https://github.com/anthony-chaudhary/fak/issues/1219),
[`context-safety-failure-mode-catalog-1217.md`](context-safety-failure-mode-catalog-1217.md)).
Its shape conforms to primitive (3) as schema'd by C6
([#1223](https://github.com/anthony-chaudhary/fak/issues/1223),
[`context-safety-visual-primitive-schemas-1217.md`](context-safety-visual-primitive-schemas-1217.md))
and re-derives via the C9 checker
([#1227](https://github.com/anthony-chaudhary/fak/issues/1227),
[`context-safety-rederivation-checker-1217.md`](context-safety-rederivation-checker-1217.md))._

---

## The lie a single axis tells, in one line

> A compaction reports "signal-to-noise rose from 0.6 to 0.95." It looks like a
> win. But the ratio rose because the shed **elided spans the next turns then
> needed** — every one of which was **demand-paged back as a fault**. The window
> got *leaner* by getting *insufficient*. A one-axis "S/N improved" chart renders
> a starving window as a triumph. **Only the second axis — Faults, charted on the
> same time base — catches it.**

This is the S/N + Faults paired strip (#1217 visual primitive 3). It is the
guarantee-G3 surface: *the shed dropped noise and did not induce faults.* It is
the cheapest honest cross-check fak makes against its own shedding, because both
axes already exist as a pure function of one `ctxplan.Outcome` — the series is a
fold over data fak already records, not a new measurement.

---

## Why the two axes are orthogonal (and both already in the tree)

The package comment on `internal/ctxplan/signalnoise.go:5-34` states the
property this primitive renders. Provider cache-hit % is a **denominator
artifact** (`cache_read / (cache_read + fresh)`) that climbs to ~1.0
mechanically with session length regardless of quality — measured on fak's own
247-session corpus, 0.88 short → 0.99 long, with two ~99%-cache-hit sessions
differing 10× in density (`signalnoise.go:8-14`). So fak's signal axis is **not**
cache-hit; it is the fraction of the *resident* window that *pulled weight* this
turn, computed token-weighted from a witnessed `Outcome`.

The two axes the strip pairs are both methods on the `SignalNoise` struct
(`signalnoise.go:40-60`):

### Axis 1 — the signal ratio (`Ratio`, `signalnoise.go:69`)

`Ratio() = SignalTokens / ResidentTokens` (`signalnoise.go:69-74`). It is the
caching/length-invariant signal axis — the answer to the cache-hit denominator
artifact. It is **invariant to caching**: re-reading a wasted span cheaply
(cached) does not make it signal — it is still resident-but-untouched
(`signalnoise.go:26-30`). A high ratio means the resident window ≈ the desired
window. The signal axis ALONE is exactly the gameable number, because…

### Axis 2 — the fault ratio (`FaultRatio`, `signalnoise.go:81`)

`FaultRatio() = FaultTokens / (ResidentTokens + FaultTokens)`
(`signalnoise.go:81-87`) — the **orthogonal under-resident axis**.
`FaultTokens` is the cost of spans the turn **needed** but the plan had
**elided** (`signalnoise.go:52-56`), and it is deliberately **NOT part of the
ratio**: "Reported so 'raise S/N' cannot be gamed by dropping needed spans (that
just moves cost here)" (`signalnoise.go:54-56`). The two axes are read together
by design — the doc on `FaultRatio` says it outright: high Ratio + low FaultRatio
is the target (lean and sufficient); high Ratio + **high** FaultRatio is
over-trimmed (lean but **starving**) (`signalnoise.go:77-80`). That second
combination is the gaming this primitive exists to expose.

### How the axes are computed FROM an Outcome (the relationship, stated correctly)

`Ratio` / `FaultRatio` / `Grade` are methods on **`SignalNoise`**
(`signalnoise.go:40`, fields `SignalTokens` / `NoiseTokens` /
`UnaccountedTokens` / `FaultTokens` / `ResidentTokens`, `signalnoise.go:44-59`).
A `SignalNoise` is *computed from* a `Plan` and its witnessed `ctxplan.Outcome`
by `ComputeSignalNoise` (`signalnoise.go:100-122`) — a pure, deterministic fold.
The `Outcome` (`learn.go:25-29`) is **span-ID string slices**:
`Hits` / `Faults` / `Wasted` (`learn.go:26-28`). The fold classifies each
resident `Plan.Selected` span's planned `Cost` by those labels — pinned-or-Hit →
signal, Wasted → noise, otherwise → unaccounted (`signalnoise.go:104-118`) — and
sums the **fault** cost from `Plan.Elided` spans named in `Outcome.Faults` via
`faultTokens` (`signalnoise.go:119-120,139-155`). So the **token-count fields**
live on `SignalNoise`; the **span-ID slices** live on `Outcome`; the former is
derived from the latter plus the `Plan`. (The witnessed-attention twin
`SignalNoiseFromAttention`, `signalnoise.go:207-238`, produces the same struct
with a continuous mass split, so `Ratio` / `FaultRatio` / `Grade` are unchanged —
the series can carry either derivation under one schema.)

---

## The paired-series datum

The C5 deliverable upgrades the **single-point read** the tree exposes today
(one `SignalNoise` per turn) into a **time-series sample stream** keyed on the
turn/event, so the operator sees both axes *move together* across a session:

```
SNFaultSample {
  TurnSeq        uint64   // monotonic turn/event index (the x-axis; binds to journal Seq where the turn shed context)
  EventKind      string   // "" (steady turn) | evict | page_out | compact | reset  (shed-marker on the ribbon)
  SignalRatio    float64  // Axis 1: SignalNoise.Ratio()      — signalnoise.go:69  (signal, [0,1], cache/length-invariant)
  FaultRatio     float64  // Axis 2: SignalNoise.FaultRatio() — signalnoise.go:81  (under-resident pressure, [0,1])
  Grade          string   // SignalNoise.Grade() — signalnoise.go:249: lean | ok | bloated | starving (the verdict)
  SignalTokens   int      // conservation segment: signalnoise.go:44
  NoiseTokens    int      // conservation segment: signalnoise.go:47
  Unaccounted    int      // conservation segment, KEPT IN DENOMINATOR: signalnoise.go:48,51
  ResidentTokens int      // == SignalTokens + NoiseTokens + Unaccounted (signalnoise.go:57); the sum the eye checks
  FaultTokens    int      // the under-resident mass (NOT in the resident sum): signalnoise.go:52-56
  Provenance     string   // always "WITNESSED" (fak's own Outcome); never blended with OBSERVED cache_read (fence b)
}
```

The strip plots `SignalRatio` and `FaultRatio` as **two lanes on one shared time
axis** (`TurnSeq`), with `EventKind != ""` samples marked as shed events. The
defining read: **a shed that raised `SignalRatio` while spiking `FaultRatio` on
the same `TurnSeq` is the starving failure** — visible as the two lanes diverging
*after* a shed marker, and stamped by the `Grade == "starving"` verdict on those
samples.

---

## The cross-checks this series carries

Per the C1 self-checking-visual doctrine, the strip names the mechanisms that
let a reader falsify it **from the visual itself**. It carries **A + B + D**:

- **A — paired-axis invariant.** The two axes *are* `SignalRatio` **with**
  `FaultRatio`, on one time base. Showing only S/N is the lie this primitive
  exists to defeat. The witness that this catches gaming is already a test:
  `TestSignalNoise_FaultsAreSeparateAxis` (`signalnoise_test.go:89`) proves that
  paging a needed span out does **not** raise the score — it moves cost to the
  fault axis (`FaultRatio() > 0.9`, `signalnoise_test.go:108-109`) and forces
  `Grade() == "starving"` (`signalnoise_test.go:105-106`). The strip is the
  time-series rendering of exactly that assertion: the raise-S/N-by-starving
  manoeuvre **spikes the visible Faults lane**, by construction of the metric.

- **B — conservation roll-up.** `ResidentTokens == SignalTokens + NoiseTokens +
  UnaccountedTokens` (`signalnoise.go:57`). The three resident segments must sum
  to the resident total — if they don't, the strip is *visibly* broken and the
  eye does the check. Crucially, `UnaccountedTokens` is **kept in the
  denominator** of `Ratio` (`signalnoise.go:48-51,65-66`) so the signal ratio
  **can never over-claim**: an unlabeled resident span sits as honest "unknown"
  in the denominator, never silently promoted to signal. `FaultTokens` is held
  *outside* the resident sum on its own axis (`signalnoise.go:52-56`) — it is not
  a segment of resident cost, it is the under-resident pressure, so the
  conservation bar and the fault lane are distinct quantities that the eye must
  not see merged.

- **D — re-derivation.** Both axes re-derive from the same tamper-evident source:
  the `(Plan, Outcome)` pair the sample was folded from. `ComputeSignalNoise`
  (`signalnoise.go:100`) is **pure and deterministic** — "same (plan, outcome)
  yields the same breakdown" (`signalnoise.go:98-99`) — so the C9 checker
  (#1227) re-runs the fold from the stored `Outcome` (and, where a turn shed
  context, anchors the sample to the journal `Seq` of the `CAP_EVICT` /
  `CAP_FAULT` row, `journal.go:389-391`, replayable via `Verify`
  `journal.go:577` / `VerifyRows` `journal.go:619`) and reds when a rendered
  `SignalRatio` / `FaultRatio` does not reproduce. The strip is RED unless both
  axes re-derive.

---

## The named failure verdict — `starving`

`Grade()` (`signalnoise.go:249-261`) folds the two axes into one operator-legible
word. The order is load-bearing: it checks `FaultRatio > 0.25` **first**
(`signalnoise.go:251-252`) and returns `"starving"` — so a window that is lean
*and* faulting is graded `starving` **even when its signal ratio is high**. That
is the whole defence: the gaming move (raise S/N by dropping needed spans) cannot
buy a good grade, because the fault axis is consulted before the ratio. The other
grades — `bloated` (Ratio < 0.5, the cache-hit trap in the open,
`signalnoise.go:254-255`), `lean` (Ratio ≥ 0.8 and FaultRatio ≤ 0.1, the goal,
`signalnoise.go:257-258`), `ok` (everything between, `signalnoise.go:260`) — are
the other resident states. On the strip, `Grade` is the per-sample color/label
band; a run of `starving` samples *after* a shed marker is the rendered form of
failure **F4** (S/N-up-but-faults-up) from the C2 catalog.

---

## The provenance fence — WITNESSED, never blended

The single load-bearing honesty fence on this primitive (#1217 fence b):

> Both axes are **WITNESSED** — they fold from fak's own `ctxplan.Outcome`
> (`learn.go:25-29`), which is fak's record of which resident spans the turn
> referenced, which elided spans it demand-paged, and which it never touched.
> The provider's `cache_read` is **OBSERVED** and is **never plotted on either
> axis**, never summed into the ratio, never substituted for the fault axis.

This is not stylistic. `cache_read` is the denominator artifact the whole metric
exists to route around (`signalnoise.go:5-14`); plotting it on the signal lane
would re-import the exact ~1.0-on-long-sessions inflation `Ratio` is invariant
to. If `cache_read` ever appears on the strip it is a **separate, visibly-labeled
OBSERVED lane** (doctrine C, the provenance channel) — never merged with the two
WITNESSED axes.

---

## Honest `not yet` gaps (per fence c)

- **The single-point → time-series upgrade is `not yet`.** The tree computes a
  `SignalNoise` per turn (`ComputeSignalNoise`, `signalnoise.go:100`), but there
  is **no `SNFaultSample` stream** persisted and keyed on `TurnSeq` with the
  shed-event marker — the per-turn struct is read at a point, not retained as a
  series. Building the sample stream (and binding the shed samples to the journal
  `CAP_EVICT` / `CAP_FAULT` `Seq`) is the one wiring this primitive asks of the
  event source — named here, not assumed. Until then, "S/N improved across the
  shed" is a claim no surface can falsify over time.

- **No rendered strip exists, and no checker re-validates it.** `Ratio` /
  `FaultRatio` / `Grade` are unit-tested as a point function
  (`signalnoise_test.go`), but there is **no visual** that renders the two axes
  on a shared time base and **no re-derivation checker** that reds when a rendered
  lane disagrees with a re-fold of the stored `Outcome`. The fold is pure and
  re-derivable (the witness exists); the rendered surface and its C9 cross-check
  (#1227) are the `not yet`. Per doctrine D this stays a *design target*, not a
  shipped guarantee.

- **The starving-after-shed *attribution* is observable but not yet a closed
  witness.** The strip can show `SignalRatio` up and `FaultRatio` up on the same
  `TurnSeq` as a shed marker — but proving the *specific shed* caused the
  *specific later fault* (rather than coincident drift) needs the per-event
  saved-vs-realized binding C4 (#1221) supplies on the saved side; cross-linked,
  not duplicated here. This series shows the contradiction; it does not by itself
  prove causation.

---

## Acceptance check (against #1222)

The issue's acceptance: *both axes re-derivable, charted together so a
raise-S/N-by-starving gaming spikes the visible Faults lane; the 'starving' grade
is the named failure verdict.*

- **Both axes re-derivable.** `SignalRatio` = `Ratio()` (`signalnoise.go:69`) and
  `FaultRatio` = `FaultRatio()` (`signalnoise.go:81`) both fold from one
  `(Plan, Outcome)` via the **pure, deterministic** `ComputeSignalNoise`
  (`signalnoise.go:98-122`); the shed samples anchor to the journal `CAP_EVICT` /
  `CAP_FAULT` `Seq` (`journal.go:389-391`) replayable via `Verify` /
  `VerifyRows` (`journal.go:577,619`). The C9 checker re-runs the fold and reds on
  disagreement (cross-check D).
- **Charted together so raise-S/N-by-starving spikes the visible Faults lane.**
  The two axes share one time base (`TurnSeq`); the metric is *built* so dropping
  a needed span moves cost to `FaultTokens` rather than raising the ratio
  (`signalnoise.go:52-56`), proven by `TestSignalNoise_FaultsAreSeparateAxis`
  (`signalnoise_test.go:89,105-109`). Conservation `resident = signal + noise +
  unaccounted` (`signalnoise.go:57`) with `unaccounted` in the denominator
  (`signalnoise.go:48,65-66`) keeps the ratio from over-claiming (cross-check B).
- **The `starving` grade is the named failure verdict.** `Grade()`
  (`signalnoise.go:249`) checks `FaultRatio > 0.25` first
  (`signalnoise.go:251-252`) and returns `"starving"` regardless of how high the
  signal ratio climbed — the rendered form of catalog failure F4
  (`context-safety-failure-mode-catalog-1217.md`).
- **Provenance fence.** Both axes WITNESSED (fold from fak's own `Outcome`);
  provider `cache_read` stays OBSERVED and is never plotted on or summed into
  either axis (#1217 fence b).

---

_Filed as research / planning only under epic #1217. Parent: #1217. Binds
guarantee G3 of the C1 doctrine (#1218); renders failure F4 of the C2 catalog
(#1219); conforms to primitive (3) as schema'd by C6 (#1223); re-derived by the
C9 checker (#1227); cross-linked to the saved-vs-realized gap C4 (#1221) for
per-shed causal attribution and kept strictly distinct from the #1147
self-tax / mediation-overhead plane (fence a). Design-only — no implementation
ships under #1217 until the notes are reviewed._
