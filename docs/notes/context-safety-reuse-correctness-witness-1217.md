# Context-safety reuse-correctness witness (#1232, C14 of epic #1217)

_Research / design only. This is the **C14 second-derivation spec** for the
context-safety epic [#1217](https://github.com/anthony-chaudhary/fak/issues/1217).
It specs the reuse-**correctness** witness — the *second* independent derivation
that the C4 saved-vs-realized reuse **count** must reconcile against, so a high
reuse count over a *broken* prefix is caught as value that only looks real. No
code ships here — the deliverable is this committed spec: the witness seam, the
reconciliation rule, and the structured refusal that fires on disagreement. It
deepens guarantee **G2(b)** of the C1 doctrine note
([#1218](https://github.com/anthony-chaudhary/fak/issues/1218),
[`context-safety-visuals-tracking-1217.md`](context-safety-visuals-tracking-1217.md))
and is the reconciliation partner of the C4 gap spec
([#1221](https://github.com/anthony-chaudhary/fak/issues/1221),
[`context-safety-saved-vs-realized-gap-1217.md`](context-safety-saved-vs-realized-gap-1217.md))._

---

## Why a count is not enough

> The C4 saved-vs-realized gap proves the saved bytes were **reused** — a
> *count*. But a count is blind to *correctness*: a prefix can be reused in full
> and still be **wrong** — stale, re-rotated incorrectly, or coherent only by
> coincidence. A high reuse count over a broken prefix is value that looks real
> and isn't. Two numbers that *should* agree — "how much was reused" and "was
> what we reused actually coherent" — must be **reconciled**, or the gap chart
> renders a confident green over silent corruption.

This is the doctrine's mechanism **D**, two-derivation reconciliation, applied to
G2: the realized-reuse *count* (C4) is derivation one; the reuse *correctness*
witness specced here is derivation two. They must reconcile; disagreement is the
failure.

---

## The witness seam (already in the tree)

The reuse-correctness witness is **not new** — it is two existing,
fail-closed verdicts in `internal/cachemeta`:

### EvaluatePrefixCoherence — the coherence verdict (fail-closed)

`internal/cachemeta/prefix_coherence.go:51`
`EvaluatePrefixCoherence(deps []PrefixDependency, current map[string]string)
PrefixBreakDirective` evaluates whether a reused prefix is **coherent** — that
the cached KV being reused is the KV that the current prefix would have produced,
not a stale or mismatched one — and returns a **`PrefixBreakDirective`**: the
fail-closed directive to **break** (drop and recompute) the prefix when it is
*not* coherent. The fail-closed polarity (uncertainty → break, never an
optimistic "probably fine") is exactly what a correctness witness for a value
claim must have — the burden is on proving coherence, not on disproving it. The
revoked-witness variant is `EvaluatePrefixCoherenceRevoked` (`:73`).

### StabilityReport — the stability roll-up

`internal/cachemeta/prefix_stability.go:193` `StabilityReport` (the struct,
produced by `AnalyzeStability(turns [][]PromptSegment) StabilityReport` at `:203`)
rolls per-prefix stability into a report — how stable the reused prefix has been
across turns, the second correctness axis. A prefix that is reused but *churning*
(re-derived differently turn to turn) is reuse the count credits but stability
refutes.

Together these are the **correctness** half of G2: the count (C4) says *how
much* bit; coherence + stability say *whether what bit was right*.

---

## The reconciliation rule

```
ReuseReconciliation {
  EventSeq          uint64  // the C4 SavedRealizedGap.EventSeq this reconciles
  RealizedReused    int     // C4 derivation 1: the WITNESSED reuse count
  Coherent          bool    // C14 derivation 2a: EvaluatePrefixCoherence verdict (fail-closed)
  Stable            bool    // C14 derivation 2b: StabilityReport above threshold
  Reconciled        bool    // RealizedReused > 0 IMPLIES (Coherent AND Stable)
  Refusal           string  // "" | VALUE_UNRECONCILED   (C13, #1231)
}
```

**The rule:** a non-zero `RealizedReused` (C4 credited reuse value) is reconciled
**only if** the reused prefix is `Coherent` **and** `Stable`. Formally:

> `Reconciled  ⟺  (RealizedReused == 0)  ∨  (Coherent ∧ Stable)`

- **Reused and coherent and stable** → reconciled; the gap chart's realized
  value is genuine. Green.
- **Reused but incoherent or unstable** → **not reconciled**: the count credits
  value the correctness witness refutes. The datum fires `VALUE_UNRECONCILED`
  (C13) and the gap panel refuses green — it does *not* render the high count as
  realized value.
- **Not reused at all** (`RealizedReused == 0`) → trivially reconciled (there is
  no value claim to refute); this is the ordinary saved-but-never-reused case
  (C2 failure F3), surfaced by the gap itself, not by this reconciliation.

The reconciliation is the doctrine-D check on the gap: two independent
derivations of "was this reuse real value," and the panel is green only when
they agree.

---

## Honest `not yet` gaps (per fence c)

- **No reconciliation between the reuse count and the coherence verdict exists
  today.** The count (`cachewitness` / `cacheobs`) and the coherence/stability
  verdicts (`prefix_coherence.go` / `prefix_stability.go`) are computed on
  *separate paths* and are **never compared** — that comparison is precisely
  the witness this note specs and the build epic implements. Until then, the C4
  gap's `Reconciled` field defaults to *unknown* and the datum carries
  `SAFETY_UNWITNESSED` (C13) rather than crediting the count alone.
- **The `Stable` threshold is unspecified here.** `StabilityReport` produces the
  stability axis; the *cut* (how stable is stable enough) is a tuning decision
  for the build epic, named as open rather than silently hard-coded.
- **`VALUE_UNRECONCILED` is not yet a real reason token.** Confirmed via
  `dos_check_reason` (returns `UNCLASSIFIED`); child C13 (#1231) specs its
  declaration. Until declared, this reconciliation can detect a disagreement but
  cannot *emit* the structured refusal — it can only flag it.

---

## Acceptance check (against #1232)

The issue's acceptance: *spec the reuse-correctness witness
(`EvaluatePrefixCoherence` + `StabilityReport`) as the second independent
derivation C4's saved-vs-realized number must reconcile against; disagreement
fires `VALUE_UNRECONCILED`._

- **The witness seam** is bound to real `file:line`: `prefix_coherence.go:51`
  `EvaluatePrefixCoherence` (fail-closed coherence) + `prefix_stability.go`
  `StabilityReport` (stability roll-up).
- **The reconciliation rule** is stated as the implication
  `Reconciled ⟺ (RealizedReused == 0) ∨ (Coherent ∧ Stable)`, with the
  `ReuseReconciliation` datum binding it to the C4 `SavedRealizedGap.EventSeq`.
- **Disagreement fires `VALUE_UNRECONCILED`** (C13), confirmed UNCLASSIFIED
  today via `dos_check_reason` and named as a `not yet` until C13 declares it.
- **The honest gap** — that no count-vs-coherence reconciliation exists in the
  tree yet — is named, not assumed; it is the exact surface the build epic
  closes.

---

_Filed as research / planning only under epic #1217. Parent: #1217. Deepens
guarantee G2(b) of the C1 doctrine (#1218); is the reconciliation partner of the
C4 saved-vs-realized gap (#1221); fires the C13 `VALUE_UNRECONCILED` token
(#1231) on disagreement; re-derived by the C9 checker (#1227). Design-only — no
implementation ships under #1217 until the notes are reviewed._
