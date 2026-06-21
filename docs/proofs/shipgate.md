# D12 · shipgate

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 1 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/shipgate/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

`internal/shipgate` is RSI-as-ship-gate: the propose → measure → keep-or-revert
loop with the one rule no prior auto-improver enforced — a candidate change is
**KEPT only if a witness the candidate's author did not write confirms a STRICT
metric gain**; otherwise it is **REVERTED**. The core decision procedure is the
pure function `Evaluate(Witness) (Decision, Witness)` (`shipgate.go:64`), which
folds three measured facts — a strict before/after metric gain (`improved()`,
`shipgate.go:54`), a green suite, and a clean truth syscall — into a single
non-forgeable keep-bit. A breaker (`Gate`, `shipgate.go:77`) escalates after K
consecutive non-keeps, and `ShipAdjudicator` (`adjudicate.go`) puts the in-band
ship rung on the kernel decision path. "Correct" for this regime-**D**
decision-procedure module means: (1) the keep/revert verdict is *sound* — it
admits (KEEPs) strictly less than the spec permits and fails toward REVERT — and
(2) the measurement that drives it is *deterministic*, the same inputs yielding
the same verdict every time.

---

## Theorem 1 — keep-or-revert monotonicity

**THEOREM.** For any candidate measurement `Witness w`, `Evaluate(w)` returns
`KEEP` iff `w.improved()` (a STRICT metric gain under `w.LowerBetter` direction)
**and** `w.SuiteGreen` **and** `w.TruthClean` all hold; otherwise it returns
`REVERT`. Equivalently: a candidate is kept ONLY when the measured metric
strictly improves and the non-author witness (suite + truth) is clean; every
non-strict-improving or failing-witness case reverts. The keep-bit is
non-forgeable — settable only inside `Evaluate`.

**REGIME.** D — decision-procedure soundness (sound + fail-closed verdict).

**PROOF.** The keep/revert decision is computed in `Evaluate`
(`fak/internal/shipgate/shipgate.go:64-70`):
`w.improvedBit = w.improved() && w.SuiteGreen && w.TruthClean`, returning `KEEP`
iff that bit is set, else `REVERT`. `improved()`
(`fak/internal/shipgate/shipgate.go:54-59`) is a STRICT inequality —
`After < Before` when `LowerBetter`, else `After > Before` — so equality (no
gain) and regressions both yield `false`, forcing `REVERT`. The two gating bits
`SuiteGreen` / `TruthClean` are conjoined, so a strict gain alone is
insufficient: a red suite or dirty truth syscall reverts. The keep-bit
`improvedBit` is an **unexported** field set ONLY inside `Evaluate`; `Kept()`
(`shipgate.go:74`) reads it, and the zero value is `false`, so a caller cannot
fabricate a `KEEP`. `TuneCacheSize` (`shipgate.go:147-156`) is a thin
`LowerBetter=false` adapter over the same `Evaluate`, so its keep/revert
inherits the monotonicity.

**WITNESS.**
```
(go test ./internal/shipgate/ -count=1 -timeout 120s \
  -run 'TestEvaluateKeepsStrictGain|TestEvaluateRevertsNoGain|TestEvaluateRevertsIfSuiteRed|TestTuneCacheSizeRevertsNonImproving|TestKeepBitNonForgeable' -v)
```
`TestEvaluateKeepsStrictGain` pins KEEP on a strict gain + green witness;
`TestEvaluateRevertsNoGain` covers `After==Before` (no change) **and** a
regression, both REVERT; `TestEvaluateRevertsIfSuiteRed` shows a STRICT gain
blocked by `SuiteGreen=false` and separately by `TruthClean=false` (both
REVERT); `TestTuneCacheSizeRevertsNonImproving` exercises the one-shot adapter
(equal hit-rate REVERTs, strictly-higher KEEPs); `TestKeepBitNonForgeable`
confirms a zero `Witness` reports `Kept()==false`.

**VERDICT.** **PROVEN** (2026-06-20). All five witnesses PASS:
`--- PASS: TestEvaluateKeepsStrictGain`, `…RevertsNoGain`,
`…RevertsIfSuiteRed`, `…TuneCacheSizeRevertsNonImproving`,
`…KeepBitNonForgeable`; `ok github.com/anthony-chaudhary/fak/internal/shipgate 0.272s`.

**DOS.** Mechanism + witnesses ship in `04e4b23`
(`feat(adjudicator,shipgate): real dev-agent floor + wire the CICD pillars (#11)`).
bound at ship.

---

## Theorem 2 — the measurement is deterministic

**THEOREM.** `Evaluate` is deterministic: for any fixed `Witness w`, repeated
evaluations `Evaluate(w)` yield the identical `(Decision, keep-bit)` — no
dependence on RNG, wall-clock, goroutine scheduling, map-iteration order, or
mutable global state.

**REGIME.** D — decision-procedure soundness (determinism of the verdict input).

**PROOF.** `Evaluate` (`fak/internal/shipgate/shipgate.go:64-70`) is a pure
function of its by-value `Witness` argument: `improved()`
(`shipgate.go:54-59`) compares `After` against `Before` with a strict IEEE-754
float ordering, and `Evaluate` ANDs in the `SuiteGreen` / `TruthClean`
booleans. Float comparison `<` / `>` is deterministic; boolean AND is
deterministic; the result is stored in a *local copy's* `improvedBit` and
returned. There is no `rand`, no `time`, no range-over-map, and no read of any
package-level mutable state on this path (the package globals `shipTools` /
`DefaultAdjudicator` are not consulted by `Evaluate`). Hence the function is
referentially transparent: same `Witness` in ⇒ same `(Decision, keep-bit)` out.
This argument is sound, but the honesty rule forbids promoting it to PROVEN
without a deterministic witness that actually re-runs the property.

**WITNESS.**
```
(go test ./internal/shipgate/ -count=1 -timeout 120s)
```
The package run is green, but **no existing test asserts repeatability** — a
`grep -i 'determin|same input|twice|repeat|idempot'` over
`internal/shipgate/` returns nothing, and the existing tests check single
input→output pairs, not that a repeated call returns the same verdict. A green
package run does NOT witness determinism.

**To close it:** add a stdlib table test (`TestEvaluateDeterministic`) that, for
a fixed set of `Witness`es (gain / no-gain / regression × `SuiteGreen` ×
`TruthClean`), calls `Evaluate` — and `TuneCacheSize` — N times and asserts
byte-identical `(Decision, Kept())` across all N runs. That converts this row to
PROVEN.

**VERDICT.** **OPEN** (2026-06-20). The function is structurally
deterministic (pure scalar fold, no RNG/clock/global/map-iteration), but un-
witnessed by any repeated-call test.

**DOS.** Would bind to the commit that lands `TestEvaluateDeterministic`.
bound at ship.

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **measurement-deterministic** → ✅ PROVEN by `TestEvaluateDeterministicRepeat`. Evaluate(w) is deterministic over a fixed-seed sweep (seed 0x5eed1234) of the full exported Witness input surface, including NaN/Inf/equal-boundary Before/After and all bool combinations. A single reference (Decision,keep-bit) is taken per case; 64 repeats must each equal it bit-identically. Non-vacuous: it pins KEEP<=>Kept() / REVERT<=>!Kept() consistency (Evaluate never returns ESCALATE) and asserts BOTH KEEP and REVERT outcomes actually occur in the sweep. Complemented by TestEvaluateDeterministicConcurrent (32 goroutines barrier-released on the same witness yield exactly one result, ruling out shared mutable global state / scheduling dependence; also green under `go test -race`) and TestEvaluateDeterministicQuick (testing/quick, fixed seed 0xABCD4242, MaxCount 5000, independent generator). Mechanism is pure: shipgate.go:54 improved() and shipgate.go:64 Evaluate() read only the witness's own fields with no RNG/clock/map/global; the test confirms it empirically. Full package go test passes with the new file present; new tests also pass under -race.
