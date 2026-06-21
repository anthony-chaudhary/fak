# N2 · model/norm

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 2 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/model/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

This module implements the per-position normalization primitives of the transformer block: **RMSNorm** (`rmsnorm`, the Llama convention `x · w / sqrt(mean(x²)+eps)`), its **NormGain1p** variant (Gemma's `(1+w)` gain), and mean-subtracting **LayerNorm** (`x ↦ (x−mean)·w/sqrt(var+eps) + bias`, used by StableLM and friends), plus the in-place / allocation-free twins (`applyRMSNormInPlaceCfg`, `rmsnormInto`) and the config router (`normCfg`). "Correct" here is **regime N (numerical)**: the computed tensor must equal the mathematically-defined function within a stated float error model — checked by recomputing the closed form independently in the witness (exact reference parity) and, where no cheap oracle exists, by metamorphic relations (shift/scale equivariance). The sum-of-squares reduction is deliberately scalar in-order in `rmsnorm`/`layernorm` so the f32 bit-exact forward rungs stay reproducible; the quant twin `rmsnormInto` is the one permitted to use the vectorized `fdot`.

---

## Theorem 1 — RMSNorm computes the definition

**THEOREM.** For all `x, w` of equal length and `eps > 0`,
`rmsnorm(x,w,eps)[i] == x[i]·w[i]/sqrt(mean(x²)+eps)` within 1e-6, and the
`NormGain1p` variant computes `x[i]·(1+w[i])/sqrt(mean(x²)+eps)`.

**REGIME.** N — numerical reference parity.

**PROOF.** `rmsnorm` (`fak/internal/model/forward.go:356`) accumulates `ss = Σ x²`
in scalar in-order f32, sets `inv = 1/sqrt(ss/len(x)+eps)` (float64 sqrt, narrowed),
and writes `out[i] = x[i]·inv·w[i]` — exactly the definition. The Gain1p branch
substitutes `gain = (1+w[i])` (`fak/internal/model/arch.go:445`–`453`, and the
in-place twin `applyRMSNormInPlaceCfg` `arch.go:250`–`256`). `TestNormGain1p`
(`arch_test.go:261`) independently recomputes the same `inv` from the same in-order
reduction and asserts the Gain1p output equals `x[i]·inv·(1+w[i])` to 1e-6
(`arch_test.go:279`–`282`), then binds the plain path **bit-for-bit** to that `inv`
via `assertFloat32BitsEqual("norm gain off == plain", plain, off)`
(`arch_test.go:292`–`293`). The literal closed form is thus the oracle, not a
restatement. *Scope note:* the explicit `plain[i] == x[i]·inv·w[i]` equality is
implied transitively (off==plain, gain checked) rather than written as its own line.

**WITNESS.** `go test -run 'TestNormGain1p' ./internal/model/ -count=1 -timeout 120s -v`

**VERDICT.** PROVEN — 2026-06-20. `--- PASS: TestNormGain1p (0.00s)`, `ok …/internal/model 0.216s` (native, macOS arm64, go1.26).

**DOS.** bound at ship.

---

## Theorem 2 — LayerNorm matches the reference

**THEOREM.** For all `x, w, bias` of equal length and `eps > 0`,
`layernorm(x,w,bias,eps)[i] == (x[i]−mean)·w[i]/sqrt(var+eps) + bias[i]` within
1e-6, where `mean = avg(x)`, `var = mean((x−mean)²)`.

**REGIME.** N — numerical reference parity.

**PROOF.** `layernorm` (`fak/internal/model/arch.go:457`) subtracts the row mean,
computes the centered `ss = Σ(x−mean)²`, sets `inv = 1/sqrt(ss/len+eps)`, and writes
`(x[i]−mean)·inv·w[i]` (plus `bias[i]` when `bias != nil`) — the mean-subtracting
LayerNorm definition. `normCfg` (`arch.go:438`–`441`) routes `cfg.LayerNorm` here.
`TestLayerNormAxis` (`arch_test.go:296`) recomputes `mean`, `ss`, `inv` from `x`
and asserts `got[i] == (x[i]−mean)·inv·w[i]` to 1e-6 (`arch_test.go:316`–`319`) and
the `+bias` path `got == want + b[i]` (`arch_test.go:320`–`322`) — an independent
reference recomputation.

**WITNESS.** `go test -run 'TestLayerNormAxis' ./internal/model/ -count=1 -timeout 120s -v`

**VERDICT.** PROVEN — 2026-06-20. `--- PASS: TestLayerNormAxis (0.00s)`, `ok …/internal/model 0.216s`.

**DOS.** bound at ship.

---

## Theorem 3 — LayerNorm is shift+scale equivariant (RMSNorm scale-invariant)

**THEOREM.** LayerNorm is invariant to affine input transforms on the normalized
axis: for `a > 0, b`, `layernorm(a·x+b)[i] == layernorm(x)[i]` in the `eps → 0`
limit (mean-subtraction cancels `b`, division by stddev cancels `a`). RMSNorm is
invariant to positive scaling up to the learned gain: `rmsnorm(c·x, 1)` direction
== `rmsnorm(x, 1)` direction for `c > 0`.

**REGIME.** N — metamorphic relation (00-METHOD.md §3.2).

**PROOF.** The mechanism supports it — `layernorm` (`arch.go:457`) subtracts the
mean before scaling, and `rmsnorm` (`forward.go:356`) divides by the RMS so a
positive global scale cancels — **but a PROOF needs a witness, not the argument.**
No existing test runs a *second* normalization on an affine-transformed input and
compares. `grep` for `equivar|shift|scale.*invar` over `internal/model/*_test.go`
returns only unrelated hits. `TestLayerNormAxis`/`TestNormGain1p` check the closed
form at a single input only.

**WITNESS.** `go test -run 'TestLayerNormAxis|TestNormGain1p' ./internal/model/ -count=1 -timeout 120s` (these run green but do **not** assert this relation).

**VERDICT.** OPEN — 2026-06-20. **Closing test:** add `TestLayerNormShiftScaleEquivariant`
feeding `y = a·x + b` for several `(a>0, b)` and asserting `max|layernorm(y)−layernorm(x)| < tol`
(small `eps`); and `TestRMSNormScaleInvariant` asserting the normalized direction is
invariant to `c>0`. Not promoted by argument alone.

**DOS.** bound at ship.

---

## Theorem 4 — Normalization is numerically stable on large-magnitude inputs

**THEOREM.** For large-magnitude *finite* inputs, `rmsnorm`/`layernorm` produce only
finite outputs (no NaN, no ±Inf): the sum-of-squares does not overflow and the
`1/sqrt` never divides by zero/inf.

**REGIME.** N — numerical stability (metamorphic / boundedness).

**PROOF.** No deterministic witness asserts this against `rmsnorm`/`layernorm`
directly. `TestGemmaStackChangesOutput` (`arch_test.go:631`) is a **forward-pass
smoke gate** on normal-magnitude `NewSynthetic` inputs that checks the *logits* are
finite (`arch_test.go:648`) after the whole stack — it never drives the norm
primitives with large inputs. The other `IsNaN/IsInf` checks (glm/moe/quant tests)
are likewise downstream forward smoke gates. **Boundary note:** `ss` is accumulated
in **f32** (`forward.go:357`–`360`; `arch.go:246`–`249`, `463`–`467`); `|x| ≳ 1.8e19`
overflows f32 `ss` to `+Inf`, giving `inv = 0` → output `0` (finite but degenerate),
and a single `Inf` input yields `NaN`. So the claim holds only on a *bounded* domain,
and the witness must pin that boundary.

**WITNESS.** `go test -run 'TestGemmaStackChangesOutput' ./internal/model/ -count=1 -timeout 120s` (green, `ok …/internal/model 0.217s`, but does not witness this theorem).

**VERDICT.** OPEN — 2026-06-20. **Closing test:** add `TestNormFiniteOnLargeInputs`
calling `rmsnorm`/`layernorm` on `x` with `|x|` up to a stated f32-safe bound
(e.g. 1e15) and asserting every output is finite (`!math.IsNaN && !math.IsInf`),
documenting the ~1.8e19 f32-`ss` overflow boundary. A stronger fix — accumulate `ss`
in float64 — is a code change, out of scope for this proof pass.

**DOS.** bound at ship.

---

### Honest ledger summary

| # | Theorem | Verdict | Witness |
|---|---|---|---|
| 1 | RMSNorm = `x·gain/sqrt(mean(x²)+eps)` | **PROVEN** | `TestNormGain1p` |
| 2 | LayerNorm matches reference (float tol) | **PROVEN** | `TestLayerNormAxis` |
| 3 | LayerNorm shift+scale equivariant | **OPEN** | needs `TestLayerNormShiftScaleEquivariant` |
| 4 | Stable (finite) on large inputs | **OPEN** | needs `TestNormFiniteOnLargeInputs` (with overflow boundary) |

Nothing is REFUTED. The two PROVEN rows ran green on this macOS node; the two OPEN
rows are honestly un-witnessed and each names the exact closing test.

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **layernorm-shift-scale-equivariant** → ✅ PROVEN by `TestProofLayerNormShiftScaleEquivariant`, `TestProofRMSNormPositiveScaleInvariant`. 100 trials with eps=1e-12 (eps->0 limit): layernorm(a*x+b,w,nil) == layernorm(x,w,nil) for random a>0,b, within magnitude-scaled tol 1e-4*(1+|y|). RMSNorm positive-scale invariance is a separate test below. Calls the real unexported layernorm (arch.go:457).
- **norm-numerically-stable-large-inputs** → ✅ PROVEN by `TestProofNormNumericallyStableLargeInputs`. 4 magnitudes (1e15,1e16,1e18,1e20) x 50 trials each: rmsnorm (forward.go:356), layernorm (arch.go:457), and applyRMSNormInPlace (arch.go:241/245) all produce only finite f32 outputs (no NaN, no +/-Inf). Witnesses the sum-of-squares does not overflow f32 and 1/sqrt does not divide by zero/inf.
