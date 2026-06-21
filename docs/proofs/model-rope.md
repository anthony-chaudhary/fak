# N3 · model/rope

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 2 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/model/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

Rotary Position Embedding (RoPE) injects absolute position into the query/key
vectors by rotating each adjacent dimension-pair `(h[j], h[j+half])` by an angle
`p · inv_freq[j]` that is linear in the token position `p`. The module is implemented
in `fak/internal/model` across three files: `kv.go` builds the inverse-frequency table
(`invFreq`), materialises the per-position `cos/sin` rows (`ropeRow*`, `ropeRowInto`),
and applies the rotation (`applyRopeRow`); `arch.go` rescales `inv_freq` for the
long-context scaling schemes (`applyRopeScaling`: `llama3` piecewise + `yarn`
linear-ramp); and `longrope.go` carries Phi's per-dimension factor vector plus the
attention-temperature warm-up. "Correct" here (regime **N**) means three things: the
rotation is a genuine rotation (it preserves the per-pair norm), the resulting
attention score depends on the two positions only through their *difference* `(m−n)`,
and each scaling variant rescales the frequencies exactly per its published reference
formula. The witnesses below are run on the macOS node natively
(`go test ./internal/model`); the heavy weight/oracle rungs are scoped per theorem.

---

## THEOREM (1) — RoPE preserves the per-(head,pair) vector norm (it is a rotation)

**REGIME** N (numerical / metamorphic relation).

**STATEMENT** For the unscaled (Llama) rotary map, `applyRopeRow` acts on each pair
`(h[j], h[j+half])` as a Givens rotation by `p · inv_freq[j]`; because `cos²+sin²=1`
it preserves the per-pair Euclidean norm — and hence the whole per-head vector norm —
for every position `p` and every input vector.

**PROOF** `applyRopeRow` (`fak/internal/model/kv.go:473`) sets
`h[j] = a·cos − b·sin`, `h[j+half] = b·cos + a·sin` with `a=h[j], b=h[j+half]` — the
standard 2×2 rotation, so it preserves `a²+b²` per pair when `cos²+sin²=1`. The
`cos/sin` come from `ropeRowInto` (`kv.go:465`) as `Cos/Sin` of `p·inv_freq[j]`, which
satisfies the Pythagorean identity to float rounding. The explicit `float32()` pins at
`kv.go:485-486` make the result bit-deterministic across architecture/call-site.
*Domain caveat (keeps the theorem honest and narrow):* `ropeRowFromInvScaled`
(`kv.go:451`) multiplies **both** `cos` and `sin` by a non-unit `scale` on the `yarn`
attention-factor path (`arch.go:138`) and the longrope temperature folds into
`cfg.attnScale`; on those configs `cos²+sin²=scale²≠1`, so the per-pair norm is *scaled*,
not preserved. Norm preservation is a property of the bare rotation, not of every served
config.

**WITNESS** `(go test -run 'Rope|RoPE|Rotary|Scaling|Longrope|Yarn' ./internal/model/ -count=1 -v -timeout 120s)`
— relevant tests `TestForwardRopeAppliesSharedRotation`, `TestRopeRowsShareInvFreqBitExact`.

**VERDICT** OPEN (2026-06-20). The package run is green, but **no test asserts norm
preservation**: `TestForwardRopeAppliesSharedRotation` only checks `applyRopeRow`
matches `rope.apply` bit-for-bit (a *consistency* check, not a norm MR), and
`TestRopeRowsShareInvFreqBitExact` only checks two `inv_freq` builders agree. Neither
computes `‖v‖` before vs after. **Closeable** by a zero-dependency metamorphic test:
for an unscaled config (`RopeScaling==""`), over random `hv` and several positions,
assert `|Σ(h[j]²+h[j+half]²)_after − _before| ≤ tol` — scoped to the unscaled path
because the yarn/longrope `scale` breaks `cos²+sin²=1` by construction.

**DOS** bound at ship.

---

## THEOREM (2) — the rotated dot product depends only on the relative position (m−n)

**REGIME** N (numerical / metamorphic relation).

**STATEMENT** For unscaled RoPE, `⟨R_m q, R_n k⟩ = ⟨R_{m−n} q, R_0 k⟩` — the rotated
query·key score depends on `m,n` only through the offset `(m−n)`, the defining property
of RoPE.

**PROOF** Per pair, `R_p` is rotation by `p·inv_freq[j]` (`kv.go:465`,`kv.go:473`).
Rotation matrices satisfy `R_mᵀ R_n = R_{n−m}`, hence
`⟨R_m q, R_n k⟩ = qᵀ R_mᵀ R_n k = qᵀ R_{n−m} k`, a function of `(n−m)` alone, summed over
pairs. The angle is *exactly* linear in position (`a = float64(p)·inv[j]`, `kv.go:467`),
which is the algebraic precondition for the identity.

**WITNESS** `(go test -run 'Rope|RoPE|Rotary|Scaling|Longrope|Yarn' ./internal/model/ -count=1 -v -timeout 120s)`
— no existing test bears on it.

**VERDICT** OPEN (2026-06-20). A grep of `internal/model/*_test.go` for a
relative-position / `(m−n)` / rotated-dot metamorphic relation returns nothing relevant
(the "relative" hits are quant-tolerance comments). **Closeable** by a stdlib MR test:
random `q,k` and offsets `(m,n)` and `(m+d,n+d)`; apply `applyRopeRow` to copies at each
absolute position; assert `dot(q_m,k_n) ≈ dot(q_{m+d},k_{n+d})` within a float tolerance,
on the unscaled path (note the yarn/longrope `scale` multiplies both rotated vectors, so
the dot picks up a constant `scale²` factor — it does not break relative-position
dependence but must be accounted for in the assertion).

**DOS** bound at ship.

---

## THEOREM (3) — RoPE scaling variants match their reference formula

**REGIME** N (numerical / oracle-against-hand-reference).

**STATEMENT** Each implemented RoPE scaling variant rescales `inv_freq` (and `cos/sin`)
per its published reference: `llama3` piecewise low/high-frequency-wavelength rescale
(HF `_compute_llama3_parameters`), `yarn` linear-ramp interpolation + attention
temperature, and Phi `longrope` per-dimension factor division + `sqrt(1+ln(max/orig)/ln(orig))²`
attention warm-up — each checked against a *hand-computed* reference, not the impl.

**PROOF** `applyRopeScaling` (`fak/internal/model/arch.go:26`) is the single rescale
site. `llama3` divides low-freq bands by `factor`, leaves high-freq bare, and
smooth-interpolates the middle band (`arch.go:39-54`); `yarn` blends `inv` and
`inv/factor` via `yarnCorrectionRange` + `yarnLinearRamp` (`arch.go:55-92`,`arch.go:104`)
and `ropeAttentionFactor` (`arch.go:138`) scales `cos/sin`. Phi longrope divides
`inv_freq` by the pinned per-dim factor (`kv.go:325` via `ropeLongFactor`,
`longrope.go:36`) and warms the score by the sqrt-log formula (`longrope.go:69`).
`TestRopeScalingLlama3` (`arch_test.go:124`) re-derives the band thresholds independently
and asserts `bare/factor` on low-freq, unchanged on high-freq, in-range on interp, plus
the misconfigured fail-safe-to-bare branch. `TestYarnRopeScalesCosSin` (`arch_test.go:582`)
asserts `cos==0.1·ln(32)+1`, `sin==0` at `p=0` (the attention-factor reference).
`TestLongropeInvFreqAppliesFactorPerDim` (`longrope_test.go:36`) asserts `inv[j]==base/factor[j]`
vs a hand reference; `TestLongropeAttnScaleTemperature` (`longrope_test.go:88`) asserts the
multiplier equals the literal sqrt-log formula and folds into `cfg.attnScale`.

**WITNESS** `(go test -run 'Rope|RoPE|Rotary|Scaling|Longrope|Yarn' ./internal/model/ -count=1 -v -timeout 120s)`
— `TestRopeScalingLlama3`, `TestYarnRopeScalesCosSin`, `TestLongropeInvFreqAppliesFactorPerDim`,
`TestLongropeAttnScaleTemperature`, `TestLongropeLlamaNoOp`.

**VERDICT** PROVEN (2026-06-20). All five ran green:
`--- PASS: TestRopeScalingLlama3`, `--- PASS: TestYarnRopeScalesCosSin`,
`--- PASS: TestLongropeInvFreqAppliesFactorPerDim`,
`--- PASS: TestLongropeAttnScaleTemperature`, `--- PASS: TestLongropeLlamaNoOp`. Each
asserts against an independent hand-computed reference. *Scope note on the theorem as
posed:* the codebase implements `llama3` and `yarn` (plus Phi `longrope`); there is **no**
separate `linear` (position-interpolation) or bare-NTK case in the `applyRopeScaling`
switch (`arch.go:27-96`, cases `""/"none"/"llama3"/"yarn"/default`). The "linear / NTK"
wording is therefore discharged only insofar as those two schemes are not present here;
the variants that *do* exist match reference and are PROVEN. Two end-to-end PyTorch/HF
oracle rungs (`TestOptionalLlama3OracleCoversScalingAndEOSList`,
`TestOptionalPhi3LongropeOracleCoversLongFactor`) **SKIPPED** for absent
`.cache/oracle-llama3` / `.cache/oracle-phi3-longrope-local` fixtures — they would add a
forward-parity witness on top of the unit references; regenerate via
`python internal/model/export_oracle.py --out .cache/oracle-llama3` to close that
heavier rung.

**DOS** bound at ship.

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **rope-preserves-pair-norm** → ✅ PROVEN by `TestProofRopePreservesPairNorm`. 100 trials. Uses real unexported applyRopeRow (kv.go:473) + ropeRowFromInv (kv.go:447) with an independently-built standard Llama inv_freq (1/10000^(2j/dim)). Asserts each pair (h[j],h[j+half]) Euclidean norm^2 and the whole-vector norm^2 are preserved within 1e-4*(1+norm) for random positions p in [0,4096). Witnesses the Givens-rotation (cos^2+sin^2=1) property.
- **rope-dot-relative-position** → ✅ PROVEN by `TestProofRopeDotRelativePosition`. 100 trials. <R_m q, R_n k> == <R_{m-n} q, R_0 k> (R_0 identity) within 1e-3*(1+|lhs|), for random m in [1,2048] and n in [0,m]. Uses real applyRopeRow + ropeRowFromInv + dot. Witnesses RoPE's defining relative-position property (dot depends on m,n only through m-n).
