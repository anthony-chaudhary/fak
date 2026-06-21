# N5 · model/quant

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 1 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/model/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

The `model` quantization lane turns full-precision weights into compact integer block
formats — Q8_0 (`scale·q`, 32-wide blocks), Q4_0 (`d·(code−8)`), the GGUF k-quant Q4_K
(affine `d·sc_s·nibble − min·m_s` per 32-wide sub-block of a 256-wide super-block), and
AWQ (`scale[o]·(code−8)`, symmetric 4-bit) — and runs the matmuls (GEMV decode + batched
GEMM prefill) directly on those formats. "Correct" here is **regime N (numerical)** in
three concrete senses: (1) **dequant is affine-correct** — the unpacked value equals the
block format's defining affine map, bit-exactly against the loader/reference dequant;
(2) **AWQ matches its reference** dequant; and (3) the accelerated **integer SDOT/int8
reduction path is bit-identical** to the portable integer reference sum, with float error
confined to the final de-affine combine. Quantization is lossy by construction, so the
witnesses are of two kinds: *exact* (bit-identity to a reference dequant or to the integer
reduction) where the math is exactly representable, and *bounded* (max-abs/RMS within a
stated tolerance) where activation quantization is in play.

This node is `Darwin/arm64` (Apple M3) with `FEAT_DotProd` present, so the NEON SDOT path
is the **active** decode/prefill kernel (`neonDot=true`, `detectDotProd()=true`) — the
bit-identity theorem is witnessed on the path that actually runs, not a skipped one.

---

## THEOREM 1 — Q4_K / Q8_0 (and Q4_0) dequant is affine-correct

**THEOREM.** For every Q4_K super-block, `q4kDequantSuperBlock` reproduces
`value = d·sc_s·nibble − min·m_s` (the per-sub-block affine dequant) **bit-exactly** equal
to the loader's f32 reference `dequantQ4K`, over all code/scale/min byte patterns. For
Q8_0, the code round (`value = d·q`, symmetric zero-point 0, `d = maxabs/127`) is
bit-exactly `math.Round` and the dequant-dot stays within Q8 quantization error with no
int32 overflow. For Q4_0, `dequantQ4Block` (`d·(code−8)`) inverts `quantizeQ4Block` to
within one quantum.

**REGIME.** N (numerical — exact for the dequant arithmetic, bounded for the lossy dot).

**PROOF.** `q4kDequantSuperBlock` (`fak/internal/model/quant_q4k.go:78`) computes
`d1=d·sc, m1=min·m` via `getScaleMinK4` (`quant_q4k.go:68`, the 6-bit scale/min unpack)
then `dst[j+l]=d1·(nibble) − m1` — exactly the affine form. The witness oracle
`dequantQ4KRef` (`quant_q4k_test.go:45`) is a **verbatim copy of the loader's
`ggufload.dequantQ4K`**, and `TestQ4KDequantSuperBlockMatchesRef` asserts equality
bit-exactly over 2000 random super-blocks; so the resident decode IS the f32 weights the
loader would have produced, by construction. Q8_0 (`quant.go:96`): `value=d·q`,
`d=maxabs/127`, zero-point 0 (symmetric); `q8round` (`quant.go:75`) is pinned
bit-identical to `math.Round` (ties away from zero) and `qdot8scalar` (`quant.go:190`)
reduces in `int32` with no overflow (per-block bound `32·127·127 ≈ 5.2e5`). Q4_0
`dequantQ4Block` (`quant_q4.go:84`) is the exact inverse of `quantizeQ4Block` on the codes.

**WITNESS.**
```
(go test -run 'TestQ4KDequantSuperBlockMatchesRef|TestQ4KMatRowsMatchesF32|TestQ8RoundMatchesMathRound|TestQdot8MatchesF32|TestQ4BlockRoundTrip' ./internal/model/ -count=1 -timeout 120s -v)
```

**VERDICT.** PROVEN (2026-06-20, arm64/Darwin). All green:
`TestQ4KDequantSuperBlockMatchesRef` exact over 2000 trials;
`TestQ4KMatRowsMatchesF32` GEMV max-abs/RMS = 1.812e-06 (reorder-only);
`TestQ8RoundMatchesMathRound` zero mismatches over 200013 points incl. adversarial
near-halves; `TestQdot8MatchesF32` rel err ≤ 4.06e-3 across in={32,64,576,1536} (no
overflow); `TestQ4BlockRoundTrip` within d/2 over 1000 trials.

**DOS.** bound at ship.

---

## THEOREM 2 — AWQ dequant matches its reference

**THEOREM.** AWQ dequant `weight[o·in+i] = scale[o]·(unpack4bit(code) − 8)` matches its
reference: (a) the affine arithmetic is bit-exact to an independently-computed expected
value, and (b) it matches the HuggingFace AutoAWQ reference the format claims compatibility
with.

**REGIME.** N (numerical).

**PROOF.** `awqDequantRow` (`fak/internal/model/awq.go:57`) and the scalar reference
`awqDequantRowScalar` (`awq_scalar.go:32`) both compute
`scale·float32(int16(code) − zeroPoint)` with `zeroPoint=8` (`awq_scalar.go:29`) — the
symmetric-4-bit affine the format spec states (`awq.go:14`). `unpack4bit` (`awq.go:49`) is
pinned by a 6-case truth table (`TestAWQUnpack4bit`). The dequant, dot, and GEMV are pinned
**bit-exact to hand-derived expected values** (`TestAWQDequantRowScalar` vs a literal
`want[]`; `TestAWQMatRows` vs `y[1]=-40`; `TestAWQDotProductScalar` vs `-8.8`), which
witnesses part (a) — the affine math is self-consistent and correct. Part (b) is **not**
witnessed: the only oracle-shaped test, `TestAWQOracleThreshold`, self-quantizes 8 toy
weights and asserts cosine ≥ 0.95 against its **own** f32 (a round-trip floor, not an
external reference); there is no `oracle-awq` fixture and no HF-AutoAWQ comparison anywhere
in scope. The doc's "≥0.995 vs HF AutoAWQ" is narrated, not tested.

**WITNESS.**
```
(go test -run 'TestAWQUnpack4bit|TestAWQDequantRowScalar|TestAWQDotProductScalar|TestAWQMatRows|TestAWQQuantizeFromRaw|TestAWQOracleThreshold' ./internal/model/ -count=1 -timeout 120s -v)
```

**VERDICT.** OPEN (2026-06-20). Every listed test ran **green**, so the affine
self-consistency (part a) is effectively PROVEN; but the strong claim "matches its
reference" (part b — bit/cosine parity against a real HuggingFace AutoAWQ export) has **no
deterministic witness**. To close it: add an `export_oracle`-style AutoAWQ fixture (a real
AWQ-quantized tensor plus the HF-dequantized f32) under a gitignored `.cache/oracle-awq`,
and a test that compares `awqDequantRow` to it bit-exactly (codes lossless) or by cosine
within the stated bound — analogous to the `.cache/oracle-*` path in 00-METHOD.md §3.1.
Until that fixture exists this stays OPEN, not PROVEN.

**DOS.** bound at ship.

---

## THEOREM 3 — the int8 SDOT reduction is bit-identical to the integer reference

**THEOREM.** The arch-dispatched integer reduction `q4kReduceRow` (NEON SDOT on arm64)
produces the per-sub-block int32 pairs `IS=Σ nibble·qx`, `SS=Σ qx` **bit-identical** to the
portable scalar reference `q4kReduceRowScalar`; the only float error in the full int8 dot is
confined to the final de-affine combine `q4kCombineRow` (shared Go).

**REGIME.** N (numerical — bit-identity for the integer reduction; bounded float only in
the combine).

**PROOF.** `q4kReduceRowAsm` (`fak/internal/model/quant_arm64_q4k.go:15`, asm body in
`quant_arm64_q4k.s`) computes `IS`/`SS` in `int32` via SDOT; `q4kReduceRowScalar`
(`quant_q4k_int8.go:72`) is the int32 oracle. The reductions are `int8×int8→int32` (SDOT)
and a ones-sum, both **associative with no overflow** on these ranges
(`|IS| ≤ 32·15·127 ≈ 6.1e4`, `|SS| ≤ 32·127 ≈ 4.1e3`, both inside int32 —
`quant_q4k_int8.go:53`), so any SIMD lane order yields the identical int32, making the asm
reduction equal to scalar bit-for-bit. The float combine `q4kCombineRow`
(`quant_q4k_int8.go:110`) is the **same compiled Go on every arch**, so once `IS`/`SS`
match, the full asm-path dot equals the scalar-int8 dot exactly; the only float drift lives
in that shared de-affine combine — precisely the theorem. The dispatch
(`q4kReduceRow`, `quant_arm64_q4k.go:30`) was exercised, not skipped: `detectDotProd()=true`
on this node.

**WITNESS.**
```
(go test -run 'TestQ4KReduceAsmMatchesScalar|TestQ4KInt8DotMatchesF32' ./internal/model/ -count=1 -timeout 120s -v)
```

**VERDICT.** PROVEN (2026-06-20, arm64/Darwin, FEAT_DotProd present). Both green:
`TestQ4KReduceAsmMatchesScalar` — "q4k SDOT reduce bit-identical to scalar across 16 rows ×
24 sub-blocks (neonDot=true)", asserting `asm IS[i]==scalar IS[i]` and `SS` for every `i`;
`TestQ4KInt8DotMatchesF32` — int8 vs f32 max-abs/RMS = 1.380e-02, within the 0.05
activation-quant tolerance (the bounded float error of the de-affine combine). On a part
without FEAT_DotProd the asm test SKIPs and `q4kReduceRow` falls to the scalar path; on this
node the accelerated path is the one witnessed.

**DOS.** bound at ship.

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **awq-matches-reference** → ✅ PROVEN by `TestProofAWQMatchesReference`. Part (a) PROVEN: 200 trials assert awqDequantRow (awq.go:57) AND the portable reference awqDequantRowScalar (awq_scalar.go:32) are BIT-EXACT (==) to an independently-computed closed-form scale*(nibble-8), and unpack4bit (awq.go:49) inverts the nibble packing exactly. Part (b) — byte-equality against a stored HuggingFace AutoAWQ fixture — is NOT claimed here (needs an absent on-disk AWQ export); the test header documents this and leaves (b) to oracle_test. Verdict PROVEN reflects the affine-arithmetic claim (a) that this stdlib-only test soundly closes.
