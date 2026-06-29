---
title: "fak proof: compute HAL GEMM numerical parity"
description: "Proof that fak's compute backend GEMM equals the reference inner product bit-identically on the F32 path and within the Approx gate on device paths."
---

# N8 · compute/gemm

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 1 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/compute/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

The `internal/compute` package is fak's hardware-abstraction seam (HAL): it lifts the forward pass's seven baked-in hardware assumptions into a `Backend` interface so a GPU/NPU/WASM target is a registration, not a fork of the loop (`compute.go:1-45`). Day-1 it ships exactly one backend — a pure-Go scalar **Reference** CPU backend (`cpuref.go`) whose kernels reproduce the model's exact reduction order byte-for-byte. For this proof obligation "correct" means, in regime **N (numerical / linear-algebra)**: the HAL's GEMM family — `MatMul` (`y=x@Wᵀ`) and `BatchedMatMul` (the prefill GEMM) — computes the mathematically-defined matrix-vector / matrix-matrix product equal to an independently-computed reference, **bit-identically** for the Reference (`F32`) path and within the **Approx gate** (argmax-stable + logit-cosine≈1) for every quantized/device path. The seam's `CorrectnessClass` makes that scoping mechanical: only a `Reference` backend is ever held to `max|Δ|=0`.

---

THEOREM   The HAL's F32 `MatMul`/`BatchedMatMul` computes `y[o]=Σ_i w[o,i]·x[i]` equal to the package's canonical reference inner product (`fdot`, the model's 8-accumulator pairwise tree) for every output row, at `max|Δ|=0` (bit-identity) — strictly inside any float tolerance.
REGIME    N
PROOF     `MatMul` (`cpuref.go:119-141`) dispatches on `w.Dtype`; the F32 branch computes `y[o]=fdot(wf[o*in:o*in+in], xf)` — literally the reference dot per row (`cpuref.go:126-128`). `BatchedMatMul` (`cpuref.go:145-178`) does the same per `(t,o)`. `fdot` (`cpuref.go:376-395`) is the defined reference reduction (8 accumulators, fixed pairwise combine). The witness reconstructs the `fdot` body independently and asserts `math.Float32bits(got)==math.Float32bits(want)`, so routing the loop through the interface adds only a method indirection, not a numeric change. **Honesty caveat:** the reference here is `fdot`, *not* a literal naive single-accumulator triple loop; `TestReductionOrderIsTheModelTree` deliberately demonstrates that `fdot` differs in bits from the naive serial sum on an order-sensitive input (a float summation-*order* difference — both are exact orderings of the same bilinear map). So "equals the reference within float tolerance" is witnessed at the *stronger* `max|Δ|=0` against the canonical reduction; a separate naive-triple-loop-within-tolerance comparison is not asserted.
WITNESS   `go test -run 'TestMatMulDelegatesVerbatim|TestReductionOrderIsTheModelTree|TestRMSNormAndSwiGLUVerbatim' ./internal/compute/ -count=1 -timeout 120s -v`
VERDICT   **PROVEN** — 2026-06-20: `PASS TestMatMulDelegatesVerbatim`, `PASS TestReductionOrderIsTheModelTree` (run on this M3/darwin-arm64 node, native go1.26).
DOS       bound at ship — ship commit `7f7732b` (`compute: cross-platform backend (HAL) seam + pure-Go CPU reference`); audit via `dos commit-audit 7f7732b` / `dos verify` from the fleet repo root.

---

THEOREM   GEMM is bilinear: `A(B+C)=AB+AC` and `(αA)B=α(AB)`, checked against the naive computation up to float tolerance (the metamorphic relation named in 00-METHOD §3.2).
REGIME    N
PROOF     Structurally, `MatMul` reduces with IEEE `+`/`*` (`cpuref.go:119-141`, `fdot` `cpuref.go:376`), so the map is additive and homogeneous up to float rounding. But 00-METHOD §4's honesty rule forbids promoting this to `PROVEN` on the structural argument alone: **no deterministic witness re-checks it.** A `grep` for `bilinear|naive|A(B+C)|AB+AC|alpha|distribut` over `internal/compute/*_test.go` returns only comment text inside `TestReductionOrderIsTheModelTree` (which contrasts `fdot` with a naive *sum* for order-sensitivity — not additivity/homogeneity of the matmul), and the `-run 'Bilinear|Distribut|Homog'` filter selects zero tests.
WITNESS   `go test -run 'Bilinear|Distribut|Homog' ./internal/compute/ -count=1 -timeout 120s -v` (selects 0 tests today)
VERDICT   **OPEN** — 2026-06-20: un-witnessed. CLOSES WITH a stdlib metamorphic test (reusing the in-file `lcg`/`randVec`/`cosine` helpers, `compute_test.go:8-35`) that builds random `A,B,C`, asserts `cosine(MatMul(A,B+C), AddInPlace(MatMul(A,B),MatMul(A,C)))≈1` within an EM/ED float band, and `MatMul(αA,B)==α·MatMul(A,B)`.
DOS       bound at ship — no commit yet (OPEN); the witness above must ship and be `dos commit-audit` `diff-witnessed` before this row can move to PROVEN.

---

THEOREM   Every registered backend (CPU / Metal / CUDA / Vulkan, by platform/build tag) agrees with the CPU Reference: an Approx backend's `MatMul`/forward is argmax-stable (same next token) and logit-cosine≈1 vs cpuref; the Reference itself is bit-identical.
REGIME    N
PROOF     The agreement contract is type-enforced by `CorrectnessClass` (`compute.go:197-217`): only a `Reference` backend faces `max|Δ|=0`; an `Approx` backend faces the argmax-exact + per-backend cosine gate, with `RequireReference` (`compute.go:385-389`) making it mechanically impossible to expect bit-identity of a device. The **CPU↔Q8(Approx)** leg is witnessed and green: `TestQ8DispatchIsApproxAndGated` (`compute_test.go:136-159`) routes a Q8 weight to the int8 path and asserts `cosine(yf,yq)≥0.995` plus byte-deterministic re-run; `TestCorrectnessClassEnforcement` asserts cpu-ref is `Reference` and a device is `Approx` and barred from the exact rungs. The **GPU** legs assert exactly the same gate per backend — Metal op-level cosine≥0.9999 and forward logit-cosine≥0.999 + device-argmax==host-argmax + next-token-exact across greedy steps (`metal_test.go:166-210`), CUDA identically (`cuda_test.go:163-218`), Vulkan cosine≥0.9999 (`vulkan_test.go:87/113/122`) — but these files are gated `//go:build darwin && arm64 && cgo` / `//go:build cuda` / `//go:build vulkan && windows && cgo` and only compile on the matching platform/toolchain.
WITNESS   `go test -run 'TestQ8DispatchIsApproxAndGated|TestCorrectnessClassEnforcement' ./internal/compute/ -count=1 -timeout 120s -v` (CPU/Q8 leg, runs here); GPU legs need the matching platform/device (Metal: darwin/arm64+cgo; CUDA/Vulkan: their build tags + hardware).
VERDICT   **SCOPED-OUT** — 2026-06-20, updated for #62: the CPU↔Q8(Approx) leg ran green (`PASS TestQ8DispatchIsApproxAndGated`, `PASS TestCorrectnessClassEnforcement`); the device legs are out of scope for the default run. UPGRADE PATH: on an M3 Apple-Silicon node, `go test -run 'TestMetalMatMulApproxMatchesRef|TestMetalForwardMatchesRef' ./internal/compute/` closes the Metal leg; CUDA/Vulkan need their own hardware+toolchain.
DOS       bound at ship — CPU/Q8 leg ships in `7f7732b`; Metal in `2c88580`; CUDA in `54f8b58`. Audit each via `dos commit-audit <ref>` from the fleet repo root.

---

## Reproduce

```bash
# all CPU witnesses for this obligation (11/11 green on 2026-06-20, darwin/arm64, go1.26):
go test -run 'MatMul|Reduction|RMSNorm|SwiGLU|Q8|Attention|Evict|Clone|Device|Correctness|Registry|Dtype' ./internal/compute/ -count=1 -timeout 120s -v
# Metal device parity (darwin/arm64+cgo only):
go test -run 'TestMetalMatMulApproxMatchesRef|TestMetalForwardMatchesRef' ./internal/compute/ -count=1
```

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **gemm-bilinear** → ✅ PROVEN by `TestGEMMBilinear`. Reference GEMM (cpuref.go MatMul -> fdot) is bilinear, witnessed as a metamorphic relation up to 1e-5 relative tolerance across all four legs: additivity in the vector arg W(x1+x2)~Wx1+Wx2, additivity in the matrix arg (A+B)x~Ax+Bx, homogeneity in the vector arg W(ax)~a(Wx), and homogeneity in the matrix arg (aA)x~a(Wx) (alpha=2.75). Deterministic (fixed lcg seeds 1009/2027). NOT bit-identity: float arithmetic is non-distributive so the legs differ in ~1ulp; tolerance per 00-METHOD 3.2 is the correct instrument, and that is exactly what the OPEN theorem states ('up to float tolerance'). Non-vacuity guarded: the witnessed output is asserted finite and not identically zero, so the identity holds because inputs are non-trivial. A companion test TestGEMMBatchedBilinear proves additivity in the matrix arg on the separate BatchedMatMul prefill-GEMM path (Y[t,o]=Sum W[o,i]X[t,i]) so both reference matmul entry points are covered. Both tests pass and the full internal/compute package still passes with the new file present.
