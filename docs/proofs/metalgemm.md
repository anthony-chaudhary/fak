---
title: "fak proof: Metal GPU GEMM matches CPU reference"
description: "Proof that fak's Apple-Silicon Metal GEMM matches the f32 CPU reference within the half-precision error model, argmax-exact at logit-cosine 1.0."
---

# N9 · metalgemm

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 1 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/metalgemm/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

Package `metalgemm` (and its sibling Metal backend in `internal/compute`) is the optional Apple-Silicon GPU acceleration lane for the prefill projections. It holds each weight matrix as f16 in unified memory and runs `Y = X·Wᵀ` through `MPSMatrixMultiplication` (plus runtime-compiled MSL kernels for RMSNorm/RoPE/attention/SwiGLU/argmax in `internal/compute`). It is compiled automatically on `darwin/arm64` when cgo is enabled; non-Apple-Silicon and `CGO_ENABLED=0` builds use the pure-Go stub. **"Correct" here is regime N (numerical):** the device path must compute the *same matmul* the f32 CPU reference computes, within the half-precision error model — not bit-identity (MPS reduction order differs from the CPU `fdot`), so the witness is **error-to-scale `< 1%`** for the op-level GEMM and **logit-cosine `= 1.0` with argmax-exact next-token** for the full decode. A second obligation is that the *stub* build (no Metal) presents the same interface and introduces **no** math divergence — it does no GPU math and degrades to "unavailable" so callers fall back to the pure-Go CPU path.

All witnesses below were run on this node — **Apple M3 Pro, go1.26 darwin/arm64, native toolchain** (the macOS node; the Windows/WSL test machinery in the root `CLAUDE.md` does not apply here).

---

## Theorem 1 — Metal MPS GEMM matches the CPU reference (cosine = 1.0 parity)

**THEOREM.** For random `W[out,in]` and `X[P,in]`, the Metal f16 GEMM `Y = X·Wᵀ` equals the f32 CPU-reference matmul within the half-precision error model: `err/scale = max|Δ| / max|ref| < 1%` at the op level, and **logit-cosine = 1.0 with per-step argmax-exact next-token** over a multi-layer Llama decode.

**REGIME.** N — Numerical / linear-algebra.

**PROOF.** `metalgemm.Weight.MatMul` (`fak/internal/metalgemm/metalgemm.go:77`) routes to `mg_matmul` (MPS f16 GEMM). `TestMatMulMatchesReference` (`fak/internal/metalgemm/metalgemm_test.go:28`) compares it against `refMatMul` — a `float64`-accumulated f32 triple loop (`metalgemm_test.go:12`) — and gates on `err/scale < 0.01`, the meaningful number for an f16 GEMM whose outputs cross zero (per-element relative error is rejected as ill-conditioned at near-zero entries, `metalgemm_test.go:52-55`). In `internal/compute`, `metalBackend.MatMul` (`fak/internal/compute/metal.go:200`) calls `fmetal_matmul_f32`; `TestMetalMatMulApproxMatchesRef` (`fak/internal/compute/metal_test.go:153`) gates `cosine ≥ 0.9999` against the cpuref backend, and `TestMetalForwardMatchesRef` (`fak/internal/compute/metal_test.go:182`) runs the **same** Llama-decode op chain through the `Backend` interface on both cpuref and metal — RMSNorm / MatMul / RoPE / GQA-Attention / SwiGLU / residual (`metal_test.go:88-116`) — requiring **argmax-exact** next-token AND `cosine ≥ 0.999` at every prefill + greedy step. The backend is declared `Approx` (`metal.go:98`) precisely because MPS's reduction order differs from `fdot`; so the witness is cosine / error-to-scale, the *correct* error model for the path, not a brittle bit-compare.

**WITNESS.**
```
CGO_ENABLED=1 go test -run 'MatMul|Reset' ./internal/metalgemm/ -count=1 -v
CGO_ENABLED=1 go test -run 'Metal'       ./internal/compute/   -count=1 -v
```
Tests: `TestMatMulMatchesReference`, `TestResetReclaimsTable`, `TestMetalMatMulApproxMatchesRef`, `TestMetalForwardMatchesRef`.

**VERDICT.** **PROVEN** (2026-06-20, Apple M3 Pro). All four tests PASS on the real GPU. `metalgemm` err/scale: `[16,64,96]=0.00039`, `[64,512,256]=0.00041`, `[256,1536,1536]=0.00041` — all ~25× under the 1% bound. `compute`: `TestMetalMatMulApproxMatchesRef` `cosine=1.00000000 maxAbs=1.91e-06`; `TestMetalForwardMatchesRef` "6 prompt + 8 greedy steps, argmax-exact, final cosine=1.00000000". Deterministic seeds (`rand` source 7; `lcg` seeds 99 and `0x1234567`).

**DOS.** bound at ship — `dos commit-audit` of the proof-shipping commit (the witnesses ship in `2c88580 feat(metal): Apple Silicon Metal backend first light` and `74fedc6 fix(metalgemm): free the f16 weight table`).

---

## Theorem 2 — Stub and metal builds present the same interface (no math divergence by build tag)

**THEOREM.** The stub (default) build and the metal build of `metalgemm` present the **same exported interface**, and the stub introduces **no** math divergence: it does no GPU math but degrades cleanly to "unavailable" (`Available()=false`, `Compiled()=false`, `Prefill` `ok=false`) so callers fall back to the pure-Go CPU path.

**REGIME.** N — Numerical (the "no math divergence" half) with a structural compile component.

**PROOF.** The two files carry complementary build tags — `metalgemm.go:1` `darwin && arm64 && cgo`, `metalgemm_stub.go:1` `!(darwin && arm64 && cgo)` — so exactly one compiles per build. Their exported surfaces are diff-identical (`Available, Compiled, Upload, MatMul, Free, ID, UploadVec, FwdConfig, FwdLayer, FwdFinalNorm, Reset, Prefill`, and the `Weight` type). Interface-no-divergence is corroborated by a **real compile witness**: the shared caller `internal/model/metal_prefill.go` (calls `metalgemm.Upload/Prefill/FwdConfig/...`, `metal_prefill.go:152`) plus `cmd/fakchat` and `cmd/modelbench` all build green under both the stub build and the Apple-Silicon+cgo Metal build — had the surfaces diverged, one of those builds would fail to compile. The stub is math-free by construction: `Available`/`Compiled` return `false` (`metalgemm_stub.go:9,11`), `Prefill` returns `ok=false` (`metalgemm_stub.go:45`), `MatMul` is an empty body (`metalgemm_stub.go:21`); so the caller's `if !ok { fall back }` (`metal_prefill.go:153`) routes to the pure-Go CPU path and the stub *cannot* introduce numerical drift.

**WITNESS.**
```
go build ./internal/model/ ./cmd/fakchat/ ./cmd/modelbench/
CGO_ENABLED=1 go build ./internal/model/ ./cmd/fakchat/ ./cmd/modelbench/
```
(No `go test -run` target exists — see verdict.)

**VERDICT.** **OPEN** (2026-06-20; build tags updated for #62). Both builds compiled green at the time (`STUB-BUILD OK`, `FAKMETAL-BUILD OK`), and the exported-surface diff was empty (`IDENTICAL EXPORTED SURFACE`). But this row was **not promoted to PROVEN** then, per the honesty rule: there was **no committed unit test** asserting the stub's behavioral contract. `go test ./internal/metalgemm/` on the default build reported `[no test files]`, because `metalgemm_test.go` was itself tagged for the Metal build and excluded from the stub build entirely. The evidence was a build witness (lives in `go build`), not a re-runnable assertion over the contract. **What closed it later:** the stub-tagged `internal/metalgemm/proofs_witness_test.go` (build tag `!(darwin && arm64 && cgo)`) asserts `!Available() && !Compiled()`, that stub `Prefill` returns `ok==false`, and that `Upload` returns `nil`.

**DOS.** bound at ship — `dos commit-audit` of the commit that adds the stub-contract test (currently unwritten); the existing surface ships in `74fedc6` / `af3a68e` / `ba08d0e` (`fak metalgemm` trailers).

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **stub-metal-interface-parity** → ✅ PROVEN by `TestStubInterfaceParity_CompilesAndScalarContract`. The witness file uses the same build tag as the stub `//go:build !(darwin && arm64 && cgo)` + `package metalgemm`. Two layers of proof: (1) COMPILE-TIME interface parity — a package-level var block binds every exported symbol (Available, Compiled, Upload, UploadVec, FwdConfig, FwdLayer, FwdFinalNorm, Reset, Prefill, (*Weight).MatMul/Free/ID, Weight{Out,In}) to a value of the documented metalgemm.go signature; any drift in the stub's exported surface (rename/arity/type) fails to compile and so fails the whole package. (2) RUNTIME degradation, asserted non-vacuously over multiple shapes incl. degenerate ones: Available()==false & Compiled()==false (stable across repeated calls); Upload(...)==nil for 6 shapes; UploadVec(...)==-1 for 5 lengths; Prefill(...) returns ok==false AND all-four output slices nil for 5 (P,nLayers,w,H) cases incl. large-magnitude X (no NaN/Inf, just declines) — exactly the `if !ok { return nil }` fallback internal/model/metal_prefill.go:152-155 keys on, so the stub introduces NO math divergence (model only folds the pure-Go CPU prefill state). No-math purity: (*Weight).MatMul leaves a sentinel-filled output byte-identical, (*Weight).ID()==-1, Free() on nil is a safe no-op, FwdConfig/FwdLayer/FwdFinalNorm/Reset total no-ops (no panic on benign or adversarial ids, Reset idempotent). The metal-build half of parity is enforced structurally by the shared `package metalgemm` clause + mutually-exclusive build tags (exactly one of the two files compiles) plus the signatures pinned here matching metalgemm.go's declarations.
