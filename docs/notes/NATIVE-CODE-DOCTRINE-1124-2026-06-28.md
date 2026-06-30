---
title: "The native-code doctrine — where raw C / Go-assembly earns its place in fak (2026-06-28)"
description: "fak's written rule for dropping below portable Go: Go-assembly (.s) is welcome on the model/compute path and is NOT cgo; cgo/import \"C\" is justified only for GPU device code behind an opt-in build tag; neither is ever allowed on the request/adjudication path. Plus the verified backlog of hot scalar kernels worth vectorizing (C1–C6) with their status, and the four places native code must never go."
---

# The native-code doctrine (#1124)

This is the decision framework for epic [#1124](https://github.com/anthony-chaudhary/fak/issues/1124):
**where does dropping below portable Go (hand-written Go-assembly, or cgo C/C++/CUDA/Metal)
genuinely buy speed, and where would it only buy a static-binary regression, a portability
tax, or no measurable win?** It is a *rule first, work list second*. The rule is already
half-enforced by `internal/architest`; this note writes it down so we stop re-debating it.

## Two zones, one bright line — enforced by `internal/architest`

| Zone | Packages | Native-code policy | Enforced by |
|---|---|---|---|
| **Request / adjudication path** | `adjudicator, kernel, vdso, grammar, preflight, ctxmmu, ratelimit` + everything in `requestPathClosure` | **NEVER cgo, NEVER os/exec, NEVER `import "C"`.** Pure static Go only. | `TestHotPathHasNoExec`, `TestEveryAdjudicatorIsExecFree`, `TestRequestPathDefaultBuildIsCgoFree` |
| **Model / compute path** | `internal/model`, `internal/compute`, `internal/metalgemm` | Native code is **allowed and welcome**, but always behind an opt-in `//go:build` tag, off the default-build closure. | the cgo gate passes only because the cgo files are tag-gated out of `go build ./cmd/fak` |

### The sharp distinction the rule turns on — Go-assembly is NOT cgo

- **Go-assembly (`.s` files)** links into the *static* binary, pulls **no C toolchain**, and
  passes `TestRequestPathDefaultBuildIsCgoFree` untouched. It is the **correct and preferred**
  tool for CPU SIMD (AVX2/AVX-512/VNNI/NEON). The `.s` files in `internal/model` are exactly
  this. **More of this is good** wherever a hot CPU kernel is still scalar.
- **cgo (`import "C"` → C/C++/CUDA/Metal/Vulkan)** pulls a C toolchain + a dynamically-linked
  C runtime onto whatever links it. It is justified **only** for GPU *device* code (there is
  no other way to reach CUDA/Metal), and **only** behind an opt-in tag so the shipped `fak`
  binary stays one static Go executable. **More of this is good only on the GPU device
  kernels** — never as a CPU shortcut, never on the request path.

## Where raw C / asm does NOT make sense (the "don't" half — these are deliverables too)

1. **Anywhere on the request/adjudication path.** Non-negotiable; architest reds the trunk.
   A static, toolchain-free, single-binary decision path is a load-bearing property
   (`DIRECTION.md`), not a nicety.
2. **Dense f32 dot / GEMV.** `fdot` (`parallel.go`) is pure-Go with 8 ILP accumulators and a
   fixed combine order; the compiler already auto-vectorizes it and it hits memory bandwidth
   at decode. Hand-asm here buys ~nothing and costs bit-identity risk. **Leave it Go.**
3. **The CUDA *prefill* gap (~380× behind llama.cpp).** Not a kernel-quality problem — the
   GEMMs already call cuBLAS. It is a per-op launch-overhead / batching defect; the fix is
   **Go-side orchestration** (CUDA graphs / continuous batching), not more `.cu`.
4. **Re-implementing what cuBLAS / MPS already do well.** GEMM stays on the vendor BLAS; our
   native device code is the *dequant-fused* and *whole-forward-resident* glue around it.

## Where MORE native code DOES earn its place — the verified backlog

Every one is a *hot* kernel currently running **scalar Go** on a path the roofline says
matters, on a device/arch where it is measurable.

| # | Gap | Status |
|---|---|---|
| **C1 [#1125]** | Q6_K decode reduce is scalar on every arch → AVX2/VNNI + NEON | open (the deferred 3rd reducer from #209) |
| **C2 [#1126]** | Q5_K decode reduce is scalar on arm64 → NEON SDOT | open (amd64 already has AVX2/VNNI) |
| **C3 [#1127]** | AVX2 Q8 prefill GEMM falls to scalar → register-blocked AVX2 tile | open (the CPU server / AVX2-only-host prefill floor) |
| **C4 [#1128]** | AWQ amd64 SIMD bodies were scalar placeholders → real AVX2/AVX-512 dequant+dot | **SHIPPED 2026-06-28** — `awq_amd64_simd.s`; dequant bit-identical, dot cosine-parity, vet(asmdecl)-clean, witnessed on Zen 5 (AVX2 **and** AVX-512). |
| **C5 [#1129]** | CPU attention inner loop is pure scalar → measure share, then Go-ILP / flash / SIMD | open (lower-confidence; measure-first) |
| **C6 [#1130]** | Dequant-at-LOAD body is scalar (and serial) → SIMD it, compose with #1102's parFor | open |

## The acceptance bar every child owes (the `net-true-value` + #209 pattern)

1. **Bit-identity first** where there is no reduction (dequant): the asm output equals the
   scalar reference bit-for-bit. Where there *is* a float reduction (dot/GEMV), the lane order
   differs from the sequential scalar sum, so the bar is **cosine-parity** (≈1.0) with a tight
   relative bound — bit-identity is not achievable for a reduction.
2. **`go vet` (asmdecl) clean** + the #209 GO-ASM LAWS (correct frame/arg sizes, SS/stack
   segment, X↔Y aliasing, block-stride constants, `·`-prefixed package vars).
3. **A microbench ladder** (scalar → AVX2/NEON → VNNI where applicable) on the matching ISA,
   with the witnessed multiple over scalar quoted.
4. **Default path provably untouched** — dispatch behind the existing CPUID / feature-tier
   gate; the portable scalar stays the fallback on non-matching arch.
5. **Provenance-labeled** (witnessed vs modeled) per `docs/standards/net-true-value.md`; no
   e2e tok/s claim without the host that can actually run it.

## Definition of done

The doctrine above lives here (tracked), and C1–C6 are each either shipped-with-witness or
closed-as-not-worth-it with the measurement that killed it. The "where it doesn't make sense"
verdicts (1–4) are as much a deliverable as the kernels. **C4 is the first child shipped.**
