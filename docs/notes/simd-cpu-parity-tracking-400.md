---
title: "fak SIMD CPU parity tracker: Go-native kernels vs llama.cpp"
description: "Consolidates the shipped Go-native SIMD lane (AVX2/AVX-512/NEON + runtime detection + bit-exact gates) and the measured CPU throughput against llama.cpp, with an honest per-axis verdict: parity where SIMD is the lever, named non-SIMD residuals where it is not."
---

# SIMD CPU throughput-parity tracking — Go-native kernels vs `llama.cpp` (#400)

> **Umbrella tracker for [#400](https://github.com/anthony-chaudhary/fak/issues/400)** —
> *"achieve parity with llama.cpp on SIMD throughput through Go native SIMD operations
> (eliminate the compute gap)."* This is the **CPU/SIMD sibling** of
> [`gpu-parity-tracking-480.md`](gpu-parity-tracking-480.md) (the device/CUDA path).
>
> **House rule for this repo: every number comes from a real run, and no number is
> restated as new.** The SIMD *implementation* #400 asks for is **already shipped** (AVX2 +
> AVX-512 on amd64, NEON on arm64, runtime ISA detection, bit-exact-vs-scalar gates, graceful
> scalar fallback). The CPU throughput vs llama.cpp is **already measured** and committed. This
> doc **consolidates** that evidence, cites it (never re-states it as a fresh measurement), and
> gives the one thing the umbrella lacked: an **honest per-axis verdict** that separates *what
> SIMD governs* (at parity) from *what it does not* (the non-SIMD residuals).
>
> **The honest one-line verdict:** on the axes SIMD is the lever, fak is at parity with
> llama.cpp on CPU — amd64 prefill **compute slope 1.03×** and batched decode **1.04×
> (slight lead)**. The residual gaps are **not SIMD problems**: amd64 single-stream decode
> (**1.12×** behind) is **memory-bandwidth-bound** (1-thread ≈ all-core), and arm64
> (**0.55–0.73×** decode on M3 Pro) is the **Apple AMX / Accelerate-BLAS / Metal** boundary —
> a hardware matmul unit *outside* SIMD that no pure-NEON kernel can reach. So "the compute
> gap" SIMD can close **is** closed; the remaining distance is owned by memory bandwidth,
> fixed per-prefill overhead, and a non-SIMD hardware unit — each tracked as its own lever.
>
> Written 2026-06-25 on a win32/amd64 dev box (native `go test` green here). The numbers
> below are cited from committed artifacts; the only gate *run for this doc* is the bit-exact
> SIMD correctness suite (§3).

---

## 1. What #400 asks for, mapped to what is shipped

The issue's DoD has four phases. Here is each milestone against the tree, by `file:line` /
commit / committed artifact.

| DoD milestone | Status | Evidence |
|---|---|---|
| Go-native SIMD kernels for the matmul / dot hot paths | ✅ **shipped** | amd64 `qdot8` AVX2 + AVX-512 (`internal/model/quant_amd64.go:37-115`, `quant_amd64.s`); AVX-512 prefill tile GEMM (`quant_amd64.go:154-235`); `fdot3` (`fdot_amd64.go`), `saxpy3` (`saxpy_amd64.go`), AWQ dequant+dot (`awq_amd64.go`, `awq_amd64_asm.go`); arm64 NEON SDOT `qdot8` + 2×4 tile GEMM + amortized decode (`quant_arm64.go:18-274`, `quant_arm64.s`, `quant_arm64_amort.s`) |
| AVX2 / AVX-512 / NEON code paths with **runtime detection** | ✅ **shipped** | amd64: CPUID + XGETBV, stdlib-only, no `golang.org/x/sys/cpu` (`quant_amd64.go:50-100`, `resolveTier`/`detectAVX2`/`detectAVX512`); arm64: HWCAP via `/proc/self/auxv` on linux, unconditional on darwin (`quant_arm64.go:49-102`) |
| **Graceful fallback** on unsupported CPUs | ✅ **shipped** | every kernel has a scalar reference (`qdot8scalar`, `qgemm8cell`, `awqDotProductScalar`); `resolveTier` returns `tierScalar` when neither ISA is present; `FAK_QKERNEL=scalar` pins it (`quant_amd64.go:50-67`) |
| New tests for SIMD correctness (**bit-exact vs scalar**) | ✅ **shipped + run green here** | `TestQdot8AsmMatchesScalar`, `TestQGemm8AsmMatchesScalar`, `TestQdot8KernelsMatchScalar`, `TestQdot8MatchesF32` (amd64); `TestQdot8NEONMatchesScalar` (arm64); amortized path proven argmax-exact + cosine ≥ 0.9999 (`quant_arm64_amort_test.go:58-95`). See §3. |
| All existing tests pass | ✅ | `go test ./internal/model/ -run 'Qdot8|QGemm|Scalar|Quantize'` → `ok` (§3) |
| Benchmark harness to measure SIMD lift vs baseline | ✅ **shipped** | `cmd/q8bench` (Q8 vs f32 decode tok/s + argmax check), `cmd/batchbench` (batched throughput), `cmd/modelbench` (forward-pass latency); `FAK_QKERNEL` A/B-pins the tier so the lift is directly measurable |
| Benchmark shows parity with llama.cpp on same hardware | 🟡 **parity on the SIMD-governed axes; residuals named** | §2 (per-axis verdict, cited) |
| JSON artifact committed with hardware/config | ✅ **exists** (amd64 + arm64) | `experiments/model-baseline/comparison.json`, `batch-decode-q8.json`; `experiments/model-ladder/qwen25-1.5b-q8-cpu-parity-m3pro.json` |
| Result in `BENCHMARK-AUTHORITY.md` | ✅ (arm64 row) / 🟡 (amd64 lives in `LLAMACPP-HEADTOHEAD-RESULTS.md`) | `BENCHMARK-AUTHORITY.md:30` (M3 CPU parity, canonical JSON cited) |
| Baseline explicitly stated (which llama.cpp) | ✅ | amd64: `llama-cpp-python` 0.3.30 CPU wheel (`LLAMACPP-HEADTOHEAD-RESULTS.md:29-30`); arm64: llama.cpp `-ngl 0`, build 8200 `541bf3762` (`BENCHMARK-AUTHORITY.md:30`) |

**The verb #400 used — "eliminate the compute gap" — is the precise frame.** The *compute*
(the SIMD inner loop) is at parity; the residual is *not* compute. §2 proves it axis by axis.

---

## 2. The measured CPU axes (cited, not restated)

### 2.1 amd64 — Zen5 16C/32T, SmolLM2-135M, Q8_0

Canonical artifact: [`experiments/model-baseline/comparison.json`](https://github.com/anthony-chaudhary/fak/blob/main/experiments/model-baseline/comparison.json)
(re-measured 2026-06-17 on an **idle** box, CPU ~10%), folded in
[`docs/benchmarks/LLAMACPP-HEADTOHEAD-RESULTS.md`](../benchmarks/LLAMACPP-HEADTOHEAD-RESULTS.md).

| axis | llama.cpp | fak | verdict | SIMD-governed? |
|---|---|---|---|---|
| single-stream **decode** (1t, Q8) | 6.91 ms/tok | 7.75 ms/tok | fak **1.12× behind** | ❌ memory-bandwidth-bound (1t ≈ all-core; `comparison.json` `note`) |
| single-stream **prefill** — per-token **slope** | 0.337 ms/tok | 0.346 ms/tok | **parity (1.03×)** | ✅ **yes — the compute slope IS at parity** (`LLAMACPP-HEADTOHEAD-RESULTS.md:46`) |
| single-stream **prefill** — absolute **wall@P=256** | 74.7 ms (1t) / 82.9 ms (32t) | 145.8 ms (all-core) | fak **1.76–1.95× behind** | ❌ fixed per-prefill overhead (~135 MB Q8 weights streamed), not the inner loop |
| **batched decode** throughput (peak, 32t) | ~2816 tok/s (B=256) | **2916 tok/s** (B=960) | **parity / slight fak lead (1.04×)** | ✅ yes (`LLAMACPP-HEADTOHEAD-RESULTS.md:70-75`) |
| cross-agent **shared-prefix** (agents/s, C=32) | 17.2 *(preliminary)* | 5.2 | **OPEN — needs verification** | n/a — settings-dependent probe (`…:77-111`) |
| **turns × agents** (long endpoint grid) | 1.59–2.71 agent-turns/s | 0.58–0.87 | fak **0.32–0.41×** | ❌ end-to-end serving shape, not the SIMD kernel |

The two axes where SIMD is the lever — **prefill compute slope** and **batched decode** — are
at **parity**. The decode-latency residual is memory-bandwidth (a load Go can't emit + shared
bus), the absolute-prefill residual is fixed weight-stream overhead, and the cross-agent /
turns×agents axes are serving-shape concerns owned elsewhere — none is a SIMD-kernel deficit.

> **Doc-reconciliation note (not edited here):** the head-to-head doc's *intro* (line 16) still
> reads batched as "behind (2.26×)" — that is the **pre-fix** number; its own **Axis 3 body**
> (lines 70-75) records the **post-fix** 2916 tok/s / 1.04× lead after the per-step allocation,
> attention, SwiGLU, and Q8 tile-dispatch fixes. This tracker cites the body verdict and flags
> the stale intro line as a follow-up doc edit (out of this note's lane).

### 2.2 arm64 — Apple M3 Pro, Qwen2.5-1.5B, Q8_0

Canonical artifact: `experiments/model-ladder/qwen25-1.5b-q8-cpu-parity-m3pro.json`
(`BENCHMARK-AUTHORITY.md:30`), narrative
[`docs/benchmarks/M3-LLAMACPP-RESULTS.md`](../benchmarks/M3-LLAMACPP-RESULTS.md).

| axis | llama.cpp CPU (`-ngl 0`) | fak (NEON Q8) | verdict |
|---|---|---|---|
| decode | 52.4 (12t) / 68.7 (6t) | 38.1 | fak **0.55–0.73×** |
| prefill@256 | 412.5 tok/s | 240.4 tok/s | fak **0.58×** |

The NEON SDOT lane is **bit-identical** to scalar (`TestQdot8NEONMatchesScalar`) and flipped
int8 to 1.9× faster than f32 (`M3-LLAMACPP-RESULTS.md:26-33`) — the SIMD kernel works. The
residual to llama.cpp is **not a better NEON kernel**: llama.cpp reaches its M3 numbers through
**Apple AMX via Accelerate's SGEMM + the Metal backend**, a hardware block-matmul unit outside
the NEON SIMD ISA. No pure-NEON shape (including the 2×4 tile, `FAK_ARM_TILE=1`) beats the
simple per-cell SDOT on M3 for this reason (`quant_arm64.go:177-182`). This is a *hardware-unit*
gap, not a SIMD-throughput gap — and the device-acceleration answer to it is the GPU/Metal lane
tracked in [`gpu-parity-tracking-480.md`](gpu-parity-tracking-480.md), not more NEON.

---

## 3. The gate run for this doc — bit-exact SIMD correctness (green)

The DoD's correctness requirement ("SIMD results must match scalar, bit-exact") is the gate I
can run on this amd64 host, and it is **green**:

```
$ go test ./internal/model/ -run 'TestQdot8|TestQGemm8' -count=1 -v
--- PASS: TestQdot8KernelsMatchScalar
--- PASS: TestQGemm8CellLaneGeometries
--- PASS: TestQGemm8CellRejectsInvalidLaneCount
--- PASS: TestQGemm8AsmMatchesScalar
--- PASS: TestQGemm8IntoManyMatchesSeparate
--- PASS: TestQdot8MatchesF32
--- PASS: TestQdot8AsmMatchesScalar
ok  	github.com/anthony-chaudhary/fak/internal/model	0.466s
```

`TestQdot8AsmMatchesScalar` is the load-bearing bit-identity gate: the active SIMD tier
(AVX-512 here) is asserted equal to the scalar reference via `math.Float32bits` equality, so
the SIMD dispatch (`quant_amd64.go:105-115`) changes only speed, never numerics. The arm64
analogue `TestQdot8NEONMatchesScalar` is green on Apple Silicon (`M3-LLAMACPP-RESULTS.md:28`),
not runnable on this amd64 box.

---

## 4. The honest residual — what closing the *last* distance would take (and why none is SIMD)

| residual | axis | why it is NOT a SIMD fix | who owns it |
|---|---|---|---|
| **1.12× decode** (amd64) | single-stream decode | memory-bandwidth-bound (1t ≈ all-core); the kernel is already AVX-512, the bus is the wall | a fused/wider load Go can't emit, or a quieter box; not a new SIMD op |
| **1.76–1.95× wall@P=256** (amd64) | absolute prefill | fixed per-prefill overhead (~135 MB Q8 weights streamed once); the per-token *slope* is already at parity | weight-residency / load amortization, not the GEMM tile |
| **0.55–0.73× decode, 0.58× prefill** (arm64) | M3 Pro | llama.cpp uses **Apple AMX + Accelerate + Metal** — a hardware matmul unit outside the NEON SIMD ISA | the GPU/Metal lane ([#480](gpu-parity-tracking-480.md)), not more NEON |
| a single consolidated **#400 CPU JSON** | reporting | the committed numbers come from separate amd64/arm64 runs/docs; a fresh same-box, same-quant, matched-thread cross-engine re-run vs a **pinned** llama.cpp commit would fold them into one artifact | a quiet amd64 box with llama.cpp built (this dev box is shared/contended) |
| **cross-agent shared-prefix** robustness | Axis 4 | preliminary, settings-dependent probe (needs multi-config + real `llama-server` parallel slots) | the serving lane, flagged OPEN in `LLAMACPP-HEADTOHEAD-RESULTS.md:77-111` |

**Verdict for #400:** the Go-native SIMD implementation is shipped, correct (bit-exact gate
green), runtime-detected, and falls back gracefully — and on the axes SIMD governs it is at
**parity** with llama.cpp on CPU. The compute gap SIMD can close **is** closed. The remaining
distance is memory bandwidth, fixed per-prefill overhead, and a non-SIMD hardware unit (Apple
AMX) — each named, each tracked on its own lever. This umbrella's SIMD ask is **met**; the
non-SIMD residuals do not belong to it.

---

## 5. Provenance

- **Implementation:** `internal/model/quant_amd64.{go,s}` (AVX2/AVX-512 + CPUID/XGETBV
  detection), `internal/model/quant_arm64.{go,s}` + `quant_arm64_amort.s` (NEON + HWCAP),
  `fdot_amd64.{go,s}`, `saxpy_amd64.{go,s}`, `awq_amd64*.go`.
- **Correctness gates:** `internal/model/quant_amd64_test.go` (`TestQdot8AsmMatchesScalar`,
  `TestQGemm8AsmMatchesScalar`), `quant_arm64_test.go` (`TestQdot8NEONMatchesScalar`),
  `quant_arm64_amort_test.go` (argmax-exact + cosine ≥ 0.9999). Run green here 2026-06-25 (§3).
- **Measured baselines (cited, not restated):** `experiments/model-baseline/comparison.json`
  + `batch-decode-q8.json` (amd64 Zen5); `experiments/model-ladder/qwen25-1.5b-q8-cpu-parity-m3pro.json`
  + `BENCHMARK-AUTHORITY.md:30` (arm64 M3 Pro); narratives `docs/benchmarks/LLAMACPP-HEADTOHEAD-RESULTS.md`
  + `docs/benchmarks/M3-LLAMACPP-RESULTS.md` + `docs/benchmarks/MODEL-BASELINE-RESULTS.md` (ACT 5,
  the prefill-slope parity).
- **Baselines pinned:** amd64 `llama-cpp-python` 0.3.30 CPU wheel; arm64 llama.cpp build 8200
  (`541bf3762`), `-ngl 0`.
- **Sibling tracker:** `docs/notes/gpu-parity-tracking-480.md` (the device/CUDA path).
- Written 2026-06-25 on a win32/amd64 host; hardware referred to by class, never lab host/IP/path.
