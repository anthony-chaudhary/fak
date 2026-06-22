//go:build arm64

package model

import (
	"encoding/binary"
	"os"
	"runtime"
)

// quant_arm64.go — the arm64 SIMD entry point for the Q8_0 inner product, the twin of
// quant_amd64.go. The hand-written NEON kernel (quant_arm64.s, SDOT) ships in the same static
// binary — no cgo, no FFI, no external process — so the in-kernel thesis is intact; it is
// "pure Go" in the one sense that matters (one binary, the kernel owns the math), just not
// pure scalar. Falls back to qdot8scalar on any arm64 CPU without FEAT_DotProd.

//go:noescape
func qdot8asm(qw, qx *int8, dw, dx *float32, nblk int) float32

//go:noescape
func qdot8amortNEON(qw, qx *int8, dw, dx *float32, nblk int) float32

// qdot8 kernel tiers for arm64, mirroring the amd64 FAK_QKERNEL tier scheme in
// quant_amd64.go (tierScalar/tierAVX2/tierAVX512). Here the ladder is:
//
//	tierScalar — no SDOT (qdot8scalar), the safe fallback;
//	tierNEON   — the bit-identical SDOT path (qdot8asm) + the default decode GEMV (unroll4);
//	tierAmort  — issue #477's amortized-FP-reduction decode kernel (qdot8amortNEON), which
//	             forfeits bit-identity for the llama.cpp-style deferred reduction.
//
// The SMMLA/i8mm kernel is a documented follow-up (see detectI8MM); when it lands it becomes a
// higher tier reached on i8mm-capable parts, and the dispatch below already routes to it.
const (
	tierScalar = iota
	tierNEON
	tierAmort
)

// qkernelTier is resolved once at init. FAK_QKERNEL pins the tier for A/B measurement
// ("scalar"|"neon"/"sdot"/"asimddp"|"amort"; "i8mm"/"smmla" alias the amortized kernel until
// the SMMLA tier ships). neonDot is the derived bool the SDOT-based paths (qdot8, the prefill
// GEMM) already gate on — true for any tier at or above tierNEON. The DEFAULT (no env) keeps
// the prior behavior exactly: detectDotProd() → tierNEON or tierScalar, so the bit-identical
// qdot8asm path and TestQdot8NEONMatchesScalar are untouched.
var qkernelTier = resolveQKernelTier()

var neonDot = qkernelTier >= tierNEON

func resolveQKernelTier() int {
	switch os.Getenv("FAK_QKERNEL") {
	case "scalar":
		return tierScalar
	case "neon", "sdot", "asimddp":
		return tierNEON // operator asserts the hardware; the asm is built in regardless
	case "amort", "i8mm", "smmla":
		// i8mm/smmla alias amort until the SMMLA tier lands (honest-block follow-up). The
		// amortized kernel itself needs only FEAT_DotProd, so it is the correct selection on
		// any SDOT-capable part the operator points it at.
		return tierAmort
	}
	if detectDotProd() {
		return tierNEON
	}
	return tierScalar
}

// detectDotProd reports FEAT_DotProd (SDOT/UDOT) support. Every Apple Silicon part (M1+) has
// it, so darwin/arm64 is unconditionally true. On linux/arm64 it reads HWCAP from
// /proc/self/auxv (AT_HWCAP, bit 20 = asimddp) using stdlib only — no golang.org/x/sys/cpu
// dependency, matching the amd64 path's self-contained CPUID detection. Any other arm64 OS
// falls back to scalar (safe: a #UD on SDOT would crash, so absent proof we assume absent).
func detectDotProd() bool {
	switch runtime.GOOS {
	case "darwin", "ios":
		return true
	case "linux", "android":
		return linuxHasASIMDDP()
	default:
		return false
	}
}

func linuxHasASIMDDP() bool {
	const atHWCAP = 16
	const hwcapASIMDDP = 1 << 20
	b, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		return false
	}
	// auxv is a flat array of (uint64 type, uint64 value) pairs, terminated by type 0.
	for i := 0; i+16 <= len(b); i += 16 {
		typ := binary.LittleEndian.Uint64(b[i:])
		val := binary.LittleEndian.Uint64(b[i+8:])
		if typ == atHWCAP {
			return val&hwcapASIMDDP != 0
		}
		if typ == 0 {
			break
		}
	}
	return false
}

// i8mmAvailable reports FEAT_I8MM (the SMMLA / int8 matrix-multiply-accumulate extension).
// It is the SELECTION gate for the (follow-up) SMMLA decode tier; the amortized-reduction
// kernel that ships now uses SDOT (FEAT_DotProd) and does NOT require i8mm, so it stays
// reachable on any SDOT part. Detection is the i8mm twin of neonDot's FEAT_DotProd path.
var i8mmAvailable = detectI8MM()

// detectI8MM reports FEAT_I8MM. ARMv8.6-A mandates it and every Apple part from M2/A15 on has
// it (M3 — issue #477's target — does); per the issue spec darwin/ios is taken as true. (M1/A14
// have FEAT_DotProd but NOT i8mm; that imprecision is harmless here because i8mm only gates the
// not-yet-shipped SMMLA tier — the amortized kernel runs on M1 via SDOT regardless.) On linux it
// reads AT_HWCAP2 bit 13 (HWCAP2_I8MM) from /proc/self/auxv — the AT_HWCAP2 twin of
// linuxHasASIMDDP's AT_HWCAP read, stdlib-only. Any other arm64 OS: false (safe — absent proof
// of the feature we assume absent, so an SMMLA #UD can never be reached).
func detectI8MM() bool {
	switch runtime.GOOS {
	case "darwin", "ios":
		return true
	case "linux", "android":
		return linuxHasI8MM()
	default:
		return false
	}
}

func linuxHasI8MM() bool {
	const atHWCAP2 = 26
	const hwcap2I8MM = 1 << 13
	b, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		return false
	}
	// auxv is a flat array of (uint64 type, uint64 value) pairs, terminated by type 0.
	for i := 0; i+16 <= len(b); i += 16 {
		typ := binary.LittleEndian.Uint64(b[i:])
		val := binary.LittleEndian.Uint64(b[i+8:])
		if typ == atHWCAP2 {
			return val&hwcap2I8MM != 0
		}
		if typ == 0 {
			break
		}
	}
	return false
}

// qdot8 dispatches the Q8_0 GEMV inner product to the NEON kernel when available, else the
// scalar reference. The NEON kernel is bit-identical to qdot8scalar (TestQdot8NEONMatchesScalar),
// so this dispatch changes only speed, never the quantized decode path's numerics.
func qdot8(qw []int8, dw []float32, qv q8Vec, nblk int) float32 {
	if neonDot && nblk > 0 {
		return qdot8asm(&qw[0], &qv.q[0], &dw[0], &qv.d[0], nblk)
	}
	return qdot8scalar(qw, dw, qv, nblk)
}

// qGemm8 / qGemm8Into: the batched prefill GEMM. On arm64 there is no register-blocked tile
// kernel (qgemm8tile is amd64 asm), so the amd64 "scalar" fallback — qgemm8cell, a pure-Go
// per-cell dot — is what ran here, and it left prefill at scalar speed (the NEON kernel was
// reached only by decode). Now each output cell goes through the NEON qdot8 (qdot8asm) instead:
// the same int8×int8→int32 SDOT the decode GEMV uses, with the weight row held in cache across
// the P activation columns. This is the prefill twin of the decode win. The reduction order is
// qdot8's (per-block int sum → sequential FMA), not qgemm8cell's deferred-lane order — both are
// valid Q8_0 dots within the quantized gate's tolerance (argmax-exact vs the HF oracle), and no
// arm64 rung pins the GEMM to a specific order (the bit-exact GEMM pin is amd64-only). Falls
// back to the scalar reference on an arm64 part without FEAT_DotProd.
func qGemm8(qt *q8Tensor, qp *q8Panel) []float32 {
	Y := make([]float32, qp.P*qt.out)
	qGemm8Into(qt, qp, Y)
	return Y
}

// qGemm8Into is the buffer-reuse form (see quant_amd64.go's doc comment). The default arm64 prefill
// GEMM is the per-cell NEON SDOT sweep: on Apple Silicon (M3 Pro) it is the FASTEST Q8_0 GEMM we
// have — the register-blocked deferred-reduction tile (qGemm8TileInto), which wins big on AVX-512,
// REGRESSES here (per-cell ~100 MAC/ns vs tile ~85 agg; ~15 vs ~11 single-core). The measured tell
// is that llama.cpp CPU's 7.6× prefill/decode ratio comes from Apple AMX via Accelerate's SGEMM,
// not a better NEON kernel, so no pure-NEON shape beats the simple per-cell SDOT here. The tile is
// kept correct (TestQGemm8IntoMatchesScalarNEON) and opt-in via FAK_ARM_TILE=1 for non-Apple arm64
// parts where the AVX-512-style amortization may pay off.
func qGemm8Into(qt *q8Tensor, qp *q8Panel, Y []float32) {
	if qgemmMode == qgemmModeLegacy {
		qGemm8legacyInto(qt, qp, Y)
		return
	}
	if !neonDot {
		qGemm8scalarInto(qt, qp, 4, Y)
		return
	}
	if armUseTile {
		qGemm8TileInto(qt, qp, Y)
		return
	}
	out, in, nblk, P := qt.out, qt.in, qt.nblk, qp.P
	Pmain := P &^ 3   // tokens handled 4-at-a-time
	Omain := out &^ 1 // rows handled in 2-row bands by the 2×4 tile (73 MAC/ns/core; ~2.7× row4)
	// Each band runs the 2×4 load-reusing tile over the token blocks, with the token remainder
	// (P%4) via qgemm8cell. Tile/row4/cell are all bit-identical to qgemm8cell(...,4), so the whole
	// GEMM stays bit-identical to qGemm8scalar(...,4) (TestQGemm8IntoMatchesScalarNEON).
	tileRow := func(lo, hi int) {
		for p := lo; p < hi; p++ {
			o := p * 2
			for t := 0; t < Pmain; t += 4 {
				qgemm8tile2x4NEON(&qt.q[o*in], &qp.q[t*in], &qt.d[o*nblk], &qp.d[t*nblk], in, nblk, out, &Y[t*out+o])
			}
			for t := Pmain; t < P; t++ {
				qx := qp.q[t*in : t*in+in]
				dx := qp.d[t*nblk : t*nblk+nblk]
				Y[t*out+o] = qgemm8cell(qt.q[o*in:o*in+in], qt.d[o*nblk:o*nblk+nblk], qx, dx, nblk, 4)
				Y[t*out+o+1] = qgemm8cell(qt.q[(o+1)*in:(o+1)*in+in], qt.d[(o+1)*nblk:(o+1)*nblk+nblk], qx, dx, nblk, 4)
			}
		}
	}
	nPairs := Omain / 2
	if out*in*P < parThreshold {
		tileRow(0, nPairs)
	} else {
		parFor(nPairs, numWorkers, tileRow)
	}
	// odd last row (out%2): the 1×4 row kernel + cell remainder.
	if Omain < out {
		o := out - 1
		qw := qt.q[o*in : o*in+in]
		dw := qt.d[o*nblk : o*nblk+nblk]
		for t := 0; t < Pmain; t += 4 {
			qgemm8row4NEON(&qw[0], &qp.q[t*in], &dw[0], &qp.d[t*nblk], in, nblk, out, &Y[t*out+o])
		}
		for t := Pmain; t < P; t++ {
			Y[t*out+o] = qgemm8cell(qw, dw, qp.q[t*in:t*in+in], qp.d[t*nblk:t*nblk+nblk], nblk, 4)
		}
	}
}

func qGemm8IntoMany(qp *q8Panel, targets ...qgemm8Target) {
	for _, tg := range targets {
		qGemm8Into(tg.qt, qp, tg.Y)
	}
}

// q8PreextendVec / extendQ8ToQ16 / qdot8GEMV / qMatRowsRangeFast — the four Q8 helpers the
// arch-neutral callers (quant.go, quant_forward.go) reference unconditionally. amd64 supplies
// AVX-accelerated forms (quant_amd64.go) and every other arch the portable scalar ones
// (quant_noasm.go); arm64 was the gap — it carried its own qdot8/qGemm8 NEON path but not
// these, so the package failed to build here. The arm64 forms mirror the portable semantics,
// except qdot8GEMV routes through the NEON SDOT qdot8 (its bit-identical fast path) rather than
// the scalar reference: arm64's SDOT consumes int8 directly, so no int16 pre-extension is
// wanted (q8PreextendVec=false) and the row-range fast path declines to the generic caller.
func q8PreextendVec() bool { return false }

func extendQ8ToQ16(q []int8, q16 []int16, nblk int) {
	extendQ8ToQ16Scalar(q, q16, nblk)
}

// qdot8GEMV is the decode matvec dot. It routes through the latency-hiding qdot8unroll4NEON (four
// independent float accumulators + four independent SDOT scratch regs) rather than the
// bit-identical-to-scalar qdot8asm: the unrolled kernel is ~2.8-3.3× faster single-core (the
// per-cell kernel's per-block SDOT->SCVTF->FMLA chain was latency-bound, not throughput-bound).
// The 4-accumulator reduction order is NOT bit-identical to qdot8scalar — it is the same
// llama.cpp-style deferred order the AVX-512 GEMV already ships, gated by the Q8 oracle-fidelity
// tests (argmax-exact), not by scalar bit-identity (which qdot8/qdot8asm still hold for their test).
func qdot8GEMV(qw []int8, dw []float32, qv q8Vec, nblk int) float32 {
	if neonDot && nblk > 0 {
		// FAK_QKERNEL=amort (tierAmort) selects the issue #477 amortized-reduction kernel; the
		// default (tierNEON) keeps the existing unroll4 GEMV. Both are deferred-reduction (NOT
		// bit-identical to scalar); amort additionally amortizes the float reduce ggml-style.
		if qkernelTier >= tierAmort {
			return qdot8amortNEON(&qw[0], &qv.q[0], &dw[0], &qv.d[0], nblk)
		}
		return qdot8unroll4NEON(&qw[0], &qv.q[0], &dw[0], &qv.d[0], nblk)
	}
	return qdot8(qw, dw, qv, nblk)
}

// qMatRowsRangeFast declines on arm64: the decode GEMV is memory-bandwidth-bound, and the
// per-row qdot8GEMV (qdot8asm, single sequential weight stream) already saturates ~105 GB/s at
// full cores — a register-tiled multi-row kernel only interleaves weight streams and adds FP
// overhead with no token reuse to amortize it (measured slower). The prefill GEMM, where the
// register tile DOES pay off (token reuse, compute-bound), routes through qGemm8Into instead.
func qMatRowsRangeFast(qt *q8Tensor, qv q8Vec, y []float32, lo, hi int) bool {
	return false
}
