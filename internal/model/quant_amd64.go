//go:build amd64

package model

import "os"

// quant_amd64.go — the amd64 SIMD entry point for the Q8_0 inner product. This is the
// change that turns the quantization byte-savings into actual speed: the scalar int8 dot
// is compute-bound (Go emits per-byte sign-extend + scalar imul, ~2.5× slower than the f32
// FMA path), so streaming 4× fewer weight bytes bought nothing until the dot itself goes
// SIMD. The kernel is hand-written Go assembly (quant_amd64.s) — it ships in the same
// static binary, no cgo, no FFI, no external process, so the in-kernel thesis is intact;
// it is "pure Go" in the sense that matters (one binary, the kernel owns the math), just
// not pure scalar. Falls back to qdot8scalar on any CPU without AVX2.

//go:noescape
func qdot8asm(qw, qx *int8, dw, dx *float32, nblk int) float32

//go:noescape
func qdot8asm512(qw, qx *int8, dw, dx *float32, nblk int) float32

//go:noescape
func qdot8gemv512(qw, qx *int8, dw, dx *float32, nblk int) float32

//go:noescape
func qdot8gemv512q16(qw *int8, qx *int16, dw, dx *float32, nblk int) float32

//go:noescape
func q8ToQ16Asm512(q *int8, q16 *int16, nblk int)

//go:noescape
func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)

//go:noescape
func xgetbv() (eax, edx uint32)

// qdot8 kernel tiers, fastest first. The active tier is resolved once at init.
const (
	tierScalar = iota
	tierAVX2
	tierAVX512
)

// qtier is the resolved kernel tier. Feature detection uses CPUID + XGETBV (the OS must
// have enabled the relevant vector state, else the instructions would #UD); no dependency
// on golang.org/x/sys/cpu — this module is deliberately stdlib-only. FAK_QKERNEL pins the
// tier ("scalar"|"avx2"|"avx512") for A/B measurement, capped at what the hardware has.
var qtier = resolveTier()

func resolveTier() int {
	want := tierAVX512
	switch os.Getenv("FAK_QKERNEL") {
	case "scalar":
		want = tierScalar
	case "avx2":
		want = tierAVX2
	case "avx512":
		want = tierAVX512
	}
	if want >= tierAVX512 && detectAVX512() {
		return tierAVX512
	}
	if want >= tierAVX2 && detectAVX2() {
		return tierAVX2
	}
	return tierScalar
}

// cpuFeatureBits returns the CPUID leaf-7 EBX feature bits, but only after confirming the OS
// has enabled the XSAVE state selected by xcr0Mask (XMM/YMM for VEX-256, +opmask/ZMM for
// AVX-512). ok is false when OSXSAVE is clear or any required XCR0 state bit is unset, so a
// feature whose register state the OS never saves is reported absent. Shared probe of
// detectAVX2 / detectAVX512, which differ only in the XCR0 mask and the feature-bit test.
func cpuFeatureBits(xcr0Mask uint32) (uint32, bool) {
	_, _, ecx1, _ := cpuid(1, 0)
	const osxsave = 1 << 27
	if ecx1&osxsave == 0 {
		return 0, false
	}
	xcr0, _ := xgetbv()
	if xcr0&xcr0Mask != xcr0Mask {
		return 0, false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	return ebx7, true
}

func detectAVX2() bool {
	// bit1 = SSE (XMM) state, bit2 = AVX (YMM) state — both required for VEX-256.
	ebx7, ok := cpuFeatureBits(0x6)
	if !ok {
		return false
	}
	const avx2 = 1 << 5
	return ebx7&avx2 != 0
}

func detectAVX512() bool {
	// need XMM(1)+YMM(2)+opmask(5)+ZMM_Hi256(6)+Hi16_ZMM(7) state = 0xe6 for zmm ops.
	ebx7, ok := cpuFeatureBits(0xe6)
	if !ok {
		return false
	}
	const avx512f = 1 << 16
	const avx512bw = 1 << 30
	return ebx7&avx512f != 0 && ebx7&avx512bw != 0
}

// qdot8 dispatches to the resolved kernel tier. All three are bit-identical
// (TestQdot8AsmMatchesScalar), so this dispatch changes only speed, never the quantized
// path's numerics.
func qdot8(qw []int8, dw []float32, qv q8Vec, nblk int) float32 {
	if nblk > 0 {
		switch qtier {
		case tierAVX512:
			return qdot8asm512(&qw[0], &qv.q[0], &dw[0], &qv.d[0], nblk)
		case tierAVX2:
			return qdot8asm(&qw[0], &qv.q[0], &dw[0], &qv.d[0], nblk)
		}
	}
	return qdot8scalar(qw, dw, qv, nblk)
}

func q8PreextendVec() bool {
	return qtier == tierAVX512
}

func extendQ8ToQ16(q []int8, q16 []int16, nblk int) {
	if nblk > 0 && qtier == tierAVX512 {
		q8ToQ16Asm512(&q[0], &q16[0], nblk)
		return
	}
	extendQ8ToQ16Scalar(q, q16, nblk)
}

// qdot8GEMV is the decode matvec dot. The public qdot8 kernel remains bit-identical to
// qdot8scalar for its tests; on AVX-512, GEMV uses the same deferred FMA lane reduction as
// the batched Q8 GEMM, which is the faster llama.cpp-style reduction order already gated by
// Q8 oracle-fidelity tests.
func qdot8GEMV(qw []int8, dw []float32, qv q8Vec, nblk int) float32 {
	if nblk > 0 && qtier == tierAVX512 && len(qv.q16) >= nblk*qBlk {
		return qdot8gemv512q16(&qw[0], &qv.q16[0], &dw[0], &qv.d[0], nblk)
	}
	return qdot8(qw, dw, qv, nblk)
}

func qMatRowsRangeFast(qt *q8Tensor, qv q8Vec, y []float32, lo, hi int) bool {
	if qtier != tierAVX512 || qt.nblk <= 0 {
		return false
	}
	in, nblk := qt.in, qt.nblk
	if len(qv.q16) < nblk*qBlk {
		return false
	}
	for o := lo; o < hi; o++ {
		y[o] = qdot8gemv512q16(&qt.q[o*in], &qv.q16[0], &qt.d[o*nblk], &qv.d[0], nblk)
	}
	return true
}

// qgemm8tile512 computes a 5(row)×4(token) Q8_0 output tile with AVX-512BW, deferred
// reduction (see quant_gemm.go). Weight row i, block b at qw + i*in + b*qBlk (scale dw +
// i*nblk + b); token j, block b at qx + j*in + b*qBlk (scale dx + j*nblk + b); result
// (row i, token j) written to dst + j*outStride + i (floats). Bit-identical to
// qgemm8cell(...,16) — TestQGemm8AsmMatchesScalar pins it.
//
//go:noescape
func qgemm8tile512(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)

//go:noescape
func qgemm8tile512x1(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)

// qgemm8tile256 is the AVX2 sibling of qgemm8tile512: a 3(row)×2(token) register-blocked Q8
// tile for hosts without AVX-512. Bit-identical to qgemm8cell(...,8) — TestQGemm8AVX2MatchesScalar
// pins it. Same operand layout as qgemm8tile512 (see quant_amd64.s).
//
//go:noescape
func qgemm8tile256(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)

const qgemmMR = 5    // full AVX-512 tile rows; row remainders use qgemm8tile512x1 plus scalar token tail
const qgemmMR256 = 3 // full AVX2 tile rows
const qgemmNR256 = 2 // full AVX2 tile tokens

// qGemm8 is the batched Q8_0 prefill GEMM dispatcher: register-blocked tile kernel on
// AVX-512, with a scalar reference for the row/token remainder (and as the AVX2/scalar
// fallback until an AVX2 tile lands). Output row-major [P, out], identical layout to the
// old qMatMulBatch.
func qGemm8(qt *q8Tensor, qp *q8Panel) []float32 {
	Y := make([]float32, qp.P*qt.out)
	qGemm8Into(qt, qp, Y)
	return Y
}

// qGemm8Into runs the Q8 GEMM into a caller-provided Y (len >= P*out), so the hot multi-user
// decode step can reuse one output buffer per projection across steps instead of allocating
// P*out floats every call (the per-step allocation/GC the batching doc names as a measured
// gap-to-roofline component). Bit-identical to qGemm8 — only Y's backing memory changes.
func qGemm8Into(qt *q8Tensor, qp *q8Panel, Y []float32) {
	if qgemmMode == qgemmModeLegacy {
		qGemm8legacyInto(qt, qp, Y)
		return
	}
	if qtier == tierAVX2 {
		// AVX2-only host (the EPYC-7742 CPU-server floor): the register-blocked AVX2 tile, with
		// lanes=8 scalar-ref remainders so the whole path is bit-identical to qGemm8scalar(...,8).
		qGemm8avx2Into(qt, qp, Y)
		return
	}
	if qtier != tierAVX512 {
		// Pure scalar (no AVX2): the portable reference (correct, slower). lanes=16 keeps the
		// scalar fallback's numerics independent of which non-vector box runs it.
		qGemm8scalarInto(qt, qp, 16, Y)
		return
	}

	out, in, nblk, P := qt.out, qt.in, qt.nblk, qp.P
	Pmain := P &^ (4 - 1) // tokens handled by the NR=4 tile; remainder done by the cell ref
	nTiles := out / qgemmMR

	tile := func(lo, hi int) {
		for tt := lo; tt < hi; tt++ {
			o := tt * qgemmMR
			for t := 0; t < Pmain; t += 4 {
				qgemm8tile512(
					&qt.q[o*in], &qp.q[t*in],
					&qt.d[o*nblk], &qp.d[t*nblk],
					in, nblk, out, &Y[t*out+o],
				)
			}
		}
	}
	if out*in*P < parThreshold {
		tile(0, nTiles)
	} else {
		parFor(nTiles, numWorkers, tile)
	}

	// Remainder rows (out % MR): every token, via the matching scalar reference.
	for o := nTiles * qgemmMR; o < out; o++ {
		qw := qt.q[o*in : o*in+in]
		dw := qt.d[o*nblk : o*nblk+nblk]
		for t := 0; t < Pmain; t += 4 {
			qgemm8tile512x1(&qw[0], &qp.q[t*in], &dw[0], &qp.d[t*nblk], in, nblk, out, &Y[t*out+o])
		}
		for t := Pmain; t < P; t++ {
			Y[t*out+o] = qgemm8cell(qw, dw, qp.q[t*in:t*in+in], qp.d[t*nblk:t*nblk+nblk], nblk, 16)
		}
	}
	// Remainder tokens (P % 4): the tiled rows still need these columns.
	for t := Pmain; t < P; t++ {
		qx := qp.q[t*in : t*in+in]
		dx := qp.d[t*nblk : t*nblk+nblk]
		for o := 0; o < nTiles*qgemmMR; o++ {
			Y[t*out+o] = qgemm8cell(qt.q[o*in:o*in+in], qt.d[o*nblk:o*nblk+nblk], qx, dx, nblk, 16)
		}
	}
}

// qGemm8avx2Into runs the Q8 prefill GEMM with the AVX2 register-blocked tile (qgemm8tile256,
// MR=3×NR=2) over the tile-aligned bulk and the lanes=8 scalar reference (qgemm8cell) for the
// row/token remainders. The whole path is Float32bits-identical to qGemm8scalar(qt, qp, 8) —
// the tile is pinned bit-for-bit to qgemm8cell(...,8) and the remainders call it directly — so
// the AVX2 host's prefill GEMM stops running the scalar reference (the #1127 CPU-server floor) while
// staying on the same authoritative argmax-vs-f32 Q8 gate. Callable directly (not gated on the
// resolved qtier) so it can be exercised on any AVX2-capable host; wired into qGemm8Into for
// tierAVX2. Output row-major [P, out].
func qGemm8avx2Into(qt *q8Tensor, qp *q8Panel, Y []float32) {
	out, in, nblk, P := qt.out, qt.in, qt.nblk, qp.P
	Pmain := P &^ (qgemmNR256 - 1) // tokens handled by the NR=2 tile; remainder via the cell ref
	nTiles := out / qgemmMR256

	tile := func(lo, hi int) {
		for tt := lo; tt < hi; tt++ {
			o := tt * qgemmMR256
			for t := 0; t < Pmain; t += qgemmNR256 {
				qgemm8tile256(
					&qt.q[o*in], &qp.q[t*in],
					&qt.d[o*nblk], &qp.d[t*nblk],
					in, nblk, out, &Y[t*out+o],
				)
			}
		}
	}
	if out*in*P < parThreshold {
		tile(0, nTiles)
	} else {
		parFor(nTiles, numWorkers, tile)
	}

	// Remainder rows (out % MR): every token, via the matching lanes=8 scalar reference.
	for o := nTiles * qgemmMR256; o < out; o++ {
		qw := qt.q[o*in : o*in+in]
		dw := qt.d[o*nblk : o*nblk+nblk]
		for t := 0; t < P; t++ {
			Y[t*out+o] = qgemm8cell(qw, dw, qp.q[t*in:t*in+in], qp.d[t*nblk:t*nblk+nblk], nblk, 8)
		}
	}
	// Remainder tokens (P % NR): the tiled rows still need these columns.
	for t := Pmain; t < P; t++ {
		qx := qp.q[t*in : t*in+in]
		dx := qp.d[t*nblk : t*nblk+nblk]
		for o := 0; o < nTiles*qgemmMR256; o++ {
			Y[t*out+o] = qgemm8cell(qt.q[o*in:o*in+in], qt.d[o*nblk:o*nblk+nblk], qx, dx, nblk, 8)
		}
	}
}

// qGemm8IntoMany runs several GEMMs that share the same quantized activation panel under
// one parallel launch. It preserves the exact qGemm8Into arithmetic and output layout for
// each target; the win is avoiding repeated goroutine/barrier setup for q/k/v and gate/up
// groups in batched decode.
func qGemm8IntoMany(qp *q8Panel, targets ...qgemm8Target) {
	if len(targets) == 0 {
		return
	}
	if !qgemmGroup || qp.P > qgemmGroupMaxP || qgemmMode == qgemmModeLegacy || qtier != tierAVX512 || len(targets) > 4 {
		for _, tg := range targets {
			qGemm8Into(tg.qt, qp, tg.Y)
		}
		return
	}

	type plan struct {
		tg         qgemm8Target
		start, end int
	}
	var plans [4]plan
	totalTiles := 0
	for i, tg := range targets {
		nTiles := tg.qt.out / qgemmMR
		plans[i] = plan{tg: tg, start: totalTiles, end: totalTiles + nTiles}
		totalTiles += nTiles
	}

	in, nblk, P := qp.in, qp.nblk, qp.P
	Pmain := P &^ (4 - 1)
	tile := func(lo, hi int) {
		for i := 0; i < len(targets); i++ {
			pl := plans[i]
			a, b := lo, hi
			if a < pl.start {
				a = pl.start
			}
			if b > pl.end {
				b = pl.end
			}
			if a >= b {
				continue
			}
			qt, Y := pl.tg.qt, pl.tg.Y
			for tt := a; tt < b; tt++ {
				o := (tt - pl.start) * qgemmMR
				for t := 0; t < Pmain; t += 4 {
					qgemm8tile512(
						&qt.q[o*in], &qp.q[t*in],
						&qt.d[o*nblk], &qp.d[t*nblk],
						in, nblk, qt.out, &Y[t*qt.out+o],
					)
				}
			}
		}
	}
	if totalTiles*qgemmMR*in*P < parThreshold {
		tile(0, totalTiles)
	} else {
		parFor(totalTiles, numWorkers, tile)
	}

	for _, tg := range targets {
		qt, Y := tg.qt, tg.Y
		out := qt.out
		nTiles := out / qgemmMR
		for o := nTiles * qgemmMR; o < out; o++ {
			qw := qt.q[o*in : o*in+in]
			dw := qt.d[o*nblk : o*nblk+nblk]
			for t := 0; t < Pmain; t += 4 {
				qgemm8tile512x1(&qw[0], &qp.q[t*in], &dw[0], &qp.d[t*nblk], in, nblk, out, &Y[t*out+o])
			}
			for t := Pmain; t < P; t++ {
				Y[t*out+o] = qgemm8cell(qw, dw, qp.q[t*in:t*in+in], qp.d[t*nblk:t*nblk+nblk], nblk, 16)
			}
		}
		for t := Pmain; t < P; t++ {
			qx := qp.q[t*in : t*in+in]
			dx := qp.d[t*nblk : t*nblk+nblk]
			for o := 0; o < nTiles*qgemmMR; o++ {
				Y[t*out+o] = qgemm8cell(qt.q[o*in:o*in+in], qt.d[o*nblk:o*nblk+nblk], qx, dx, nblk, 16)
			}
		}
	}
}
