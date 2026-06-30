//go:build !amd64 && !arm64

package model

// quant_noasm.go — the portable dispatch for builds with no SIMD Q8 kernel (anything that is
// neither amd64 nor arm64): qdot8 is the scalar reference. amd64 has quant_amd64.{go,s}
// (AVX2/AVX-512) and arm64 has quant_arm64.{go,s} (NEON SDOT); every other arch keeps the
// correct-but-slower path here, which is fine since the quantized lane is opt-in.

func qdot8(qw []int8, dw []float32, qv q8Vec, nblk int) float32 {
	return qdot8scalar(qw, dw, qv, nblk)
}

func q8PreextendVec() bool { return false }

func extendQ8ToQ16(q []int8, q16 []int16, nblk int) {
	extendQ8ToQ16Scalar(q, q16, nblk)
}

func qdot8GEMV(qw []int8, dw []float32, qv q8Vec, nblk int) float32 {
	return qdot8(qw, dw, qv, nblk)
}

func qMatRowsRangeFast(qt *q8Tensor, qv q8Vec, y []float32, lo, hi int) bool {
	return false
}

// qGemm8 on non-amd64 is the portable batched GEMM (no tile asm). FAK_QGEMM=legacy still
// routes through the old per-element sweep for parity with the amd64 A/B.
func qGemm8(qt *q8Tensor, qp *q8Panel) []float32 {
	Y := make([]float32, qp.P*qt.out)
	qGemm8Into(qt, qp, Y)
	return Y
}

// qGemm8Into is the buffer-reuse form (see the amd64 doc comment).
func qGemm8Into(qt *q8Tensor, qp *q8Panel, Y []float32) {
	if qgemmMode == qgemmModeLegacy {
		qGemm8legacyInto(qt, qp, Y)
		return
	}
	if qgemmMode == qgemmModeAccel && qGemm8AccelInto(qt, qp, Y) {
		return
	}
	qGemm8scalarInto(qt, qp, 16, Y)
}

func qGemm8IntoMany(qp *q8Panel, targets ...qgemm8Target) {
	for _, tg := range targets {
		qGemm8Into(tg.qt, qp, tg.Y)
	}
}
