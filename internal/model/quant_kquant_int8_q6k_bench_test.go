package model

import (
	"encoding/binary"
	"testing"
)

// BenchmarkQ6K{F32,Int8}GEMV A/B the two Q6_K decode GEMV paths on a GLM-5.2-shaped down-experts row
// set (out=6144 hidden, in=2048 expert-intermediate — the ffn_down_exps direction). The f32 path
// dequants 256 f32 per super-block per row; the int8 path quantizes the activation once and runs the
// compact integer reduction (q6kReduceRow → the AVX2/VNNI kernel on amd64, scalar elsewhere).
//
// The reducer tier is resolved once at init, so the scalar → AVX2 → VNNI ladder the #1124 acceptance
// asks for is three runs of BenchmarkQ6KInt8GEMV under FAK_QKERNEL=scalar / avx2 / (unset, picks VNNI
// when the box has it). Single-worker (parThreshold folds out at this size) so it measures the kernel.
func benchQ6KFixture(b *testing.B, out, in int) (*kQuantTensor, []float32) {
	b.Helper()
	nblk := in / qkK
	bb := kindQ6K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, 0xfeedface12345678)
	for o := 0; o < out; o++ {
		for bk := 0; bk < nblk; bk++ {
			blk := raw[(o*nblk+bk)*bb:]
			binary.LittleEndian.PutUint16(blk[q6kBlockBytes-2:], f16One) // d=1.0
		}
	}
	qt := quantizeKQuantFromRaw(raw, out, in, kindQ6K)
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*13)%29) - 14
	}
	return qt, x
}

func BenchmarkQ6KF32GEMV(b *testing.B) {
	qt, x := benchQ6KFixture(b, 6144, 2048)
	y := make([]float32, qt.out)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kQuantMatRowsRange(qt, x, y, 0, qt.out)
	}
}

func BenchmarkQ6KInt8GEMV(b *testing.B) {
	qt, x := benchQ6KFixture(b, 6144, 2048)
	y := make([]float32, qt.out)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qv := quantizeVecQ8(x)
		q6kMatRowsRangeInt8(qt, qv, y, 0, qt.out)
	}
}
