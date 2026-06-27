package model

import (
	"encoding/binary"
	"testing"
)

// BenchmarkQ5K{F32,Int8}GEMV A/B the two Q5_K decode GEMV paths on a GLM-5.2-shaped expert row set
// (out=2048 expert-intermediate, in=6144 hidden). The f32 path dequants 256 f32 per super-block per
// row; the int8 path quantizes the activation once and does the compact integer reduction. Run both
// to read the per-call ns and the int8 speedup. Single-worker (parThreshold/parFor folded out) so the
// bench measures the kernel, not the scheduler.
func benchKQuantFixture(b *testing.B, out, in int) (*kQuantTensor, []float32) {
	b.Helper()
	nblk := in / qkK
	bb := kindQ5K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, 0xfeedface12345678)
	for o := 0; o < out; o++ {
		for bk := 0; bk < nblk; bk++ {
			blk := raw[(o*nblk+bk)*bb:]
			binary.LittleEndian.PutUint16(blk[0:], f16One) // d
			binary.LittleEndian.PutUint16(blk[2:], 0)      // min
		}
	}
	qt := quantizeKQuantFromRaw(raw, out, in, kindQ5K)
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*13)%29) - 14
	}
	return qt, x
}

func BenchmarkQ5KF32GEMV(b *testing.B) {
	qt, x := benchKQuantFixture(b, 2048, 6144)
	y := make([]float32, qt.out)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kQuantMatRowsRange(qt, x, y, 0, qt.out)
	}
}

func BenchmarkQ5KInt8GEMV(b *testing.B) {
	qt, x := benchKQuantFixture(b, 2048, 6144)
	y := make([]float32, qt.out)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qv := quantizeVecQ8(x)
		q5kMatRowsRangeInt8(qt, qv, y, 0, qt.out)
	}
}
