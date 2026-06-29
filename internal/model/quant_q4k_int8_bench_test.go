package model

import (
	"encoding/binary"
	"testing"
)

// BenchmarkQ4KInt8GEMV measures the Q4_K int8 decode GEMV on a GLM-5.2-shaped expert row set
// (out=2048 expert-intermediate, in=6144 hidden) — the path that dispatches through q4kReduceRow
// (AVX2 on amd64, NEON on arm64, scalar elsewhere). A/B it against BenchmarkQ4KInt8GEMVScalar (which
// pins FAK_QKERNEL=scalar via the dispatcher tier) to read the SIMD speedup over the portable
// reducer. Single-worker so the bench measures the kernel, not the scheduler.
func benchQ4KFixture(b *testing.B, out, in int) (*q4kTensor, []float32) {
	b.Helper()
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	lcgBytes(raw, 0xfeedface12345678)
	for o := 0; o < out; o++ {
		for bk := 0; bk < nblk; bk++ {
			blk := raw[(o*nblk+bk)*q4kBlockBytes:]
			binary.LittleEndian.PutUint16(blk[0:], f16One) // d
			binary.LittleEndian.PutUint16(blk[2:], 0)      // min
		}
	}
	qt := quantizeQ4KFromRaw(raw, out, in)
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*13)%29) - 14
	}
	return qt, x
}

func BenchmarkQ4KInt8GEMV(b *testing.B) {
	qt, x := benchQ4KFixture(b, 2048, 6144)
	y := make([]float32, qt.out)
	qv := quantizeVecQ8(x)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q4kMatRowsRangeInt8(qt, qv, y, 0, qt.out)
	}
}

func BenchmarkQ4KF32GEMV(b *testing.B) {
	qt, x := benchQ4KFixture(b, 2048, 6144)
	y := make([]float32, qt.out)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q4kMatRowsRange(qt, x, y, 0, qt.out)
	}
}

func benchQ4KGemmFixture(b *testing.B, out, in, P int) (*q4kTensor, []q8Vec, *q8Panel) {
	b.Helper()
	qt, x := benchQ4KFixture(b, out, in)
	X := make([]float32, P*in)
	for t := 0; t < P; t++ {
		copy(X[t*in:(t+1)*in], x)
	}
	qvs := make([]q8Vec, P)
	for t := 0; t < P; t++ {
		qvs[t] = quantizeVecQ8(X[t*in : (t+1)*in])
	}
	qp := quantizeBatchPanel(X, P, in)
	return qt, qvs, qp
}

func BenchmarkQ4KGemmInt8LegacyReducePerToken(b *testing.B) {
	const out, in, P = 17408, 5120, 22 // Qwen3.6 q4_k_m MLP gate/up shape.
	qt, qvs, _ := benchQ4KGemmFixture(b, out, in, P)
	Y := make([]float32, P*out)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q4kGemmRangeInt8(qt, qvs, P, Y, 0, out)
	}
}

func BenchmarkQ4KGemmInt8ExtractOnce(b *testing.B) {
	const out, in, P = 17408, 5120, 22 // Qwen3.6 q4_k_m MLP gate/up shape.
	qt, _, qp := benchQ4KGemmFixture(b, out, in, P)
	Y := make([]float32, P*out)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q4kGemmExtractOnceInt8Into(qt, qp, Y)
	}
}
