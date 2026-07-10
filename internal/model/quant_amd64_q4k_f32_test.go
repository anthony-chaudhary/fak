//go:build amd64

package model

import (
	"math"
	"math/rand"
	"testing"
)

// TestQ4KRowDotF32AVX2MatchesScalar pins the AVX2 f32 decode kernel (q4kMatRowsRangeArch) BIT-FOR-BIT
// against the scalar reference q4kMatRowsRange. Because the kernel adds no activation quantization and
// reproduces the scalar's stride-4 4-accumulator reduction with separate mul/sub/add (no FMA), the two
// must agree with max|Δ|==0 — the same bit-identity discipline q4kReduceRowAsmAVX2 is held to. Skips on
// a box (or FAK_QKERNEL pin) without AVX2, where the arch path declines and the scalar runs anyway.
func TestQ4KRowDotF32AVX2MatchesScalar(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skipf("AVX2 f32 Q4_K kernel inactive (qtier=%d < tierAVX2); scalar path is current", qtier)
	}
	rng := rand.New(rand.NewSource(0x9950))
	shapes := []struct{ out, in int }{
		{1, 256},   // single row, single super-block
		{7, 256},   // odd row count, one super-block
		{5, 512},   // two super-blocks
		{16, 768},  // three super-blocks
		{33, 1280}, // five super-blocks, non-multiple-of-8 rows
		{2, 5120},  // Qwen3.6 hidden width (20 super-blocks)
	}
	for _, sh := range shapes {
		nblk := sh.in / qkK
		raw := make([]byte, sh.out*nblk*q4kBlockBytes)
		blk := make([]byte, q4kBlockBytes)
		for i := 0; i < sh.out*nblk; i++ {
			randQ4KBlock(rng, blk)
			copy(raw[i*q4kBlockBytes:(i+1)*q4kBlockBytes], blk)
		}
		qt := quantizeQ4KFromRaw(raw, sh.out, sh.in)

		x := make([]float32, sh.in)
		for i := range x {
			// Mix of magnitudes/signs so a lane- or affine-min bug shows up, not just small noise.
			x[i] = float32(rng.NormFloat64()) * float32(1+(i%9))
		}

		yScalar := make([]float32, sh.out)
		q4kMatRowsRange(qt, x, yScalar, 0, sh.out)

		yArch := make([]float32, sh.out)
		if !q4kMatRowsRangeArch(qt, x, yArch, 0, sh.out) {
			t.Fatalf("shape %dx%d: q4kMatRowsRangeArch declined at qtier=%d (>= tierAVX2)", sh.out, sh.in, qtier)
		}

		for o := 0; o < sh.out; o++ {
			if yArch[o] != yScalar[o] {
				t.Fatalf("shape %dx%d row %d: AVX2 %.9g != scalar %.9g (Δbits=%#x vs %#x) — not bit-identical",
					sh.out, sh.in, o, yArch[o], yScalar[o],
					math.Float32bits(yArch[o]), math.Float32bits(yScalar[o]))
			}
		}
	}
}

// BenchmarkQ4KF32GEMVAVX2 is the AVX2 f32 kernel twin of BenchmarkQ4KF32GEMV (the scalar reference,
// same 2048x6144 shape): A/B them under FAK_QKERNEL=avx2 to read the exact-decode speedup on an
// AVX2-only host (da33). Single-worker so it measures the kernel, not the scheduler.
func BenchmarkQ4KF32GEMVAVX2(b *testing.B) {
	if qtier < tierAVX2 {
		b.Skipf("AVX2 f32 Q4_K kernel inactive (qtier=%d)", qtier)
	}
	qt, x := benchQ4KFixture(b, 2048, 6144)
	y := make([]float32, qt.out)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q4kMatRowsRangeArch(qt, x, y, 0, qt.out)
	}
}
