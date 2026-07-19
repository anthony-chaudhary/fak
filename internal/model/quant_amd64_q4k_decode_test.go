//go:build amd64

package model

import (
	"math"
	"math/rand"
	"testing"
)

// TestQ4KDecodeAffineMatchesScalar holds the affine-split FMA decode kernel (q4kMatRowsRangeAffine)
// to the scalar reference q4kMatRowsRange. Unlike the exact AVX2 kernel this path reassociates the
// dot (d-scale and min-subtract hoisted out of the per-weight loop, FMA reduction), so it is NOT
// bit-identical — the contract is a decode-grade quality gate: high cosine and small relative error,
// which is what preserves the argmax. A lane-swap, wrong sub-block/x alignment, or a dropped min term
// would blow past these bounds. Skips without AVX2 (the kernel declines and scalar runs anyway).
func TestQ4KDecodeAffineMatchesScalar(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skipf("AVX2 Q4_K affine kernel inactive (qtier=%d < tierAVX2)", qtier)
	}
	rng := rand.New(rand.NewSource(0x4f1e))
	shapes := []struct{ out, in int }{
		{1, 256},
		{7, 256},
		{5, 512},
		{16, 768},
		{33, 1280},
		{2, 5120},
		{64, 4096},
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
			x[i] = float32(rng.NormFloat64()) * float32(1+(i%9))
		}

		yScalar := make([]float32, sh.out)
		q4kMatRowsRange(qt, x, yScalar, 0, sh.out)

		yAff := make([]float32, sh.out)
		xsum := q4kDecodeXSum(x, nblk)
		if !q4kMatRowsRangeAffine(qt, x, yAff, 0, sh.out, xsum) {
			t.Fatalf("shape %dx%d: q4kMatRowsRangeAffine declined at qtier=%d", sh.out, sh.in, qtier)
		}

		var dot, na, nb, maxRel float64
		for o := 0; o < sh.out; o++ {
			a, b := float64(yAff[o]), float64(yScalar[o])
			dot += a * b
			na += a * a
			nb += b * b
			rel := math.Abs(a-b) / (math.Abs(b) + 1e-3)
			if rel > maxRel {
				maxRel = rel
			}
		}
		cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		if cos < 0.9999995 {
			t.Fatalf("shape %dx%d: cosine %.9f below 0.9999995 — affine kernel diverges from scalar", sh.out, sh.in, cos)
		}
		if maxRel > 5e-4 {
			t.Fatalf("shape %dx%d: max relative error %.3g exceeds 5e-4", sh.out, sh.in, maxRel)
		}
	}
}

// BenchmarkQ4KF32GEMVAffine is the affine-split kernel twin of BenchmarkQ4KF32GEMVAVX2 (same
// 2048x6144 shape, single worker): A/B them on an AVX2-only host (da33) to read the affine decode
// speedup over the exact AVX2 kernel. xsum is precomputed outside the timed loop, matching the
// dispatch (computed once per matmul, not per row).
func BenchmarkQ4KF32GEMVAffine(b *testing.B) {
	if qtier < tierAVX2 {
		b.Skipf("AVX2 Q4_K affine kernel inactive (qtier=%d)", qtier)
	}
	qt, x := benchQ4KFixture(b, 2048, 6144)
	y := make([]float32, qt.out)
	xsum := q4kDecodeXSum(x, qt.nblk)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q4kMatRowsRangeAffine(qt, x, y, 0, qt.out, xsum)
	}
}
