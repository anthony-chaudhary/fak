package model

import (
	"math"
	"math/rand"
	"testing"
)

// TestQ4KInt8DotMatchesF32 checks the resident-Q4_K int8 SDOT path stays close to the f32 scalar
// GEMV (the path TestQ4KMatRowsMatchesF32 pins) within activation-quantization tolerance. The int8
// path quantizes the activation to Q8_0 (per 32-block) and pairs it with the raw Q4_K nibbles, so
// it is APPROXIMATE vs f32 — the gate is the q4_k_m greedy + first-token agreement, NOT bit-exact.
// This test is a numeric safety net: a gross mismatch would mean the affine-min handling, the
// sub-block/scale indexing, or the activation-block alignment is wrong (those blow the error up by
// orders of magnitude, not percent). Skips where the int8 path is inactive (non-arm64 / scalar pin).
func TestQ4KInt8DotMatchesF32(t *testing.T) {
	if !q4kSDOTEnabled() {
		t.Skip("q4_k int8 SDOT path inactive on this arch (f32 scalar path is current)")
	}
	const out, in = 48, 768 // nblk = 3
	rng := rand.New(rand.NewSource(23))
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	blk := make([]byte, q4kBlockBytes)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			randQ4KBlock(rng, blk)
			off := (o*nblk + b) * q4kBlockBytes
			copy(raw[off:off+q4kBlockBytes], blk)
		}
	}
	qt := quantizeQ4KFromRaw(raw, out, in)

	x := make([]float32, in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	yInt8 := q4kMatRows(qt, x) // SDOT int8 path (quantizeVecQ8 once, reuse)
	yF32 := make([]float32, out)
	q4kMatRowsRange(qt, x, yF32, 0, out) // f32 scalar reference path

	var sumSq, maxRel float64
	for o := 0; o < out; o++ {
		sumSq += float64(yF32[o]) * float64(yF32[o])
	}
	rms := math.Sqrt(sumSq / float64(out))
	if rms < 1e-9 {
		t.Fatalf("f32 reference RMS ~0; bad test data")
	}
	for o := 0; o < out; o++ {
		if rel := math.Abs(float64(yInt8[o]-yF32[o])) / rms; rel > maxRel {
			maxRel = rel
		}
	}
	// Activation Q8_0 quant on gaussian inputs lands well under 1% per-dot; 5% is a generous
	// ceiling that still catches a real indexing/affine bug (which diverges by orders of magnitude).
	if maxRel > 0.05 {
		t.Fatalf("q4_k int8 vs f32 max-abs/RMS %.4f exceeds 0.05 (activation-quant tolerance)", maxRel)
	}
	t.Logf("q4_k int8 vs f32 max-abs/RMS = %.4e (in=%d, rms=%.4g)", maxRel, in, rms)
}

// BenchmarkQ4KMatRowsF32 / BenchmarkQ4KMatRowsInt8 measure the resident-Q4_K decode GEMV on a
// realistic per-layer shape, f32 scalar (the P1 path, compute-bound) vs the P2 int8 SDOT path.
// ns/op over [out] rows => the per-token weight-matmul cost; the speedup is the P2 win toward the
// q4 bandwidth ceiling. Reported as GiB/s of weight stream (in*0.5625 B/param per row).
func benchQ4KMatRows(b *testing.B, out, in int, int8 bool) {
	rng := rand.New(rand.NewSource(7))
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	blk := make([]byte, q4kBlockBytes)
	for o := 0; o < out; o++ {
		for k := 0; k < nblk; k++ {
			randQ4KBlock(rng, blk)
			off := (o*nblk + k) * q4kBlockBytes
			copy(raw[off:off+q4kBlockBytes], blk)
		}
	}
	qt := quantizeQ4KFromRaw(raw, out, in)
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	y := make([]float32, out)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if int8 {
			q4kMatRowsInto(qt, x, y) // dispatches to the SDOT int8 path when q4kSDOTEnabled
		} else {
			q4kMatRowsRange(qt, x, y, 0, out) // f32 scalar reference path
		}
	}
	weightGiB := float64(out*nblk*q4kBlockBytes) / (1 << 30)
	b.ReportMetric(weightGiB, "weightGiB")
}

// BenchmarkQ4KMatRowsF32 is the P1 baseline (scalar f32 dequant+dot, compute-bound).
func BenchmarkQ4KMatRowsF32(b *testing.B) {
	benchQ4KMatRows(b, 5120, 5120, false)
}

// BenchmarkQ4KMatRowsInt8 is the P2 SDOT path (only active on arm64 + FEAT_DotProd).
func BenchmarkQ4KMatRowsInt8(b *testing.B) {
	if !q4kSDOTEnabled() {
		b.Skip("q4_k int8 SDOT path inactive on this arch")
	}
	benchQ4KMatRows(b, 5120, 5120, true)
}
