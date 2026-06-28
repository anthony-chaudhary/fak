package model

import (
	"math"
	"math/rand"
	"testing"
)

// TestQ2BlockRoundTrip pins quantizeQ2Block/dequantQ2Block as exact inverses on the codes:
// quantize then dequant must reproduce each weight to within one 2-bit quantum (d/2 in the
// interior), and a zero block must round-trip to exactly zero. This is the safety net that
// catches any packing/sign error in the int2 format without needing a model run.
func TestQ2BlockRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	src := make([]float32, qBlk2)
	got := make([]float32, qBlk2)
	packed := make([]byte, qBlk2/4)
	for trial := 0; trial < 1000; trial++ {
		scale := float32(rng.NormFloat64())
		if scale < 0 {
			scale = -scale
		}
		if scale < 1e-3 {
			scale = 1e-3
		}
		zero := trial%50 == 0
		for i := range src {
			if zero {
				src[i] = 0
			} else {
				src[i] = float32(rng.NormFloat64()) * scale
			}
		}
		d := quantizeQ2Block(packed, src)
		dequantQ2Block(got, d, packed)
		// Max reconstruction error is bounded by half a quantum (round-to-nearest); the
		// ±amax peaks are exact. Allow a small fp margin.
		bound := d*0.5 + 1e-5
		for i := range src {
			if math.Abs(float64(src[i]-got[i])) > float64(bound) {
				t.Fatalf("trial %d idx %d: src=%v got=%v d=%v bound=%v", trial, i, src[i], got[i], d, bound)
			}
		}
		if zero && d != 0 {
			t.Fatalf("trial %d: zero block quantized to non-zero scale %v", trial, d)
		}
	}
}

// TestQ2MatRowsMatchesDequant pins the int2 decode GEMV to a reference computed by fully
// dequantizing the same tensor and running the dense f32 matRows. This proves the kernel's
// packing/indexing/dot is correct independent of quantization quality (a sign or stride bug
// would blow far past the float-reassociation tolerance), over a realistic weight shape.
func TestQ2MatRowsMatchesDequant(t *testing.T) {
	const out, in = 64, 512 // in is a multiple of qBlk2
	rng := rand.New(rand.NewSource(7))
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * 0.05 // LLM-weight-like magnitude
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	qt := quantizeQ2(w, out, in)
	yQ2 := q2MatRows(qt, x)
	yRef := matRows(dequantQ2Tensor(qt), x, out, in) // same int2 values, dense dot

	var sumSq, maxAbs float64
	for o := 0; o < out; o++ {
		sumSq += float64(yRef[o]) * float64(yRef[o])
		if e := math.Abs(float64(yQ2[o] - yRef[o])); e > maxAbs {
			maxAbs = e
		}
	}
	rms := math.Sqrt(sumSq / float64(out))
	if rms < 1e-9 {
		t.Fatalf("reference RMS ~0; bad test data")
	}
	// Only float reassociation (8-accumulator block dot vs dense matRows) separates the two;
	// a real packing/sign/stride bug would be orders of magnitude larger than this.
	if rel := maxAbs / rms; rel > 1e-4 {
		t.Fatalf("int2 GEMV max-abs/RMS %.6f exceeds 1e-4 (out=%d in=%d rms=%v)", rel, out, in, rms)
	}
}

// TestQ2MemoryReduction witnesses the memory-reduction acceptance: the int2 resident
// footprint is at least 2× smaller than int8 (it is in fact ~3×), and the quantized code
// payload is exactly 4× smaller than int8 and 2× smaller than int4. Built from one matrix
// so the three footprints are directly comparable.
func TestQ2MemoryReduction(t *testing.T) {
	const out, in = 32, 256
	rng := rand.New(rand.NewSource(11))
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * 0.05
	}
	q2 := quantizeQ2(w, out, in)
	q4 := quantizeQ4(w, out, in)
	q8 := quantizeQ8(w, out, in)

	// Quantized code payloads (the part that scales with the bit width).
	q2Codes, q4Codes, q8Codes := len(q2.q), len(q4.q), len(q8.q)
	if q2Codes*4 != q8Codes {
		t.Fatalf("int2 code bytes %d not exactly 4x smaller than int8 %d", q2Codes, q8Codes)
	}
	if q2Codes*2 != q4Codes {
		t.Fatalf("int2 code bytes %d not exactly 2x smaller than int4 %d", q2Codes, q4Codes)
	}

	// Total resident footprint (codes + f32 scales): require ≥2× reduction vs int8.
	q2Bytes := q2.footprintBytes()
	q8Bytes := q8Codes + 4*len(q8.d)
	if q2Bytes*2 > q8Bytes {
		t.Fatalf("int2 footprint %d B not ≥2x smaller than int8 %d B", q2Bytes, q8Bytes)
	}
	// Honest expected ratio: (4+8)/(4+32) = 1/3 of int8 per block.
	if got, want := float64(q8Bytes)/float64(q2Bytes), 3.0; math.Abs(got-want) > 0.05 {
		t.Fatalf("int2-vs-int8 footprint ratio %.3f not ~%.1f", got, want)
	}
}
