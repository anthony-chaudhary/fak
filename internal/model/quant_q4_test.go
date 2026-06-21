package model

import (
	"math"
	"math/rand"
	"testing"
)

// TestQ4BlockRoundTrip pins quantizeQ4Block/dequantQ4Block as exact inverses on the codes:
// quantize then dequant must reproduce each weight to within one 4-bit quantum (d), and a
// zero block must round-trip to exactly zero. This is the safety net that catches any
// packing/sign error in the int4 format without needing a 27B run.
func TestQ4BlockRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	src := make([]float32, qBlk4)
	got := make([]float32, qBlk4)
	nib := make([]byte, qBlk4/2)
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
		d := quantizeQ4Block(nib, src)
		dequantQ4Block(got, d, nib)
		// Max reconstruction error is bounded by the quantum d/2 (round-to-nearest) plus
		// the clamp at the ±7 range; allow a small margin.
		bound := d*0.5 + 1e-6
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

// TestQ4MatRowsMatchesF32 checks the int4 GEMV against the f32 reference within the Q4
// tolerance, over a realistic weight shape (hidden-wide reduction dim) and a random row.
func TestQ4MatRowsMatchesF32(t *testing.T) {
	const out, in = 64, 512 // in is a multiple of qBlk4
	rng := rand.New(rand.NewSource(7))
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * 0.05 // LLM-weight-like magnitude
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	qt := quantizeQ4(w, out, in)
	yQ4 := q4MatRows(qt, x)
	yRef := matRows(w, x, out, in)

	// Absolute error normalized by the RMS of the f32 reference — a well-conditioned Q4
	// accuracy metric (per-row relative is ill-conditioned: any row whose reference dot is
	// ~0 makes the relative error explode even though the int4 dot is fine). Q4_0 adds ~one
	// quantum per weight, so the dot error grows like sqrt(in)*quantum; this bound proves
	// correctness (no packing/sign bug) without pinning a quality number.
	var sumSq, maxAbs float64
	for o := 0; o < out; o++ {
		sumSq += float64(yRef[o]) * float64(yRef[o])
		if e := math.Abs(float64(yQ4[o] - yRef[o])); e > maxAbs {
			maxAbs = e
		}
	}
	rms := math.Sqrt(sumSq / float64(out))
	if rms < 1e-9 {
		t.Fatalf("reference RMS ~0; bad test data")
	}
	rel := maxAbs / rms
	if rel > 0.25 {
		t.Fatalf("int4 GEMV max-abs/RMS %.4f exceeds 0.25 (out=%d in=%d rms=%v)", rel, out, in, rms)
	}
}

// TestQ4KernelMatchesQ4MatRows pins the matKernel wrapper to the direct GEMV.
func TestQ4KernelMatchesQ4MatRows(t *testing.T) {
	const out, in = 16, 64
	w := make([]float32, out*in)
	rng := rand.New(rand.NewSource(3))
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * 0.1
	}
	m := &Model{Cfg: Config{HiddenSize: in}, manifest: map[string]tensorMeta{}, q4w: map[string]*q4Tensor{}}
	m.q4w["w"] = quantizeQ4(w, out, in)
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	k := q4Kernel{m}
	yK := k.mul("w", k.prep(x), out, in)
	yD := q4MatRows(m.q4("w"), x)
	for i := range yK {
		if yK[i] != yD[i] {
			t.Fatalf("kernel vs direct mismatch at %d: %v != %v", i, yK[i], yD[i])
		}
	}
}
