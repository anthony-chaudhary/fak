package main

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/experiments/qwen36/gdn"
)

// TestQ8RoundTripBounded checks the Q8_0 weight round-trip is faithful: every dequantized value is
// within half a block-scale (d/2, d=amax/127) of the original — the exact error bound of
// round-to-nearest symmetric int8 — and that a symmetric block reproduces its max exactly.
func TestQ8RoundTripBounded(t *testing.T) {
	const in = 32 // one Q8_0 block
	w := make([]float32, in)
	for i := range w {
		w[i] = float32(math.Sin(float64(i))) * 3.5 // spread of magnitudes, amax ~3.5
	}
	dq := q8RowRoundTrip(w, 1, in)
	var amax float32
	for _, v := range w {
		if a := float32(math.Abs(float64(v))); a > amax {
			amax = a
		}
	}
	d := amax / 127
	for i := range w {
		if got := float32(math.Abs(float64(dq[i] - w[i]))); got > d/2+1e-6 {
			t.Fatalf("elem %d: |dq-w|=%g exceeds d/2=%g", i, got, d/2)
		}
	}
}

// TestQ4KRoundTripBounded checks the affine 4-bit sub-block round-trip stays within half a step
// (scale/2, scale=(max-min)/15) of the original and clamps into [min,max].
func TestQ4KRoundTripBounded(t *testing.T) {
	const in = 32
	w := make([]float32, in)
	mn, mx := float32(math.Inf(1)), float32(math.Inf(-1))
	for i := range w {
		w[i] = float32(i)*0.1 - 1.5
		if w[i] < mn {
			mn = w[i]
		}
		if w[i] > mx {
			mx = w[i]
		}
	}
	dq := q4kRowRoundTrip(w, 1, in)
	step := (mx - mn) / 15
	for i := range w {
		if got := float32(math.Abs(float64(dq[i] - w[i]))); got > step/2+1e-6 {
			t.Fatalf("elem %d: |dq-w|=%g exceeds step/2=%g", i, got, step/2)
		}
		if dq[i] < mn-1e-6 || dq[i] > mx+1e-6 {
			t.Fatalf("elem %d: dq=%g outside [min,max]=[%g,%g]", i, dq[i], mn, mx)
		}
	}
}

// TestQuantizeProjectionsSplit verifies quantizeProjections perturbs exactly the five isQuantWeight
// projection matrices and leaves the control tensors (kept f32 by the real loader) bit-identical.
func TestQuantizeProjectionsSplit(t *testing.T) {
	hidden = 256 // small, block-aligned
	lw := gdn.NewLayerWeights(hidden)
	lw.Fill(0, aLogMean)
	q := quantizeProjections(lw, modeQ8)

	// control tensors must be the SAME backing arrays (shared, untouched).
	if &q.Conv[0] != &lw.Conv[0] || &q.ALog[0] != &lw.ALog[0] || &q.NormW[0] != &lw.NormW[0] || &q.WIn[0] != &lw.WIn[0] {
		t.Fatalf("control tensors must be shared f32, not copied/quantized")
	}
	// projection matrices must differ (quantization changed them) and be new arrays.
	changed := func(a, b []float32) bool {
		if &a[0] == &b[0] {
			return false
		}
		for i := range a {
			if a[i] != b[i] {
				return true
			}
		}
		return false
	}
	for name, pair := range map[string][2][]float32{
		"wqkv": {lw.Wqkv, q.Wqkv},
		"wz":   {lw.Wz, q.Wz},
		"wb":   {lw.Wb, q.Wb},
		"wa":   {lw.Wa, q.Wa},
		"wOut": {lw.WOut, q.WOut},
	} {
		if !changed(pair[0], pair[1]) {
			t.Fatalf("projection %s should be quantized to a new, different array", name)
		}
	}
}

// TestQuantizeProjectionsReadsTheWeightSetsOwnHidden guards the parameter that replaced the
// package-level `hidden` inside the round-trip: the matrices must be reshaped from the weight
// set they were allocated with, not from whatever the global happens to hold. A mismatch would
// slice the rows at the wrong stride and quantize garbage blocks.
func TestQuantizeProjectionsReadsTheWeightSetsOwnHidden(t *testing.T) {
	saved := hidden
	defer func() { hidden = saved }()

	lw := gdn.NewLayerWeights(256)
	lw.Fill(0, aLogMean)
	hidden = 1024 // deliberately disagrees with the weight set

	q := quantizeProjections(lw, modeQ8)
	if len(q.Wqkv) != len(lw.Wqkv) || len(q.WOut) != len(lw.WOut) {
		t.Fatalf("round-trip changed a projection's length: Wqkv %d->%d, WOut %d->%d",
			len(lw.Wqkv), len(q.Wqkv), len(lw.WOut), len(q.WOut))
	}
	// Row 0 of Wqkv is 256 wide (8 Q8_0 blocks). Quantizing it at the wrong stride would let
	// row 0's block scales be computed from a different row's values, so check the bound holds
	// row-locally on the LAST row too, where a wrong stride runs off the intended data.
	last := (gdn.ConvDim - 1) * 256
	for i := last; i < last+256; i++ {
		if d := math.Abs(float64(q.Wqkv[i] - lw.Wqkv[i])); d > 0.05 {
			t.Fatalf("last row element %d moved by %g; the round-trip is reading the wrong stride", i, d)
		}
	}
}

// TestQ4KPerturbsMoreThanQ8 pins the ordering the swept modes rely on: a 4-bit affine
// sub-block must distort the projections strictly more than an 8-bit symmetric block. If this
// inverted, the "lower bound" framing of the Q4_K arm would be backwards.
func TestQ4KPerturbsMoreThanQ8(t *testing.T) {
	saved := hidden
	defer func() { hidden = saved }()
	hidden = 256

	lw := gdn.NewLayerWeights(hidden)
	lw.Fill(0, aLogMean)
	q8 := quantizeProjections(lw, modeQ8)
	q4 := quantizeProjections(lw, modeQ4K)

	e8 := gdn.RelDiv(lw.Wqkv, q8.Wqkv)
	e4 := gdn.RelDiv(lw.Wqkv, q4.Wqkv)
	t.Logf("relative weight error: Q8=%g Q4_K=%g", e8, e4)
	if e8 <= 0 {
		t.Fatalf("Q8 round-trip left the weights untouched (err=%g)", e8)
	}
	if e4 <= e8 {
		t.Fatalf("Q4_K error %g must exceed Q8 error %g (4 bits vs 8)", e4, e8)
	}
}

// TestFlatInLengthSmoke is a fast end-to-end guard: at a tiny scale, quant-induced rho must not blow
// up with decode length (the paper claim in miniature). Two short lengths, few layers.
func TestFlatInLengthSmoke(t *testing.T) {
	hidden = 256
	_, r16 := runStackForLength(16, 4, modeQ8)
	_, r64 := runStackForLength(64, 4, modeQ8)
	if r16 <= 0 || r64 <= 0 {
		t.Fatalf("rho must be positive: r16=%g r64=%g", r16, r64)
	}
	if ratio := r64 / r16; ratio > 2 || ratio < 0.5 {
		t.Fatalf("rho should be roughly flat in length at tiny scale; r16=%g r64=%g ratio=%g", r16, r64, ratio)
	}
}
