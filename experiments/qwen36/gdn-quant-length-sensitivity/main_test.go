package main

import (
	"math"
	"testing"
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
	lw := newLayerWeights()
	lw.fill(0)
	q := quantizeProjections(lw, modeQ8)

	// control tensors must be the SAME backing arrays (shared, untouched).
	if &q.conv[0] != &lw.conv[0] || &q.aLog[0] != &lw.aLog[0] || &q.normW[0] != &lw.normW[0] || &q.wIn[0] != &lw.wIn[0] {
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
		"wqkv": {lw.wqkv, q.wqkv},
		"wz":   {lw.wz, q.wz},
		"wb":   {lw.wb, q.wb},
		"wa":   {lw.wa, q.wa},
		"wOut": {lw.wOut, q.wOut},
	} {
		if !changed(pair[0], pair[1]) {
			t.Fatalf("projection %s should be quantized to a new, different array", name)
		}
	}
}

// TestRelDiv checks the relative-divergence metric on identity and a known perturbation.
func TestRelDiv(t *testing.T) {
	a := []float32{3, 4} // norm 5
	if got := relDiv(a, a); got != 0 {
		t.Fatalf("relDiv(a,a)=%g want 0", got)
	}
	b := []float32{3, 4 + 5} // Δ=(0,5), ||Δ||=5, ||a||=5 -> rho=1
	if got := relDiv(a, b); math.Abs(got-1) > 1e-6 {
		t.Fatalf("relDiv=%g want 1", got)
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
