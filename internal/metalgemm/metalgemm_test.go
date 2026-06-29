//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"math/rand"
	"testing"
)

// refMatMul computes the f32 reference Y[P,out] = X[P,in] · Wᵀ (W row-major [out,in]).
func refMatMul(x, w []float32, P, in, out int) []float32 {
	y := make([]float32, P*out)
	for t := 0; t < P; t++ {
		for o := 0; o < out; o++ {
			var acc float64
			for i := 0; i < in; i++ {
				acc += float64(x[t*in+i]) * float64(w[o*in+i])
			}
			y[t*out+o] = float32(acc)
		}
	}
	return y
}

// TestMatMulMatchesReference pins the Metal f16 GEMM against the f32 CPU reference within a
// half-precision tolerance — proof the device path computes the right matmul, not just fast.
func TestMatMulMatchesReference(t *testing.T) {
	if !MPSAvailable() {
		t.Skip("Metal MPS unavailable")
	}
	rng := rand.New(rand.NewSource(7))
	for _, dims := range [][3]int{{16, 64, 96}, {64, 512, 256}, {256, 1536, 1536}} {
		P, in, out := dims[0], dims[1], dims[2]
		w := make([]float32, out*in)
		x := make([]float32, P*in)
		for i := range w {
			w[i] = float32(rng.NormFloat64()) * 0.1
		}
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		wt := Upload(w, out, in)
		if wt == nil {
			t.Fatalf("upload failed for %v", dims)
		}
		y := make([]float32, P*out)
		wt.MatMul(x, P, y)
		wt.Free()

		ref := refMatMul(x, w, P, in, out)
		// Normalise the worst absolute error by the output magnitude (max|ref|), NOT
		// per-element: outputs cross zero, so a per-element relative error blows up on
		// near-zero entries while the matmul is in fact f16-accurate. Error-to-scale is the
		// meaningful number — a correct f16 GEMM lands well under 1%.
		var maxAbs, maxRef float64
		for i := range ref {
			if d := math.Abs(float64(y[i] - ref[i])); d > maxAbs {
				maxAbs = d
			}
			if a := math.Abs(float64(ref[i])); a > maxRef {
				maxRef = a
			}
		}
		rel := maxAbs / maxRef
		if rel > 0.01 {
			t.Errorf("dims %v: maxAbs=%.5f / maxRef=%.3f = %.4f exceeds f16 tolerance", dims, maxAbs, maxRef, rel)
		}
		t.Logf("dims %v: maxAbs=%.5f maxRef=%.3f err/scale=%.5f", dims, maxAbs, maxRef, rel)
	}
}

// TestResetReclaimsTable proves Reset returns the weight table to a clean, reusable
// state: uploading + matmul'ing then Reset()'ing repeatedly stays correct and never
// exhausts the table. This is the leak fix's contract — a long-lived process reloading
// models can free the f16 weight set between loads instead of accumulating it.
func TestResetReclaimsTable(t *testing.T) {
	if !MPSAvailable() {
		t.Skip("Metal MPS unavailable")
	}
	const out, in, P = 32, 64, 8
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32((i%7)-3) * 0.05
	}
	x := make([]float32, P*in)
	for i := range x {
		x[i] = float32((i%5)-2) * 0.1
	}
	ref := refMatMul(x, w, P, in, out)

	for r := 0; r < 4; r++ {
		wt := Upload(w, out, in)
		if wt == nil {
			t.Fatalf("round %d: upload failed (table not reclaimed by Reset?)", r)
		}
		y := make([]float32, P*out)
		wt.MatMul(x, P, y)

		var maxAbs, maxRef float64
		for i := range ref {
			if d := math.Abs(float64(y[i] - ref[i])); d > maxAbs {
				maxAbs = d
			}
			if a := math.Abs(float64(ref[i])); a > maxRef {
				maxRef = a
			}
		}
		if maxRef > 0 && maxAbs/maxRef > 0.01 {
			t.Fatalf("round %d: err/scale %.4f exceeds f16 tolerance after reset", r, maxAbs/maxRef)
		}
		Reset() // frees wt's buffer; wt is invalid by contract from here
	}
}
