//go:build darwin && arm64 && cgo

// q2k_test.go — on-device parity tests for Metal Q2_K GEMV and GEMM on Apple Silicon.

package metalgemm

import (
	"math"
	"math/rand"
	"testing"
)

func q2kCloseEnough(a, b float32) bool {
	diff := math.Abs(float64(a - b))
	if diff < 1e-3 {
		return true
	}
	mag := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	return diff/mag < 1e-3
}

func TestMetalQ2KGEMVMatchesReference(t *testing.T) {
	if !Available() {
		t.Skip("no Metal device available")
	}
	defer ResetQ2K()
	rng := rand.New(rand.NewSource(10868))

	shapes := [][2]int{
		{1, 256},
		{5, 512},
		{17, 512},
		{32, 1024},
		{64, 2048},
	}

	for _, s := range shapes {
		out, in := s[0], s[1]
		raw := q2kTestRaw(out, in, uint64(out*1000+in))
		w := UploadQ2K(raw, out, in)
		if w == nil {
			t.Fatalf("shape [%d,%d]: UploadQ2K returned nil", out, in)
		}
		if w.Out != out || w.In != in || w.Nblk != in/Q2KBlockWeights {
			t.Fatalf("shape [%d,%d]: handle dims = (Out=%d,In=%d,Nblk=%d)", out, in, w.Out, w.In, w.Nblk)
		}

		x := make([]float32, in)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}

		want := q2kReference(raw, out, in, x)
		got := make([]float32, out)
		w.GEMV(x, got)

		nonZero := false
		for o := 0; o < out; o++ {
			if math.IsNaN(float64(got[o])) || math.IsInf(float64(got[o]), 0) {
				t.Fatalf("shape [%d,%d] row %d: Metal GEMV produced non-finite %v", out, in, o, got[o])
			}
			if got[o] != 0 {
				nonZero = true
			}
			if !q2kCloseEnough(got[o], want[o]) {
				t.Fatalf("shape [%d,%d] row %d: Metal GEMV = %v, CPU-ref = %v (delta %v)",
					out, in, o, got[o], want[o], math.Abs(float64(got[o]-want[o])))
			}
		}
		if !nonZero {
			t.Fatalf("shape [%d,%d]: all-zero output produced", out, in)
		}
	}
}

func TestMetalQ2KGEMMParity(t *testing.T) {
	if !Available() {
		t.Skip("no Metal device available")
	}
	defer ResetQ2K()
	rng := rand.New(rand.NewSource(108682))

	const out, in, P = 16, 512, 4
	raw := q2kTestRaw(out, in, 42)
	w := UploadQ2K(raw, out, in)
	if w == nil {
		t.Fatal("UploadQ2K returned nil")
	}

	X := make([]float32, P*in)
	for i := range X {
		X[i] = float32(rng.NormFloat64())
	}

	got := make([]float32, P*out)
	w.GEMM(X, P, got)

	for p := 0; p < P; p++ {
		rowX := X[p*in : (p+1)*in]
		wantRow := q2kReference(raw, out, in, rowX)
		gotRow := got[p*out : (p+1)*out]
		for o := 0; o < out; o++ {
			if !q2kCloseEnough(gotRow[o], wantRow[o]) {
				t.Fatalf("token %d row %d: Metal GEMM = %v, CPU-ref = %v (delta %v)",
					p, o, gotRow[o], wantRow[o], math.Abs(float64(gotRow[o]-wantRow[o])))
			}
		}
	}
}

func TestMetalQ2KUploadRejectsBadShapes(t *testing.T) {
	if !Available() {
		t.Skip("no Metal device available")
	}
	defer ResetQ2K()

	const out, in = 8, 256
	raw := q2kTestRaw(out, in, 108683)

	if w := UploadQ2K(raw, 0, in); w != nil {
		t.Fatal("UploadQ2K with out=0 must return nil")
	}
	if w := UploadQ2K(raw, out, 0); w != nil {
		t.Fatal("UploadQ2K with in=0 must return nil")
	}
	if w := UploadQ2K(raw, out, 250); w != nil {
		t.Fatal("UploadQ2K with in not a multiple of 256 must return nil")
	}
	if w := UploadQ2K(raw[:len(raw)-1], out, in); w != nil {
		t.Fatal("UploadQ2K with short payload must return nil")
	}
	if w := UploadQ2K(raw, out, in); w == nil {
		t.Fatal("UploadQ2K with valid payload must return non-nil handle")
	}
}
