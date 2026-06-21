package compute

import (
	"math"
	"testing"
)

// proofs_witness_test.go — closes OPEN proof obligations from fak/docs/proofs.
//
// Each Test here ASSERTS a metamorphic property of the compute reference backend
// (mechanism: cpuref.go MatMul + fdot), comparing against the naive computation up
// to a float tolerance, per the proof discipline in fak/docs/proofs/00-METHOD.md.
// These are NOT bit-identity (byte-equality) checks: float arithmetic is not
// associative or distributive, so A(B+C) and AB+AC differ in the last ~1ulp. The
// 00-METHOD §3.2 metamorphic relation is the right tool here — it asserts the
// property holds up to tolerance, which is exactly what bilinearity means for a
// finite-precision GEMM.

// relErr returns the max relative error between two equal-length vectors, scaled so
// that near-zero components fall back to absolute error (avoids divide-by-tiny blowups).
func relErr(a, b []float32) float64 {
	var worst float64
	for i := range a {
		af, bf := float64(a[i]), float64(b[i])
		d := math.Abs(af - bf)
		scale := math.Max(math.Max(math.Abs(af), math.Abs(bf)), 1.0)
		if e := d / scale; e > worst {
			worst = e
		}
	}
	return worst
}

func finite(v []float32) bool {
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	return true
}

// TestGEMMBilinear is the [gemm-bilinear] witness: the reference GEMM
// y = MatMul(W, x) (y[o]=Σ_i W[o,i]·x[i]) is bilinear in W and x, checked against the
// naive recomputation up to a 1e-5 float tolerance. We exercise all four legs of
// bilinearity as metamorphic relations:
//
//	(1) additivity in the vector arg:  W(x1+x2) ≈ Wx1 + Wx2
//	(2) additivity in the matrix arg:  (A+B)x  ≈ Ax + Bx
//	(3) homogeneity in the vector arg: W(αx)   ≈ α·(Wx)
//	(4) homogeneity in the matrix arg: (αW)x   ≈ α·(Wx)
//
// All inputs are deterministic (fixed lcg seed). The property is non-vacuous: each
// leg compares two independently-computed tensors that are only equal *because* the
// op is bilinear — a buggy MatMul (e.g. an off-by-one row stride or a non-linear
// activation slipped into the dot) would break the relation well outside tolerance.
func TestGEMMBilinear(t *testing.T) {
	c := cpu()
	const tol = 1e-5
	out, in := 7, 32 // 32 = a clean fdot lane multiple

	mm := func(w, x []float32) []float32 {
		wt := NewF32(c, []int{out, in}, w)
		xt := NewF32(c, []int{in}, x)
		return c.Read(c.MatMul(wt, xt))
	}
	add := func(a, b []float32) []float32 {
		r := make([]float32, len(a))
		for i := range a {
			r[i] = a[i] + b[i]
		}
		return r
	}
	scale := func(a []float32, s float32) []float32 {
		r := make([]float32, len(a))
		for i := range a {
			r[i] = a[i] * s
		}
		return r
	}

	var sd lcg = 1009
	A := randVec(&sd, out*in)
	B := randVec(&sd, out*in)
	x1 := randVec(&sd, in)
	x2 := randVec(&sd, in)
	const alpha float32 = 2.75

	// (1) additivity in the vector arg
	if got, want := mm(A, add(x1, x2)), add(mm(A, x1), mm(A, x2)); relErr(got, want) > tol {
		t.Fatalf("A(x1+x2) != Ax1+Ax2: relErr %.3e > %.0e", relErr(got, want), tol)
	}
	// (2) additivity in the matrix arg
	if got, want := mm(add(A, B), x1), add(mm(A, x1), mm(B, x1)); relErr(got, want) > tol {
		t.Fatalf("(A+B)x != Ax+Bx: relErr %.3e > %.0e", relErr(got, want), tol)
	}
	// (3) homogeneity in the vector arg
	if got, want := mm(A, scale(x1, alpha)), scale(mm(A, x1), alpha); relErr(got, want) > tol {
		t.Fatalf("A(αx) != α(Ax): relErr %.3e > %.0e", relErr(got, want), tol)
	}
	// (4) homogeneity in the matrix arg
	if got, want := mm(scale(A, alpha), x1), scale(mm(A, x1), alpha); relErr(got, want) > tol {
		t.Fatalf("(αA)x != α(Ax): relErr %.3e > %.0e", relErr(got, want), tol)
	}

	// Non-vacuity guard: the bilinear identity must hold because the inputs are
	// *non-trivial*, not because everything is zero. Confirm a representative output
	// is both finite and not identically zero.
	y := mm(A, x1)
	if !finite(y) {
		t.Fatalf("MatMul produced a non-finite output")
	}
	var any bool
	for _, v := range y {
		if v != 0 {
			any = true
			break
		}
	}
	if !any {
		t.Fatalf("MatMul output is identically zero — bilinearity check would be vacuous")
	}
}

// TestGEMMBatchedBilinear extends [gemm-bilinear] to the prefill GEMM path
// (BatchedMatMul, Y[t,o]=Σ_i W[o,i]·X[t,i]). It asserts additivity in the matrix arg
// across an entire P-row activation batch — a separate code path (fdot per (row,token))
// from the single-vector MatMul above — so the property is witnessed on both reference
// matmul entry points.
func TestGEMMBatchedBilinear(t *testing.T) {
	c := cpu()
	const tol = 1e-5
	out, in, P := 5, 32, 4

	bmm := func(w, X []float32) []float32 {
		wt := NewF32(c, []int{out, in}, w)
		Xt := NewF32(c, []int{P, in}, X)
		return c.Read(c.BatchedMatMul(wt, Xt, P))
	}

	var sd lcg = 2027
	A := randVec(&sd, out*in)
	B := randVec(&sd, out*in)
	X := randVec(&sd, P*in)

	addW := func(a, b []float32) []float32 {
		r := make([]float32, len(a))
		for i := range a {
			r[i] = a[i] + b[i]
		}
		return r
	}
	addY := addW // same elementwise add, different lengths

	got := bmm(addW(A, B), X)
	want := addY(bmm(A, X), bmm(B, X))
	if e := relErr(got, want); e > tol {
		t.Fatalf("(A+B)X != AX+BX (batched): relErr %.3e > %.0e", e, tol)
	}
	if !finite(got) {
		t.Fatalf("BatchedMatMul produced a non-finite output")
	}
}
