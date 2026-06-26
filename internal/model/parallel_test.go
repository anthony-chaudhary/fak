package model

import (
	"math"
	"sync/atomic"
	"testing"
	"time"
)

// TestParallelMatchesSerial pins parMatRows and matMulBatch to the serial matRows
// reference BIT-FOR-BIT (not within tolerance). This is the contract that lets the
// performance lane keep every exact fak-vs-fak rung (R2 max|Δ|=0, R14 d==0) green: the
// parallel/batched ops reorder WORK, never a single dot-product's reduction.
func TestParallelMatchesSerial(t *testing.T) {
	// deterministic pseudo-random weights/inputs (no rng dependency)
	mk := func(n int, seed uint64) []float32 {
		v := make([]float32, n)
		s := seed
		for i := range v {
			s = s*6364136223846793005 + 1442695040888963407
			v[i] = float32(int64(s>>40))/float32(1<<23) - 0.5
		}
		return v
	}
	cases := []struct{ out, in int }{
		{576, 576}, {192, 576}, {1536, 576}, {576, 1536}, {49152, 576}, {7, 3}, {1, 1},
	}
	for _, c := range cases {
		w := mk(c.out*c.in, uint64(c.out*131+c.in))
		x := mk(c.in, 99)
		ser := matRows(w, x, c.out, c.in)
		par := parMatRows(w, x, c.out, c.in)
		for o := 0; o < c.out; o++ {
			if math.Float32bits(ser[o]) != math.Float32bits(par[o]) {
				t.Fatalf("parMatRows[%dx%d] o=%d: serial %v != parallel %v (NOT bit-identical)",
					c.out, c.in, o, ser[o], par[o])
			}
		}
		// batched: every (t,o) must equal the per-row serial result for that t's input.
		P := 5
		X := mk(P*c.in, uint64(c.in*7+3))
		bat := matMulBatch(w, X, c.out, c.in, P)
		for tt := 0; tt < P; tt++ {
			ser := matRows(w, X[tt*c.in:(tt+1)*c.in], c.out, c.in)
			for o := 0; o < c.out; o++ {
				if math.Float32bits(bat[tt*c.out+o]) != math.Float32bits(ser[o]) {
					t.Fatalf("matMulBatch[%dx%d] t=%d o=%d not bit-identical to per-row matRows",
						c.out, c.in, tt, o)
				}
			}
		}
	}
}

func TestFdot3MatchesFdot(t *testing.T) {
	mk := func(n int, seed uint64) []float32 {
		v := make([]float32, n)
		s := seed
		for i := range v {
			s = s*6364136223846793005 + 1442695040888963407
			v[i] = float32(int64(s>>40))/float32(1<<23) - 0.5
		}
		return v
	}
	for _, n := range []int{1, 7, 64, 65, 576} {
		r0, r1, r2, x := mk(n, 10), mk(n, 20), mk(n, 30), mk(n, 40)
		a, b, c := fdot3(r0, r1, r2, x)
		if math.Float32bits(a) != math.Float32bits(fdot(r0, x)) ||
			math.Float32bits(b) != math.Float32bits(fdot(r1, x)) ||
			math.Float32bits(c) != math.Float32bits(fdot(r2, x)) {
			t.Fatalf("fdot3 n=%d does not match independent fdot calls", n)
		}
		a, b, c = fdot3SIMD(r0, r1, r2, x)
		if math.Abs(float64(a-fdot(r0, x))) > 1e-4 ||
			math.Abs(float64(b-fdot(r1, x))) > 1e-4 ||
			math.Abs(float64(c-fdot(r2, x))) > 1e-4 {
			t.Fatalf("fdot3SIMD n=%d drift too large vs independent fdot calls", n)
		}
	}
}

func TestParForHonorsRequestedWorkers(t *testing.T) {
	if numWorkers < 4 {
		t.Skipf("numWorkers=%d is too small to prove a sub-budget cap", numWorkers)
	}
	var active int64
	var maxActive int64
	parFor(64, 2, func(lo, hi int) {
		now := atomic.AddInt64(&active, 1)
		for {
			old := atomic.LoadInt64(&maxActive)
			if now <= old || atomic.CompareAndSwapInt64(&maxActive, old, now) {
				break
			}
		}
		time.Sleep(2 * time.Millisecond)
		atomic.AddInt64(&active, -1)
	})
	if maxActive > 2 {
		t.Fatalf("parFor dispatched %d workers for requested budget 2", maxActive)
	}
}

func TestSaxpy3MatchesScalar(t *testing.T) {
	mk := func(n int, seed uint64) []float32 {
		v := make([]float32, n)
		s := seed
		for i := range v {
			s = s*6364136223846793005 + 1442695040888963407
			v[i] = float32(int64(s>>40))/float32(1<<23) - 0.5
		}
		return v
	}
	for _, n := range []int{1, 7, 8, 64, 65} {
		x := mk(n, 1)
		got0, got1, got2 := mk(n, 10), mk(n, 20), mk(n, 30)
		want0 := append([]float32(nil), got0...)
		want1 := append([]float32(nil), got1...)
		want2 := append([]float32(nil), got2...)
		a0, a1, a2 := float32(0.125), float32(-0.75), float32(1.5)
		saxpy3(got0, got1, got2, x, a0, a1, a2)
		saxpy3scalar(want0, want1, want2, x, a0, a1, a2)
		for i := range x {
			if math.Float32bits(got0[i]) != math.Float32bits(want0[i]) ||
				math.Float32bits(got1[i]) != math.Float32bits(want1[i]) ||
				math.Float32bits(got2[i]) != math.Float32bits(want2[i]) {
				t.Fatalf("saxpy3 n=%d i=%d does not match scalar", n, i)
			}
		}
	}
}
