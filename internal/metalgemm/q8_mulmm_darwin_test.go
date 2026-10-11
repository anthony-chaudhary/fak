//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

type q8MulMMCase struct{ out, in, P int }

func q8MulMMFixture(c q8MulMMCase, seed int64) (codes []int8, wd []float32, xq []int8, xd []float32) {
	rng := rand.New(rand.NewSource(seed))
	nblk := c.in / 32
	codes = make([]int8, c.out*c.in)
	for i := range codes {
		codes[i] = int8(rng.Intn(255) - 127)
	}
	wd = make([]float32, c.out*nblk)
	for i := range wd {
		wd[i] = float32(0.001 + 0.02*rng.Float64())
	}
	xq = make([]int8, c.P*c.in)
	for i := range xq {
		xq[i] = int8(rng.Intn(255) - 127)
	}
	xd = make([]float32, c.P*nblk)
	for i := range xd {
		xd[i] = float32(0.001 + 0.05*rng.Float64())
	}
	return
}

// q8MulMMOracle is the f64 reference Y[t*out+o] = sum_b wd*xd*sum_k wcode*xcode.
func q8MulMMOracle(c q8MulMMCase, codes []int8, wd []float32, xq []int8, xd []float32) []float64 {
	nblk := c.in / 32
	y := make([]float64, c.P*c.out)
	for t := 0; t < c.P; t++ {
		for o := 0; o < c.out; o++ {
			var acc float64
			for b := 0; b < nblk; b++ {
				s := 0
				for k := 0; k < 32; k++ {
					s += int(codes[o*c.in+b*32+k]) * int(xq[t*c.in+b*32+k])
				}
				acc += float64(s) * float64(wd[o*nblk+b]) * float64(xd[t*nblk+b])
			}
			y[t*c.out+o] = acc
		}
	}
	return y
}

func q8MulMMCompare(t *testing.T, name string, got []float32, want []float64, maxRelTol float64) {
	t.Helper()
	var dot, ng, nw, maxAbs, maxErr float64
	for i := range want {
		g := float64(got[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("%s: non-finite output at %d", name, i)
		}
		dot += g * want[i]
		ng += g * g
		nw += want[i] * want[i]
		maxAbs = math.Max(maxAbs, math.Abs(want[i]))
		maxErr = math.Max(maxErr, math.Abs(g-want[i]))
	}
	cos := dot / math.Sqrt(ng*nw)
	if cos < 0.99999 || maxErr > maxRelTol*maxAbs {
		t.Fatalf("%s: cos=%.7f maxErr/maxAbs=%.2e (tol %.1e)", name, cos, maxErr/maxAbs, maxRelTol)
	}
}

// fak-test:runtime integration est=5s lane=default
func TestQ8MulMMMatchesIndependentCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	if !Q8MulMMReady() {
		t.Fatal("q8_mul_mm pipeline unavailable on a Metal device")
	}
	defer SetQ8UseMulMM(true)
	cases := []q8MulMMCase{
		{64, 256, 32}, {96, 512, 45}, {40, 256, 128}, {32, 2560, 128}, {1024, 2560, 128}, {128, 4096, 33}, {256, 512, 512}, {64, 256, 3}, {64, 256, 24},
	}
	for i, c := range cases {
		codes, wd, xq, xd := q8MulMMFixture(c, int64(1000+i))
		want := q8MulMMOracle(c, codes, wd, xq, xd)
		w, w2 := UploadQ8(codes, wd, c.out, c.in), UploadQ8(codes, wd, c.out, c.in)
		if w == nil || w2 == nil {
			t.Fatalf("UploadQ8(%d,%d) returned nil", c.out, c.in)
		}
		outs := map[bool][]float32{}
		for _, mulmm := range []bool{true, false} {
			SetQ8UseMulMM(mulmm)
			got := make([]float32, c.P*c.out)
			w.GEMM(xq, xd, c.P, got)
			outs[mulmm] = got
			tol := 1e-5 // scalar q8_gemm: exact int8 products, f32 accumulate
			if mulmm && c.P >= Q8MulMMMinPrompt {
				tol = 2e-3 // weights rounded to half before the MMA
			}
			q8MulMMCompare(t, "single", got, want, tol)
			grouped := GEMMGroupQ8([]*Q8Weight{w, w2}, xq, xd, c.P)
			if len(grouped) != 2 {
				t.Fatalf("GEMMGroupQ8 returned %d outputs", len(grouped))
			}
			for _, g := range grouped {
				q8MulMMCompare(t, "group", g, want, tol)
			}
		}
		// Routing witness: half-rounded weights make q8_mul_mm's bits differ from q8_gemm's exactly
		// when it was dispatched (P >= Q8MulMMMinPrompt), and never below the floor.
		if same := slices.Equal(outs[true], outs[false]); same == (c.P >= Q8MulMMMinPrompt) {
			t.Fatalf("P=%d out=%d: mul_mm on/off outputs identical=%v, want mul_mm dispatched iff P>=%d", c.P, c.out, same, Q8MulMMMinPrompt)
		}
		w.Release()
		w2.Release()
	}
}
