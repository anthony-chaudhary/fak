//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"os"
	"testing"
	"time"
)

// The bandwidth-shaped Q8_0 GEMV keeps k_q8_gemm's per-lane block order, term and
// shuffle tree, so the two must agree bit for bit on the Inkling-2 decode shapes.
// FAK_INK2_Q8_BENCH=1 also logs each kernel's time and bandwidth per shape.
// fak-test:runtime integration est=2s lane=optin
func TestCUDAInk2Q8GEMVFastMatchesReference(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	defer ink2SelectQ8GEMV(ink2SelectQ8GEMV(true))
	r := rand.New(rand.NewSource(37))
	bench := os.Getenv("FAK_INK2_Q8_BENCH") == "1"
	for _, c := range []struct{ rows, in int }{
		{3072, 2304}, {1024, 2304}, {384, 2304}, {2304, 3072}, {48, 2304}, {37, 4096}, {5, 160},
	} {
		nblk := c.in / 32
		codes := make([]int8, c.rows*c.in)
		for i := range codes {
			codes[i] = int8(r.Intn(255) - 127)
		}
		scales := make([]float32, c.rows*nblk)
		for i := range scales {
			scales[i] = float32(r.ExpFloat64() * 1e-3)
		}
		w := cb.Upload(NewQ8(Default(), []int{c.rows, c.in}, codes, scales, 32), Q8_0)
		x := make([]float32, c.in)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		alloc := func(n int) Tensor {
			tt, err := cb.Ink2Alloc(n)
			if err != nil {
				t.Fatal(err)
			}
			return tt
		}
		dx, q := alloc(c.in), alloc(c.in/4+nblk)
		if err := cb.Ink2Write(Ink2At(dx, 0), x); err != nil {
			t.Fatal(err)
		}
		run := func(fast bool) ([]float32, time.Duration) {
			ink2SelectQ8GEMV(fast)
			y := alloc(c.rows)
			matvec := func(quantize bool) {
				ok, err := cb.Ink2MatVecQ8(w, Ink2At(dx, 0), Ink2Ref{}, Ink2At(q, 0), quantize, Ink2At(y, 0))
				if err != nil || !ok {
					t.Fatalf("%dx%d: ok=%v err=%v", c.rows, c.in, ok, err)
				}
			}
			matvec(true)
			out, err := cb.Ink2Read(Ink2At(y, 0), c.rows)
			if err != nil {
				t.Fatal(err)
			}
			var per time.Duration
			if bench {
				const iters = 400
				start := time.Now()
				for i := 0; i < iters; i++ {
					matvec(false)
				}
				if _, err := cb.Ink2Read(Ink2At(y, 0), 1); err != nil {
					t.Fatal(err)
				}
				per = time.Since(start) / iters
			}
			return out, per
		}
		want, tRef := run(false)
		got, tFast := run(true)
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%dx%d: y[%d] fast %v, reference %v", c.rows, c.in, i, got[i], want[i])
			}
		}
		if bench {
			bytes := float64(c.rows*c.in + c.rows*nblk*4)
			t.Logf("q8 gemv %dx%d: k_q8_gemm %v (%.0f GB/s), k_ink2_q8_gemv %v (%.0f GB/s)", c.rows, c.in,
				tRef, bytes/tRef.Seconds()/1e9, tFast, bytes/tFast.Seconds()/1e9)
		}
	}
}
