//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"math/rand"
	"testing"
)

// ink2DF2TestHostNormRope is the host reference's normRope: RMS norm in f64,
// then neox halves rotated by f64 sincos.
func ink2DF2TestHostNormRope(x, w []float32, pos int, eps float32, theta float64) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	sc := float32(1 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	half := len(x) / 2
	out := make([]float32, len(x))
	for i := 0; i < half; i++ {
		a, b := float64(x[i]*sc*w[i]), float64(x[i+half]*sc*w[i+half])
		s, c := math.Sincos(float64(pos) * math.Pow(theta, -float64(2*i)/float64(len(x))))
		out[i], out[i+half] = float32(a*c-b*s), float32(b*c+a*s)
	}
	return out
}

func ink2DF2TestClose(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	worst := 0.0
	for i := range want {
		worst = math.Max(worst, math.Abs(float64(got[i]-want[i])))
	}
	if worst > tol {
		t.Fatalf("%s: max abs error %g over %g", what, worst, tol)
	}
	t.Logf("%s: max abs error %g", what, worst)
}

// fak-test:runtime integration est=1s lane=default
func TestInk2DFlash2QKRopeRingAndSiluMatchHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(5))
	const rows, heads, dim, cap, base, pos0 = 3, 4, 128, 7, 2, 2050
	const eps, theta = float32(1e-6), 10000.0
	x := make([]float32, rows*heads*dim)
	w := make([]float32, dim)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	for i := range w {
		w[i] = 0.5 + rng.Float32()
	}
	pos := ink2OpsTestPut(t, d, 0, []float32{math.Float32frombits(uint32(pos0))})
	xr, wr := ink2OpsTestPut(t, d, 1, x), ink2OpsTestPut(t, d, 0, w)
	want := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		for h := 0; h < heads; h++ {
			o := (r*heads + h) * dim
			copy(want[o:o+dim], ink2DF2TestHostNormRope(x[o:o+dim], w, pos0+base+r, eps, theta))
		}
	}
	// row order, in place
	if err := d.DraftQKRope(xr, wr, xr, pos, rows, heads, dim, 0, base, false, eps, theta); err != nil {
		t.Fatal(err)
	}
	ink2DF2TestClose(t, "qk rope rows", ink2OpsTestRead(t, d, xr, len(x)), want, 2e-3)

	// ring slots of the same rows
	xr = ink2OpsTestPut(t, d, 0, x)
	ring := ink2TestAlloc(t, d, cap*heads*dim)
	if err := d.DraftQKRope(xr, wr, ring, pos, rows, heads, dim, cap, base, true, eps, theta); err != nil {
		t.Fatal(err)
	}
	got := ink2OpsTestRead(t, d, ring, cap*heads*dim)
	w0 := heads * dim
	for r := 0; r < rows; r++ {
		slot := (pos0 + base + r) % cap
		ink2DF2TestClose(t, "qk rope ring row", got[slot*w0:(slot+1)*w0], want[r*w0:(r+1)*w0], 2e-3)
	}

	ring2 := ink2TestAlloc(t, d, cap*w0)
	if err := d.DraftRingStore(ink2OpsTestPut(t, d, 0, x), ring2, pos, rows, w0, cap, base); err != nil {
		t.Fatal(err)
	}
	got = ink2OpsTestRead(t, d, ring2, cap*w0)
	for r := 0; r < rows; r++ {
		slot := (pos0 + base + r) % cap
		ink2DF2TestClose(t, "ring store row", got[slot*w0:(slot+1)*w0], x[r*w0:(r+1)*w0], 0)
	}

	n := len(x)
	u := make([]float32, n)
	for i := range u {
		u[i] = rng.Float32()*4 - 2
	}
	gr := ink2OpsTestPut(t, d, 0, x)
	if err := d.DraftSiluMul(gr, ink2OpsTestPut(t, d, 0, u), gr, n); err != nil {
		t.Fatal(err)
	}
	ws := make([]float32, n)
	for i, g := range x {
		ws[i] = g / (1 + float32(math.Exp(float64(-g)))) * u[i]
	}
	ink2DF2TestClose(t, "silu mul", ink2OpsTestRead(t, d, gr, n), ws, 1e-6)
}

// The draft head against the verify rows kernel on the same weight: equal to
// rounding at every covered type, and timed at the shipped head geometry.
//
// fak-test:runtime integration est=20s lane=default
func TestInk2DFlash2HeadMatchesVerifyRows(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(23))
	store, data := ink2TestStore(t, 64)
	const rows, cols, tokens = 75, 512, 7
	x := ink2OpsTestRand(rng, 8*cols, 1)
	xr := ink2VTestPut(t, d, x)
	for _, typ := range []uint32{8, 12, 13, 14} {
		w, _ := ink2VTestWeight(t, store, data, rng, 64, typ, rows, cols)
		for _, T := range []int{1, tokens, 8} {
			want, got := ink2TestAlloc(t, d, T*rows), ink2VTestPut(t, d, make([]float32, T*rows+rows))
			if err := d.VerifyRows(w, xr, cols, want, rows, T); err != nil {
				t.Fatal(err)
			}
			ok, err := d.DraftHead(w, xr, cols, got, rows, T)
			if err != nil || !ok {
				t.Fatalf("type %d: draft head ok=%v err=%v", typ, ok, err)
			}
			wv, gv := ink2OpsTestRead(t, d, want, T*rows), ink2OpsTestRead(t, d, got, T*rows+rows)
			worst, scale := 0.0, 0.0
			for i, v := range wv {
				worst = math.Max(worst, math.Abs(float64(gv[i]-v)))
				scale = math.Max(scale, math.Abs(float64(v)))
			}
			if worst > 1e-5*scale+1e-6 {
				t.Fatalf("type %d, %d rows: max abs error %g of %g", typ, T, worst, scale)
			}
			for i := T * rows; i < len(gv); i++ {
				if gv[i] != 0 {
					t.Fatalf("type %d, %d rows wrote past its outputs", typ, T)
				}
			}
		}
	}
	if testing.Short() {
		return
	}
	const vocab, hidden = 201024, 2304
	big, bigData := ink2TestStore(t, 21000)
	w, _ := ink2VTestWeight(t, big, bigData, rng, 64, 13, vocab, hidden)
	hx := ink2VTestPut(t, d, ink2OpsTestRand(rng, 8*hidden, 1))
	y := ink2TestAlloc(t, d, 8*vocab)
	rowsT := ink2DFTestGPU(t, d, func() error { return d.VerifyRows(w, hx, hidden, y, vocab, tokens) })
	headT := ink2DFTestGPU(t, d, func() error {
		_, err := d.DraftHead(w, hx, hidden, y, vocab, tokens)
		return err
	})
	t.Logf("q5_K head %dx%d, %d rows: verify rows %v, draft head %v", vocab, hidden, tokens, rowsT, headT)
}
