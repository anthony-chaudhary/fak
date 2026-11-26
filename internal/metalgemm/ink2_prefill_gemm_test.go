//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// The GEMM kernels sum in another order than the decode GEMV, so they are held
// to it within a rounding bound, and to the host forward under the
// conformance bound, instead of bit for bit.

func ink2PGemmTypes() []struct {
	name string
	typ  uint32
	cols int
} {
	types := append(ink2PTestTypes[:0:0], ink2PTestTypes...)
	for i := range types {
		// The kernels step K by 32.
		types[i].cols = (types[i].cols + ink2GemmBK - 1) / ink2GemmBK * ink2GemmBK
		if block, _, _ := packedBlock(types[i].typ); types[i].cols%block != 0 {
			types[i].cols = block
		}
	}
	return types
}

// ink2PGemmClose compares got with the decode kernel's want, bounded by the
// largest magnitude of the outputs: a misplaced tile is off by that much.
func ink2PGemmClose(t *testing.T, what string, got, want []float32) {
	t.Helper()
	scale := 0.0
	for _, w := range want {
		scale = max(scale, math.Abs(float64(w)))
	}
	for i := range want {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > 1e-5*scale+1e-6 {
			t.Fatalf("%s[%d] = %v, decode kernel %v (outputs up to %v)", what, i, got[i], want[i], scale)
		}
	}
}

// fak-test:runtime integration est=6s lane=default
func TestInk2PGemmRowsMatchesMatVecAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(91))
	// Two weight tiles, the second partial.
	rows := ink2GemmRows.bn() + 6
	for _, c := range ink2PGemmTypes() {
		t.Run(c.name, func(t *testing.T) {
			store, data := ink2TestStore(t, 64)
			m, _ := ink2PTestWeights(t, store, data, rng, 64, c.typ, rows, c.cols, 1)
			w, err := m.stack.Ink2Weight(0)
			if err != nil {
				t.Fatal(err)
			}
			xStride, yStride := c.cols+4, rows+3
			maxT := 2*ink2GemmRows.bm() + 5
			x := ink2OpsTestRand(rng, maxT*xStride, .5)
			xr := ink2OpsTestPut(t, d, 8, x)
			for _, T := range []int{1, ink2GemmRows.bm(), maxT} {
				serial, batched := ink2TestAlloc(t, d, T*yStride), ink2TestAlloc(t, d, T*yStride)
				for tok := range T {
					if err := d.MatVec(w, xr.At(tok*xStride), serial.At(tok*yStride)); err != nil {
						t.Fatal(err)
					}
				}
				before := d.Stats()
				if ok, err := d.RowsGEMM(w, xr, xStride, batched, yStride, T); err != nil || !ok {
					t.Fatalf("rows GEMM: ok %v, %v", ok, err)
				}
				if s := d.Stats(); s.Submissions != before.Submissions || s.Pending != before.Pending+1 {
					t.Fatalf("rows GEMM did not only encode one launch: %+v after %+v", s, before)
				}
				got, want := ink2OpsTestRead(t, d, batched, T*yStride), ink2OpsTestRead(t, d, serial, T*yStride)
				for tok := range T {
					ink2PGemmClose(t, fmt.Sprintf("%d rows: token %d", T, tok), got[tok*yStride:][:rows], want[tok*yStride:][:rows])
					if m.ref == nil {
						continue
					}
					for r := range rows {
						host, mag := m.dot(0, r, x[tok*xStride:])
						if g := got[tok*yStride+r]; !ink2TestClose(g, host, mag) {
							t.Fatalf("%d rows, token %d row %d: device %v, host %v (|w||x| %v)", T, tok, r, g, host, mag)
						}
					}
				}
			}
		})
	}
}

// fak-test:runtime integration est=8s lane=default
func TestInk2PGemmGroupedMatchesExpertMatVecAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(92))
	const experts, k = 5, 3
	// Enough pairs that expert 1 runs past one tile, and rows past one
	// column tile of every grouped tile.
	rows, T := max(ink2GemmGrouped.bn(), 128)+6, 40*max(1, ink2GemmGrouped.bm()/32)
	ink2GroupedForce = 0
	defer func() { ink2GroupedForce = -1 }()
	npairs := T * k
	// Expert 4 is never routed; expert 1 holds runs of more than one tile, and
	// the others end in partial tiles.
	ids := make([]int32, npairs)
	for i := range ids {
		ids[i] = []int32{1, 0, 1, 3, 1, 2, 1, 1}[rng.Intn(8)]
	}
	_, offsets, order := ink2PTestBucket(ids, experts)
	idr := ink2OpsTestPut(t, d, 0, ink2PTestInts(ids))
	orr := ink2OpsTestPut(t, d, 1, ink2PTestInts(order))
	ofr := ink2OpsTestPut(t, d, 3, ink2PTestInts(offsets))
	scale := []float32{1.5, -3, .25, .75, 2}
	scr := ink2OpsTestPut(t, d, 2, scale)
	for _, c := range ink2PGemmTypes() {
		t.Run(c.name, func(t *testing.T) {
			store, data := ink2TestStore(t, 256)
			m, _ := ink2PTestWeights(t, store, data, rng, 64, c.typ, rows, c.cols, experts)
			stack, ok := m.stack.Ink2Stack()
			if !ok {
				t.Fatal("stack split across views")
			}
			xStride := c.cols + 4
			for _, form := range []struct {
				name   string
				xDiv   int
				scaled bool
			}{{"token rows scaled", k, true}, {"pair rows", 1, false}} {
				inRows := (npairs-1)/form.xDiv + 1
				x := ink2OpsTestRand(rng, inRows*xStride, .5)
				perPair := make([]float32, npairs*xStride)
				for i := range npairs {
					copy(perPair[i*xStride:(i+1)*xStride], x[(i/form.xDiv)*xStride:])
				}
				xr, pr := ink2OpsTestPut(t, d, 4, x), ink2OpsTestPut(t, d, 0, perPair)
				var sc Ink2Ref
				if form.scaled {
					sc = scr
				}
				serial, batched := ink2TestAlloc(t, d, npairs*rows), ink2TestAlloc(t, d, npairs*rows)
				if ok, err := d.ExpertMatVec(stack, idr, pr, xStride, serial, npairs, sc, nil); err != nil || !ok {
					t.Fatalf("%s: decode expert matvec: ok %v, %v", form.name, ok, err)
				}
				if ok, err := d.PExpertGEMM(stack, idr, orr, ofr, xr, form.xDiv, xStride, batched, npairs, sc); err != nil || !ok {
					t.Fatalf("%s: grouped GEMM: ok %v, %v", form.name, ok, err)
				}
				got, want := ink2OpsTestRead(t, d, batched, npairs*rows), ink2OpsTestRead(t, d, serial, npairs*rows)
				// The adaptive kernel sums each element in the same order.
				if ink2GemmGroupedAdaptKernel[c.typ] != "" {
					ink2GroupedForce = 1
					ok, err := d.PExpertGEMM(stack, idr, orr, ofr, xr, form.xDiv, xStride, batched, npairs, sc)
					ink2GroupedForce = 0
					if err != nil || !ok {
						t.Fatalf("%s: adaptive grouped GEMM: ok %v, %v", form.name, ok, err)
					}
					ink2OpsTestSame(t, form.name+" adaptive", ink2OpsTestRead(t, d, batched, npairs*rows), got)
				}
				for i, e := range ids {
					ink2PGemmClose(t, fmt.Sprintf("%s: pair %d (expert %d)", form.name, i, e), got[i*rows:][:rows], want[i*rows:][:rows])
					if m.ref == nil {
						continue
					}
					for r := range rows {
						host, mag := m.dot(int(e), r, x[(i/form.xDiv)*xStride:])
						if form.scaled {
							host, mag = host*float64(scale[e]), mag*math.Abs(float64(scale[e]))
						}
						if g := got[i*rows+r]; !ink2TestClose(g, host, mag) {
							t.Fatalf("%s: pair %d (expert %d) row %d: device %v, host %v (|w||x| %v)", form.name, i, e, r, g, host, mag)
						}
					}
				}
			}
		})
	}
}

// fak-test:runtime integration est=2s lane=default
func TestInk2PGemmControlMatchesControlAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(93))
	// The tanh of a GEMM-ordered g strays past the decode controller's host
	// bound by a little; this holds it to twice that.
	worst := 0.0
	for _, c := range []struct{ n, rank, T int }{{256, 32, 40}, {2304, 64, 33}} {
		what := fmt.Sprintf("width %d, rank %d, %d rows", c.n, c.rank, c.T)
		cur, msg := ink2OpsTestRand(rng, c.T*c.n, 2), ink2OpsTestRand(rng, c.T*c.n, .5)
		down := ink2OpsTestRand(rng, c.rank*2*c.n, 1/math.Sqrt(float64(2*c.n)))
		gain := ink2OpsTestRand(rng, c.n*c.rank, 1/math.Sqrt(float64(c.rank)))
		forget := ink2OpsTestRand(rng, c.n*c.rank, 1/math.Sqrt(float64(c.rank)))
		cr, mr, dr := ink2OpsTestPut(t, d, 1, cur), ink2OpsTestPut(t, d, 2, msg), ink2OpsTestPut(t, d, 0, down)
		gr, fr := ink2OpsTestPut(t, d, 3, gain), ink2OpsTestPut(t, d, 0, forget)
		g, out := ink2TestAlloc(t, d, c.T*c.rank), ink2TestAlloc(t, d, c.T*c.n)
		wantG, want := ink2TestAlloc(t, d, c.T*c.rank), ink2TestAlloc(t, d, c.T*c.n)
		scratch := ink2TestAlloc(t, d, 2*c.T*c.n)
		if err := d.PCtrlDownGEMM(cr, mr, dr, g, scratch, c.n, c.rank, c.T); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		if err := d.PCtrlApplyGEMM(cr, mr, g, gr, fr, out, scratch, c.n, c.rank, c.T); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		if err := d.PCtrlDown([]Ink2Ref{cr}, []Ink2Ref{mr}, []Ink2Ref{dr}, []Ink2Ref{wantG}, c.n, c.rank, c.T); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		if err := d.PCtrlApply([]Ink2Ref{cr}, []Ink2Ref{mr}, []Ink2Ref{wantG}, []Ink2Ref{gr}, []Ink2Ref{fr}, []Ink2Ref{want}, c.n, c.rank, c.T); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		gotG, wg := ink2OpsTestRead(t, d, g, c.T*c.rank), ink2OpsTestRead(t, d, wantG, c.T*c.rank)
		got, w := ink2OpsTestRead(t, d, out, c.T*c.n), ink2OpsTestRead(t, d, want, c.T*c.n)
		for tok := range c.T {
			where := fmt.Sprintf("%s: row %d", what, tok)
			ink2PGemmClose(t, where+" g", gotG[tok*c.rank:][:c.rank], wg[tok*c.rank:][:c.rank])
			ink2PGemmClose(t, where, got[tok*c.n:][:c.n], w[tok*c.n:][:c.n])
			host, mag := ink2OpsTestHostControl(cur[tok*c.n:][:c.n], msg[tok*c.n:][:c.n], down, gain, forget, c.rank)
			for i, h := range host {
				bound := 1e-5*mag[i] + 1e-6
				d := math.Abs(float64(got[tok*c.n+i]) - float64(h))
				worst = max(worst, d/bound)
				if d > 2*bound {
					t.Fatalf("%s vs host[%d] = %v, want %v (magnitude %v)", where, i, got[tok*c.n+i], h, mag[i])
				}
			}
		}
	}
	t.Logf("worst drift is %.3g of the decode bound", worst)
}

// fak-test:runtime integration est=3s lane=default
func TestInk2PFlashAttendMatchesAttendAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	mini := Ink2AttnParams{Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16, Scale: float32(1 / math.Sqrt(128))}
	bare := Ink2AttnParams{Heads: 6, KVHeads: 2, HeadDim: 128, Scale: float32(1 / math.Sqrt(128))}
	rng := rand.New(rand.NewSource(94))
	for _, c := range []struct {
		p                   Ink2AttnParams
		cap, window, extent int
		pos0, T             int
	}{
		// a first chunk, a local ring filling, cut by its window and wrapped
		{p: mini, cap: 1024 + 64, window: 1024, extent: 1024, pos0: 0, T: 41},
		{p: mini, cap: 1024 + 64, window: 1024, extent: 1024, pos0: 1000, T: 33},
		{p: mini, cap: 1024 + 64, window: 1024, extent: 1024, pos0: 2050, T: 64},
		// a global slot whose history outruns the bias table
		{p: mini, cap: 4096, extent: 1024, pos0: 1500, T: 37},
		{p: bare, cap: 256, pos0: 5, T: 50},
	} {
		what := fmt.Sprintf("heads %d/%d rel %d cap %d window %d extent %d pos0 %d rows %d",
			c.p.Heads, c.p.KVHeads, c.p.RelLen, c.cap, c.window, c.extent, c.pos0, c.T)
		p := c.p
		qn, rn, kvDim := p.Heads*p.HeadDim, p.Heads*p.RelLen, p.KVHeads*p.HeadDim
		q, r := ink2OpsTestRand(rng, c.T*qn, 1), ink2OpsTestRand(rng, c.T*rn, 1)
		sinks := ink2OpsTestRand(rng, c.T*p.Heads, 1)
		rel := ink2OpsTestRand(rng, c.extent*p.RelLen, .1)
		kc, vc := ink2OpsTestRand(rng, c.cap*kvDim, .3), ink2OpsTestRand(rng, c.cap*kvDim, 1)
		qr, sr := ink2OpsTestPut(t, d, 0, q), ink2OpsTestPut(t, d, 0, sinks)
		rr, relr := ink2TestAttendUpload(t, d, r), ink2TestAttendUpload(t, d, rel)
		kcr, vcr := ink2OpsTestPut(t, d, 0, kc), ink2OpsTestPut(t, d, 3, vc)
		serial, flash := ink2TestAlloc(t, d, c.T*qn), ink2TestAlloc(t, d, c.T*qn)
		scores, meta := ink2TestAlloc(t, d, p.Heads*c.cap), ink2TestAlloc(t, d, 4)
		for tok := range c.T {
			ink2PTestMeta(t, d, meta, c.pos0+tok)
			if err := d.Attend(qr.At(tok*qn), rr.At(tok*rn), sr.At(tok*p.Heads), relr, kcr, vcr, scores, serial.At(tok*qn),
				meta, p, c.extent, c.window, c.cap); err != nil {
				t.Fatalf("%s: %v", what, err)
			}
			if err := d.Flush(); err != nil {
				t.Fatal(err)
			}
		}
		if ok, err := d.PFlashAttend(qr, rr, sr, relr, kcr, vcr, flash, p, c.extent, c.window, c.cap, c.pos0, c.T); err != nil || !ok {
			t.Fatalf("%s: flash attend: ok %v, %v", what, ok, err)
		}
		got, want := ink2OpsTestRead(t, d, flash, c.T*qn), ink2OpsTestRead(t, d, serial, c.T*qn)
		for tok := range c.T {
			ink2PGemmClose(t, fmt.Sprintf("%s: row %d", what, tok), got[tok*qn:][:qn], want[tok*qn:][:qn])
			host, mag := ink2TestAttendHost(p, q[tok*qn:][:qn], r[tok*rn:][:rn], sinks[tok*p.Heads:][:p.Heads], rel, kc, vc,
				c.pos0+tok, c.extent, c.window, c.cap)
			ink2TestAttendCheck(t, fmt.Sprintf("%s: row %d vs host", what, tok), got[tok*qn:][:qn], host, mag)
		}
	}
}
