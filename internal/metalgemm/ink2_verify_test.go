//go:build darwin && arm64 && cgo

package metalgemm

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// The verify widths every op is proven invariant over: a row's bits must not
// depend on how many rows share its launch.
var ink2VTestWidths = []int{1, 2, 4, 8}

func ink2VTestPut(t *testing.T, d *Ink2Device, vals []float32) Ink2Ref {
	t.Helper()
	r := ink2TestAlloc(t, d, max(len(vals), 1))
	if err := d.Write(r, vals); err != nil {
		t.Fatal(err)
	}
	return r
}

func ink2VTestInts(vals ...int) []float32 {
	out := make([]float32, len(vals))
	for i, v := range vals {
		out[i] = math.Float32frombits(uint32(int32(v)))
	}
	return out
}

func ink2VTestFinite(t *testing.T, what string, v []float32) {
	t.Helper()
	for i, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			t.Fatalf("%s[%d] = %v: the fixture must stay finite", what, i, x)
		}
	}
}

// ink2VTestWeight writes rows x cols weights of a GGML type as random block
// bytes with sane scales and returns its view and the first byte after it.
func ink2VTestWeight(t *testing.T, store *PackedGGUFStore, data []byte, rng *rand.Rand, offset int, typ uint32, rows, cols int) (Ink2Weight, int) {
	t.Helper()
	block, blockBytes, err := packedBlock(typ)
	if err != nil {
		t.Fatal(err)
	}
	half := func() uint16 { return uint16(0x3000 + rng.Intn(1024)) }
	n := rows * cols / block * blockBytes
	raw := data[offset : offset+n]
	switch typ {
	case 0:
		for i := 0; i < n; i += 4 {
			binary.LittleEndian.PutUint32(raw[i:], math.Float32bits(rng.Float32()-.5))
		}
	case 1:
		for i := 0; i < n; i += 2 {
			binary.LittleEndian.PutUint16(raw[i:], half()|uint16(rng.Intn(2))<<15)
		}
	case 30:
		for i := 0; i < n; i += 2 {
			binary.LittleEndian.PutUint16(raw[i:], uint16(math.Float32bits(rng.Float32()-.5)>>16))
		}
	default:
		rng.Read(raw)
		// Byte offsets of the half-float scales inside one block.
		scales := map[uint32][]int{8: {0}, 11: {108}, 12: {0, 2}, 13: {0, 2}, 14: {208}, 17: {0}, 18: {0}, 20: {0}, 21: {0}, 22: {0}, 40: nil}
		at, ok := scales[typ]
		if !ok {
			t.Fatalf("no synthesizer for GGML type %d", typ)
		}
		for b := 0; b < n; b += blockBytes {
			for _, o := range at {
				binary.LittleEndian.PutUint16(raw[b+o:], half())
			}
		}
	}
	m, err := store.Matrix(offset, typ, rows, cols, 1)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.Ink2Weight(0)
	if err != nil {
		t.Fatal(err)
	}
	return w, (offset + n + 31) &^ 31
}

// fak-test:runtime integration est=10s lane=default
func TestInk2VRowsAreWidthInvariantAndMatchMatVec(t *testing.T) {
	d := ink2TestDevice(t)
	store, data := ink2TestStore(t, 64)
	rng := rand.New(rand.NewSource(71))
	const rows, cols, maxT = 13, 512, 19
	const xStride, yStride = cols + 64, rows + 3
	x := ink2OpsTestRand(rng, maxT*xStride, 1)
	xr := ink2VTestPut(t, d, x)
	for _, c := range []struct {
		name string
		typ  uint32
	}{{"f32", 0}, {"f16", 1}, {"bf16", 30}, {"q8_0", 8}, {"q3_K", 11}, {"q4_K", 12}, {"q5_K", 13}, {"q6_K", 14},
		{"iq2_xs", 17}, {"iq3_xxs", 18}, {"iq4_nl", 20}, {"iq3_s", 21}, {"iq2_s", 22}, {"nvfp4", 40}} {
		t.Run(c.name, func(t *testing.T) {
			w, _ := ink2VTestWeight(t, store, data, rng, 64, c.typ, rows, cols)
			// The decode GEMV of each row, and that row verified alone.
			decode, alone := ink2TestAlloc(t, d, maxT*rows), ink2TestAlloc(t, d, maxT*rows)
			for r := range maxT {
				if err := d.MatVec(w, xr.At(r*xStride), decode.At(r*rows)); err != nil {
					t.Fatal(err)
				}
				if err := d.VerifyRows(w, xr.At(r*xStride), xStride, alone.At(r*rows), rows, 1); err != nil {
					t.Fatal(err)
				}
			}
			want := ink2OpsTestRead(t, d, decode, maxT*rows)
			ink2VTestFinite(t, "decode", want)
			ink2OpsTestSame(t, "a row verified alone vs the decode GEMV", ink2OpsTestRead(t, d, alone, maxT*rows), want)
			for _, T := range append([]int{3, 5, 6, 7, 9, 10, 11, 12, maxT}, ink2VTestWidths...) {
				y := ink2VTestPut(t, d, make([]float32, T*yStride))
				if err := d.VerifyRows(w, xr, xStride, y, yStride, T); err != nil {
					t.Fatal(err)
				}
				got := ink2OpsTestRead(t, d, y, T*yStride)
				for r := range T {
					ink2OpsTestSame(t, fmt.Sprintf("width %d row %d", T, r), got[r*yStride:r*yStride+rows], want[r*rows:(r+1)*rows])
					for i := rows; i < yStride; i++ {
						if got[r*yStride+i] != 0 {
							t.Fatalf("width %d row %d wrote past its %d outputs", T, r, rows)
						}
					}
				}
			}
		})
	}
}

// fak-test:runtime integration est=5s lane=default
func TestInk2VExpertsMatchExpertMatVecPerPair(t *testing.T) {
	d := ink2TestDevice(t)
	store, data := ink2TestStore(t, 128)
	rng := rand.New(rand.NewSource(73))
	const rows, cols, nexp, k, maxT = 13, 512, 6, 3, 11
	const xStride = cols + 32
	x := ink2OpsTestRand(rng, maxT*k*xStride, 1)
	xr := ink2VTestPut(t, d, x)
	scale := make([]float32, nexp)
	for i := range scale {
		scale[i] = 0.5 + rng.Float32()
	}
	sr := ink2VTestPut(t, d, scale)
	// k distinct experts a token, so an expert holds at most one pair a token
	// and up to maxT pairs a chunk, more than the kernel's tile.
	ids := make([]int, 0, maxT*k)
	for range maxT {
		perm := rng.Perm(nexp)
		if rng.Intn(3) == 0 {
			perm = []int{0, 1, 2}
		}
		ids = append(ids, perm[:k]...)
	}
	ir := ink2VTestPut(t, d, ink2VTestInts(ids...))
	tab := ink2TestAlloc(t, d, VerifyExpertsTable(len(ids)))
	for _, c := range []struct {
		name string
		typ  uint32
	}{{"q8_0", 8}, {"q3_K", 11}, {"q4_K", 12}, {"q5_K", 13}, {"q6_K", 14},
		{"iq2_xs", 17}, {"iq3_xxs", 18}, {"iq4_nl", 20}, {"iq3_s", 21}, {"iq2_s", 22}, {"nvfp4", 40}} {
		t.Run(c.name, func(t *testing.T) {
			ink2VTestWeight(t, store, data, rng, 64, c.typ, rows*nexp, cols)
			m, err := store.Matrix(64, c.typ, rows, cols, nexp)
			if err != nil {
				t.Fatal(err)
			}
			w, ok := m.Ink2Stack()
			if !ok {
				t.Fatal("expert stack split across buffers")
			}
			for _, xDiv := range []int{1, k} {
				for _, T := range []int{1, 2, 5, 8, maxT} {
					npairs := T * k
					// The decode expert GEMV over each pair's own input row.
					rep := make([]float32, npairs*xStride)
					for i := range npairs {
						copy(rep[i*xStride:(i+1)*xStride], x[(i/xDiv)*xStride:(i/xDiv+1)*xStride])
					}
					repr := ink2VTestPut(t, d, rep)
					want, got := ink2TestAlloc(t, d, npairs*rows), ink2TestAlloc(t, d, npairs*rows)
					if _, err := d.ExpertMatVec(w, ir, repr, xStride, want, npairs, sr, nil); err != nil {
						t.Fatal(err)
					}
					ok, err := d.VerifyExperts(w, ir, xr, xDiv, xStride, got, npairs, sr, tab)
					if err != nil || !ok {
						t.Fatalf("verify experts: ok=%v err=%v", ok, err)
					}
					wv := ink2OpsTestRead(t, d, want, npairs*rows)
					ink2VTestFinite(t, "decode experts", wv)
					ink2OpsTestSame(t, fmt.Sprintf("xDiv %d width %d", xDiv, T), ink2OpsTestRead(t, d, got, npairs*rows), wv)
				}
			}
		})
	}
	if ok, err := d.VerifyExperts(Ink2Weight{Type: 30}, ir, xr, 1, xStride, xr, 1, Ink2Ref{}, tab); ok || err != nil {
		t.Fatalf("a bf16 stack: ok=%v err=%v, want a decline", ok, err)
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2VRouteAndCombineRowsMatchPerToken(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(79))
	const nexp, k, n, maxT = 128, 8, 96, 9
	logits := ink2OpsTestRand(rng, maxT*nexp, 1.5)
	lr, br := ink2VTestPut(t, d, logits), ink2VTestPut(t, d, ink2OpsTestRand(rng, nexp, 0.2))
	down := ink2VTestPut(t, d, ink2OpsTestRand(rng, maxT*k*n, 1))
	for _, T := range []int{1, 2, 5, maxT} {
		ids, wts := ink2TestAlloc(t, d, T*k), ink2TestAlloc(t, d, T*k)
		oneIds, oneWts := ink2TestAlloc(t, d, T*k), ink2TestAlloc(t, d, T*k)
		out, one := ink2TestAlloc(t, d, T*n), ink2TestAlloc(t, d, T*n)
		if err := d.RouteRows(lr, br, ids, wts, nexp, k, 2.5, T); err != nil {
			t.Fatal(err)
		}
		if err := d.CombineRows(down, wts, out, k, n, T); err != nil {
			t.Fatal(err)
		}
		for r := range T {
			if err := d.Route(lr.At(r*nexp), br, oneIds.At(r*k), oneWts.At(r*k), nexp, k, 2.5); err != nil {
				t.Fatal(err)
			}
			if err := d.Combine(down.At(r*k*n), oneWts.At(r*k), one.At(r*n), k, n); err != nil {
				t.Fatal(err)
			}
		}
		gotIds, err := d.ReadInts(ids, T*k)
		if err != nil {
			t.Fatal(err)
		}
		wantIds, err := d.ReadInts(oneIds, T*k)
		if err != nil {
			t.Fatal(err)
		}
		for i := range wantIds {
			if gotIds[i] != wantIds[i] {
				t.Fatalf("width %d: slot %d routes to %d, alone %d", T, i, gotIds[i], wantIds[i])
			}
		}
		ink2OpsTestSame(t, fmt.Sprintf("width %d route weights", T), ink2OpsTestRead(t, d, wts, T*k), ink2OpsTestRead(t, d, oneWts, T*k))
		ink2OpsTestSame(t, fmt.Sprintf("width %d combine", T), ink2OpsTestRead(t, d, out, T*n), ink2OpsTestRead(t, d, one, T*n))
	}
}

// fak-test:runtime integration est=3s lane=default
func TestInk2VMixIsWidthInvariantAndMatchesMix(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(72))
	const n, ns, nq, maxT = 300, 5, 3, 8
	const eps = 1e-6
	states := make([]Ink2Ref, ns)
	for s := range states {
		states[s] = ink2VTestPut(t, d, ink2OpsTestRand(rng, maxT*n, float64(1+s)))
	}
	queries := ink2VTestPut(t, d, ink2OpsTestRand(rng, nq*n, 1))
	for _, c := range []struct {
		name  string
		norms []Ink2Ref
	}{
		{"plain", nil},
		{"folded norm", []Ink2Ref{ink2VTestPut(t, d, ink2OpsTestRand(rng, n, 1)), {}, ink2VTestPut(t, d, ink2OpsTestRand(rng, n, 1))}},
	} {
		t.Run(c.name, func(t *testing.T) {
			ref := ink2TestAlloc(t, d, maxT*nq*n)
			for r := range maxT {
				row := make([]Ink2Ref, ns)
				for s := range row {
					row[s] = states[s].At(r * n)
				}
				if err := d.Mix(row, queries, nq, c.norms, eps, ref.At(r*nq*n), n); err != nil {
					t.Fatal(err)
				}
			}
			want := ink2OpsTestRead(t, d, ref, maxT*nq*n)
			ink2VTestFinite(t, "mix", want)
			for _, T := range ink2VTestWidths {
				out := ink2TestAlloc(t, d, nq*T*n)
				if err := d.VerifyMix(states, queries, nq, c.norms, eps, out, n, T); err != nil {
					t.Fatal(err)
				}
				got := ink2OpsTestRead(t, d, out, nq*T*n)
				for q := range nq {
					for r := range T {
						ink2OpsTestSame(t, fmt.Sprintf("width %d query %d row %d", T, q, r),
							got[(q*T+r)*n:(q*T+r+1)*n], want[(r*nq+q)*n:(r*nq+q+1)*n])
					}
				}
			}
		})
	}
}

// fak-test:runtime integration est=3s lane=default
func TestInk2VControlIsWidthInvariantAndMatchesControl(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(73))
	const n, rank, nj, maxT = 300, 6, 3, 8
	var ctl Ink2Controls
	for range nj {
		ctl.Cur = append(ctl.Cur, ink2VTestPut(t, d, ink2OpsTestRand(rng, maxT*n, 1)))
		ctl.Msg = append(ctl.Msg, ink2VTestPut(t, d, ink2OpsTestRand(rng, maxT*n, 2)))
		ctl.Down = append(ctl.Down, ink2VTestPut(t, d, ink2OpsTestRand(rng, rank*2*n, .1)))
		ctl.Gain = append(ctl.Gain, ink2VTestPut(t, d, ink2OpsTestRand(rng, n*rank, .3)))
		ctl.Forget = append(ctl.Forget, ink2VTestPut(t, d, ink2OpsTestRand(rng, n*rank, .3)))
	}
	refG, refOut := ink2TestAlloc(t, d, maxT*nj*rank), ink2TestAlloc(t, d, maxT*nj*n)
	for r := range maxT {
		row := Ink2Controls{Down: ctl.Down, Gain: ctl.Gain, Forget: ctl.Forget}
		for j := range nj {
			row.Cur, row.Msg = append(row.Cur, ctl.Cur[j].At(r*n)), append(row.Msg, ctl.Msg[j].At(r*n))
		}
		if err := d.Control(row, refG.At(r*nj*rank), refOut.At(r*nj*n), n, rank, nil, 0); err != nil {
			t.Fatal(err)
		}
	}
	wantG, wantOut := ink2OpsTestRead(t, d, refG, maxT*nj*rank), ink2OpsTestRead(t, d, refOut, maxT*nj*n)
	ink2VTestFinite(t, "control", wantOut)
	for _, T := range ink2VTestWidths {
		// One contiguous buffer per output, as the driver lays them out.
		gs, outs := ink2TestAlloc(t, d, nj*T*rank), ink2TestAlloc(t, d, nj*T*n)
		g, out := make([]Ink2Ref, nj), make([]Ink2Ref, nj)
		for j := range nj {
			g[j], out[j] = gs.At(j*T*rank), outs.At(j*T*n)
		}
		if err := d.VerifyControl(ctl, g, out, n, rank, T); err != nil {
			t.Fatal(err)
		}
		gotG, gotOut := ink2OpsTestRead(t, d, gs, nj*T*rank), ink2OpsTestRead(t, d, outs, nj*T*n)
		for j := range nj {
			for r := range T {
				what := fmt.Sprintf("width %d controller %d row %d", T, j, r)
				ink2OpsTestSame(t, what+" g", gotG[(j*T+r)*rank:(j*T+r+1)*rank], wantG[(r*nj+j)*rank:(r*nj+j+1)*rank])
				ink2OpsTestSame(t, what+" out", gotOut[(j*T+r)*n:(j*T+r+1)*n], wantOut[(r*nj+j)*n:(r*nj+j+1)*n])
			}
		}
	}
}

type ink2VTestConv struct {
	name            string
	n, heads, kn    int
	dynamic, resid  bool
	x, w, dyn, hist []float32
	res             []float32
	norm            float32
}

func (c *ink2VTestConv) fill(rng *rand.Rand, maxT int) {
	c.x, c.w = ink2OpsTestRand(rng, maxT*c.n, 1), ink2OpsTestRand(rng, c.n*c.kn, .5)
	c.hist = ink2OpsTestRand(rng, c.n*(c.kn-1), 1)
	c.norm = float32(1 / math.Sqrt(float64(c.kn+1)))
	if c.dynamic {
		c.dyn = ink2OpsTestRand(rng, maxT*c.heads*c.kn, .5)
		c.norm = float32(1 / math.Sqrt(float64(2*c.kn+1)))
	}
	if c.resid {
		c.res = ink2OpsTestRand(rng, maxT*c.n, 1)
	}
}

// serial runs the decode conv token by token and returns the outputs and the
// history after each token count 0..maxT.
func (c *ink2VTestConv) serial(t *testing.T, d *Ink2Device, maxT int) (out []float32, hists [][]float32) {
	t.Helper()
	x, w, hist := ink2VTestPut(t, d, c.x), ink2VTestPut(t, d, c.w), ink2VTestPut(t, d, c.hist)
	var dyn, res Ink2Ref
	if c.dynamic {
		dyn = ink2VTestPut(t, d, c.dyn)
	}
	if c.resid {
		res = ink2VTestPut(t, d, c.res)
	}
	o := ink2TestAlloc(t, d, maxT*c.n)
	past := c.kn - 1
	hists = append(hists, c.hist)
	for r := range maxT {
		var dr, rr Ink2Ref
		if c.dynamic {
			dr = dyn.At(r * c.heads * c.kn)
		}
		if c.resid {
			rr = res.At(r * c.n)
		}
		if err := d.Conv(x.At(r*c.n), w, dr, hist, o.At(r*c.n), rr, c.n, c.heads, c.kn, c.norm); err != nil {
			t.Fatal(err)
		}
		hists = append(hists, ink2OpsTestRead(t, d, hist, c.n*past))
	}
	return ink2OpsTestRead(t, d, o, maxT*c.n), hists
}

// fak-test:runtime integration est=4s lane=default
func TestInk2VConvMatchesSerialConvAndCommitRewinds(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(74))
	const maxT = 8
	for _, c := range []ink2VTestConv{
		{name: "dynamic taps with a residual", n: 24, heads: 4, kn: 4, dynamic: true, resid: true},
		{name: "static taps", n: 300, heads: 1, kn: 4},
		{name: "kernel of two", n: 9, heads: 3, kn: 2, dynamic: true},
		{name: "widest kernel", n: 16, heads: 2, kn: 8, resid: true},
	} {
		t.Run(c.name, func(t *testing.T) {
			c.fill(rng, maxT)
			want, hists := c.serial(t, d, maxT)
			ink2VTestFinite(t, "conv", want)
			past := c.kn - 1
			for _, T := range ink2VTestWidths {
				L := past + T
				x, w, hist := ink2VTestPut(t, d, c.x), ink2VTestPut(t, d, c.w), ink2VTestPut(t, d, c.hist)
				var dyn, res Ink2Ref
				if c.dynamic {
					dyn = ink2VTestPut(t, d, c.dyn)
				}
				if c.resid {
					res = ink2VTestPut(t, d, c.res)
				}
				out, log := ink2TestAlloc(t, d, T*c.n), ink2TestAlloc(t, d, c.n*L)
				if err := d.VerifyConv(x, w, dyn, hist, log, out, res, c.n, c.heads, c.kn, c.norm, T); err != nil {
					t.Fatal(err)
				}
				ink2OpsTestSame(t, fmt.Sprintf("width %d outputs", T), ink2OpsTestRead(t, d, out, T*c.n), want[:T*c.n])
				ink2OpsTestSame(t, fmt.Sprintf("width %d final history", T), ink2OpsTestRead(t, d, hist, c.n*past), hists[T])
				gotLog := ink2OpsTestRead(t, d, log, c.n*L)
				for ch := range c.n {
					row := append(append([]float32(nil), c.hist[ch*past:(ch+1)*past]...), make([]float32, T)...)
					for r := range T {
						row[past+r] = c.x[r*c.n+ch]
					}
					ink2OpsTestSame(t, fmt.Sprintf("width %d log of channel %d", T, ch), gotLog[ch*L:(ch+1)*L], row)
				}
				// Without a log the op is the same conv.
				hist2, out2 := ink2VTestPut(t, d, c.hist), ink2TestAlloc(t, d, T*c.n)
				if err := d.VerifyConv(x, w, dyn, hist2, Ink2Ref{}, out2, res, c.n, c.heads, c.kn, c.norm, T); err != nil {
					t.Fatal(err)
				}
				ink2OpsTestSame(t, fmt.Sprintf("width %d unlogged outputs", T), ink2OpsTestRead(t, d, out2, T*c.n), want[:T*c.n])
				for keep := 0; keep <= T; keep++ {
					if err := d.VerifyConvCommit(hist, log, Ink2Ref{}, c.n, past, L, keep); err != nil {
						t.Fatal(err)
					}
					ink2OpsTestSame(t, fmt.Sprintf("width %d commit of %d", T, keep), ink2OpsTestRead(t, d, hist, c.n*past), hists[keep])
					keepDev := ink2VTestPut(t, d, append(make([]float32, 3), ink2VTestInts(keep)...))
					if err := d.Zero(hist, c.n*past); err != nil {
						t.Fatal(err)
					}
					// The host keep is ignored when the device one is bound.
					if err := d.VerifyConvCommit(hist, log, keepDev.At(3), c.n, past, L, -1); err != nil {
						t.Fatal(err)
					}
					ink2OpsTestSame(t, fmt.Sprintf("width %d device commit of %d", T, keep), ink2OpsTestRead(t, d, hist, c.n*past), hists[keep])
				}
			}
		})
	}
}

// fak-test:runtime integration est=2s lane=default
func TestInk2VConvPairMatchesTwoConvs(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(75))
	const n, heads, kn, maxT = 40, 4, 4, 8
	past := kn - 1
	convs := []ink2VTestConv{{n: n, heads: heads, kn: kn, dynamic: true}, {n: n, heads: heads, kn: kn}}
	var want [2][]float32
	var hists [2][][]float32
	for j := range convs {
		convs[j].fill(rng, maxT)
		want[j], hists[j] = convs[j].serial(t, d, maxT)
	}
	for _, T := range ink2VTestWidths {
		L := past + T
		var x, w, dyn, hist, log, out []Ink2Ref
		// One history and one log hold both convs' channels, as a slot's do.
		histBoth := ink2VTestPut(t, d, append(append([]float32(nil), convs[0].hist...), convs[1].hist...))
		logBoth := ink2TestAlloc(t, d, 2*n*L)
		for j, c := range convs {
			x, w = append(x, ink2VTestPut(t, d, c.x)), append(w, ink2VTestPut(t, d, c.w))
			var dr Ink2Ref
			if c.dynamic {
				dr = ink2VTestPut(t, d, c.dyn)
			}
			dyn = append(dyn, dr)
			hist, log = append(hist, histBoth.At(j*n*past)), append(log, logBoth.At(j*n*L))
			out = append(out, ink2TestAlloc(t, d, T*n))
		}
		if err := d.VerifyConvPair(x, w, dyn, hist, log, out, []float32{convs[0].norm, convs[1].norm}, n, heads, kn, T); err != nil {
			t.Fatal(err)
		}
		gotHist, gotLog := ink2OpsTestRead(t, d, histBoth, 2*n*past), ink2OpsTestRead(t, d, logBoth, 2*n*L)
		for j, c := range convs {
			what := fmt.Sprintf("width %d conv %d", T, j)
			ink2OpsTestSame(t, what+" outputs", ink2OpsTestRead(t, d, out[j], T*n), want[j][:T*n])
			ink2OpsTestSame(t, what+" history", gotHist[j*n*past:(j+1)*n*past], hists[j][T])
			for ch := range n {
				for r := range T {
					if got := gotLog[j*n*L+ch*L+past+r]; math.Float32bits(got) != math.Float32bits(c.x[r*n+ch]) {
						t.Fatalf("%s log[%d][%d] = %v, want the input %v", what, ch, past+r, got, c.x[r*n+ch])
					}
				}
			}
		}
	}
}

// fak-test:runtime integration est=4s lane=default
func TestInk2VQKVPostMatchesSerialQKVPost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(76))
	const maxT = 8
	for _, c := range []struct {
		name      string
		p         Ink2AttnParams
		cap, pos0 int
	}{
		{"global", Ink2AttnParams{Heads: 6, KVHeads: 2, HeadDim: 40, RelLen: 20, QKClip: 2, RelClip: 1.5, Eps: 1e-6, Global: true, Alpha: .35, Floor: 3}, 1100, 37},
		{"local ring that wraps", Ink2AttnParams{Heads: 4, KVHeads: 4, HeadDim: 64, RelLen: 33, QKClip: 1.5, RelClip: 2, Eps: 1e-5}, 11, 7},
		{"no relative states at position zero", Ink2AttnParams{Heads: 4, KVHeads: 2, HeadDim: 16, QKClip: .7, RelClip: 1, Eps: 1e-6, Global: true, Alpha: .5, Floor: 2}, 9, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := c.p
			qn, rn, kvDim := p.Heads*p.HeadDim, p.Heads*p.RelLen, p.KVHeads*p.HeadDim
			q, r := ink2OpsTestRand(rng, maxT*qn, 1), ink2OpsTestRand(rng, maxT*rn, 1)
			// Head amplitudes straddle the clip thresholds.
			for i := range q {
				q[i] *= .15 * float32(1+i/p.HeadDim%p.Heads)
			}
			k, v := ink2OpsTestRand(rng, maxT*kvDim, 1), ink2OpsTestRand(rng, maxT*kvDim, 1)
			qclip, rclip := ink2VTestPut(t, d, ink2OpsTestRand(rng, qn, 1.5)), ink2VTestPut(t, d, ink2OpsTestRand(rng, max(rn, 1), 1.5))
			kc0, vc0 := ink2OpsTestRand(rng, c.cap*kvDim, 9), ink2OpsTestRand(rng, c.cap*kvDim, 9)
			positions := make([]int, maxT)
			for i := range positions {
				positions[i] = c.pos0 + i
			}
			metas := ink2VTestPut(t, d, ink2VTestInts(positions...))
			type state struct{ q, r, kc, vc Ink2Ref }
			fresh := func() state {
				return state{ink2VTestPut(t, d, q), ink2VTestPut(t, d, r), ink2VTestPut(t, d, kc0), ink2VTestPut(t, d, vc0)}
			}
			kr, vr := ink2VTestPut(t, d, k), ink2VTestPut(t, d, v)
			read := func(s state) [4][]float32 {
				return [4][]float32{ink2OpsTestRead(t, d, s.q, maxT*qn), ink2OpsTestRead(t, d, s.r, max(maxT*rn, 1)),
					ink2OpsTestRead(t, d, s.kc, c.cap*kvDim), ink2OpsTestRead(t, d, s.vc, c.cap*kvDim)}
			}
			for _, T := range ink2VTestWidths {
				serial := fresh()
				for i := range T {
					if err := d.QKVPost(serial.q.At(i*qn), serial.r.At(i*rn), kr.At(i*kvDim), vr.At(i*kvDim), qclip, rclip,
						serial.kc, serial.vc, metas.At(i), p, c.cap, nil); err != nil {
						t.Fatal(err)
					}
				}
				want := read(serial)
				ink2VTestFinite(t, "q", want[0])
				for _, form := range []string{"host position", "device position"} {
					s := fresh()
					var posDev Ink2Ref
					pos0 := c.pos0
					if form == "device position" {
						posDev, pos0 = metas, 0
					}
					if err := d.VerifyQKVPost(s.q, s.r, kr, vr, qclip, rclip, s.kc, s.vc, posDev, p, c.cap, pos0, T); err != nil {
						t.Fatal(err)
					}
					got := read(s)
					for i, name := range []string{"q", "r", "k cache", "v cache"} {
						ink2OpsTestSame(t, fmt.Sprintf("width %d, %s: %s", T, form, name), got[i], want[i])
					}
				}
			}
		})
	}
}

// fak-test:runtime integration est=5s lane=default
func TestInk2VAttendIsWidthInvariantAndMatchesAttend(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(77))
	const maxT = 8
	p := Ink2AttnParams{Heads: 6, KVHeads: 3, HeadDim: 32, RelLen: 16, Scale: float32(1 / math.Sqrt(32))}
	odd := Ink2AttnParams{Heads: 4, KVHeads: 2, HeadDim: 30, RelLen: 5, Scale: .2}
	for _, c := range []struct {
		name                      string
		p                         Ink2AttnParams
		cap, window, extent, pos0 int
	}{
		{"global", p, 200, 0, 64, 60},
		{"local ring that wraps", p, 48 + maxT, 48, 48, 130},
		{"first tokens", p, 64, 0, 16, 0},
		{"ragged head", odd, 40, 12, 20, 17},
	} {
		t.Run(c.name, func(t *testing.T) {
			pp := c.p
			n, rn, kvDim := pp.Heads*pp.HeadDim, pp.Heads*pp.RelLen, pp.KVHeads*pp.HeadDim
			q := ink2VTestPut(t, d, ink2OpsTestRand(rng, maxT*n, 1))
			r := ink2VTestPut(t, d, ink2OpsTestRand(rng, maxT*rn, 1))
			sinks := ink2VTestPut(t, d, ink2OpsTestRand(rng, maxT*pp.Heads, 1))
			rel := ink2VTestPut(t, d, ink2OpsTestRand(rng, c.extent*pp.RelLen, .1))
			kc, vc := ink2VTestPut(t, d, ink2OpsTestRand(rng, c.cap*kvDim, .3)), ink2VTestPut(t, d, ink2OpsTestRand(rng, c.cap*kvDim, 1))
			positions := make([]int, maxT)
			for i := range positions {
				positions[i] = c.pos0 + i
			}
			metas := ink2VTestPut(t, d, ink2VTestInts(positions...))
			ref, scratch := ink2TestAlloc(t, d, maxT*n), ink2TestAlloc(t, d, pp.Heads*c.cap)
			for i := range maxT {
				if err := d.Attend(q.At(i*n), r.At(i*rn), sinks.At(i*pp.Heads), rel, kc, vc, scratch, ref.At(i*n), metas.At(i),
					pp, c.extent, c.window, c.cap); err != nil {
					t.Fatal(err)
				}
			}
			want := ink2OpsTestRead(t, d, ref, maxT*n)
			ink2VTestFinite(t, "attend", want)
			for _, T := range ink2VTestWidths {
				for _, form := range []string{"host position", "device position", "one row per launch"} {
					var posDev Ink2Ref
					pos0 := c.pos0
					if form != "host position" {
						posDev, pos0 = metas, 0
					}
					floats := T * pp.Heads * VerifyAttendSpan(posDev.Valid(), c.window, c.cap, pos0, T)
					if form == "one row per launch" {
						floats /= T
					}
					scores, out := ink2TestAlloc(t, d, floats), ink2TestAlloc(t, d, T*n)
					if err := d.VerifyAttend(q, r, sinks, rel, kc, vc, scores, floats, out, posDev, pp, c.extent, c.window, c.cap, pos0, T); err != nil {
						t.Fatal(err)
					}
					ink2OpsTestSame(t, fmt.Sprintf("width %d, %s", T, form), ink2OpsTestRead(t, d, out, T*n), want[:T*n])
				}
			}
		})
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2VRepeatRowsAndEncodeOnly(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(78))
	const n, rows, k = 70, 9, 8
	x := ink2OpsTestRand(rng, rows*n, 1)
	xr, out := ink2VTestPut(t, d, x), ink2TestAlloc(t, d, rows*k*n)
	before := d.Stats()
	if err := d.VerifyRepeatRows(xr, out, n, rows, k); err != nil {
		t.Fatal(err)
	}
	if s := d.Stats(); s.Submissions != before.Submissions || s.Pending != 1 {
		t.Fatalf("repeat rows submitted instead of encoding: %+v after %+v", s, before)
	}
	got := ink2OpsTestRead(t, d, out, rows*k*n)
	for r := range rows {
		for j := range k {
			ink2OpsTestSame(t, fmt.Sprintf("row %d copy %d", r, j), got[(r*k+j)*n:(r*k+j+1)*n], x[r*n:(r+1)*n])
		}
	}
}

func TestInk2VOpsRejectBadOperands(t *testing.T) {
	d := ink2TestDevice(t)
	a, b := ink2TestAlloc(t, d, 4096), ink2TestAlloc(t, d, 4096)
	p := Ink2AttnParams{Heads: 2, KVHeads: 1, HeadDim: 8, RelLen: 4, Scale: 1}
	for name, err := range map[string]error{
		"conv output aliasing its input": d.VerifyConv(a, b, Ink2Ref{}, b.At(512), Ink2Ref{}, a, Ink2Ref{}, 16, 1, 4, 1, 2),
		"conv kernel past the limit":     d.VerifyConv(a, b, Ink2Ref{}, b.At(512), Ink2Ref{}, a.At(1024), Ink2Ref{}, 16, 1, 9, 1, 2),
		"commit past the log row":        d.VerifyConvCommit(a, b, Ink2Ref{}, 16, 3, 5, 3),
		"more rows than cache slots":     d.VerifyQKVPost(a, a.At(256), b, b.At(256), a.At(512), a.At(768), b.At(1024), b.At(2048), Ink2Ref{}, p, 3, 0, 4),
		"attend without a scores row":    d.VerifyAttend(a, a.At(256), a.At(512), a.At(768), b, b.At(1024), b.At(2048), 3, a.At(2048), Ink2Ref{}, p, 4, 0, 8, 0, 2),
		"five controllers":               d.VerifyControl(Ink2Controls{Cur: make([]Ink2Ref, 5)}, nil, nil, 8, 2, 1),
		"repeat onto its input":          d.VerifyRepeatRows(a, a.At(8), 16, 2, 2),
		"rows weight without a buffer":   d.VerifyRows(Ink2Weight{Rows: 4, Cols: 32, Experts: 1}, a, 32, b, 4, 2),
	} {
		if err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if s := d.Stats(); s.Pending != 0 {
		t.Fatalf("a rejected op encoded %d launches", s.Pending)
	}
}
