//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// Every batched kernel is held to two references: the decode op it batches,
// row by row and bit for bit, and the host forward under the conformance bound
// the decode op's own test uses.

func ink2PTestInts(vals []int32) []float32 {
	out := make([]float32, len(vals))
	for i, v := range vals {
		out[i] = math.Float32frombits(uint32(v))
	}
	return out
}

func ink2PTestMeta(t *testing.T, d *Ink2Device, meta Ink2Ref, pos int) {
	t.Helper()
	if err := d.Write(meta, ink2PTestInts([]int32{int32(pos)})); err != nil {
		t.Fatal(err)
	}
}

// ink2PTestWeights writes a matrix of any packed type. The types with a
// synthesizer carry a float64 decode; the rest are random blocks with finite
// scales, which only the decode kernel can be compared against.
func ink2PTestWeights(t *testing.T, store *PackedGGUFStore, data []byte, rng *rand.Rand, offset int, typ uint32, rows, cols, experts int) (*ink2TestMatrix, int) {
	t.Helper()
	switch typ {
	case 0, 8, 30:
		return ink2TestFill(t, store, data, rng, offset, typ, rows, cols, experts)
	}
	block, blockBytes, err := packedBlock(typ)
	if err != nil {
		t.Fatal(err)
	}
	// Byte offsets of the half-float scales inside one block.
	halves := map[uint32][]int{1: {0}, 11: {108}, 12: {0, 2}, 13: {0, 2}, 14: {208}, 17: {0}, 18: {0}, 20: {0}, 21: {0}, 22: {0}, 40: nil}
	at, ok := halves[typ]
	if !ok {
		t.Fatalf("no synthesizer for GGML type %d", typ)
	}
	n := experts * rows * cols / block * blockBytes
	raw := data[offset : offset+n]
	rng.Read(raw)
	for b := 0; b < n; b += blockBytes {
		for _, h := range at {
			bits := uint16(0x3000 + rng.Intn(1024))
			raw[b+h], raw[b+h+1] = byte(bits), byte(bits>>8)
		}
	}
	stack, err := store.Matrix(offset, typ, rows, cols, experts)
	if err != nil {
		t.Fatal(err)
	}
	return &ink2TestMatrix{stack: stack, rows: rows, cols: cols, experts: experts}, (offset + n + 31) &^ 31
}

var ink2PTestTypes = []struct {
	name string
	typ  uint32
	cols int
}{
	{"f32", 0, 37}, {"f16", 1, 50}, {"bf16", 30, 50}, {"q8_0", 8, 96}, {"q3_K", 11, 512}, {"q4_K", 12, 512},
	{"q5_K", 13, 512}, {"q6_K", 14, 512}, {"iq2_xs", 17, 512}, {"iq2_s", 22, 512}, {"iq3_xxs", 18, 512},
	{"iq3_s", 21, 512}, {"iq4_nl", 20, 96}, {"nvfp4", 40, 128},
}

// fak-test:runtime integration est=4s lane=default
func TestInk2PRowsMatMulMatchesMatVecAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(71))
	const rows, maxT = 9, 21
	for _, c := range ink2PTestTypes {
		t.Run(c.name, func(t *testing.T) {
			store, data := ink2TestStore(t, 4)
			m, _ := ink2PTestWeights(t, store, data, rng, 64, c.typ, rows, c.cols, 1)
			w, err := m.stack.Ink2Weight(0)
			if err != nil {
				t.Fatal(err)
			}
			// Strides wider than the rows, as an arena row of another width has.
			xStride, yStride := (c.cols+15)&^3, rows+3
			x := ink2OpsTestRand(rng, maxT*xStride, .5)
			xr := ink2OpsTestPut(t, d, 8, x)
			for _, T := range []int{1, 7, 8, 9, maxT} {
				serial, batched := ink2TestAlloc(t, d, T*yStride), ink2TestAlloc(t, d, T*yStride)
				for tok := range T {
					if err := d.MatVec(w, xr.At(tok*xStride), serial.At(tok*yStride)); err != nil {
						t.Fatal(err)
					}
				}
				before := d.Stats()
				if err := d.RowsMatMul(w, xr, xStride, batched, yStride, T); err != nil {
					t.Fatal(err)
				}
				if s := d.Stats(); s.Submissions != before.Submissions || s.Pending != before.Pending+1 {
					t.Fatalf("rows matmul did not only encode one launch: %+v after %+v", s, before)
				}
				got, want := ink2OpsTestRead(t, d, batched, T*yStride), ink2OpsTestRead(t, d, serial, T*yStride)
				ink2OpsTestSame(t, fmt.Sprintf("%d rows", T), got, want)
				if m.ref == nil {
					continue
				}
				for tok := range T {
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

// fak-test:runtime integration est=2s lane=default
func TestInk2PMixMatchesMixAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(72))
	var drift ink2OpsTestDrift
	for _, c := range []struct{ states, nq, n, T int }{
		{1, 1, 300, 1}, {5, 3, 301, 9}, {16, 4, 64, 2}, {7, 1, 2304, 33}, {3, 4, 513, 5},
	} {
		what := fmt.Sprintf("%d states, %d queries, width %d, %d rows", c.states, c.nq, c.n, c.T)
		host := make([][]float32, c.states)
		refs := make([]Ink2Ref, c.states)
		for s := range refs {
			host[s] = ink2OpsTestRand(rng, c.T*c.n, 1+float64(s))
			refs[s] = ink2OpsTestPut(t, d, s, host[s])
		}
		queries := ink2OpsTestRand(rng, c.nq*c.n, 2)
		qr := ink2OpsTestPut(t, d, 1, queries)
		serial, batched := ink2TestAlloc(t, d, c.nq*c.T*c.n), ink2TestAlloc(t, d, c.nq*c.T*c.n+2).At(2)
		rowOut := ink2TestAlloc(t, d, c.nq*c.n)
		want := make([]float32, 0, c.nq*c.T*c.n)
		for tok := range c.T {
			rows := make([]Ink2Ref, c.states)
			for s := range rows {
				rows[s] = refs[s].At(tok * c.n)
			}
			if err := d.Mix(rows, qr, c.nq, nil, 0, rowOut, c.n); err != nil {
				t.Fatalf("%s: %v", what, err)
			}
			want = append(want, ink2OpsTestRead(t, d, rowOut, c.nq*c.n)...)
		}
		_ = serial
		if err := d.PMix(refs, qr, c.nq, batched, c.n, c.T); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		got := ink2OpsTestRead(t, d, batched, c.nq*c.T*c.n)
		for q := range c.nq {
			for tok := range c.T {
				row := got[(q*c.T+tok)*c.n:][:c.n]
				ink2OpsTestSame(t, fmt.Sprintf("%s: query %d row %d", what, q, tok), row, want[(tok*c.nq+q)*c.n:][:c.n])
				states := make([][]float32, c.states)
				for s := range states {
					states[s] = host[s][tok*c.n:][:c.n]
				}
				hostRow, mag := ink2OpsTestHostMix(states, queries[q*c.n:][:c.n])
				drift.parity(t, fmt.Sprintf("%s: query %d row %d vs host", what, q, tok), row, hostRow, mag)
			}
		}
	}
	drift.report(t)
}

// fak-test:runtime integration est=2s lane=default
func TestInk2PControlMatchesControlAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(73))
	var drift ink2OpsTestDrift
	for _, c := range []struct{ nj, n, rank, T int }{{1, 300, 7, 1}, {4, 257, 6, 5}, {3, 64, 64, 9}, {1, 2304, 64, 3}, {2, 100, 1, 17}} {
		what := fmt.Sprintf("%d controllers, width %d, rank %d, %d rows", c.nj, c.n, c.rank, c.T)
		var cur, msg, down, gain, forget [][]float32
		var ctl Ink2Controls
		for j := range c.nj {
			cur = append(cur, ink2OpsTestRand(rng, c.T*c.n, 2))
			msg = append(msg, ink2OpsTestRand(rng, c.T*c.n, .5))
			// Unit-variance projections: larger ones amplify the device's dot
			// rounding through tanh past the conformance bound.
			down = append(down, ink2OpsTestRand(rng, c.rank*2*c.n, 1/math.Sqrt(float64(2*c.n))))
			gain = append(gain, ink2OpsTestRand(rng, c.n*c.rank, 1/math.Sqrt(float64(c.rank))))
			forget = append(forget, ink2OpsTestRand(rng, c.n*c.rank, 1/math.Sqrt(float64(c.rank))))
			ctl.Cur = append(ctl.Cur, ink2OpsTestPut(t, d, j, cur[j]))
			ctl.Msg = append(ctl.Msg, ink2OpsTestPut(t, d, 1, msg[j]))
			ctl.Down = append(ctl.Down, ink2OpsTestPut(t, d, 0, down[j]))
			ctl.Gain = append(ctl.Gain, ink2OpsTestPut(t, d, 3, gain[j]))
			ctl.Forget = append(ctl.Forget, ink2OpsTestPut(t, d, 0, forget[j]))
		}
		// One contiguous g and out, as the driver lays its controllers out.
		gAll, outAll := ink2TestAlloc(t, d, c.nj*c.T*c.rank), ink2TestAlloc(t, d, c.nj*c.T*c.n)
		g, out := make([]Ink2Ref, c.nj), make([]Ink2Ref, c.nj)
		for j := range c.nj {
			g[j], out[j] = gAll.At(j*c.T*c.rank), outAll.At(j*c.T*c.n)
		}
		if err := d.PCtrlDown(ctl.Cur, ctl.Msg, ctl.Down, g, c.n, c.rank, c.T); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		if err := d.PCtrlApply(ctl.Cur, ctl.Msg, g, ctl.Gain, ctl.Forget, out, c.n, c.rank, c.T); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		got, gotG := ink2OpsTestRead(t, d, outAll, c.nj*c.T*c.n), ink2OpsTestRead(t, d, gAll, c.nj*c.T*c.rank)
		rowG, rowOut := ink2TestAlloc(t, d, c.nj*c.rank), ink2TestAlloc(t, d, c.nj*c.n)
		for tok := range c.T {
			row := Ink2Controls{Down: ctl.Down, Gain: ctl.Gain, Forget: ctl.Forget}
			for j := range c.nj {
				row.Cur = append(row.Cur, ctl.Cur[j].At(tok*c.n))
				row.Msg = append(row.Msg, ctl.Msg[j].At(tok*c.n))
			}
			if err := d.Control(row, rowG, rowOut, c.n, c.rank, nil, 0); err != nil {
				t.Fatalf("%s: %v", what, err)
			}
			wantG, want := ink2OpsTestRead(t, d, rowG, c.nj*c.rank), ink2OpsTestRead(t, d, rowOut, c.nj*c.n)
			for j := range c.nj {
				where := fmt.Sprintf("%s: controller %d row %d", what, j, tok)
				ink2OpsTestSame(t, where+" g", gotG[(j*c.T+tok)*c.rank:][:c.rank], wantG[j*c.rank:][:c.rank])
				gotRow := got[(j*c.T+tok)*c.n:][:c.n]
				ink2OpsTestSame(t, where, gotRow, want[j*c.n:][:c.n])
				host, mag := ink2OpsTestHostControl(cur[j][tok*c.n:][:c.n], msg[j][tok*c.n:][:c.n], down[j], gain[j], forget[j], c.rank)
				drift.parity(t, where+" vs host", gotRow, host, mag)
			}
		}
	}
	drift.report(t)
}

// fak-test:runtime integration est=3s lane=default
func TestInk2PConvMatchesSerialConvAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(74))
	var drift ink2ConvTestDrift
	for _, c := range []struct {
		n, heads, kn, T int
		dyn, resid      bool
	}{
		{n: 300, heads: 1, kn: 4, T: 1}, {n: 96, heads: 8, kn: 4, T: 2, dyn: true}, {n: 257, heads: 1, kn: 2, T: 9, resid: true},
		{n: 64, heads: 4, kn: 8, T: 7, dyn: true, resid: true}, {n: 1024, heads: 8, kn: 4, T: 64, dyn: true},
		{n: 2304, heads: 1, kn: 4, T: 3, resid: true},
	} {
		what := fmt.Sprintf("%d channels, %d heads, kernel %d, %d rows, dyn %v, resid %v", c.n, c.heads, c.kn, c.T, c.dyn, c.resid)
		past := c.kn - 1
		norm := float32(1 / math.Sqrt(float64(c.kn+1)))
		x, w := ink2ConvTestRand(rng, c.T*c.n, 1), ink2ConvTestRand(rng, c.n*c.kn, .5)
		hist := ink2ConvTestRand(rng, c.n*past, 1)
		xr, wr := ink2ConvTestPut(t, d, 3, x), ink2ConvTestPut(t, d, 0, w)
		// The channels sit inside a larger slot history, as k, v and the post convs do.
		histSerial, histBatched := ink2ConvTestPut(t, d, 5*past, hist), ink2ConvTestPut(t, d, 5*past, hist)
		var dyn, resid []float32
		var dynRef, residRef Ink2Ref
		if c.dyn {
			dyn = ink2ConvTestRand(rng, c.T*c.heads*c.kn, .5)
			dynRef = ink2ConvTestPut(t, d, 1, dyn)
			norm = float32(1 / math.Sqrt(float64(2*c.kn+1)))
		}
		if c.resid {
			resid = ink2ConvTestRand(rng, c.T*c.n, 1)
			residRef = ink2ConvTestPut(t, d, 0, resid)
		}
		serial, batched := ink2TestAlloc(t, d, c.T*c.n), ink2TestAlloc(t, d, c.T*c.n+1).At(1)
		for tok := range c.T {
			var dr, rr Ink2Ref
			if c.dyn {
				dr = dynRef.At(tok * c.heads * c.kn)
			}
			if c.resid {
				rr = residRef.At(tok * c.n)
			}
			if err := d.Conv(xr.At(tok*c.n), wr, dr, histSerial, serial.At(tok*c.n), rr, c.n, c.heads, c.kn, norm); err != nil {
				t.Fatalf("%s: %v", what, err)
			}
		}
		if err := d.PConv(xr, wr, dynRef, histBatched, batched, residRef, c.n, c.heads, c.kn, norm, c.T); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		got := ink2ConvTestRead(t, d, batched, c.T*c.n)
		ink2ConvTestSame(t, what, got, ink2ConvTestRead(t, d, serial, c.T*c.n))
		gotHist := ink2ConvTestRead(t, d, histBatched, c.n*past)
		ink2ConvTestSame(t, what+": history", gotHist, ink2ConvTestRead(t, d, histSerial, c.n*past))

		hostHist := append([]float32(nil), hist...)
		for tok := range c.T {
			var dr []float32
			if c.dyn {
				dr = dyn[tok*c.heads*c.kn:][:c.heads*c.kn]
			}
			want, mag := ink2ConvTestHost(x[tok*c.n:][:c.n], w, dr, hostHist, c.heads, c.kn, norm)
			if c.resid {
				for i := range want {
					want[i] = resid[tok*c.n+i] + want[i]
					mag[i] += math.Abs(float64(resid[tok*c.n+i]))
				}
			}
			drift.parity(t, fmt.Sprintf("%s: row %d vs host", what, tok), got[tok*c.n:][:c.n], want, mag)
		}
		// The history is the last kn-1 inputs: no arithmetic touches it.
		ink2ConvTestSame(t, what+": history vs host", gotHist, hostHist)
	}
	t.Logf("%d values differ from the host in some bit, the worst by %.3f of the bound", drift.inexact, drift.worst)
}

// fak-test:runtime integration est=3s lane=default
func TestInk2PQKVPostMatchesSerialAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(75))
	var drift ink2ConvTestDrift
	for _, c := range []struct {
		name         string
		p            Ink2AttnParams
		cap, pos0, T int
	}{
		{"global", Ink2AttnParams{Heads: 6, KVHeads: 2, HeadDim: 40, RelLen: 20, QKClip: 2, RelClip: 1.5, Eps: 1e-6, Global: true, Alpha: .35, Floor: 3}, 64, 0, 9},
		{"global past the floor", Ink2AttnParams{Heads: 8, KVHeads: 2, HeadDim: 128, RelLen: 16, QKClip: 5, RelClip: 3, Eps: 1e-6, Global: true, Alpha: .2, Floor: 4}, 4096, 1000, 5},
		{"local ring wrapping", Ink2AttnParams{Heads: 4, KVHeads: 4, HeadDim: 64, RelLen: 33, QKClip: 1.5, RelClip: 2, Eps: 1e-5}, 12, 20, 7},
		{"chunk fills the ring", Ink2AttnParams{Heads: 3, KVHeads: 3, HeadDim: 7, RelLen: 1, QKClip: .8, RelClip: .1, Eps: 1e-6, Global: true, Alpha: 1, Floor: 1}, 5, 3, 5},
		{"no relative states", Ink2AttnParams{Heads: 4, KVHeads: 2, HeadDim: 16, QKClip: .7, RelClip: 1, Eps: 1e-6, Global: true, Alpha: .5, Floor: 2}, 9, 4, 3},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := c.p
			qn, rn, kvDim := p.Heads*p.HeadDim, p.Heads*p.RelLen, p.KVHeads*p.HeadDim
			qclip, rclip := ink2ConvTestRand(rng, qn, 1.5), ink2ConvTestRand(rng, max(rn, 1), 1.5)
			kcache, vcache := ink2ConvTestRand(rng, c.cap*kvDim, 9), ink2ConvTestRand(rng, c.cap*kvDim, 9)
			q, r := ink2ConvTestRand(rng, c.T*qn, 1), ink2ConvTestRand(rng, max(c.T*rn, 1), 1)
			k, v := ink2ConvTestRand(rng, c.T*kvDim, 1), ink2ConvTestRand(rng, c.T*kvDim, 1)
			// Head amplitudes straddle the clip thresholds, so some heads shrink and some pass.
			for tok := range c.T {
				for hd := range p.Heads {
					amp := .15 * float32(1+hd)
					for i := range p.HeadDim {
						q[(tok*p.Heads+hd)*p.HeadDim+i] *= amp
					}
					for i := range p.RelLen {
						r[(tok*p.Heads+hd)*p.RelLen+i] *= amp
					}
				}
				for hd := range p.KVHeads {
					for i := range p.HeadDim {
						k[(tok*p.KVHeads+hd)*p.HeadDim+i] *= .2 * float32(1+2*hd)
					}
				}
			}
			qclipRef, rclipRef := ink2ConvTestPut(t, d, 0, qclip), ink2ConvTestPut(t, d, 3, rclip)
			kRef, vRef := ink2ConvTestPut(t, d, 2, k), ink2ConvTestPut(t, d, 0, v)
			var rSerial, rBatched Ink2Ref
			if rn > 0 {
				rSerial, rBatched = ink2ConvTestPut(t, d, 1, r), ink2ConvTestPut(t, d, 0, r)
			}
			qSerial, qBatched := ink2ConvTestPut(t, d, 0, q), ink2ConvTestPut(t, d, 4, q)
			kcSerial, vcSerial := ink2ConvTestPut(t, d, 0, kcache), ink2ConvTestPut(t, d, 5, vcache)
			kcBatched, vcBatched := ink2ConvTestPut(t, d, 5, kcache), ink2ConvTestPut(t, d, 0, vcache)
			meta := ink2TestAlloc(t, d, 4)
			for tok := range c.T {
				ink2PTestMeta(t, d, meta, c.pos0+tok)
				var rr Ink2Ref
				if rn > 0 {
					rr = rSerial.At(tok * rn)
				}
				if err := d.QKVPost(qSerial.At(tok*qn), rr, kRef.At(tok*kvDim), vRef.At(tok*kvDim), qclipRef, rclipRef,
					kcSerial, vcSerial, meta, p, c.cap, nil); err != nil {
					t.Fatal(err)
				}
				if err := d.Flush(); err != nil {
					t.Fatal(err)
				}
			}
			before := d.Stats()
			if err := d.PQKVPost(qBatched, rBatched, kRef, vRef, qclipRef, rclipRef, kcBatched, vcBatched, p, c.cap, c.pos0, c.T); err != nil {
				t.Fatal(err)
			}
			if s := d.Stats(); s.Submissions != before.Submissions || s.Pending != 1 {
				t.Fatalf("prefill qkv post did not only encode: %+v after %+v", s, before)
			}
			gotQ := ink2ConvTestRead(t, d, qBatched, c.T*qn)
			ink2ConvTestSame(t, "q", gotQ, ink2ConvTestRead(t, d, qSerial, c.T*qn))
			var gotR []float32
			if rn > 0 {
				gotR = ink2ConvTestRead(t, d, rBatched, c.T*rn)
				ink2ConvTestSame(t, "r", gotR, ink2ConvTestRead(t, d, rSerial, c.T*rn))
			}
			gotK, gotV := ink2ConvTestRead(t, d, kcBatched, c.cap*kvDim), ink2ConvTestRead(t, d, vcBatched, c.cap*kvDim)
			ink2ConvTestSame(t, "k cache", gotK, ink2ConvTestRead(t, d, kcSerial, c.cap*kvDim))
			ink2ConvTestSame(t, "v cache", gotV, ink2ConvTestRead(t, d, vcSerial, c.cap*kvDim))

			// The host forward's attention prep, token by token.
			clipped := 0
			for tok := range c.T {
				pos := c.pos0 + tok
				qt, kt, vt := q[tok*qn:][:qn], k[tok*kvDim:][:kvDim], v[tok*kvDim:][:kvDim]
				var rt []float32
				if rn > 0 {
					rt = r[tok*rn:][:rn]
				}
				qMag, rMag := make([]float64, 0, qn), make([]float64, 0, rn)
				for hd := range p.Heads {
					qh := qt[hd*p.HeadDim:][:p.HeadDim]
					qMag = append(qMag, ink2ConvTestClip(qh, qclip[hd*p.HeadDim:][:p.HeadDim], p.QKClip)...)
					if math.Abs(float64(qh[0])) < qMag[hd*p.HeadDim] {
						clipped++
					}
					if rn > 0 {
						rMag = append(rMag, ink2ConvTestClip(rt[hd*p.RelLen:][:p.RelLen], rclip[hd*p.RelLen:][:p.RelLen], p.RelClip)...)
					}
				}
				vMag := make([]float64, kvDim)
				kMag := make([]float64, 0, kvDim)
				for hd := range p.KVHeads {
					kh, vh := kt[hd*p.HeadDim:][:p.HeadDim], vt[hd*p.HeadDim:][:p.HeadDim]
					kMag = append(kMag, ink2ConvTestClip(kh, nil, p.QKClip)...)
					var ss float64
					for _, x := range vh {
						ss += float64(x) * float64(x)
					}
					s := float32(1 / math.Sqrt(ss/float64(p.HeadDim)+float64(p.Eps)))
					for i := range vh {
						vh[i] *= s
						vMag[hd*p.HeadDim+i] = math.Abs(float64(vh[i]))
					}
				}
				if p.Global {
					tau := float32(1 + float64(p.Alpha)*math.Log(math.Max(1, float64(pos+1)/float64(p.Floor))))
					for i := range qt {
						qt[i] *= tau
						qMag[i] *= float64(tau)
					}
					for i := range rt {
						rt[i] *= tau
						rMag[i] *= float64(tau)
					}
				}
				where := fmt.Sprintf("position %d", pos)
				drift.parity(t, where+" q vs host", gotQ[tok*qn:][:qn], qt, qMag)
				if rn > 0 {
					drift.parity(t, where+" r vs host", gotR[tok*rn:][:rn], rt, rMag)
				}
				row := (pos % c.cap) * kvDim
				drift.parity(t, where+" cached k vs host", gotK[row:][:kvDim], kt, kMag)
				drift.parity(t, where+" cached v vs host", gotV[row:][:kvDim], vt, vMag)
				copy(kcache[row:], gotK[row:][:kvDim])
				copy(vcache[row:], gotV[row:][:kvDim])
			}
			if p.Heads > 3 && (clipped == 0 || clipped == c.T*p.Heads) {
				t.Fatalf("%d of %d q heads clipped: the case does not cover both sides of the threshold", clipped, c.T*p.Heads)
			}
			// Every cache row outside the chunk keeps its bits.
			ink2ConvTestSame(t, "untouched k rows", gotK, kcache)
			ink2ConvTestSame(t, "untouched v rows", gotV, vcache)
		})
	}
	t.Logf("%d values differ from the host in some bit, the worst by %.3f of the bound", drift.inexact, drift.worst)
}

// fak-test:runtime integration est=4s lane=default
func TestInk2PAttendMatchesSerialAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	tiny := Ink2AttnParams{Heads: 4, KVHeads: 2, HeadDim: 8, RelLen: 3, Scale: float32(1 / math.Sqrt(8))}
	odd := Ink2AttnParams{Heads: 6, KVHeads: 2, HeadDim: 39, RelLen: 35, Scale: float32(1 / math.Sqrt(39))}
	bare := Ink2AttnParams{Heads: 3, KVHeads: 3, HeadDim: 7, Scale: 1. / 3}
	mini := Ink2AttnParams{Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16, Scale: float32(1 / math.Sqrt(128))}
	rng := rand.New(rand.NewSource(76))
	var worst float64
	for _, c := range []struct {
		p                    Ink2AttnParams
		cap, window, extent  int
		pos0, T, launchLimit int
	}{
		// the first chunk of a prompt: the first query sees one key
		{p: tiny, cap: 64, extent: 10, pos0: 0, T: 9},
		{p: tiny, cap: 64, extent: 10, pos0: 0, T: 33, launchLimit: 2},
		// a local ring with the chunk's room past its window, filling and wrapped
		{p: tiny, cap: 16 + 7, window: 16, extent: 16, pos0: 0, T: 7},
		{p: tiny, cap: 16 + 7, window: 16, extent: 16, pos0: 12, T: 7},
		{p: tiny, cap: 16 + 7, window: 16, extent: 16, pos0: 40, T: 7, launchLimit: 3},
		// a bias table that ends before the history does, and one past it
		{p: odd, cap: 300, extent: 70, pos0: 60, T: 20},
		{p: odd, cap: 300, extent: 500, pos0: 205, T: 5},
		{p: bare, cap: 40, pos0: 30, T: 9},
		{p: mini, cap: 1024 + 16, window: 1024, extent: 1024, pos0: 2000, T: 16, launchLimit: 5},
		{p: mini, cap: 4096, extent: 1024, pos0: 500, T: 12},
	} {
		what := fmt.Sprintf("heads %d/%d dim %d rel %d cap %d window %d extent %d pos0 %d rows %d limit %d",
			c.p.Heads, c.p.KVHeads, c.p.HeadDim, c.p.RelLen, c.cap, c.window, c.extent, c.pos0, c.T, c.launchLimit)
		p := c.p
		qn, rn, kvDim := p.Heads*p.HeadDim, p.Heads*p.RelLen, p.KVHeads*p.HeadDim
		q, r := ink2OpsTestRand(rng, c.T*qn, 1), ink2OpsTestRand(rng, c.T*rn, 1)
		sinks := ink2OpsTestRand(rng, c.T*p.Heads, 1)
		rel := ink2OpsTestRand(rng, c.extent*p.RelLen, .1)
		kc, vc := ink2OpsTestRand(rng, c.cap*kvDim, .3), ink2OpsTestRand(rng, c.cap*kvDim, 1)
		qr, sr := ink2OpsTestPut(t, d, 1, q), ink2OpsTestPut(t, d, 0, sinks)
		rr, relr := ink2TestAttendUpload(t, d, r), ink2TestAttendUpload(t, d, rel)
		kcr, vcr := ink2OpsTestPut(t, d, 0, kc), ink2OpsTestPut(t, d, 3, vc)
		serial, batched := ink2TestAlloc(t, d, c.T*qn), ink2TestAlloc(t, d, c.T*qn+3).At(3)
		serialScores, meta := ink2TestAlloc(t, d, p.Heads*c.cap), ink2TestAlloc(t, d, 4)
		for tok := range c.T {
			ink2PTestMeta(t, d, meta, c.pos0+tok)
			if err := d.Attend(qr.At(tok*qn), rr.At(tok*rn), sr.At(tok*p.Heads), relr, kcr, vcr, serialScores, serial.At(tok*qn),
				meta, p, c.extent, c.window, c.cap); err != nil {
				t.Fatalf("%s: %v", what, err)
			}
			if err := d.Flush(); err != nil {
				t.Fatal(err)
			}
		}
		span := PAttendSpan(c.window, c.cap, c.pos0, c.T)
		floats := c.T * p.Heads * span
		launches := 1
		if c.launchLimit > 0 {
			floats = c.launchLimit*p.Heads*span + span - 1
			launches = (c.T + c.launchLimit - 1) / c.launchLimit
		}
		scores := ink2TestAlloc(t, d, floats)
		before := d.Stats()
		if err := d.PAttend(qr, rr, sr, relr, kcr, vcr, scores, floats, batched, p, c.extent, c.window, c.cap, c.pos0, c.T); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		if s := d.Stats(); s.Submissions != before.Submissions || s.Pending != launches {
			t.Fatalf("%s: %d launches pending after %+v, want %d and no submission: %+v", what, s.Pending, before, launches, s)
		}
		got := ink2OpsTestRead(t, d, batched, c.T*qn)
		ink2OpsTestSame(t, what, got, ink2OpsTestRead(t, d, serial, c.T*qn))
		for tok := range c.T {
			want, mag := ink2TestAttendHost(p, q[tok*qn:][:qn], r[tok*rn:][:rn], sinks[tok*p.Heads:][:p.Heads], rel, kc, vc,
				c.pos0+tok, c.extent, c.window, c.cap)
			ink2TestAttendCheck(t, fmt.Sprintf("%s: row %d vs host", what, tok), got[tok*qn:][:qn], want, mag)
			for i := range want {
				worst = max(worst, math.Abs(float64(got[tok*qn+i])-float64(want[i]))/(mag[i]+1e-30))
			}
		}
	}
	t.Logf("worst |device-host| / sum w|v| = %.3g", worst)
}

// fak-test:runtime integration est=2s lane=default
func TestInk2PRouteMatchesRouteAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(77))
	for _, c := range []struct {
		nexp, k, T int
		bias, tie  bool
		scale      float32
	}{
		{nexp: 128, k: 8, T: 9, bias: true, scale: 8}, {nexp: 100, k: 1, T: 3}, {nexp: 512, k: 16, T: 5, bias: true},
		{nexp: 33, k: 16, T: 17, scale: .5}, {nexp: 64, k: 8, T: 4, bias: true, tie: true, scale: 2},
	} {
		what := fmt.Sprintf("%d experts to %d slots, %d rows, bias %v, ties %v", c.nexp, c.k, c.T, c.bias, c.tie)
		logits := ink2OpsTestRand(rng, c.T*c.nexp, 2)
		if c.tie {
			// Equal logits everywhere: the selection has to fall to the lower index.
			for i := range logits {
				logits[i] = float32(i / c.nexp)
			}
		}
		var bias []float32
		var br Ink2Ref
		if c.bias {
			bias = ink2OpsTestRand(rng, c.nexp, .1)
			if c.tie {
				clear(bias)
			}
			br = ink2OpsTestPut(t, d, 2, bias)
		}
		lr := ink2OpsTestPut(t, d, 1, logits)
		ids, wts := ink2TestAlloc(t, d, c.T*c.k), ink2TestAlloc(t, d, c.T*c.k+1).At(1)
		rowIDs, rowWts := ink2TestAlloc(t, d, c.k), ink2TestAlloc(t, d, c.k)
		if err := d.PRoute(lr, br, ids, wts, c.nexp, c.k, c.scale, c.T); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		gotIDs, err := d.ReadInts(ids, c.T*c.k)
		if err != nil {
			t.Fatal(err)
		}
		gotWts := ink2OpsTestRead(t, d, wts, c.T*c.k)
		for tok := range c.T {
			if err := d.Route(lr.At(tok*c.nexp), br, rowIDs, rowWts, c.nexp, c.k, c.scale); err != nil {
				t.Fatalf("%s: %v", what, err)
			}
			wantIDs, err := d.ReadInts(rowIDs, c.k)
			if err != nil {
				t.Fatal(err)
			}
			where := fmt.Sprintf("%s: row %d", what, tok)
			probs, _, top := ink2OpsTestHostRoute(logits[tok*c.nexp:][:c.nexp], bias, c.k)
			hostWts := ink2OpsTestHostWeights(probs, top, c.scale)
			for s := range c.k {
				if g := gotIDs[tok*c.k+s]; g != wantIDs[s] || g != top[s] {
					t.Fatalf("%s slot %d routes to %d, decode op %d, host %d", where, s, g, wantIDs[s], top[s])
				}
				if c.tie && gotIDs[tok*c.k+s] != int32(s) {
					t.Fatalf("%s slot %d routes to %d among equals", where, s, gotIDs[tok*c.k+s])
				}
				if g, h := gotWts[tok*c.k+s], hostWts[s]; math.Abs(float64(g-h)) > 1e-5*math.Abs(float64(h))+1e-6 {
					t.Fatalf("%s slot %d weight %v, host %v", where, s, g, h)
				}
			}
			ink2OpsTestSame(t, where+" weights", gotWts[tok*c.k:][:c.k], ink2OpsTestRead(t, d, rowWts, c.k))
		}
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2PCombineMatchesCombineAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(78))
	for _, c := range []struct{ k, n, T int }{{1, 300, 1}, {8, 257, 9}, {8, 2304, 3}, {3, 5, 40}} {
		what := fmt.Sprintf("%d slots of width %d, %d rows", c.k, c.n, c.T)
		down, wts := ink2OpsTestRand(rng, c.T*c.k*c.n, 1), ink2OpsTestRand(rng, c.T*c.k, 1)
		dr, wr := ink2OpsTestPut(t, d, 2, down), ink2OpsTestPut(t, d, 1, wts)
		out, row := ink2TestAlloc(t, d, c.T*c.n), ink2TestAlloc(t, d, c.n)
		if err := d.PCombine(dr, wr, out, c.k, c.n, c.T); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		got := ink2OpsTestRead(t, d, out, c.T*c.n)
		for tok := range c.T {
			if err := d.Combine(dr.At(tok*c.k*c.n), wr.At(tok*c.k), row, c.k, c.n); err != nil {
				t.Fatalf("%s: %v", what, err)
			}
			where := fmt.Sprintf("%s: row %d", what, tok)
			ink2OpsTestSame(t, where, got[tok*c.n:][:c.n], ink2OpsTestRead(t, d, row, c.n))
			// The host sums the rounded products in routing order, as the kernel does.
			ink2OpsTestSame(t, where+" vs host", got[tok*c.n:][:c.n],
				ink2OpsTestHostCombine(down[tok*c.k*c.n:][:c.k*c.n], wts[tok*c.k:][:c.k], c.n))
		}
	}
}

// ink2PTestBucket is the stable counting sort of ids by expert.
func ink2PTestBucket(ids []int32, nexp int) (counts, offsets, order []int32) {
	counts, offsets, order = make([]int32, nexp), make([]int32, nexp+1), make([]int32, len(ids))
	for _, e := range ids {
		counts[e]++
	}
	for e := range nexp {
		offsets[e+1] = offsets[e] + counts[e]
	}
	next := append([]int32(nil), offsets[:nexp]...)
	for i, e := range ids {
		order[next[e]] = int32(i)
		next[e]++
	}
	return counts, offsets, order
}

func ink2PTestSameInts(t *testing.T, what string, got, want []int32) {
	t.Helper()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %d, want %d", what, i, got[i], want[i])
		}
	}
}

// fak-test:runtime integration est=2s lane=default
func TestInk2PBucketIsAStableSortByExpert(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(79))
	for _, c := range []struct {
		npairs, nexp int
		pick         func(i int) int32
	}{
		{1, 1, func(int) int32 { return 0 }},
		{37, 128, func(int) int32 { return 5 }},
		{4096, 128, func(int) int32 { return int32(rng.Intn(128)) }},
		{1003, 512, func(int) int32 { return int32(rng.Intn(512)) }},
		// most experts unused, and long runs of one
		{250, 64, func(i int) int32 { return int32(60 - i%3*30) }},
	} {
		what := fmt.Sprintf("%d pairs over %d experts", c.npairs, c.nexp)
		ids := make([]int32, c.npairs)
		for i := range ids {
			ids[i] = c.pick(i)
		}
		idr := ink2OpsTestPut(t, d, 3, ink2PTestInts(ids))
		// One scratch region with a guard word on every side of the three tables.
		scratch := ink2TestAlloc(t, d, 2*c.nexp+c.npairs+5)
		guard := float32(math.Float32frombits(0xdeadbeef))
		fill := make([]float32, 2*c.nexp+c.npairs+5)
		for i := range fill {
			fill[i] = guard
		}
		if err := d.Write(scratch, fill); err != nil {
			t.Fatal(err)
		}
		counts, offsets, order := scratch.At(1), scratch.At(c.nexp+2), scratch.At(2*c.nexp+4)
		if err := d.PBucket(idr, counts, offsets, order, c.npairs, c.nexp); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		got, err := d.ReadInts(scratch, len(fill))
		if err != nil {
			t.Fatal(err)
		}
		wantCounts, wantOffsets, wantOrder := ink2PTestBucket(ids, c.nexp)
		ink2PTestSameInts(t, what+": counts", got[1:], wantCounts)
		ink2PTestSameInts(t, what+": offsets", got[c.nexp+2:], wantOffsets)
		ink2PTestSameInts(t, what+": order", got[2*c.nexp+4:], wantOrder)
		for _, g := range []int{0, c.nexp + 1, 2*c.nexp + 3, 2*c.nexp + 4 + c.npairs} {
			if uint32(got[g]) != 0xdeadbeef {
				t.Fatalf("%s: guard word %d overwritten with %#x", what, g, uint32(got[g]))
			}
		}
	}
}

// fak-test:runtime integration est=6s lane=default
func TestInk2PExpertGroupedMatchesExpertMatVecAndHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(80))
	const experts, rows, k, T = 5, 6, 3, 9
	const npairs = T * k
	// Expert 4 is never routed; expert 1 holds a run longer than a tile and
	// the runs of 0, 2 and 3 start mid-tile.
	ids := make([]int32, npairs)
	for i := range ids {
		ids[i] = []int32{1, 0, 1, 3, 1, 2, 1, 1}[rng.Intn(8)]
	}
	_, _, order := ink2PTestBucket(ids, experts)
	idr := ink2OpsTestPut(t, d, 0, ink2PTestInts(ids))
	orr := ink2OpsTestPut(t, d, 1, ink2PTestInts(order))
	scale := []float32{1.5, -3, .25, .75, 2}
	scr := ink2OpsTestPut(t, d, 2, scale)
	for _, c := range ink2PTestTypes {
		t.Run(c.name, func(t *testing.T) {
			store, data := ink2TestStore(t, 8)
			m, _ := ink2PTestWeights(t, store, data, rng, 64, c.typ, rows, c.cols, experts)
			stack, ok := m.stack.Ink2Stack()
			if !ok {
				t.Fatal("stack split across views")
			}
			xStride := (c.cols + 15) &^ 3
			for _, form := range []struct {
				name   string
				xDiv   int
				scaled bool
			}{{"token rows scaled", k, true}, {"pair rows", 1, false}, {"pair rows scaled", 1, true}} {
				inRows := (npairs-1)/form.xDiv + 1
				x := ink2OpsTestRand(rng, inRows*xStride, .5)
				// The decode op reads one input row per slot.
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
				before := d.Stats()
				if err := d.PExpertGrouped(stack, idr, orr, xr, form.xDiv, xStride, batched, npairs, sc); err != nil {
					t.Fatalf("%s: %v", form.name, err)
				}
				if s := d.Stats(); s.Submissions != before.Submissions || s.Pending != before.Pending+1 {
					t.Fatalf("%s: grouped experts did not only encode one launch: %+v after %+v", form.name, s, before)
				}
				got := ink2OpsTestRead(t, d, batched, npairs*rows)
				ink2OpsTestSame(t, form.name, got, ink2OpsTestRead(t, d, serial, npairs*rows))
				if m.ref == nil {
					continue
				}
				for i, e := range ids {
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

// fak-test:runtime integration est=1s lane=default
func TestInk2PRejectsBadOperands(t *testing.T) {
	d := ink2TestDevice(t)
	a := ink2TestAlloc(t, d, 4096)
	at := func(off int) Ink2Ref { return a.At(off) }
	one := func(r Ink2Ref) []Ink2Ref { return []Ink2Ref{r} }
	p := Ink2AttnParams{Heads: 2, KVHeads: 1, HeadDim: 8, RelLen: 2, QKClip: 1, RelClip: 1, Eps: 1e-6, Scale: 1}
	dense := Ink2DenseF32(at(0), 4, 8)
	for name, err := range map[string]error{
		"mix without states":           d.PMix(nil, at(0), 1, at(64), 8, 2),
		"mix output over a state":      d.PMix(one(at(64)), at(0), 1, at(72), 8, 2),
		"mix of five queries":          d.PMix(one(at(64)), at(0), 5, at(512), 8, 2),
		"five controllers":             d.PCtrlDown(make([]Ink2Ref, 5), make([]Ink2Ref, 5), make([]Ink2Ref, 5), make([]Ink2Ref, 5), 8, 2, 2),
		"controller without a message": d.PCtrlDown(one(at(0)), one(Ink2Ref{}), one(at(64)), one(at(128)), 8, 2, 2),
		"apply with uneven operands":   d.PCtrlApply(one(at(0)), one(at(16)), one(at(32)), one(at(64)), nil, one(at(128)), 8, 2, 2),
		"conv kernel of nine":          d.PConv(at(0), at(64), Ink2Ref{}, at(256), at(512), Ink2Ref{}, 8, 1, 9, 1, 2),
		"conv output over its input":   d.PConv(at(0), at(64), Ink2Ref{}, at(256), at(8), Ink2Ref{}, 8, 1, 4, 1, 2),
		"conv output over the resid":   d.PConv(at(0), at(64), Ink2Ref{}, at(256), at(512), at(520), 8, 1, 4, 1, 2),
		"conv heads not dividing":      d.PConv(at(0), at(64), at(128), at(256), at(512), Ink2Ref{}, 8, 3, 4, 1, 2),
		"qkv chunk longer than cache":  d.PQKVPost(at(0), at(64), at(128), at(192), at(256), at(320), at(1024), at(2048), p, 3, 0, 4),
		"qkv without a cache":          d.PQKVPost(at(0), at(64), at(128), at(192), at(256), at(320), Ink2Ref{}, at(2048), p, 8, 0, 2),
		"attend without scores":        d.PAttend(at(0), at(64), at(128), at(192), at(1024), at(2048), Ink2Ref{}, 64, at(3072), p, 4, 0, 8, 0, 2),
		"attend scratch below a row":   d.PAttend(at(0), at(64), at(128), at(192), at(1024), at(2048), at(3500), 3, at(3072), p, 4, 0, 8, 0, 2),
		"attend heads not grouping":    d.PAttend(at(0), at(64), at(128), at(192), at(1024), at(2048), at(3500), 64, at(3072), Ink2AttnParams{Heads: 3, KVHeads: 2, HeadDim: 8}, 0, 0, 8, 0, 2),
		"route of more slots":          d.PRoute(at(0), Ink2Ref{}, at(64), at(128), 4, 5, 0, 2),
		"combine without weights":      d.PCombine(at(0), Ink2Ref{}, at(64), 2, 8, 2),
		"bucket over too many experts": d.PBucket(at(0), at(64), at(1024), at(2048), 8, 513),
		"rows matmul narrow stride":    d.RowsMatMul(dense, at(64), 7, at(512), 4, 2),
		"rows matmul without output":   d.RowsMatMul(dense, at(64), 8, Ink2Ref{}, 4, 2),
		"grouped experts narrow input": d.PExpertGrouped(dense, at(64), at(128), at(256), 1, 7, at(512), 2, Ink2Ref{}),
	} {
		if !errors.Is(err, ErrInk2Shape) {
			t.Errorf("%s: %v, want ErrInk2Shape", name, err)
		}
	}
	if s := d.Stats(); s.Pending != 0 {
		t.Fatalf("a rejected op encoded a launch: %+v", s)
	}
}

// A chunk's ops only encode: a whole layer visit of them is one submission.
// fak-test:runtime integration est=1s lane=default
func TestInk2PEncodesWithoutSubmitting(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(81))
	const n, T, k, nexp = 64, 5, 2, 8
	p := Ink2AttnParams{Heads: 4, KVHeads: 2, HeadDim: 16, RelLen: 4, QKClip: 1, RelClip: 1, Eps: 1e-6, Scale: .25}
	kvDim, cap := p.KVHeads*p.HeadDim, 16
	put := func(count int) Ink2Ref { return ink2OpsTestPut(t, d, 0, ink2OpsTestRand(rng, count, 1)) }
	x, w := put(T*n), Ink2DenseF32(put(n*n), n, n)
	y, mixed, conv := ink2TestAlloc(t, d, T*n), ink2TestAlloc(t, d, T*n), ink2TestAlloc(t, d, T*n)
	q, r, sinks := put(T*p.Heads*p.HeadDim), put(T*p.Heads*p.RelLen), put(T*p.Heads)
	kc, vc := ink2TestAlloc(t, d, cap*kvDim), ink2TestAlloc(t, d, cap*kvDim)
	attn, scores := ink2TestAlloc(t, d, T*p.Heads*p.HeadDim), ink2TestAlloc(t, d, T*p.Heads*cap)
	ids, wts, out := ink2TestAlloc(t, d, T*k), ink2TestAlloc(t, d, T*k), ink2TestAlloc(t, d, T*n)
	scratch, pairs := ink2TestAlloc(t, d, 2*nexp+1+T*k), ink2TestAlloc(t, d, T*k*n)
	stack := Ink2Weight{Buf: put(nexp * n * n).Buf, Rows: n, Cols: n, Experts: nexp}
	// Every host write is a fence, so the operands are all uploaded first.
	query, down, convW, hist := put(n), put(4*2*n), put(n*4), ink2TestAlloc(t, d, n*3)
	g, kx, vx := ink2TestAlloc(t, d, T*4), put(T*kvDim), put(T*kvDim)
	qclip, rclip, rel, logits := put(p.Heads*p.HeadDim), put(p.Heads*p.RelLen), put(8*p.RelLen), put(T*nexp)
	before := d.Stats()
	for _, err := range []error{
		d.RowsMatMul(w, x, n, y, n, T),
		d.PMix([]Ink2Ref{x, y}, query, 1, mixed, n, T),
		d.PCtrlDown([]Ink2Ref{x}, []Ink2Ref{y}, []Ink2Ref{down}, []Ink2Ref{g}, n, 4, T),
		d.PConv(mixed, convW, Ink2Ref{}, hist, conv, Ink2Ref{}, n, 1, 4, .5, T),
		d.PQKVPost(q, r, kx, vx, qclip, rclip, kc, vc, p, cap, 3, T),
		d.PAttend(q, r, sinks, rel, kc, vc, scores, T*p.Heads*cap, attn, p, 8, 0, cap, 3, T),
		d.PRoute(logits, Ink2Ref{}, ids, wts, nexp, k, 1, T),
		d.PBucket(ids, scratch, scratch.At(nexp), scratch.At(2*nexp+1), T*k, nexp),
		d.PExpertGrouped(stack, ids, scratch.At(2*nexp+1), conv, k, n, pairs, T*k, Ink2Ref{}),
		d.PCombine(pairs, wts, out, k, n, T),
	} {
		if err != nil {
			t.Fatal(err)
		}
	}
	s := d.Stats()
	if s.Submissions != before.Submissions || s.Pending != 11 {
		t.Fatalf("a chunk's ops submitted or miscounted their launches: %+v after %+v", s, before)
	}
	got := ink2OpsTestRead(t, d, out, T*n)
	if s := d.Stats(); s.Submissions != before.Submissions+1 || s.Pending != 0 {
		t.Fatalf("the read fence was not one submission: %+v", s)
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("out[%d] = %v", i, v)
		}
	}
}
