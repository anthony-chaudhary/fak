//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

func ink2ConvTestRand(rng *rand.Rand, n int, amp float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = (rng.Float32() - .5) * 2 * amp
	}
	return out
}

// ink2ConvTestPut allocates pad+len(vals)+pad elements and returns the
// reference of vals inside them, so an op runs at a nonzero offset.
func ink2ConvTestPut(t *testing.T, d *Ink2Device, pad int, vals []float32) Ink2Ref {
	t.Helper()
	r := ink2TestAlloc(t, d, 2*pad+len(vals)).At(pad)
	if err := d.Write(r, vals); err != nil {
		t.Fatal(err)
	}
	return r
}

func ink2ConvTestRead(t *testing.T, d *Ink2Device, r Ink2Ref, n int) []float32 {
	t.Helper()
	got, err := d.Read(r, n)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

// ink2ConvTestDrift is how far a run strayed from the host: the values that
// differ in any bit, and the largest error as a fraction of the allowed one.
type ink2ConvTestDrift struct {
	inexact int
	worst   float64
}

// parity holds got to the CUDA conformance bound around want, mag being the
// magnitude of the terms that produced each element.
func (s *ink2ConvTestDrift) parity(t *testing.T, what string, got, want []float32, mag []float64) {
	t.Helper()
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			s.inexact++
		}
		s.worst = max(s.worst, math.Abs(float64(got[i])-float64(want[i]))/(1e-5*mag[i]+1e-6))
		if !ink2TestClose(got[i], float64(want[i]), mag[i]) {
			t.Fatalf("%s element %d = %v, want %v (magnitude %v)", what, i, got[i], want[i], mag[i])
		}
	}
}

func ink2ConvTestSame(t *testing.T, what string, got, want []float32) {
	t.Helper()
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s element %d = %v, want exactly %v", what, i, got[i], want[i])
		}
	}
}

// ink2ConvTestHost is the host forward's convolve over hist ([n][kn-1]), with
// every product rounded before its add as the CUDA kernel does. mag is the
// magnitude of each output's terms.
func ink2ConvTestHost(x, w, dyn, hist []float32, heads, kn int, norm float32) (out []float32, mag []float64) {
	past := kn - 1
	out, mag = make([]float32, len(x)), make([]float64, len(x))
	for c, xc := range x {
		hc, wc := hist[c*past:(c+1)*past], w[c*kn:(c+1)*kn]
		var acc float32
		m := math.Abs(float64(xc))
		for t := range past {
			acc += float32(hc[t] * wc[t])
			m += math.Abs(float64(hc[t]) * float64(wc[t]))
		}
		acc += float32(xc * wc[past])
		m += math.Abs(float64(xc) * float64(wc[past]))
		if dyn != nil {
			dh := dyn[c/(len(x)/heads)*kn:][:kn]
			for lag := range kn {
				xv := xc
				if lag > 0 {
					xv = hc[past-lag]
				}
				acc += float32(dh[lag] * xv)
				m += math.Abs(float64(dh[lag]) * float64(xv))
			}
		}
		out[c], mag[c] = float32(xc+acc)*norm, m*math.Abs(float64(norm))
		copy(hc, hc[1:])
		hc[past-1] = xc
	}
	return out, mag
}

// fak-test:runtime integration est=2s lane=default
func TestInk2ConvMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(11))
	for _, c := range []struct {
		name         string
		n, heads, kn int
		dyn, resid   bool
	}{
		{"static", 300, 1, 4, false, false},
		{"dynamic with a residual", 512, 4, 4, true, true},
		{"two taps", 37, 1, 2, false, true},
		{"dynamic wide kernel", 96, 3, 7, true, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			past := c.kn - 1
			norm := float32(1 / math.Sqrt(float64(c.kn+1)))
			if c.dyn {
				norm = float32(1 / math.Sqrt(float64(2*c.kn+1)))
			}
			w := ink2ConvTestRand(rng, c.n*c.kn, 1)
			// Channels on either side belong to other convs of the slot.
			const guard = 5
			all := ink2ConvTestRand(rng, (c.n+2*guard)*past, 1)
			hist := append([]float32(nil), all[guard*past:][:c.n*past]...)
			wRef := ink2ConvTestPut(t, d, 0, w)
			histRef := ink2ConvTestPut(t, d, 0, all).At(guard * past)
			xRef, outRef := ink2TestAlloc(t, d, c.n+3).At(3), ink2TestAlloc(t, d, c.n+1).At(1)
			var dynRef, residRef Ink2Ref
			if c.dyn {
				dynRef = ink2TestAlloc(t, d, c.heads*c.kn+2).At(2)
			}
			if c.resid {
				residRef = ink2TestAlloc(t, d, c.n)
			}
			var drift ink2ConvTestDrift
			for range 2*c.kn + 1 {
				x := ink2ConvTestRand(rng, c.n, 2)
				if err := d.Write(xRef, x); err != nil {
					t.Fatal(err)
				}
				var dyn, resid []float32
				if c.dyn {
					dyn = ink2ConvTestRand(rng, c.heads*c.kn, 1)
					if err := d.Write(dynRef, dyn); err != nil {
						t.Fatal(err)
					}
				}
				if c.resid {
					resid = ink2ConvTestRand(rng, c.n, 3)
					if err := d.Write(residRef, resid); err != nil {
						t.Fatal(err)
					}
				}
				before := d.Stats()
				if err := d.Conv(xRef, wRef, dynRef, histRef, outRef, residRef, c.n, c.heads, c.kn, norm); err != nil {
					t.Fatal(err)
				}
				if s := d.Stats(); s.Submissions != before.Submissions || s.Pending != 1 {
					t.Fatalf("conv did not only encode: %+v after %+v", s, before)
				}
				want, mag := ink2ConvTestHost(x, w, dyn, hist, c.heads, c.kn, norm)
				for i := range resid {
					want[i] = resid[i] + want[i]
					mag[i] += math.Abs(float64(resid[i]))
				}
				drift.parity(t, "out", ink2ConvTestRead(t, d, outRef, c.n), want, mag)
				ink2ConvTestSame(t, "history", ink2ConvTestRead(t, d, histRef, c.n*past), hist)
			}
			copy(all[guard*past:], hist)
			ink2ConvTestSame(t, "history with its neighbours", ink2ConvTestRead(t, d, histRef.At(-guard*past), len(all)), all)
			t.Logf("%d outputs differ from the unfused host order in some bit", drift.inexact)
		})
	}
}

// fak-test:runtime integration est=2s lane=default
func TestInk2CombineConvMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(12))
	for _, c := range []struct {
		name     string
		n, k, kn int
		resid    bool
	}{
		{"four slots with a residual", 384, 4, 4, true},
		{"one slot", 70, 1, 4, false},
		{"eight slots, two taps", 257, 8, 2, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			past := c.kn - 1
			norm := float32(1 / math.Sqrt(float64(c.kn+1)))
			w := ink2ConvTestRand(rng, c.n*c.kn, 1)
			hist := ink2ConvTestRand(rng, c.n*past, 1)
			// The history holds combines, so it carries their rounding too.
			histMag := make([]float64, len(hist))
			for i, h := range hist {
				histMag[i] = math.Abs(float64(h))
			}
			wRef := ink2ConvTestPut(t, d, 0, w)
			histRef := ink2ConvTestPut(t, d, 4, hist)
			downRef, wtsRef := ink2TestAlloc(t, d, c.k*c.n+2).At(2), ink2TestAlloc(t, d, c.k+1).At(1)
			outRef := ink2TestAlloc(t, d, c.n)
			var residRef Ink2Ref
			if c.resid {
				residRef = ink2TestAlloc(t, d, c.n)
			}
			var drift ink2ConvTestDrift
			for range c.kn + 2 {
				down, wts := ink2ConvTestRand(rng, c.k*c.n, 2), ink2ConvTestRand(rng, c.k, 1)
				if err := d.Write(downRef, down); err != nil {
					t.Fatal(err)
				}
				if err := d.Write(wtsRef, wts); err != nil {
					t.Fatal(err)
				}
				var resid []float32
				if c.resid {
					resid = ink2ConvTestRand(rng, c.n, 3)
					if err := d.Write(residRef, resid); err != nil {
						t.Fatal(err)
					}
				}
				if err := d.CombineConv(downRef, wtsRef, c.k, wRef, histRef, outRef, residRef, c.n, c.kn, norm); err != nil {
					t.Fatal(err)
				}
				// The combine sums the routed slots in routing order.
				x, xMag := make([]float32, c.n), make([]float64, c.n)
				for i := range x {
					for s := range c.k {
						x[i] += float32(wts[s] * down[s*c.n+i])
						xMag[i] += math.Abs(float64(wts[s]) * float64(down[s*c.n+i]))
					}
				}
				// A combine's rounding reaches the output through x and every tap it passes.
				tapped := make([]float64, c.n)
				for i := range tapped {
					hm := histMag[i*past:][:past]
					tapped[i] = xMag[i] * (1 + math.Abs(float64(w[i*c.kn+past])))
					for s, m := range hm {
						tapped[i] += m * math.Abs(float64(w[i*c.kn+s]))
					}
					copy(hm, hm[1:])
					hm[past-1] = xMag[i]
				}
				want, mag := ink2ConvTestHost(x, w, nil, hist, 1, c.kn, norm)
				for i := range want {
					mag[i] += tapped[i] * float64(norm)
					if c.resid {
						want[i] = resid[i] + want[i]
						mag[i] += math.Abs(float64(resid[i]))
					}
				}
				drift.parity(t, "out", ink2ConvTestRead(t, d, outRef, c.n), want, mag)
				drift.parity(t, "history", ink2ConvTestRead(t, d, histRef, c.n*past), hist, histMag)
			}
			t.Logf("%d values differ from the unfused host order in some bit", drift.inexact)
		})
	}
}

// ink2ConvTestClip is the host forward's inkling2Clip, with the magnitude each
// output is held to.
func ink2ConvTestClip(x, scale []float32, threshold float32) (mag []float64) {
	mag = make([]float64, len(x))
	var ss float32
	for i := range x {
		if scale != nil {
			x[i] *= scale[i]
		}
		ss += float32(x[i] * x[i])
		mag[i] = math.Abs(float64(x[i]))
	}
	f := threshold / (float32(math.Sqrt(float64(ss))) + 1e-6)
	if f > 1 {
		f = 1
	}
	if f < 0 {
		f = 0
	}
	for i := range x {
		x[i] *= f
	}
	return mag
}

type ink2QKVTestCase struct {
	name      string
	p         Ink2AttnParams
	cap, kn   int
	conv, dyn bool
	positions []int
}

// fak-test:runtime integration est=3s lane=default
func TestInk2QKVPostMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(13))
	for _, c := range []ink2QKVTestCase{
		{name: "global with dynamic convs", conv: true, dyn: true, cap: 1100, kn: 4, positions: []int{0, 1, 2, 3, 4, 5, 37, 1023},
			p: Ink2AttnParams{Heads: 6, KVHeads: 2, HeadDim: 40, RelLen: 20, QKClip: 2, RelClip: 1.5, Eps: 1e-6, Global: true, Alpha: .35, Floor: 3}},
		{name: "local ring with static convs", conv: true, cap: 3, kn: 4, positions: []int{0, 1, 2, 3, 4, 5, 6, 7},
			p: Ink2AttnParams{Heads: 4, KVHeads: 4, HeadDim: 64, RelLen: 33, QKClip: 1.5, RelClip: 2, Eps: 1e-5}},
		{name: "global without convs", cap: 16, kn: 4, positions: []int{0, 7, 8, 15},
			p: Ink2AttnParams{Heads: 8, KVHeads: 2, HeadDim: 256, RelLen: 96, QKClip: 5, RelClip: 3, Eps: 1e-6, Global: true, Alpha: .2, Floor: 4}},
		{name: "narrow heads", conv: true, dyn: true, cap: 4, kn: 2, positions: []int{0, 1, 5, 9, 2},
			p: Ink2AttnParams{Heads: 3, KVHeads: 3, HeadDim: 7, RelLen: 1, QKClip: .8, RelClip: .1, Eps: 1e-6, Global: true, Alpha: 1, Floor: 1}},
		{name: "no relative states", cap: 5, kn: 2, positions: []int{0, 4, 5},
			p: Ink2AttnParams{Heads: 4, KVHeads: 2, HeadDim: 16, QKClip: .7, RelClip: 1, Eps: 1e-6, Global: true, Alpha: .5, Floor: 2}},
	} {
		t.Run(c.name, func(t *testing.T) { ink2QKVTestRun(t, d, rng, c) })
	}
}

func ink2QKVTestRun(t *testing.T, d *Ink2Device, rng *rand.Rand, c ink2QKVTestCase) {
	p := c.p
	qn, rn, kvDim, past := p.Heads*p.HeadDim, p.Heads*p.RelLen, p.KVHeads*p.HeadDim, c.kn-1
	qclip, rclip := ink2ConvTestRand(rng, qn, 1.5), ink2ConvTestRand(rng, rn, 1.5)
	kw, vw := ink2ConvTestRand(rng, kvDim*c.kn, 1), ink2ConvTestRand(rng, kvDim*c.kn, 1)
	// One slot history: the k channels, then the v channels.
	hist := ink2ConvTestRand(rng, 2*kvDim*past, 1)
	kcache, vcache := ink2ConvTestRand(rng, c.cap*kvDim, 9), ink2ConvTestRand(rng, c.cap*kvDim, 9)

	qRef, rRef := ink2TestAlloc(t, d, qn), ink2TestAlloc(t, d, rn+1).At(1)
	kRef, vRef := ink2TestAlloc(t, d, kvDim+2).At(2), ink2TestAlloc(t, d, kvDim)
	qclipRef, rclipRef := ink2ConvTestPut(t, d, 0, qclip), ink2ConvTestPut(t, d, 3, rclip)
	kcRef, vcRef := ink2ConvTestPut(t, d, 0, kcache), ink2ConvTestPut(t, d, 5, vcache)
	// The token position follows the input row, as the resident driver lays it out.
	metaRef := ink2TestAlloc(t, d, 9).At(8)
	var cv *Ink2KVConv
	var kdynRef, vdynRef Ink2Ref
	if c.conv {
		histRef := ink2ConvTestPut(t, d, 0, hist)
		cv = &Ink2KVConv{KW: ink2ConvTestPut(t, d, 0, kw), VW: ink2ConvTestPut(t, d, 1, vw),
			KHist: histRef, VHist: histRef.At(kvDim * past), Kn: c.kn,
			KNorm: float32(1 / math.Sqrt(float64(c.kn+1))), VNorm: float32(1 / math.Sqrt(float64(c.kn+1)))}
		if c.dyn {
			kdynRef, vdynRef = ink2TestAlloc(t, d, p.KVHeads*c.kn), ink2TestAlloc(t, d, p.KVHeads*c.kn+1).At(1)
			cv.KDyn, cv.VDyn = kdynRef, vdynRef
			cv.KNorm, cv.VNorm = float32(1/math.Sqrt(float64(2*c.kn+1))), float32(1/math.Sqrt(float64(2*c.kn+1)))
		}
	}

	for _, pos := range c.positions {
		// Head amplitudes straddle the clip thresholds, so some heads shrink and some pass.
		q, r := ink2ConvTestRand(rng, qn, 1), ink2ConvTestRand(rng, rn, 1)
		for hd := range p.Heads {
			amp := .15 * float32(1+hd)
			for i := range p.HeadDim {
				q[hd*p.HeadDim+i] *= amp
			}
			for i := range p.RelLen {
				r[hd*p.RelLen+i] *= amp
			}
		}
		k, v := ink2ConvTestRand(rng, kvDim, 1), ink2ConvTestRand(rng, kvDim, 1)
		for hd := range p.KVHeads {
			for i := range p.HeadDim {
				k[hd*p.HeadDim+i] *= .2 * float32(1+2*hd)
			}
		}
		type put struct {
			ref  Ink2Ref
			vals []float32
		}
		writes := []put{{qRef, q}, {rRef, r}, {kRef, k}, {vRef, v}, {metaRef, []float32{math.Float32frombits(uint32(pos))}}}
		var kdyn, vdyn []float32
		if c.dyn {
			kdyn, vdyn = ink2ConvTestRand(rng, p.KVHeads*c.kn, 1), ink2ConvTestRand(rng, p.KVHeads*c.kn, 1)
			writes = append(writes, put{kdynRef, kdyn}, put{vdynRef, vdyn})
		}
		for _, w := range writes {
			if err := d.Write(w.ref, w.vals); err != nil {
				t.Fatal(err)
			}
		}
		before := d.Stats()
		if err := d.QKVPost(qRef, rRef, kRef, vRef, qclipRef, rclipRef, kcRef, vcRef, metaRef, p, c.cap, cv); err != nil {
			t.Fatal(err)
		}
		if s := d.Stats(); s.Submissions != before.Submissions || s.Pending != 1 {
			t.Fatalf("qkv post did not only encode: %+v after %+v", s, before)
		}

		// The host forward's attention prep.
		kMag, vMag := make([]float64, kvDim), make([]float64, kvDim)
		for i := range kvDim {
			kMag[i], vMag[i] = math.Abs(float64(k[i])), math.Abs(float64(v[i]))
		}
		if c.conv {
			k, kMag = ink2ConvTestHost(k, kw, kdyn, hist[:kvDim*past], p.KVHeads, c.kn, cv.KNorm)
			v, vMag = ink2ConvTestHost(v, vw, vdyn, hist[kvDim*past:], p.KVHeads, c.kn, cv.VNorm)
		}
		qMag, rMag := make([]float64, 0, qn), make([]float64, 0, rn)
		clipped := 0
		for hd := range p.Heads {
			qh, rh := q[hd*p.HeadDim:][:p.HeadDim], r[hd*p.RelLen:][:p.RelLen]
			qMag = append(qMag, ink2ConvTestClip(qh, qclip[hd*p.HeadDim:][:p.HeadDim], p.QKClip)...)
			rMag = append(rMag, ink2ConvTestClip(rh, rclip[hd*p.RelLen:][:p.RelLen], p.RelClip)...)
			if math.Abs(float64(qh[0])) < qMag[hd*p.HeadDim] {
				clipped++
			}
		}
		if p.Heads > 3 && (clipped == 0 || clipped == p.Heads) {
			t.Fatalf("%d of %d q heads clipped: the case does not cover both sides of the threshold", clipped, p.Heads)
		}
		for hd := range p.KVHeads {
			kh, vh := k[hd*p.HeadDim:][:p.HeadDim], v[hd*p.HeadDim:][:p.HeadDim]
			ink2ConvTestClip(kh, nil, p.QKClip)
			var ss float64
			for _, x := range vh {
				ss += float64(x) * float64(x)
			}
			s := float32(1 / math.Sqrt(ss/float64(p.HeadDim)+float64(p.Eps)))
			for i := range vh {
				vh[i] *= s
				// The scale carries the relative error of the whole head's sum.
				vMag[hd*p.HeadDim+i] = math.Max(vMag[hd*p.HeadDim+i]*float64(s), math.Abs(float64(vh[i])))
			}
		}
		if p.Global {
			tau := float32(1 + float64(p.Alpha)*math.Log(math.Max(1, float64(pos+1)/float64(p.Floor))))
			for i := range q {
				q[i] *= tau
				qMag[i] *= float64(tau)
			}
			for i := range r {
				r[i] *= tau
				rMag[i] *= float64(tau)
			}
		}

		var drift ink2ConvTestDrift
		drift.parity(t, "q", ink2ConvTestRead(t, d, qRef, qn), q, qMag)
		drift.parity(t, "r", ink2ConvTestRead(t, d, rRef, rn), r, rMag)
		row := (pos % c.cap) * kvDim
		gotK, gotV := ink2ConvTestRead(t, d, kcRef, c.cap*kvDim), ink2ConvTestRead(t, d, vcRef, c.cap*kvDim)
		drift.parity(t, "cached k", gotK[row:][:kvDim], k, kMag)
		drift.parity(t, "cached v", gotV[row:][:kvDim], v, vMag)
		// Every other cache row keeps its bits.
		copy(kcache[row:], gotK[row:][:kvDim])
		copy(vcache[row:], gotV[row:][:kvDim])
		ink2ConvTestSame(t, "k cache", gotK, kcache)
		ink2ConvTestSame(t, "v cache", gotV, vcache)
		if c.conv {
			ink2ConvTestSame(t, "history", ink2ConvTestRead(t, d, cv.KHist, len(hist)), hist)
		}
		t.Logf("position %d: %d of %d values differ from the host in some bit, the worst by %.3f of the bound", pos, drift.inexact, qn+rn+2*kvDim, drift.worst)
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2ConvAndQKVPostRejectBadOperands(t *testing.T) {
	d := ink2TestDevice(t)
	r := ink2TestAlloc(t, d, 64)
	var none Ink2Ref
	p := Ink2AttnParams{Heads: 2, KVHeads: 1, HeadDim: 4, RelLen: 2, QKClip: 1, RelClip: 1}
	cv := Ink2KVConv{KW: r, VW: r, KHist: r, VHist: r, Kn: 2, KNorm: 1, VNorm: 1}
	noTaps, shortKernel := cv, cv
	noTaps.KW, shortKernel.Kn = none, 1
	for name, err := range map[string]error{
		"conv without an input":          d.Conv(none, r, none, r, r, none, 4, 1, 2, 1),
		"conv without taps":              d.Conv(r, none, none, r, r, none, 4, 1, 2, 1),
		"conv without a history":         d.Conv(r, r, none, none, r, none, 4, 1, 2, 1),
		"conv without an output":         d.Conv(r, r, none, r, none, none, 4, 1, 2, 1),
		"conv of no channels":            d.Conv(r, r, none, r, r, none, 0, 1, 2, 1),
		"conv of one tap":                d.Conv(r, r, none, r, r, none, 4, 1, 1, 1),
		"conv of no heads":               d.Conv(r, r, none, r, r, none, 4, 0, 2, 1),
		"conv with uneven dynamic heads": d.Conv(r, r, r, r, r, none, 4, 3, 2, 1),
		"combine conv without weights":   d.CombineConv(r, none, 2, r, r, r, none, 4, 2, 1),
		"combine conv of no slots":       d.CombineConv(r, r, 0, r, r, r, none, 4, 2, 1),
		"qkv post without heads":         d.QKVPost(r, r, r, r, r, r, r, r, r, Ink2AttnParams{}, 1, nil),
		"qkv post without a position":    d.QKVPost(r, r, r, r, r, r, r, r, none, p, 1, nil),
		"qkv post without a cache":       d.QKVPost(r, r, r, r, r, r, none, r, r, p, 1, nil),
		"qkv post of an empty cache":     d.QKVPost(r, r, r, r, r, r, r, r, r, p, 0, nil),
		"qkv post conv without taps":     d.QKVPost(r, r, r, r, r, r, r, r, r, p, 1, &noTaps),
		"qkv post conv of one tap":       d.QKVPost(r, r, r, r, r, r, r, r, r, p, 1, &shortKernel),
	} {
		if !errors.Is(err, ErrInk2Shape) {
			t.Errorf("%s: %v, want ErrInk2Shape", name, err)
		}
	}
	// An operand shorter than the op's extent is refused before it is encoded.
	for name, err := range map[string]error{
		"conv past its history":   d.Conv(r, r, none, r.At(40), r, none, 16, 1, 4, 1),
		"combine conv past down":  d.CombineConv(r.At(8), r, 8, r, r, r, none, 8, 2, 1),
		"qkv post past its cache": d.QKVPost(r, r, r, r, r, r, r, r, r, p, 17, &cv),
	} {
		if err == nil {
			t.Errorf("%s: encoded", name)
		}
	}
	if s := d.Stats(); s.Dispatches != 0 || s.Pending != 0 {
		t.Fatalf("a rejected op encoded work: %+v", s)
	}
}
