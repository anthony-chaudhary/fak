//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

// The global-memory score rows run the shared-memory attend's arithmetic unchanged,
// so the two placements (and any token tiling of the global one) must agree bit for
// bit; spans past the shared-memory opt-in have only the global path, so there two
// tilings are compared and a few (head, token) rows are checked against a float64
// host reference.
// fak-test:runtime integration est=10s lane=optin
func TestCUDAInk2PAttendGlobalMatchesShared(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	defer ink2PSelectAttendGlobal(ink2PSelectAttendGlobal(false))
	defer ink2PAttendGlobalBudget(ink2PAttendGlobalBudget(0))
	r := rand.New(rand.NewSource(41))
	p := Ink2AttnParams{Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16, Scale: float32(1 / math.Sqrt(128))}
	kvDim := p.KVHeads * p.HeadDim
	alloc := func(n int) Tensor {
		tt, err := cb.Ink2Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		return tt
	}
	fill := func(tt Tensor, n int, sd float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(sd * r.NormFloat64())
		}
		if err := cb.Ink2Write(Ink2At(tt, 0), v); err != nil {
			t.Fatal(err)
		}
		return v
	}
	for _, c := range []struct {
		cap, window, extent, pos0, T int
		sharedFits                   bool
	}{
		{1024, 1024, 1024, 2000, 64, true},
		{4096, 0, 1024, 4096 - 48, 48, true},
		{30000, 0, 1024, 30000 - 16, 16, false},
	} {
		span := c.pos0 + c.T
		if c.window > 0 && span > c.window {
			span = c.window
		}
		if span > c.cap {
			span = c.cap
		}
		qn, rn := c.T*p.Heads*p.HeadDim, c.T*p.Heads*p.RelLen
		q, rr, sinks, rel := alloc(qn), alloc(rn), alloc(c.T*p.Heads), alloc(c.extent*p.RelLen)
		kc, vc := alloc(c.cap*kvDim), alloc(c.cap*kvDim)
		qv := fill(q, qn, 1)
		rv := fill(rr, rn, 1)
		sv := fill(sinks, c.T*p.Heads, 1)
		relv := fill(rel, c.extent*p.RelLen, 0.1)
		kv := fill(kc, c.cap*kvDim, 0.3)
		vv := fill(vc, c.cap*kvDim, 1)
		run := func(global bool, budget uint64) []float32 {
			ink2PSelectAttendGlobal(global)
			ink2PAttendGlobalBudget(budget)
			out := alloc(c.T * p.Heads * p.HeadDim)
			if err := cb.Ink2PAttend(Ink2At(q, 0), Ink2At(rr, 0), Ink2At(sinks, 0), Ink2At(rel, 0), Ink2At(kc, 0), Ink2At(vc, 0),
				Ink2At(out, 0), p, c.extent, c.window, c.cap, c.pos0, c.T); err != nil {
				t.Fatalf("cap %d span %d global %v budget %d: %v", c.cap, span, global, budget, err)
			}
			got, err := cb.Ink2Read(Ink2At(out, 0), c.T*p.Heads*p.HeadDim)
			if err != nil {
				t.Fatal(err)
			}
			return got
		}
		row := uint64(p.Heads * span * 4)
		want := run(false, 64<<20)
		for _, v := range []struct {
			name   string
			budget uint64
		}{{"one tile", 64 << 20}, {"5-token tiles", 5 * row}, {"1-token tiles", 1}} {
			got := run(true, v.budget)
			for i := range want {
				if math.IsNaN(float64(want[i])) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("cap %d span %d %s: out[%d] global %v, reference %v", c.cap, span, v.name, i, got[i], want[i])
				}
			}
		}
		if c.sharedFits {
			continue
		}
		for _, hp := range [][2]int{{0, 0}, {5, c.T - 1}, {17, c.T / 2}, {p.Heads - 1, 3}} {
			h, tok := hp[0], hp[1]
			pos := c.pos0 + tok
			n := pos + 1
			if c.window > 0 && n > c.window {
				n = c.window
			}
			if n > c.cap {
				n = c.cap
			}
			first := pos + 1 - n
			kvh := h / (p.Heads / p.KVHeads)
			qh := qv[(tok*p.Heads+h)*p.HeadDim:][:p.HeadDim]
			rh := rv[(tok*p.Heads+h)*p.RelLen:][:p.RelLen]
			sink := float64(sv[tok*p.Heads+h])
			sc := make([]float64, n)
			m := math.Inf(-1)
			for j := range sc {
				pp := first + j
				dist := pos - pp
				kr := kv[(pp%c.cap)*kvDim+kvh*p.HeadDim:][:p.HeadDim]
				var d float64
				for i := range kr {
					d += float64(qh[i]) * float64(kr[i])
				}
				s := d*float64(p.Scale) - sink
				if dist < c.extent {
					rb := relv[dist*p.RelLen:][:p.RelLen]
					for i := range rb {
						s += float64(rb[i]) * float64(rh[i])
					}
				}
				sc[j] = s
				m = math.Max(m, s)
			}
			m = math.Max(m, 0)
			denom := math.Exp(-m)
			for j := range sc {
				sc[j] = math.Exp(sc[j] - m)
				denom += sc[j]
			}
			for dd := 0; dd < p.HeadDim; dd++ {
				var acc, mag float64
				for j, w := range sc {
					x := w / denom * float64(vv[((first+j)%c.cap)*kvDim+kvh*p.HeadDim+dd])
					acc += x
					mag += math.Abs(x)
				}
				g := float64(want[(tok*p.Heads+h)*p.HeadDim+dd])
				if math.Abs(g-acc) > 1e-4*mag+1e-6 {
					t.Fatalf("cap %d span %d head %d tok %d dim %d: device %v, host %v (mag %v)", c.cap, span, h, tok, dd, g, acc, mag)
				}
			}
		}
	}
}
