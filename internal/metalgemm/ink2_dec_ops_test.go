//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// ink2DecTestArms runs encode once with the lean kernels and once with every
// legacy kernel selected, returning what read fetches after each.
func ink2DecTestArms[T any](t *testing.T, encode func() error, read func() T) (lean, legacy T) {
	t.Helper()
	saved := ink2OpsLegacy
	defer func() { ink2OpsLegacy = saved }()
	for _, arm := range []struct {
		ops ink2DecLegacyOps
		dst *T
	}{{ink2DecLegacyOps{}, &lean}, {ink2DecLegacyOps{mix: true, ctrl: true, rms: true, route: true}, &legacy}} {
		ink2OpsLegacy = arm.ops
		if err := encode(); err != nil {
			t.Fatal(err)
		}
		*arm.dst = read()
	}
	return lean, legacy
}

func TestInk2DecLegacyEnvNamesTheOpFamilies(t *testing.T) {
	for in, want := range map[string]ink2DecLegacyOps{
		"":                    {},
		"mix":                 {mix: true},
		" Route , ctrl":       {ctrl: true, route: true},
		"gemv,scalar,rms,mix": {mix: true, rms: true},
		"mix,ctrl,route,rms":  {mix: true, ctrl: true, rms: true, route: true},
	} {
		if got := parseInk2OpsLegacy(in); got != want {
			t.Errorf("parseInk2OpsLegacy(%q) = %+v, want %+v", in, got, want)
		}
	}
}

// fak-test:runtime integration est=2s lane=default
func TestInk2LeanMixMatchesLegacyBitForBit(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(31))
	var drift ink2OpsTestDrift
	for _, c := range []struct {
		ns, nq, n int
		// norms: 0 none, 1 every query weighted, 2 query 0 normalized without a weight.
		norms int
	}{{1, 1, 9, 0}, {2, 4, 2304, 0}, {2, 1, 2304, 1}, {3, 3, 2304, 1}, {4, 3, 2304, 0}, {5, 1, 2304, 1}, {5, 3, 2304, 1},
		{5, 1, 2304, 0}, {9, 1, 2304, 1}, {12, 3, 2304, 1}, {16, 1, 2304, 1}, {16, 4, 2304, 0}, {11, 2, 1000, 2}, {8, 4, 2048, 2}, {7, 2, 777, 1}, {6, 4, 257, 0}, {8, 1, 255, 2}, {3, 2, 4099, 1}, {4, 3, 3072, 1}, {5, 2, 3071, 2}, {2, 1, 3, 1}} {
		what := fmt.Sprintf("mix %d states x %d queries of %d, norms %d", c.ns, c.nq, c.n, c.norms)
		states, refs := make([][]float32, c.ns), make([]Ink2Ref, c.ns)
		arena := ink2TestAlloc(t, d, c.ns*(c.n+3))
		for i := range states {
			states[i] = ink2OpsTestRand(rng, c.n, math.Pow(10, float64(i%5-2)))
			refs[i] = arena.At(i*(c.n+3) + 1)
			if err := d.Write(refs[i], states[i]); err != nil {
				t.Fatal(err)
			}
		}
		queries := ink2OpsTestRand(rng, c.nq*c.n, 1.5)
		var norms []Ink2Ref
		weights := make([][]float32, c.nq)
		const eps = 1e-5
		if c.norms > 0 {
			norms = make([]Ink2Ref, c.nq)
			for q := range norms {
				if c.norms == 2 && q == 0 {
					continue
				}
				weights[q] = ink2OpsTestRand(rng, c.n, 1)
				norms[q] = ink2OpsTestPut(t, d, q, weights[q])
			}
		}
		qr, out := ink2OpsTestPut(t, d, 4, queries), ink2TestAlloc(t, d, c.nq*c.n+2).At(2)
		lean, legacy := ink2DecTestArms(t,
			func() error { return d.Mix(refs, qr, c.nq, norms, eps, out, c.n) },
			func() []float32 { return ink2OpsTestRead(t, d, out, c.nq*c.n) })
		ink2OpsTestSame(t, what, lean, legacy)
		for q := range c.nq {
			want, mag := ink2OpsTestHostMix(states, queries[q*c.n:(q+1)*c.n])
			if c.norms > 0 {
				want, mag = ink2OpsTestHostNorm(want, mag, weights[q], eps)
			}
			drift.parity(t, fmt.Sprintf("%s query %d", what, q), lean[q*c.n:(q+1)*c.n], want, mag)
		}
	}
	drift.report(t)
}

// fak-test:runtime integration est=1s lane=default
func TestInk2LeanRMSMatchesLegacyBitForBit(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(32))
	var drift ink2OpsTestDrift
	for _, c := range []struct {
		rows, n int
		// weights: 0 none, 1 one shared, 2 one per row with row 0 unweighted.
		weights int
		eps     float32
	}{{1, 5, 0, 1e-6}, {1, 2304, 1, 1e-5}, {3, 2304, 2, 1e-5}, {4, 1000, 2, 1e-5}, {2, 257, 1, 0}, {3, 4099, 0, 1e-6}, {2, 3072, 2, 1e-5}, {1, 3071, 1, 1e-5}} {
		what := fmt.Sprintf("rms %dx%d, weights %d", c.rows, c.n, c.weights)
		x := ink2OpsTestRand(rng, c.rows*c.n, 3)
		host := make([][]float32, c.rows)
		var w []Ink2Ref
		switch c.weights {
		case 0:
			w = []Ink2Ref{{}}
		case 1:
			host[0] = ink2OpsTestRand(rng, c.n, 1)
			w = []Ink2Ref{ink2OpsTestPut(t, d, 3, host[0])}
			for r := range host {
				host[r] = host[0]
			}
		case 2:
			w = make([]Ink2Ref, c.rows)
			for r := 1; r < c.rows; r++ {
				host[r] = ink2OpsTestRand(rng, c.n, 1)
				w[r] = ink2OpsTestPut(t, d, r, host[r])
			}
		}
		xr, yr := ink2OpsTestPut(t, d, 7, x), ink2TestAlloc(t, d, c.rows*c.n+9).At(9)
		// The second launch normalizes the first one's output in place.
		lean, legacy := ink2DecTestArms(t, func() error {
			if err := d.rmsRows(xr, yr, w, c.weights == 2, c.rows, c.n, c.eps, ink2OpsLegacy.rms); err != nil {
				return err
			}
			return d.rmsRows(yr, yr, w, c.weights == 2, c.rows, c.n, c.eps, ink2OpsLegacy.rms)
		}, func() []float32 { return ink2OpsTestRead(t, d, yr, c.rows*c.n) })
		ink2OpsTestSame(t, what, lean, legacy)
		for r := range c.rows {
			once := ink2OpsTestHostRMS(x[r*c.n:(r+1)*c.n], host[r], c.eps)
			drift.parity(t, fmt.Sprintf("%s row %d", what, r), lean[r*c.n:(r+1)*c.n], ink2OpsTestHostRMS(once, host[r], c.eps), nil)
		}
	}
	drift.report(t)
}

// fak-test:runtime integration est=2s lane=default
func TestInk2LeanRouteMatchesLegacyBitForBit(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(33))
	type routed struct {
		ids []int32
		wts []float32
	}
	for round := range 40 {
		for _, c := range []struct {
			nexp, k int
			biased  bool
			scale   float32
		}{{128, 8, false, 8}, {128, 8, true, 8}, {128, 16, true, 0}, {4, 4, false, 0}, {33, 1, true, 2.5}, {96, 8, true, 8},
			{70, 16, false, 1}, {127, 9, false, 0}, {5, 5, true, 3}} {
			what := fmt.Sprintf("route %d experts to %d, biased %v, scale %v, round %d", c.nexp, c.k, c.biased, c.scale, round)
			logits := ink2OpsTestRand(rng, c.nexp, 1.5)
			switch round % 4 {
			case 1:
				// Ties across lanes and within one, among them saturated ones.
				for i := range logits {
					logits[i] = float32(rng.Intn(4)) - 1
				}
				logits[rng.Intn(c.nexp)], logits[rng.Intn(c.nexp)] = 60, 50
			case 2:
				// More slots than distinct scores, and probabilities of zero.
				for i := range logits {
					logits[i] = -200
				}
				logits[rng.Intn(c.nexp)] = 1
			}
			var bias []float32
			var br Ink2Ref
			if c.biased {
				bias = ink2OpsTestRand(rng, c.nexp, 0.2)
				if round%4 == 3 {
					clear(bias)
				}
				br = ink2OpsTestPut(t, d, 2, bias)
			}
			lr := ink2OpsTestPut(t, d, 1, logits)
			ids, wts := ink2TestAlloc(t, d, c.k+3).At(3), ink2TestAlloc(t, d, c.k+1).At(1)
			lean, legacy := ink2DecTestArms(t,
				func() error { return d.Route(lr, br, ids, wts, c.nexp, c.k, c.scale) },
				func() routed {
					got, err := d.ReadInts(ids, c.k)
					if err != nil {
						t.Fatal(err)
					}
					return routed{got, ink2OpsTestRead(t, d, wts, c.k)}
				})
			if !slices.Equal(lean.ids, legacy.ids) {
				t.Fatalf("%s: ids %v, legacy %v", what, lean.ids, legacy.ids)
			}
			ink2OpsTestSame(t, what+" weights", lean.wts, legacy.wts)
			if round%4 != 0 {
				// Exact ties: the host's order is the only right one.
				probs, sel, top := ink2OpsTestHostRoute(logits, bias, c.k)
				for s, id := range lean.ids {
					if id != top[s] && sel[id] != sel[top[s]] {
						t.Fatalf("%s: slot %d selects %d, host %d", what, s, id, top[s])
					}
				}
				var drift ink2OpsTestDrift
				drift.parity(t, what+" weights", lean.wts, ink2OpsTestHostWeights(probs, lean.ids, c.scale), nil)
			}
		}
	}
}

// fak-test:runtime integration est=2s lane=default
func TestInk2LeanControlMatchesLegacyBitForBit(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(34))
	var drift ink2OpsTestDrift
	for _, c := range []struct{ nj, n, rank int }{{1, 2304, 64}, {3, 2304, 64}, {4, 1021, 7}, {2, 3072, 16}, {4, 777, 48}, {1, 9, 32}, {2, 4100, 80}, {3, 300, 20}, {2, 2304, 128}, {1, 512, 96}} {
		what := fmt.Sprintf("control %d of width %d, rank %d", c.nj, c.n, c.rank)
		var ctl Ink2Controls
		type host struct{ cur, msg, down, gain, forget, norm []float32 }
		hosts, norms := make([]host, c.nj), make([]Ink2Ref, c.nj)
		const eps = 1e-5
		for j := range hosts {
			h := &hosts[j]
			h.cur, h.msg = ink2OpsTestRand(rng, c.n, 2), ink2OpsTestRand(rng, c.n, 0.5)
			h.down = ink2OpsTestRand(rng, c.rank*2*c.n, 1/math.Sqrt(float64(2*c.n)))
			h.gain = ink2OpsTestRand(rng, c.n*c.rank, 1/math.Sqrt(float64(c.rank)))
			h.forget = ink2OpsTestRand(rng, c.n*c.rank, 1/math.Sqrt(float64(c.rank)))
			h.norm = ink2OpsTestRand(rng, c.n, 1)
			ctl.Cur = append(ctl.Cur, ink2OpsTestPut(t, d, j, h.cur))
			ctl.Msg = append(ctl.Msg, ink2OpsTestPut(t, d, 3, h.msg))
			ctl.Down = append(ctl.Down, ink2OpsTestPut(t, d, 1, h.down))
			ctl.Gain = append(ctl.Gain, ink2OpsTestPut(t, d, 2, h.gain))
			ctl.Forget = append(ctl.Forget, ink2OpsTestPut(t, d, 5, h.forget))
			norms[j] = ink2OpsTestPut(t, d, 6, h.norm)
		}
		g, out := ink2TestAlloc(t, d, c.nj*c.rank+1).At(1), ink2TestAlloc(t, d, c.nj*c.n+3).At(3)
		lean, legacy := ink2DecTestArms(t,
			func() error { return d.Control(ctl, g, out, c.n, c.rank, norms, eps) },
			func() []float32 { return ink2OpsTestRead(t, d, out, c.nj*c.n) })
		ink2OpsTestSame(t, what, lean, legacy)
		for j, h := range hosts {
			want, mag := ink2OpsTestHostControl(h.cur, h.msg, h.down, h.gain, h.forget, c.rank)
			want, mag = ink2OpsTestHostNorm(want, mag, h.norm, eps)
			drift.parity(t, fmt.Sprintf("%s controller %d", what, j), lean[j*c.n:(j+1)*c.n], want, mag)
		}
	}
	drift.report(t)
}

// fak-test:runtime integration est=1s lane=default
func TestInk2LeanRouteRowsMatchLegacyBitForBit(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(35))
	type routed struct {
		ids []int32
		wts []float32
	}
	for _, c := range []struct{ nexp, k, T int }{{128, 8, 1}, {128, 8, 5}, {128, 8, 9}, {96, 16, 3}, {33, 4, 2}} {
		what := fmt.Sprintf("route rows %d experts to %d for %d tokens", c.nexp, c.k, c.T)
		lr := ink2OpsTestPut(t, d, 1, ink2OpsTestRand(rng, c.T*c.nexp, 1.5))
		br := ink2OpsTestPut(t, d, 2, ink2OpsTestRand(rng, c.nexp, 0.2))
		ids, wts := ink2TestAlloc(t, d, c.T*c.k+3).At(3), ink2TestAlloc(t, d, c.T*c.k+1).At(1)
		lean, legacy := ink2DecTestArms(t,
			func() error { return d.RouteRows(lr, br, ids, wts, c.nexp, c.k, 2.5, c.T) },
			func() routed {
				got, err := d.ReadInts(ids, c.T*c.k)
				if err != nil {
					t.Fatal(err)
				}
				return routed{got, ink2OpsTestRead(t, d, wts, c.T*c.k)}
			})
		if !slices.Equal(lean.ids, legacy.ids) {
			t.Fatalf("%s: ids %v, legacy %v", what, lean.ids, legacy.ids)
		}
		ink2OpsTestSame(t, what+" weights", lean.wts, legacy.wts)
	}
}
