//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func ink2OpsTestRand(rng *rand.Rand, n int, amp float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64() * amp)
	}
	return v
}

// ink2OpsTestPut uploads vals pad elements into a buffer of its own, so an op
// is exercised at a nonzero element offset.
func ink2OpsTestPut(t *testing.T, d *Ink2Device, pad int, vals []float32) Ink2Ref {
	t.Helper()
	r := ink2TestAlloc(t, d, pad+len(vals)+1).At(pad)
	if err := d.Write(r, vals); err != nil {
		t.Fatal(err)
	}
	return r
}

func ink2OpsTestRead(t *testing.T, d *Ink2Device, r Ink2Ref, n int) []float32 {
	t.Helper()
	got, err := d.Read(r, n)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

// ink2OpsTestDrift checks device values against the host reference under the
// CUDA expert conformance bound and remembers the worst use of that bound.
type ink2OpsTestDrift struct{ worst float64 }

func (s *ink2OpsTestDrift) parity(t *testing.T, what string, got, want []float32, mag []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range want {
		m := math.Abs(float64(want[i]))
		if mag != nil {
			m = mag[i]
		}
		diff := math.Abs(float64(got[i]) - float64(want[i]))
		s.worst = max(s.worst, diff/(1e-5*m+1e-6))
		if !ink2TestClose(got[i], float64(want[i]), m) {
			t.Fatalf("%s[%d] = %v, want %v (magnitude %v)", what, i, got[i], want[i], m)
		}
	}
}

func (s *ink2OpsTestDrift) report(t *testing.T) {
	t.Logf("worst drift is %.3g of the bound", s.worst)
}

func ink2OpsTestSame(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] = %v (%#x), want %v (%#x)", what, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

// The host references below restate internal/model's inkling2_loop forward
// (inkling2RMS, inkling2Mix, controls, moe, situ), which this package cannot
// import. They round to float32 where the host does, except that a dot
// accumulates in float64, as the CUDA conformance reference does: the host's
// sequential float32 dot strays further from that sum than the bound allows.

func ink2OpsTestHostRMS(x, w []float32, eps float32) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	s := float32(1 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * s
		if w != nil {
			out[i] *= w[i]
		}
	}
	return out
}

func ink2OpsTestHostDot(a, b []float32) float32 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return float32(s)
}

// ink2OpsTestHostMix returns the mix and, per output, the summed magnitude of
// its terms.
func ink2OpsTestHostMix(states [][]float32, query []float32) ([]float32, []float64) {
	n := len(query)
	normed := make([][]float32, len(states))
	for i, s := range states {
		normed[i] = ink2OpsTestHostRMS(s, nil, 1e-6)
	}
	inv := 1 / math.Sqrt(float64(n))
	scores := make([]float64, len(states))
	maxS := math.Inf(-1)
	for i, s := range normed {
		scores[i] = float64(ink2OpsTestHostDot(s, query)) * inv
		maxS = math.Max(maxS, scores[i])
	}
	var sum float64
	for i := range scores {
		scores[i] = math.Exp(scores[i] - maxS)
		sum += scores[i]
	}
	out, mag := make([]float32, n), make([]float64, n)
	for i, s := range normed {
		w := float32(scores[i] / sum)
		for j := range out {
			out[j] += w * s[j]
			mag[j] += math.Abs(float64(w) * float64(s[j]))
		}
	}
	return out, mag
}

// ink2OpsTestHostNorm RMS-normalizes a row whose error scales with mag.
func ink2OpsTestHostNorm(x []float32, mag []float64, w []float32, eps float32) ([]float32, []float64) {
	out, outMag := ink2OpsTestHostRMS(x, w, eps), make([]float64, len(x))
	unit := ink2OpsTestHostRMS(x, nil, eps)
	var scale float64
	for i, v := range x {
		if v != 0 {
			scale = float64(unit[i]) / float64(v)
			break
		}
	}
	for i := range x {
		outMag[i] = mag[i] * scale
		if w != nil {
			outMag[i] *= math.Abs(float64(w[i]))
		}
	}
	return out, outMag
}

// ink2OpsTestHostControl is one loop controller: down is [rank][2n], gain and
// forget are [n][rank].
func ink2OpsTestHostControl(cur, msg, down, gain, forget []float32, rank int) ([]float32, []float64) {
	n := len(cur)
	pair := append(ink2OpsTestHostRMS(cur, nil, 1e-6), ink2OpsTestHostRMS(msg, nil, 1e-6)...)
	g := make([]float32, rank)
	for r := range g {
		g[r] = ink2OpsTestHostDot(down[r*2*n:(r+1)*2*n], pair)
	}
	out, mag := make([]float32, n), make([]float64, n)
	for j, x := range cur {
		f := float32(math.Tanh(float64(ink2OpsTestHostDot(forget[j*rank:(j+1)*rank], g))))
		tg := float32(math.Tanh(float64(ink2OpsTestHostDot(gain[j*rank:(j+1)*rank], g))))
		out[j] = x - f*x + tg*msg[j]
		mag[j] = math.Abs(float64(x))*(1+math.Abs(float64(f))) + math.Abs(float64(tg)*float64(msg[j]))
	}
	return out, mag
}

func ink2OpsTestSigmoid(x float32) float32 { return float32(1 / (1 + math.Exp(float64(-x)))) }

// ink2OpsTestHostRoute returns the host's probabilities, selection scores and
// top-k (ties to the lower index).
func ink2OpsTestHostRoute(logits, bias []float32, k int) (probs, sel []float32, top []int32) {
	probs, sel = make([]float32, len(logits)), make([]float32, len(logits))
	for i, l := range logits {
		probs[i] = ink2OpsTestSigmoid(l)
		sel[i] = probs[i]
		if bias != nil {
			sel[i] += bias[i]
		}
	}
	used := make([]bool, len(sel))
	for range k {
		best := -1
		for i, x := range sel {
			if !used[i] && (best < 0 || x > sel[best]) {
				best = i
			}
		}
		used[best] = true
		top = append(top, int32(best))
	}
	return probs, sel, top
}

func ink2OpsTestHostWeights(probs []float32, top []int32, scale float32) []float32 {
	weights := make([]float32, len(top))
	var sum float32
	for i, e := range top {
		weights[i] = probs[e]
		sum += weights[i]
	}
	for i := range weights {
		weights[i] = weights[i] / (sum + 1e-20)
		if scale != 0 {
			weights[i] *= scale
		}
	}
	return weights
}

func ink2OpsTestHostSitu(gu []float32, width int, beta, lb float64) []float32 {
	act := make([]float32, width)
	for j := range act {
		g, u := float64(gu[j]), float64(gu[width+j])
		a := beta * math.Tanh(g/beta) * (1 / (1 + math.Exp(-g)))
		if lb > 0 {
			u = lb * math.Tanh(u/lb)
		}
		act[j] = float32(a * u)
	}
	return act
}

func ink2OpsTestHostCombine(down, wts []float32, n int) []float32 {
	out := make([]float32, n)
	for t, w := range wts {
		for j := range out {
			out[j] += float32(w * down[t*n+j])
		}
	}
	return out
}

// fak-test:runtime integration est=1s lane=default
func TestInk2OpsRMSMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(11))
	var drift ink2OpsTestDrift
	for _, c := range []struct {
		rows, n  int
		weighted bool
		eps      float32
	}{{1, 5, false, 1e-6}, {1, 256, true, 1e-5}, {3, 1000, true, 1e-5}, {5, 2048, false, 1e-6}, {2, 257, true, 0}} {
		x := ink2OpsTestRand(rng, c.rows*c.n, 3)
		var w []float32
		var wr Ink2Ref
		if c.weighted {
			w = ink2OpsTestRand(rng, c.n, 1)
			wr = ink2OpsTestPut(t, d, 3, w)
		}
		xr, yr := ink2OpsTestPut(t, d, 7, x), ink2TestAlloc(t, d, c.rows*c.n+9).At(9)
		if err := d.RMS(xr, wr, yr, c.rows, c.n, c.eps); err != nil {
			t.Fatal(err)
		}
		// In place over the same input.
		if err := d.RMS(xr, wr, xr, c.rows, c.n, c.eps); err != nil {
			t.Fatal(err)
		}
		got, inPlace := ink2OpsTestRead(t, d, yr, c.rows*c.n), ink2OpsTestRead(t, d, xr, c.rows*c.n)
		for r := range c.rows {
			want := ink2OpsTestHostRMS(x[r*c.n:(r+1)*c.n], w, c.eps)
			what := fmt.Sprintf("rms %dx%d row %d", c.rows, c.n, r)
			drift.parity(t, what, got[r*c.n:(r+1)*c.n], want, nil)
			ink2OpsTestSame(t, what+" in place", inPlace[r*c.n:(r+1)*c.n], got[r*c.n:(r+1)*c.n])
		}
	}
	drift.report(t)
}

// fak-test:runtime integration est=1s lane=default
func TestInk2OpsCtrlPairMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(12))
	var drift ink2OpsTestDrift
	for _, n := range []int{3, 512, 1001} {
		cur, msg := ink2OpsTestRand(rng, n, 2), ink2OpsTestRand(rng, n, 0.01)
		pair := ink2TestAlloc(t, d, 2*n+5).At(5)
		if err := d.CtrlPair(ink2OpsTestPut(t, d, 1, cur), ink2OpsTestPut(t, d, 2, msg), pair, n); err != nil {
			t.Fatal(err)
		}
		want := append(ink2OpsTestHostRMS(cur, nil, 1e-6), ink2OpsTestHostRMS(msg, nil, 1e-6)...)
		drift.parity(t, fmt.Sprintf("ctrl pair %d", n), ink2OpsTestRead(t, d, pair, 2*n), want, nil)
	}
	drift.report(t)
}

// fak-test:runtime integration est=2s lane=default
func TestInk2OpsMixMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(13))
	var drift ink2OpsTestDrift
	for _, c := range []struct {
		ns, nq, n int
		// norms: 0 none, 1 every query weighted, 2 query 0 normalized without a weight.
		norms int
	}{{1, 1, 9, 0}, {5, 3, 777, 0}, {5, 3, 777, 1}, {16, 4, 2048, 2}, {16, 4, 2048, 0}, {9, 1, 1024, 1}, {2, 2, 300, 2}} {
		what := fmt.Sprintf("mix %d states x %d queries of %d, norms %d", c.ns, c.nq, c.n, c.norms)
		states, refs := make([][]float32, c.ns), make([]Ink2Ref, c.ns)
		// One arena, as the resident forward's scratch holds every state.
		arena := ink2TestAlloc(t, d, c.ns*(c.n+3))
		for i := range states {
			// States of very different scale: the mix normalizes each.
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
		out := ink2TestAlloc(t, d, c.nq*c.n+2).At(2)
		if err := d.Mix(refs, ink2OpsTestPut(t, d, 4, queries), c.nq, norms, eps, out, c.n); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		got := ink2OpsTestRead(t, d, out, c.nq*c.n)
		for q := range c.nq {
			want, mag := ink2OpsTestHostMix(states, queries[q*c.n:(q+1)*c.n])
			if c.norms > 0 {
				want, mag = ink2OpsTestHostNorm(want, mag, weights[q], eps)
			}
			drift.parity(t, fmt.Sprintf("%s query %d", what, q), got[q*c.n:(q+1)*c.n], want, mag)
		}
	}
	drift.report(t)
}

// fak-test:runtime integration est=2s lane=default
func TestInk2OpsControlMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(14))
	var drift ink2OpsTestDrift
	for _, c := range []struct {
		nj, n, rank int
		// norms: 0 none, 1 every controller weighted, 2 controller 0 normalized without a weight.
		norms int
	}{{1, 40, 6, 0}, {3, 520, 64, 1}, {4, 2048, 64, 2}, {4, 1021, 7, 0}, {2, 256, 1, 1}} {
		what := fmt.Sprintf("control %d of width %d, rank %d, norms %d", c.nj, c.n, c.rank, c.norms)
		var ctl Ink2Controls
		type host struct{ cur, msg, down, gain, forget, norm []float32 }
		hosts := make([]host, c.nj)
		var norms []Ink2Ref
		if c.norms > 0 {
			norms = make([]Ink2Ref, c.nj)
		}
		const eps = 1e-5
		for j := range hosts {
			h := &hosts[j]
			h.cur, h.msg = ink2OpsTestRand(rng, c.n, 2), ink2OpsTestRand(rng, c.n, 0.5)
			h.down = ink2OpsTestRand(rng, c.rank*2*c.n, 1/math.Sqrt(float64(2*c.n)))
			h.gain = ink2OpsTestRand(rng, c.n*c.rank, 1/math.Sqrt(float64(c.rank)))
			h.forget = ink2OpsTestRand(rng, c.n*c.rank, 1/math.Sqrt(float64(c.rank)))
			ctl.Cur = append(ctl.Cur, ink2OpsTestPut(t, d, j, h.cur))
			ctl.Msg = append(ctl.Msg, ink2OpsTestPut(t, d, 3, h.msg))
			ctl.Down = append(ctl.Down, ink2OpsTestPut(t, d, 1, h.down))
			ctl.Gain = append(ctl.Gain, ink2OpsTestPut(t, d, 2, h.gain))
			ctl.Forget = append(ctl.Forget, ink2OpsTestPut(t, d, 5, h.forget))
			if c.norms == 1 || (c.norms == 2 && j > 0) {
				h.norm = ink2OpsTestRand(rng, c.n, 1)
				norms[j] = ink2OpsTestPut(t, d, 6, h.norm)
			}
		}
		g, out := ink2TestAlloc(t, d, c.nj*c.rank+1).At(1), ink2TestAlloc(t, d, c.nj*c.n+3).At(3)
		if err := d.Control(ctl, g, out, c.n, c.rank, norms, eps); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		got := ink2OpsTestRead(t, d, out, c.nj*c.n)
		for j, h := range hosts {
			want, mag := ink2OpsTestHostControl(h.cur, h.msg, h.down, h.gain, h.forget, c.rank)
			if c.norms > 0 {
				want, mag = ink2OpsTestHostNorm(want, mag, h.norm, eps)
			}
			drift.parity(t, fmt.Sprintf("%s controller %d", what, j), got[j*c.n:(j+1)*c.n], want, mag)
		}
	}
	drift.report(t)
}

// fak-test:runtime integration est=1s lane=default
func TestInk2OpsElementwiseMatchHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(15))
	var drift ink2OpsTestDrift
	for _, n := range []int{1, 255, 1000} {
		x, gate := ink2OpsTestRand(rng, n, 2), ink2OpsTestRand(rng, n, 6)
		// Saturated gates: exp overflows to +Inf on one side.
		gate[0] = 200
		gate[n-1] = -200
		xr := ink2OpsTestPut(t, d, 3, x)
		if err := d.SigmoidMul(xr, ink2OpsTestPut(t, d, 1, gate), n); err != nil {
			t.Fatal(err)
		}
		want := make([]float32, n)
		for i := range want {
			want[i] = x[i] * ink2OpsTestSigmoid(gate[i])
		}
		drift.parity(t, fmt.Sprintf("sigmoid mul %d", n), ink2OpsTestRead(t, d, xr, n), want, nil)

		a, b := ink2OpsTestRand(rng, n, 5), ink2OpsTestRand(rng, n, 0.3)
		sum := ink2TestAlloc(t, d, n+2).At(2)
		if err := d.Add(ink2OpsTestPut(t, d, 1, a), ink2OpsTestPut(t, d, 4, b), sum, n); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			want[i] = a[i] + b[i]
		}
		ink2OpsTestSame(t, fmt.Sprintf("add %d", n), ink2OpsTestRead(t, d, sum, n), want)
	}
	drift.report(t)
}

// fak-test:runtime integration est=1s lane=default
func TestInk2OpsRouteMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(16))
	var drift ink2OpsTestDrift
	for _, c := range []struct {
		nexp, k int
		biased  bool
		scale   float32
	}{{4, 4, false, 0}, {33, 1, true, 8}, {96, 8, true, 8}, {96, 8, false, 0}, {512, 16, true, 2.5}, {64, 16, false, 1}} {
		what := fmt.Sprintf("route %d experts to %d, biased %v, scale %v", c.nexp, c.k, c.biased, c.scale)
		logits := ink2OpsTestRand(rng, c.nexp, 1.5)
		var bias []float32
		var br Ink2Ref
		if c.biased {
			bias = ink2OpsTestRand(rng, c.nexp, 0.2)
			br = ink2OpsTestPut(t, d, 2, bias)
		}
		lr := ink2OpsTestPut(t, d, 1, logits)
		ids, wts := ink2TestAlloc(t, d, c.k+3).At(3), ink2TestAlloc(t, d, c.k+1).At(1)
		if err := d.Route(lr, br, ids, wts, c.nexp, c.k, c.scale); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		got, err := d.ReadInts(ids, c.k)
		if err != nil {
			t.Fatal(err)
		}
		probs, sel, top := ink2OpsTestHostRoute(logits, bias, c.k)
		seen := map[int32]bool{}
		for s, id := range got {
			if id < 0 || int(id) >= c.nexp || seen[id] {
				t.Fatalf("%s: slot %d selects expert %d (ids %v)", what, s, id, got)
			}
			seen[id] = true
			// A float32 sigmoid may order two scores an ulp apart the other way.
			if id != top[s] && math.Abs(float64(sel[id])-float64(sel[top[s]])) > 1e-6 {
				t.Fatalf("%s: slot %d selects expert %d (score %v), host selects %d (score %v)", what, s, id, sel[id], top[s], sel[top[s]])
			}
		}
		drift.parity(t, what+" weights", ink2OpsTestRead(t, d, wts, c.k), ink2OpsTestHostWeights(probs, got, c.scale), nil)
		ink2OpsTestSame(t, what+" logits", ink2OpsTestRead(t, d, lr, c.nexp), logits)
	}
	drift.report(t)
}

// fak-test:runtime integration est=1s lane=default
func TestInk2OpsRouteBreaksTiesToTheLowerIndex(t *testing.T) {
	d := ink2TestDevice(t)
	const nexp, k = 70, 6
	logits := make([]float32, nexp)
	for i := range logits {
		logits[i] = -3
	}
	// Equal maxima in different threads' strides and within one stride, a
	// saturated tie at 1.0, and a tie on the last slot.
	logits[40], logits[3], logits[35] = 2, 2, 2
	logits[69], logits[9] = 60, 50
	logits[20], logits[52] = 1, 1
	ids, wts := ink2TestAlloc(t, d, k), ink2TestAlloc(t, d, k)
	if err := d.Route(ink2OpsTestPut(t, d, 0, logits), Ink2Ref{}, ids, wts, nexp, k, 0); err != nil {
		t.Fatal(err)
	}
	got, err := d.ReadInts(ids, k)
	if err != nil {
		t.Fatal(err)
	}
	_, _, want := ink2OpsTestHostRoute(logits, nil, k)
	for s := range want {
		if got[s] != want[s] {
			t.Fatalf("ids = %v, want %v", got, want)
		}
	}
	if want[0] != 9 || want[1] != 69 || want[5] != 20 {
		t.Fatalf("host reference ids = %v", want)
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2OpsSituMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(17))
	var drift ink2OpsTestDrift
	for _, c := range []struct {
		k, width, stride int
		beta, lb         float64
	}{{1, 5, 5, 4, 25}, {4, 100, 128, 4, 25}, {8, 300, 320, 2.5, 0}, {16, 257, 512, 4, 25}} {
		what := fmt.Sprintf("situ %d slots of %d at stride %d", c.k, c.width, c.stride)
		gu := ink2OpsTestRand(rng, c.k*2*c.width, 3)
		// Both tanh and the sigmoid saturated, in each direction.
		gu[0], gu[1], gu[c.width], gu[c.width+1] = 300, -300, -900, 900
		const sentinel = -7.5
		fill := make([]float32, c.k*c.stride)
		for i := range fill {
			fill[i] = sentinel
		}
		act := ink2OpsTestPut(t, d, 2, fill)
		if err := d.Situ(ink2OpsTestPut(t, d, 1, gu), act, c.k, c.width, c.stride, c.beta, c.lb); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		got := ink2OpsTestRead(t, d, act, c.k*c.stride)
		for s := range c.k {
			want := ink2OpsTestHostSitu(gu[s*2*c.width:(s+1)*2*c.width], c.width, c.beta, c.lb)
			drift.parity(t, fmt.Sprintf("%s slot %d", what, s), got[s*c.stride:s*c.stride+c.width], want, nil)
			for j := c.width; j < c.stride; j++ {
				if got[s*c.stride+j] != sentinel {
					t.Fatalf("%s: slot %d padding element %d overwritten with %v", what, s, j, got[s*c.stride+j])
				}
			}
		}
	}
	drift.report(t)
}

// fak-test:runtime integration est=1s lane=default
func TestInk2OpsCombineMatchesHostBitForBit(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(18))
	for _, c := range []struct{ k, n int }{{1, 7}, {8, 1000}, {16, 2048}} {
		down, wts := ink2OpsTestRand(rng, c.k*c.n, 2), ink2OpsTestRand(rng, c.k, 1)
		out := ink2TestAlloc(t, d, c.n+1).At(1)
		if err := d.Combine(ink2OpsTestPut(t, d, 3, down), ink2OpsTestPut(t, d, 5, wts), out, c.k, c.n); err != nil {
			t.Fatal(err)
		}
		ink2OpsTestSame(t, fmt.Sprintf("combine %dx%d", c.k, c.n), ink2OpsTestRead(t, d, out, c.n), ink2OpsTestHostCombine(down, wts, c.n))
	}
	// A fused multiply-add would keep the product's rounding error here
	// instead of cancelling to zero.
	const n = 64
	down, wts := make([]float32, 2*n), []float32{1, 1.0000001}
	for j := range n {
		x := float32(1 + rng.Float64())
		down[j], down[n+j] = -(wts[1] * x), x
	}
	out := ink2TestAlloc(t, d, n)
	if err := d.Combine(ink2OpsTestPut(t, d, 0, down), ink2OpsTestPut(t, d, 0, wts), out, 2, n); err != nil {
		t.Fatal(err)
	}
	got, fused := ink2OpsTestRead(t, d, out, n), 0
	for j := range n {
		if float64(wts[1])*float64(down[n+j]) != float64(-down[j]) {
			fused++
		}
		if got[j] != 0 {
			t.Fatalf("combine[%d] = %v, want the products to cancel exactly", j, got[j])
		}
	}
	if fused == 0 {
		t.Fatal("no input distinguishes a fused multiply-add")
	}
}

func ink2OpsTestArgmax(v []float32) int32 {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return int32(best)
}

// fak-test:runtime integration est=1s lane=default
func TestInk2OpsSoftcapArgmaxMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(19))
	var drift ink2OpsTestDrift
	// One index slot reused across cases, as the resident forward reuses its own.
	slot := ink2TestAlloc(t, d, 9).At(3)
	if err := d.Write(slot.At(-3), []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		n       int
		softcap float32
		// tie: 0 none, 1 two equal maxima, 2 every logit negative with two equal maxima.
		tie int
	}{{1, 30, 0}, {300, 30, 0}, {70001, 30, 0}, {70001, 0, 0}, {70001, 30, 1}, {5000, 0, 1}, {5000, 12, 2}, {257, 30, 2}} {
		what := fmt.Sprintf("softcap argmax of %d, cap %v, tie %d", c.n, c.softcap, c.tie)
		logits := ink2OpsTestRand(rng, c.n, 20)
		if c.tie == 2 {
			for i := range logits {
				logits[i] = -float32(math.Abs(float64(logits[i]))) - 1
			}
		}
		lo, hi := c.n/7, c.n-1-c.n/9
		if c.tie > 0 {
			// Past the softcap's saturation, so both tied logits cap to the same value.
			top := float32(500)
			if c.tie == 2 {
				top = -0.5
			}
			logits[hi], logits[lo] = top, top
		}
		lr := ink2OpsTestPut(t, d, 5, logits)
		if err := d.SoftcapArgmax(lr, c.n, c.softcap, slot); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		got := ink2OpsTestRead(t, d, lr, c.n)
		want := append([]float32(nil), logits...)
		if c.softcap != 0 {
			for i, v := range want {
				want[i] = c.softcap * float32(math.Tanh(float64(v/c.softcap)))
			}
			drift.parity(t, what, got, want, nil)
		} else {
			ink2OpsTestSame(t, what, got, want)
		}
		idx, err := d.ReadInts(slot, 1)
		if err != nil {
			t.Fatal(err)
		}
		if host := ink2OpsTestArgmax(got); idx[0] != host {
			t.Fatalf("%s: index %d, want %d", what, idx[0], host)
		}
		if c.tie > 0 && idx[0] != int32(lo) {
			t.Fatalf("%s: index %d, want the lower tied index %d", what, idx[0], lo)
		}
		if host := ink2OpsTestArgmax(want); c.tie == 0 && idx[0] != host {
			t.Fatalf("%s: index %d, host argmax %d", what, idx[0], host)
		}
	}
	// Only the four index words are the op's to write.
	around := ink2OpsTestRead(t, d, slot.At(-3), 9)
	for i, want := range []float32{1, 2, 3, 0, 0, 0, 0, 8, 9} {
		if (i < 3 || i > 6) && around[i] != want {
			t.Fatalf("word %d beside the index slot = %v, want %v", i, around[i], want)
		}
	}
	drift.report(t)
}

// fak-test:runtime integration est=1s lane=default
func TestInk2OpsEncodeWithoutSubmitting(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(20))
	const n, k, nexp = 600, 4, 32
	logits, down := ink2OpsTestRand(rng, nexp, 1.5), ink2OpsTestRand(rng, k*n, 1)
	resid, weight, gate := ink2OpsTestRand(rng, n, 1), ink2OpsTestRand(rng, n, 1), ink2OpsTestRand(rng, n, 2)
	lr, dr, rr := ink2OpsTestPut(t, d, 0, logits), ink2OpsTestPut(t, d, 0, down), ink2OpsTestPut(t, d, 0, resid)
	wr, gr := ink2OpsTestPut(t, d, 0, weight), ink2OpsTestPut(t, d, 0, gate)
	// One scratch arena: each op reads what the one before it wrote.
	arena := ink2TestAlloc(t, d, 4*n+16)
	ids, wts, mixed, sum, normed, slot := arena, arena.At(k), arena.At(8), arena.At(8+n), arena.At(8+2*n), arena.At(8+3*n)
	before := d.Stats()
	steps := []error{
		d.Route(lr, Ink2Ref{}, ids, wts, nexp, k, 8),
		d.Combine(dr, wts, mixed, k, n),
		d.Add(mixed, rr, sum, n),
		d.RMS(sum, wr, normed, 1, n, 1e-5),
		d.SigmoidMul(normed, gr, n),
		d.SoftcapArgmax(normed, n, 30, slot),
	}
	for i, err := range steps {
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
	}
	if s := d.Stats(); s.Submissions != before.Submissions || s.Pending != 8 {
		t.Fatalf("after encoding: %+v, want 8 pending dispatches and no submission", s)
	}
	idx, err := d.ReadInts(slot, 1)
	if err != nil {
		t.Fatal(err)
	}
	if s := d.Stats(); s.Submissions != before.Submissions+1 || s.Pending != 0 {
		t.Fatalf("after the fence: %+v, want one submission", s)
	}
	got, err := d.ReadInts(ids, k)
	if err != nil {
		t.Fatal(err)
	}
	probs, _, _ := ink2OpsTestHostRoute(logits, nil, k)
	want := ink2OpsTestHostRMS(ink2OpsTestAdd(ink2OpsTestHostCombine(down, ink2OpsTestHostWeights(probs, got, 8), n), resid), weight, 1e-5)
	for i := range want {
		want[i] = 30 * float32(math.Tanh(float64(want[i]*ink2OpsTestSigmoid(gate[i])/30)))
	}
	final := ink2OpsTestRead(t, d, normed, n)
	var drift ink2OpsTestDrift
	drift.parity(t, "chained ops", final, want, nil)
	if host := ink2OpsTestArgmax(final); idx[0] != host {
		t.Fatalf("chained argmax %d, want %d", idx[0], host)
	}
	drift.report(t)
}

func ink2OpsTestAdd(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// fak-test:runtime integration est=1s lane=default
func TestInk2OpsRejectBadOperands(t *testing.T) {
	d := ink2TestDevice(t)
	var none Ink2Ref
	ok := ink2TestAlloc(t, d, 64)
	four := []Ink2Ref{ok, ok, ok, ok}
	for name, err := range map[string]error{
		"rms without an input":         d.RMS(none, none, ok, 1, 4, 0),
		"rms of no rows":               d.RMS(ok, none, ok, 0, 4, 0),
		"ctrl pair without an output":  d.CtrlPair(ok, ok, none, 4),
		"mix of no states":             d.Mix(nil, ok, 1, nil, 0, ok.At(32), 4),
		"mix of 17 states":             d.Mix(make([]Ink2Ref, 17), ok, 1, nil, 0, ok.At(32), 4),
		"mix of 5 queries":             d.Mix(four[:1], ok, 5, nil, 0, ok.At(32), 4),
		"mix with an absent state":     d.Mix([]Ink2Ref{ok, none}, ok, 1, nil, 0, ok.At(32), 4),
		"mix onto a state":             d.Mix([]Ink2Ref{ok.At(34)}, ok, 1, nil, 0, ok.At(32), 4),
		"mix onto its queries":         d.Mix([]Ink2Ref{ok.At(40)}, ok.At(30), 1, nil, 0, ok.At(32), 4),
		"control of no controllers":    d.Control(Ink2Controls{}, ok, ok, 4, 2, nil, 0),
		"control of 5 controllers":     d.Control(Ink2Controls{Cur: append(four, ok), Msg: append(four, ok), Down: append(four, ok), Gain: append(four, ok), Forget: append(four, ok)}, ok, ok, 2, 2, nil, 0),
		"control missing a gain":       d.Control(Ink2Controls{Cur: four[:1], Msg: four[:1], Down: four[:1], Gain: []Ink2Ref{none}, Forget: four[:1]}, ok, ok, 4, 2, nil, 0),
		"control with uneven operands": d.Control(Ink2Controls{Cur: four[:2], Msg: four[:1], Down: four[:2], Gain: four[:2], Forget: four[:2]}, ok, ok, 4, 2, nil, 0),
		"control of rank 0":            d.Control(Ink2Controls{Cur: four[:1], Msg: four[:1], Down: four[:1], Gain: four[:1], Forget: four[:1]}, ok, ok, 4, 0, nil, 0),
		"sigmoid mul without a gate":   d.SigmoidMul(ok, none, 4),
		"route to more slots":          d.Route(ok, none, ok, ok, 4, 5, 0),
		"route of 513 experts":         d.Route(ok, none, ok, ok, 513, 4, 0),
		"route to 17 slots":            d.Route(ok, none, ok, ok, 32, 17, 0),
		"route without ids":            d.Route(ok, none, none, ok, 4, 2, 0),
		"situ with a short stride":     d.Situ(ok, ok, 2, 4, 3, 4, 25),
		"situ without an output":       d.Situ(ok, none, 2, 4, 4, 4, 25),
		"combine of no slots":          d.Combine(ok, ok, ok, 0, 4),
		"add without an operand":       d.Add(ok, none, ok, 4),
		"softcap argmax of nothing":    d.SoftcapArgmax(ok, 0, 30, ok),
		"softcap argmax without index": d.SoftcapArgmax(ok, 4, 30, none),
	} {
		if !errors.Is(err, ErrInk2Shape) {
			t.Errorf("%s: %v, want ErrInk2Shape", name, err)
		}
	}
	// An operand that does not fit its buffer is caught when the launch binds it.
	for name, err := range map[string]error{
		"rms past the buffer":        d.RMS(ok, none, ok, 1, 65, 0),
		"combine past the buffer":    d.Combine(ok, ok, ok, 9, 8),
		"softcap argmax short index": d.SoftcapArgmax(ok, 4, 30, ok.At(61)),
	} {
		if err == nil {
			t.Errorf("%s: encoded", name)
		}
	}
	if s := d.Stats(); s.Dispatches != 0 {
		t.Errorf("a rejected op encoded work: %+v", s)
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2OpsReductionsAreRepeatable(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(21))
	const n, ns, nq, nj, rank, runs = 2048, 16, 4, 4, 64, 40
	states, norms := make([]Ink2Ref, ns), make([]Ink2Ref, nq)
	for i := range states {
		states[i] = ink2OpsTestPut(t, d, i, ink2OpsTestRand(rng, n, 1))
	}
	for q := range norms {
		norms[q] = ink2OpsTestPut(t, d, q, ink2OpsTestRand(rng, n, 1))
	}
	queries := ink2OpsTestPut(t, d, 0, ink2OpsTestRand(rng, nq*n, 1))
	var ctl Ink2Controls
	for j := range nj {
		ctl.Cur, ctl.Msg = append(ctl.Cur, states[j]), append(ctl.Msg, states[ns-1-j])
		ctl.Down = append(ctl.Down, ink2OpsTestPut(t, d, 0, ink2OpsTestRand(rng, rank*2*n, 0.02)))
		ctl.Gain = append(ctl.Gain, ink2OpsTestPut(t, d, 0, ink2OpsTestRand(rng, n*rank, 0.1)))
		ctl.Forget = append(ctl.Forget, ink2OpsTestPut(t, d, 0, ink2OpsTestRand(rng, n*rank, 0.1)))
	}
	// Every run lands in its own rows of one submission.
	mixed, controlled := ink2TestAlloc(t, d, runs*nq*n), ink2TestAlloc(t, d, runs*nj*n)
	g := ink2TestAlloc(t, d, runs*nj*rank)
	for r := range runs {
		if err := d.Mix(states, queries, nq, norms, 1e-5, mixed.At(r*nq*n), n); err != nil {
			t.Fatal(err)
		}
		if err := d.Control(ctl, g.At(r*nj*rank), controlled.At(r*nj*n), n, rank, norms, 1e-5); err != nil {
			t.Fatal(err)
		}
	}
	for what, rows := range map[string][]float32{
		"mix":     ink2OpsTestRead(t, d, mixed, runs*nq*n),
		"control": ink2OpsTestRead(t, d, controlled, runs*nj*n),
	} {
		width := len(rows) / runs
		for r := 1; r < runs; r++ {
			ink2OpsTestSame(t, fmt.Sprintf("%s run %d", what, r), rows[r*width:(r+1)*width], rows[:width])
		}
	}
}
