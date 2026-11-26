//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// ink2AttendArms runs one attend with the named loops and with the one-key loops.
func ink2AttendArms(t *testing.T, d *Ink2Device, c ink2TestAttend, v ink2TestAttendData) (named, oneKey []float32) {
	t.Helper()
	n := c.p.Heads * c.p.HeadDim
	q, r, sinks := ink2TestAttendUpload(t, d, v.q), ink2TestAttendUpload(t, d, v.r), ink2TestAttendUpload(t, d, v.sinks)
	rel, kc, vc := ink2TestAttendUpload(t, d, v.rel), ink2TestAttendUpload(t, d, v.kc), ink2TestAttendUpload(t, d, v.vc)
	meta, scores, out := ink2TestAttendUpload(t, d, ink2TestAttendMeta(c.pos)), ink2TestAlloc(t, d, c.p.Heads*c.cap), ink2TestAlloc(t, d, n)
	saved := ink2TinyFuseLegacy
	defer func() { ink2TinyFuseLegacy = saved }()
	for _, legacy := range []bool{false, true} {
		ink2TinyFuseLegacy = legacy
		if err := d.Attend(q, r, sinks, rel, kc, vc, scores, out, meta, c.p, c.extent, c.window, c.cap); err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		if legacy {
			oneKey = ink2OpsTestRead(t, d, out, n)
		} else {
			named = ink2OpsTestRead(t, d, out, n)
		}
	}
	return named, oneKey
}

// fak-test:runtime integration est=2s lane=default
func TestInk2AttendNamedLoopsMatchOneKeyLoopsBitForBit(t *testing.T) {
	d := ink2TestDevice(t)
	tiny := Ink2AttnParams{Heads: 4, KVHeads: 2, HeadDim: 8, RelLen: 3, Scale: float32(1 / math.Sqrt(8))}
	odd := Ink2AttnParams{Heads: 6, KVHeads: 2, HeadDim: 40, RelLen: 35, Scale: float32(1 / math.Sqrt(40))}
	mini := Ink2AttnParams{Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16, Scale: float32(1 / math.Sqrt(128))}
	rng := rand.New(rand.NewSource(61))
	var cases []ink2TestAttend
	for _, pos := range []int{0, 1, 15, 16, 17, 47, 63, 64, 65, 100, 157, 1023, 1024, 2000, 5000} {
		cases = append(cases, ink2TestAttend{p: mini, cap: 1024, window: 1024, extent: 1024, pos: pos},
			ink2TestAttend{p: mini, cap: 4096, extent: 1024, pos: pos})
	}
	cases = append(cases, ink2TestAttend{p: tiny, cap: 16, window: 16, extent: 16, pos: 37}, ink2TestAttend{p: tiny, cap: 64, extent: 10, pos: 40, sinkShift: 12},
		ink2TestAttend{p: tiny, cap: 64, extent: 10, pos: 40, sinkShift: -12}, ink2TestAttend{p: odd, cap: 300, extent: 70, pos: 211},
		ink2TestAttend{p: odd, cap: 90, window: 90, extent: 90, pos: 1000})
	for _, c := range cases {
		named, oneKey := ink2AttendArms(t, d, c, c.data(rng))
		ink2OpsTestSame(t, c.String(), named, oneKey)
	}
}

// ink2QKVSetup uploads one qkv post's operands at the Inkling-2-Mini shape.
// encode runs the post with the general loops (legacy) or the fixed ones and
// read returns q, r, the cache rows of pos and both conv histories.
func ink2QKVSetup(t *testing.T, d *Ink2Device, rng *rand.Rand, global, dynamic bool, pos int) (encode func(legacy bool) error, reset func(), read func() [][]float32) {
	t.Helper()
	p := Ink2AttnParams{Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16, QKClip: 3, RelClip: 1.2, Eps: 1e-6, Global: global, Alpha: .3, Floor: 4}
	const kn, cap = 4, 64
	kvDim := p.KVHeads * p.HeadDim
	qn, rn := p.Heads*p.HeadDim, p.Heads*p.RelLen
	// small clip thresholds against these magnitudes exercise both sides of the clip
	q0, r0 := ink2OpsTestRand(rng, qn, 0.4), ink2OpsTestRand(rng, rn, 0.5)
	k0, v0 := ink2OpsTestRand(rng, kvDim, 0.3), ink2OpsTestRand(rng, kvDim, 2)
	kh0, vh0 := ink2OpsTestRand(rng, kvDim*(kn-1), 1), ink2OpsTestRand(rng, kvDim*(kn-1), 1)
	q, r := ink2OpsTestPut(t, d, 1, q0), ink2OpsTestPut(t, d, 2, r0)
	k, v := ink2OpsTestPut(t, d, 3, k0), ink2OpsTestPut(t, d, 4, v0)
	qclip, rclip := ink2OpsTestPut(t, d, 0, ink2OpsTestRand(rng, qn, 1)), ink2OpsTestPut(t, d, 0, ink2OpsTestRand(rng, rn, 1))
	kc, vc := ink2OpsTestPut(t, d, 0, make([]float32, cap*kvDim)), ink2OpsTestPut(t, d, 0, make([]float32, cap*kvDim))
	meta := ink2TestAttendUpload(t, d, ink2TestAttendMeta(pos))
	cv := &Ink2KVConv{KW: ink2OpsTestPut(t, d, 1, ink2OpsTestRand(rng, kvDim*kn, .5)), VW: ink2OpsTestPut(t, d, 2, ink2OpsTestRand(rng, kvDim*kn, .5)),
		KHist: ink2OpsTestPut(t, d, 5, kh0), VHist: ink2OpsTestPut(t, d, 6, vh0), Kn: kn, KNorm: 0.4472136, VNorm: 0.4472136}
	if dynamic {
		cv.KDyn, cv.VDyn = ink2OpsTestPut(t, d, 1, ink2OpsTestRand(rng, p.KVHeads*kn, .5)), ink2OpsTestPut(t, d, 2, ink2OpsTestRand(rng, p.KVHeads*kn, .5))
		cv.KNorm, cv.VNorm = 0.33333334, 0.33333334
	}
	reset = func() {
		for _, w := range []struct {
			r Ink2Ref
			v []float32
		}{{q, q0}, {r, r0}, {cv.KHist, kh0}, {cv.VHist, vh0}} {
			if err := d.Write(w.r, w.v); err != nil {
				t.Fatal(err)
			}
		}
	}
	encode = func(legacy bool) error {
		saved := ink2TinyFuseLegacy
		defer func() { ink2TinyFuseLegacy = saved }()
		ink2TinyFuseLegacy = legacy
		return d.QKVPost(q, r, k, v, qclip, rclip, kc, vc, meta, p, cap, cv)
	}
	row := pos % cap * kvDim
	read = func() [][]float32 {
		return [][]float32{ink2OpsTestRead(t, d, q, qn), ink2OpsTestRead(t, d, r, rn), ink2OpsTestRead(t, d, kc.At(row), kvDim),
			ink2OpsTestRead(t, d, vc.At(row), kvDim), ink2OpsTestRead(t, d, cv.KHist, kvDim*(kn-1)), ink2OpsTestRead(t, d, cv.VHist, kvDim*(kn-1))}
	}
	return encode, reset, read
}

// fak-test:runtime integration est=1s lane=default
func TestInk2QKVPostFixedLoopsMatchGeneralLoopsBitForBit(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(65))
	for _, c := range []struct {
		global, dynamic bool
		pos             int
	}{{false, false, 0}, {false, false, 37}, {true, true, 5}, {true, true, 3000}, {true, false, 9}, {false, true, 70}} {
		encode, reset, read := ink2QKVSetup(t, d, rng, c.global, c.dynamic, c.pos)
		reset()
		if err := encode(true); err != nil {
			t.Fatal(err)
		}
		want := read()
		reset()
		if err := encode(false); err != nil {
			t.Fatal(err)
		}
		got := read()
		for i, what := range []string{"q", "r", "k row", "v row", "k history", "v history"} {
			ink2OpsTestSame(t, fmt.Sprintf("global %v dynamic %v pos %d %s", c.global, c.dynamic, c.pos, what), got[i], want[i])
		}
	}
}

// The fixed-step qkv post and attend builds against the general ones at the
// Inkling-2-Mini decode shapes, on the decode-ops microbenchmark.
// fak-test:runtime slow est=60s
func TestInk2DecFixedLoopsMicrobench(t *testing.T) {
	d := ink2OpsBenchDevice(t)
	saved := ink2TinyFuseLegacy
	t.Cleanup(func() { ink2TinyFuseLegacy = saved })
	rng := rand.New(rand.NewSource(64))
	var cases []ink2OpsBenchCase
	// One site chains every launch on the last; eight let them overlap.
	for _, sites := range []int{1, 8} {
		for _, global := range []bool{false, true} {
			encs := make([]func(bool) error, sites)
			for i := range encs {
				encs[i], _, _ = ink2QKVSetup(t, d, rng, global, global, 40+i)
			}
			cases = append(cases, ink2OpsBenchCase{name: fmt.Sprintf("x%d qkv_post global=%v", sites, global), sites: sites,
				arms: []ink2OpsBenchArm{{"general", func(i int) error { return encs[i](true) }}, {"fixed", func(i int) error { return encs[i](false) }}}})
		}
		c := ink2TestAttend{p: Ink2AttnParams{Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16, Scale: float32(1 / math.Sqrt(128))},
			cap: 4096, extent: 1024}
		for _, pos := range []int{40, 100, 160, 1000} {
			c.pos = pos
			type site struct{ q, r, sinks, rel, kc, vc, meta, scores, out Ink2Ref }
			ss := make([]site, sites)
			for i := range ss {
				v := c.data(rng)
				ss[i] = site{ink2TestAttendUpload(t, d, v.q), ink2TestAttendUpload(t, d, v.r), ink2TestAttendUpload(t, d, v.sinks),
					ink2TestAttendUpload(t, d, v.rel), ink2TestAttendUpload(t, d, v.kc), ink2TestAttendUpload(t, d, v.vc),
					ink2TestAttendUpload(t, d, ink2TestAttendMeta(pos)), ink2TestAlloc(t, d, c.p.Heads*c.cap), ink2TestAlloc(t, d, c.p.Heads*c.p.HeadDim)}
			}
			arm := func(legacy bool) func(int) error {
				return func(i int) error {
					ink2TinyFuseLegacy = legacy
					s := ss[i]
					return d.Attend(s.q, s.r, s.sinks, s.rel, s.kc, s.vc, s.scores, s.out, s.meta, c.p, c.extent, c.window, c.cap)
				}
			}
			cases = append(cases, ink2OpsBenchCase{name: fmt.Sprintf("x%d attend pos %d", sites, pos), sites: sites,
				arms: []ink2OpsBenchArm{{"one-key", arm(true)}, {"named", arm(false)}}})
		}
	}
	ink2OpsBenchRun(t, d, cases, 200, ink2OpsBenchReps(t))
}
