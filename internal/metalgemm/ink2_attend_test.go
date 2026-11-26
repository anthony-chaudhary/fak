//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// ink2TestAttendHost is the host forward's attend over the history a slot's
// ring still holds: float32 scores, a float64 softmax with the zero sink logit
// in the denominator, float32 value sums. mag is sum_j w_j*|v_j| per output.
func ink2TestAttendHost(p Ink2AttnParams, q, r, sinks, relBias, kcache, vcache []float32, pos, extent, window, cap int) (out []float32, mag []float64) {
	kvDim := p.KVHeads * p.HeadDim
	dot := func(a, b []float32) float32 {
		var s float32
		for i := range a {
			s += a[i] * b[i]
		}
		return s
	}
	var positions []int
	for at := max(0, pos+1-cap); at <= pos; at++ {
		positions = append(positions, at)
	}
	group := p.Heads / p.KVHeads
	out, mag = make([]float32, p.Heads*p.HeadDim), make([]float64, p.Heads*p.HeadDim)
	scores := make([]float64, len(positions))
	for hd := 0; hd < p.Heads; hd++ {
		kvh := hd / group
		qh := q[hd*p.HeadDim : (hd+1)*p.HeadDim]
		maxS := math.Inf(-1)
		for j, at := range positions {
			dist := pos - at
			if window > 0 && dist >= window {
				scores[j] = math.Inf(-1)
				continue
			}
			bias := -sinks[hd]
			if dist < extent {
				bias += dot(relBias[dist*p.RelLen:(dist+1)*p.RelLen], r[hd*p.RelLen:(hd+1)*p.RelLen])
			}
			row := at % cap * kvDim
			s := float64(dot(qh, kcache[row+kvh*p.HeadDim:row+(kvh+1)*p.HeadDim])*p.Scale + bias)
			scores[j] = s
			maxS = max(maxS, s)
		}
		if maxS < 0 {
			maxS = 0
		}
		denom := math.Exp(-maxS)
		for j := range scores {
			if math.IsInf(scores[j], -1) {
				scores[j] = 0
				continue
			}
			scores[j] = math.Exp(scores[j] - maxS)
			denom += scores[j]
		}
		oh, mh := out[hd*p.HeadDim:(hd+1)*p.HeadDim], mag[hd*p.HeadDim:(hd+1)*p.HeadDim]
		for j, pr := range scores {
			if pr == 0 {
				continue
			}
			w := float32(pr / denom)
			row := positions[j] % cap * kvDim
			vj := vcache[row+kvh*p.HeadDim : row+(kvh+1)*p.HeadDim]
			for d := range oh {
				oh[d] += w * vj[d]
				mh[d] += float64(w) * math.Abs(float64(vj[d]))
			}
		}
	}
	return out, mag
}

type ink2TestAttend struct {
	p                        Ink2AttnParams
	cap, window, extent, pos int
	sinkShift                float32
}

func (c ink2TestAttend) String() string {
	return fmt.Sprintf("heads %d/%d dim %d rel %d cap %d window %d extent %d pos %d sink %+g",
		c.p.Heads, c.p.KVHeads, c.p.HeadDim, c.p.RelLen, c.cap, c.window, c.extent, c.pos, c.sinkShift)
}

type ink2TestAttendData struct {
	q, r, sinks, rel, kc, vc []float32
}

func (c ink2TestAttend) data(rng *rand.Rand) ink2TestAttendData {
	fill := func(n int, sd float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(sd * rng.NormFloat64())
		}
		return v
	}
	kvDim := c.p.KVHeads * c.p.HeadDim
	v := ink2TestAttendData{
		q: fill(c.p.Heads*c.p.HeadDim, 1), r: fill(c.p.Heads*c.p.RelLen, 1), sinks: fill(c.p.Heads, 1),
		rel: fill(c.extent*c.p.RelLen, .1), kc: fill(c.cap*kvDim, .3), vc: fill(c.cap*kvDim, 1),
	}
	for i := range v.sinks {
		v.sinks[i] += c.sinkShift
	}
	return v
}

func ink2TestAttendUpload(t *testing.T, d *Ink2Device, v []float32) Ink2Ref {
	t.Helper()
	ref := ink2TestAlloc(t, d, max(len(v), 1))
	if err := d.Write(ref, v); err != nil {
		t.Fatal(err)
	}
	return ref
}

func ink2TestAttendMeta(pos int) []float32 {
	return []float32{math.Float32frombits(uint32(int32(pos)))}
}

// The CUDA conformance bound, over the weighted value magnitudes.
func ink2TestAttendCheck(t *testing.T, what string, got, want []float32, mag []float64) {
	t.Helper()
	for i := range want {
		if diff := math.Abs(float64(got[i]) - float64(want[i])); !(diff <= 1e-5*mag[i]+1e-6) {
			t.Fatalf("%s: out[%d] device %v, host %v (sum w|v| %v)", what, i, got[i], want[i], mag[i])
		}
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2AttendMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	tiny := Ink2AttnParams{Heads: 4, KVHeads: 2, HeadDim: 8, RelLen: 3, Scale: float32(1 / math.Sqrt(8))}
	odd := Ink2AttnParams{Heads: 6, KVHeads: 2, HeadDim: 40, RelLen: 35, Scale: float32(1 / math.Sqrt(40))}
	flat := Ink2AttnParams{Heads: 3, KVHeads: 3, HeadDim: 256, RelLen: 1, Scale: 1. / 16}
	mini := Ink2AttnParams{Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16, Scale: float32(1 / math.Sqrt(128))}
	cases := []ink2TestAttend{
		// a sliding slot filling, full, and wrapped around its ring
		{p: tiny, cap: 16, window: 16, extent: 16, pos: 0},
		{p: tiny, cap: 16, window: 16, extent: 16, pos: 5},
		{p: tiny, cap: 16, window: 16, extent: 16, pos: 15},
		{p: tiny, cap: 16, window: 16, extent: 16, pos: 16},
		{p: tiny, cap: 16, window: 16, extent: 16, pos: 37},
		// a ring with room past the window, as a batched prefill keeps
		{p: tiny, cap: 24, window: 16, extent: 16, pos: 9},
		{p: tiny, cap: 24, window: 16, extent: 16, pos: 40},
		// a full slot whose bias table ends before its history does
		{p: tiny, cap: 64, extent: 10, pos: 0},
		{p: tiny, cap: 64, extent: 10, pos: 9},
		{p: tiny, cap: 64, extent: 10, pos: 10},
		{p: tiny, cap: 64, extent: 10, pos: 63},
		{p: tiny, cap: 64, extent: 10, pos: 100},
		// every score far below, and far above, the zero sink logit
		{p: tiny, cap: 64, extent: 10, pos: 40, sinkShift: 12},
		{p: tiny, cap: 64, extent: 10, pos: 40, sinkShift: -12},
		{p: odd, cap: 300, extent: 70, pos: 0},
		{p: odd, cap: 300, extent: 70, pos: 211},
		{p: odd, cap: 90, window: 90, extent: 90, pos: 1000},
		{p: flat, cap: 40, extent: 40, pos: 33},
		{p: mini, cap: 1024, window: 1024, extent: 1024, pos: 82},
		{p: mini, cap: 1024, window: 1024, extent: 1024, pos: 2000},
		{p: mini, cap: 4096, extent: 1024, pos: 2000},
		{p: mini, cap: 4096, extent: 1024, pos: 5000},
	}
	rng := rand.New(rand.NewSource(41))
	var worst float64
	for _, c := range cases {
		v := c.data(rng)
		n := c.p.Heads * c.p.HeadDim
		out := ink2TestAlloc(t, d, n)
		if err := d.Attend(ink2TestAttendUpload(t, d, v.q), ink2TestAttendUpload(t, d, v.r), ink2TestAttendUpload(t, d, v.sinks),
			ink2TestAttendUpload(t, d, v.rel), ink2TestAttendUpload(t, d, v.kc), ink2TestAttendUpload(t, d, v.vc),
			ink2TestAlloc(t, d, c.p.Heads*c.cap), out, ink2TestAttendUpload(t, d, ink2TestAttendMeta(c.pos)),
			c.p, c.extent, c.window, c.cap); err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		got, err := d.Read(out, n)
		if err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		want, mag := ink2TestAttendHost(c.p, v.q, v.r, v.sinks, v.rel, v.kc, v.vc, c.pos, c.extent, c.window, c.cap)
		ink2TestAttendCheck(t, c.String(), got, want, mag)
		for i := range want {
			worst = max(worst, math.Abs(float64(got[i])-float64(want[i]))/(mag[i]+1e-30))
		}
	}
	t.Logf("worst |device-host| / sum w|v| = %.3g", worst)
}

// The driver keeps q, r, the sinks and the output in one arena and shares one
// scores scratch between every layer of a token, all inside one submission.
// fak-test:runtime integration est=1s lane=default
func TestInk2AttendSharesAnArenaAndTheScoresScratch(t *testing.T) {
	d := ink2TestDevice(t)
	p := Ink2AttnParams{Heads: 6, KVHeads: 3, HeadDim: 32, RelLen: 16, Scale: float32(1 / math.Sqrt(32))}
	n := p.Heads * p.HeadDim
	layers := []ink2TestAttend{
		{p: p, cap: 48, window: 48, extent: 48, pos: 130},
		{p: p, cap: 200, extent: 64, pos: 130},
	}
	rng := rand.New(rand.NewSource(43))
	// per layer: [q | r | sinks | pad | out | guard]
	const guard = 8
	sinksAt := n + p.Heads*p.RelLen
	outAt := sinksAt + 8
	stride := outAt + n + guard
	arenaHost := make([]float32, stride*len(layers))
	for i := range arenaHost {
		arenaHost[i] = float32(1000 + i)
	}
	data := make([]ink2TestAttendData, len(layers))
	for l, c := range layers {
		data[l] = c.data(rng)
		copy(arenaHost[l*stride:], data[l].q)
		copy(arenaHost[l*stride+n:], data[l].r)
		copy(arenaHost[l*stride+sinksAt:], data[l].sinks)
	}
	arena := ink2TestAttendUpload(t, d, arenaHost)
	scores := ink2TestAlloc(t, d, 2*p.Heads*200)
	// meta rides behind other floats, as the position does behind the embedding
	meta := ink2TestAttendUpload(t, d, append(make([]float32, 5), ink2TestAttendMeta(130)...))
	// a host write is a fence, so every operand is uploaded before the first launch
	var rel, kc, vc []Ink2Ref
	for _, v := range data {
		rel, kc, vc = append(rel, ink2TestAttendUpload(t, d, v.rel)), append(kc, ink2TestAttendUpload(t, d, v.kc)), append(vc, ink2TestAttendUpload(t, d, v.vc))
	}
	before := d.Stats()
	for l, c := range layers {
		at := arena.At(l * stride)
		if err := d.Attend(at, at.At(n), at.At(sinksAt), rel[l], kc[l], vc[l], scores, at.At(outAt), meta.At(5),
			c.p, c.extent, c.window, c.cap); err != nil {
			t.Fatalf("layer %d: %v", l, err)
		}
	}
	if s := d.Stats(); s.Submissions != before.Submissions || s.Pending != len(layers) {
		t.Fatalf("attend submitted instead of encoding: %+v after %+v", s, before)
	}
	got, err := d.Read(arena, len(arenaHost))
	if err != nil {
		t.Fatal(err)
	}
	if s := d.Stats(); s.Submissions != before.Submissions+1 {
		t.Fatalf("%d submissions for two attends, want 1", s.Submissions-before.Submissions)
	}
	for l, c := range layers {
		v := data[l]
		want, mag := ink2TestAttendHost(c.p, v.q, v.r, v.sinks, v.rel, v.kc, v.vc, c.pos, c.extent, c.window, c.cap)
		ink2TestAttendCheck(t, fmt.Sprintf("layer %d", l), got[l*stride+outAt:l*stride+outAt+n], want, mag)
		for i := 0; i < stride; i++ {
			if i >= outAt && i < outAt+n {
				continue
			}
			if at := l*stride + i; math.Float32bits(got[at]) != math.Float32bits(arenaHost[at]) {
				t.Fatalf("layer %d: arena[%d] changed from %v to %v", l, i, arenaHost[at], got[at])
			}
		}
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2AttendWithoutHistoryOrBiasTable(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(47))
	c := ink2TestAttend{p: Ink2AttnParams{Heads: 2, KVHeads: 1, HeadDim: 16, Scale: .25}, cap: 12, pos: 30}
	v := c.data(rng)
	n := c.p.Heads * c.p.HeadDim
	run := func(pos int) []float32 {
		t.Helper()
		out := ink2TestAttendUpload(t, d, make([]float32, n+1))
		if err := d.Write(out, []float32{7, 7, 7}); err != nil {
			t.Fatal(err)
		}
		// no relative states: the query and table operands are absent
		if err := d.Attend(ink2TestAttendUpload(t, d, v.q), Ink2Ref{}, ink2TestAttendUpload(t, d, v.sinks), Ink2Ref{},
			ink2TestAttendUpload(t, d, v.kc), ink2TestAttendUpload(t, d, v.vc), ink2TestAlloc(t, d, c.p.Heads*c.cap), out,
			ink2TestAttendUpload(t, d, ink2TestAttendMeta(pos)), c.p, 0, 0, c.cap); err != nil {
			t.Fatal(err)
		}
		got, err := d.Read(out, n)
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	want, mag := ink2TestAttendHost(c.p, v.q, nil, v.sinks, nil, v.kc, v.vc, c.pos, 0, 0, c.cap)
	ink2TestAttendCheck(t, "no bias table", run(c.pos), want, mag)
	// A slot nothing was appended to attends to the sink alone.
	for i, got := range run(-1) {
		if got != 0 {
			t.Fatalf("empty slot: out[%d] = %v, want 0", i, got)
		}
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2AttendRejectsBadOperands(t *testing.T) {
	d := ink2TestDevice(t)
	p := Ink2AttnParams{Heads: 4, KVHeads: 2, HeadDim: 8, RelLen: 3, Scale: 1}
	r := ink2TestAlloc(t, d, 4096)
	attend := func(p Ink2AttnParams, out, rel Ink2Ref, extent, window, cap int) error {
		return d.Attend(r, r, r, rel, r, r, r, out, r, p, extent, window, cap)
	}
	with := func(edit func(*Ink2AttnParams)) Ink2AttnParams {
		q := p
		edit(&q)
		return q
	}
	for name, err := range map[string]error{
		"no heads":            attend(with(func(q *Ink2AttnParams) { q.Heads = 0 }), r, r, 4, 0, 8),
		"ungrouped heads":     attend(with(func(q *Ink2AttnParams) { q.KVHeads = 3 }), r, r, 4, 0, 8),
		"head past the group": attend(with(func(q *Ink2AttnParams) { q.HeadDim = 513 }), r, r, 4, 0, 8),
		"no capacity":         attend(p, r, r, 4, 0, 0),
		"negative window":     attend(p, r, r, 4, -1, 8),
		"absent output":       attend(p, Ink2Ref{}, r, 4, 0, 8),
		"absent bias table":   attend(p, r, Ink2Ref{}, 4, 0, 8),
	} {
		if !errors.Is(err, ErrInk2Shape) {
			t.Errorf("%s: %v, want ErrInk2Shape", name, err)
		}
	}
	// 4096 floats hold a cache of 256 rows, not 257.
	if err := attend(p, r, r, 4, 0, 256); err != nil {
		t.Errorf("cache that fits: %v", err)
	}
	if err := attend(p, r, r, 4, 0, 257); err == nil {
		t.Error("a cache past its buffer was encoded")
	}
	if err := d.Flush(); err != nil {
		t.Fatal(err)
	}
}
