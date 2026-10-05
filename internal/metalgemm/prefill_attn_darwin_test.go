//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"sync"
	"testing"
)

// Tolerance for the tiled GQA prefill attention against the f64 CPU reference.
//
// The kernel keeps q, k, v, scores, the online-softmax state and the PV accumulators in f32, so
// every output is a convex combination of v rows (|v| <= 1 here) perturbed by f32 rounding in
// three places: the hd-long q.k dot (error ~ sqrt(hd)*u*|score scale|, u = 2^-24), exp/rescale of
// the running max (a few ulp relative per key), and the kvLen-long PV accumulation (~ sqrt(kvLen)
// * u relative). At hd <= 256 and kvLen <= 2048 that is ~1e-6 absolute in the worst typical case
// (the implementer measured ~1.5e-7 on this normalized data), so 5e-5 is a >30x margin that still
// sits orders of magnitude below any structural bug: a wrong causal/window bound, head-to-KV
// mapping or unnormalized softmax moves outputs by O(1e-2..1). The "sharp" case scales q so raw
// scores reach O(100), which overflows exp unless the online max is tracked correctly.
const prefillAttnMaxAbs = 5e-5

// prefillAttnReference is an independent f64 causal GQA attention: query row t of the panel sits
// at absolute position base+t (base = kvLen-P) and attends keys [lo, base+t], lo = base+t-window+1
// clamped to 0 when window > 0, else 0. Head h reads KV head h/(nH/nKV).
func prefillAttnReference(q, k, v []float32, P, kvLen, nH, nKV, hd, window int, scale float32) []float32 {
	out := make([]float32, P*nH*hd)
	grp := nH / nKV
	base := kvLen - P
	units := P * nH
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for wk := 0; wk < workers; wk++ {
		wg.Add(1)
		go func(wk int) {
			defer wg.Done()
			scores := make([]float64, kvLen)
			acc := make([]float64, hd)
			for u := wk; u < units; u += workers {
				t, h := u/nH, u%nH
				kvh := h / grp
				qabs := base + t
				lo := 0
				if window > 0 && qabs-window+1 > 0 {
					lo = qabs - window + 1
				}
				qh := q[t*nH*hd+h*hd : t*nH*hd+(h+1)*hd]
				maxS := math.Inf(-1)
				for j := lo; j <= qabs; j++ {
					kj := k[j*nKV*hd+kvh*hd : j*nKV*hd+(kvh+1)*hd]
					var s float64
					for d := 0; d < hd; d++ {
						s += float64(qh[d]) * float64(kj[d])
					}
					s *= float64(scale)
					scores[j] = s
					maxS = math.Max(maxS, s)
				}
				var sum float64
				for d := range acc {
					acc[d] = 0
				}
				for j := lo; j <= qabs; j++ {
					p := math.Exp(scores[j] - maxS)
					sum += p
					vj := v[j*nKV*hd+kvh*hd : j*nKV*hd+(kvh+1)*hd]
					for d := 0; d < hd; d++ {
						acc[d] += p * float64(vj[d])
					}
				}
				o := out[t*nH*hd+h*hd : t*nH*hd+(h+1)*hd]
				for d := 0; d < hd; d++ {
					o[d] = float32(acc[d] / sum)
				}
			}
		}(wk)
	}
	wg.Wait()
	return out
}

func prefillAttnRandom(n int, seed int64, amp float32) []float32 {
	rng := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * amp
	}
	return x
}

type prefillAttnCase struct {
	name                     string
	P, base, nH, nKV, hd, ww int
	qAmp                     float32
}

func (c prefillAttnCase) run(t *testing.T, seed int64) {
	t.Helper()
	kvLen := c.base + c.P
	if !PrefillAttentionSupported(c.hd, c.nH, c.nKV) {
		t.Fatalf("PrefillAttentionSupported(%d,%d,%d)=false for a supported geometry", c.hd, c.nH, c.nKV)
	}
	amp := c.qAmp
	if amp == 0 {
		amp = 1
	}
	q := prefillAttnRandom(c.P*c.nH*c.hd, seed, amp)
	k := prefillAttnRandom(kvLen*c.nKV*c.hd, seed+1, 1)
	v := prefillAttnRandom(kvLen*c.nKV*c.hd, seed+2, 1)
	scale := float32(1 / math.Sqrt(float64(c.hd)))
	out := make([]float32, c.P*c.nH*c.hd)
	for i := range out {
		out[i] = float32(math.NaN())
	}
	timing, err := PrefillAttention(out, q, k, v, c.P, kvLen, c.nH, c.nKV, c.hd, c.ww, scale)
	if err != nil {
		t.Fatalf("PrefillAttention: %v", err)
	}
	if !(timing.GPUMilliseconds > 0) || math.IsInf(timing.GPUMilliseconds, 0) {
		t.Fatalf("non-positive GPU window %+v", timing)
	}
	ref := prefillAttnReference(q, k, v, c.P, kvLen, c.nH, c.nKV, c.hd, c.ww, scale)
	maxAbs, at := 0.0, 0
	for i := range ref {
		g := float64(out[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("non-finite output[%d]=%g (row %d)", i, g, i/(c.nH*c.hd))
		}
		if d := math.Abs(g - float64(ref[i])); d > maxAbs {
			maxAbs, at = d, i
		}
	}
	t.Logf("%s: P=%d base=%d nH=%d nKV=%d hd=%d window=%d maxAbs=%.3e gpu_ms=%.3f",
		c.name, c.P, c.base, c.nH, c.nKV, c.hd, c.ww, maxAbs, timing.GPUMilliseconds)
	if maxAbs > prefillAttnMaxAbs {
		row := at / (c.nH * c.hd)
		head := (at / c.hd) % c.nH
		t.Fatalf("%s: maxAbs=%.3e > %g at row %d head %d (got %g want %g)", c.name, maxAbs, prefillAttnMaxAbs, row, head, out[at], ref[at])
	}
}

// TestPrefillAttentionMatchesIndependentReference is the fak#13695 kernel parity witness against an
// independent f64 CPU attention over head_dim 64/128/256, GQA groups 1/4/6/7 (plus 9 and 16, which
// split a group across threadgroups), panels that are not a multiple of the 8-row Q tile, a cached
// prefix (base > 0), sliding windows, sharp (large-score) softmax, and the Qwen2.5-7B geometry.
func TestPrefillAttentionMatchesIndependentReference(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	var cases []prefillAttnCase
	for _, hd := range []int{64, 128, 256} {
		for _, grp := range []int{1, 4, 6, 7} {
			cases = append(cases, prefillAttnCase{name: fmt.Sprintf("hd%d_grp%d", hd, grp), P: 37, base: 19, nH: 2 * grp, nKV: 2, hd: hd, ww: 0})
		}
	}
	cases = append(cases,
		prefillAttnCase{name: "fresh_P8", P: 8, base: 0, nH: 8, nKV: 2, hd: 128},
		prefillAttnCase{name: "single_row_cached", P: 1, base: 23, nH: 8, nKV: 2, hd: 128},
		prefillAttnCase{name: "grp9_split", P: 29, base: 5, nH: 9, nKV: 1, hd: 64},
		prefillAttnCase{name: "grp16_split", P: 21, base: 70, nH: 32, nKV: 2, hd: 64},
		prefillAttnCase{name: "window16", P: 45, base: 30, nH: 8, nKV: 2, hd: 128, ww: 16},
		prefillAttnCase{name: "window1", P: 13, base: 4, nH: 4, nKV: 1, hd: 64, ww: 1},
		prefillAttnCase{name: "window_wider_than_kv", P: 17, base: 3, nH: 8, nKV: 2, hd: 256, ww: 4096},
		prefillAttnCase{name: "window_spans_tiles", P: 150, base: 200, nH: 12, nKV: 2, hd: 128, ww: 97},
		prefillAttnCase{name: "sharp_scores", P: 61, base: 40, nH: 28, nKV: 4, hd: 128, qAmp: 8},
		prefillAttnCase{name: "7b_P2048", P: 2048, base: 0, nH: 28, nKV: 4, hd: 128},
		prefillAttnCase{name: "7b_P777_cached", P: 777, base: 1271, nH: 28, nKV: 4, hd: 128},
	)
	for i, c := range cases {
		t.Run(c.name, func(t *testing.T) { c.run(t, int64(13695+100*i)) })
	}
}

// TestPrefillAttentionRefusesInvalidShapesUntouched pins the fail-closed contract: an invalid or
// unsupported geometry returns an error before any command buffer and leaves out untouched.
func TestPrefillAttentionRefusesInvalidShapesUntouched(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	for _, tc := range []struct {
		hd, nH, nKV int
		want        bool
	}{
		{64, 8, 2, true}, {128, 28, 4, true}, {256, 16, 16, true}, {128, 7, 7, true},
		{96, 8, 2, false}, {32, 8, 2, false}, {512, 8, 2, false}, {0, 8, 2, false},
		{128, 6, 4, false}, {128, 28, 3, false}, {128, 8, 0, false}, {128, 0, 2, false},
	} {
		if got := PrefillAttentionSupported(tc.hd, tc.nH, tc.nKV); got != tc.want {
			t.Errorf("PrefillAttentionSupported(hd=%d,nH=%d,nKV=%d)=%t want %t", tc.hd, tc.nH, tc.nKV, got, tc.want)
		}
	}

	const P, kvLen, nH, nKV, hd = 16, 24, 8, 2, 128
	q := prefillAttnRandom(P*nH*hd, 1, 1)
	k := prefillAttnRandom(kvLen*nKV*hd, 2, 1)
	v := prefillAttnRandom(kvLen*nKV*hd, 3, 1)
	// Wide enough K/V for the nH%nKV != 0 case, so only the GQA divisibility refuses it.
	kWide := prefillAttnRandom(kvLen*4*hd, 4, 1)
	vWide := prefillAttnRandom(kvLen*4*hd, 5, 1)
	marked := func(n int) []float32 {
		o := make([]float32, n)
		for i := range o {
			o[i] = 13695
		}
		return o
	}
	full := P * nH * hd
	for _, tc := range []struct {
		name                        string
		out, q, k, v                []float32
		P, kvLen, nH, nKV, hd, wndw int
	}{
		{"kvLen_below_P", marked(full), q, k, v, P, P - 1, nH, nKV, hd, 0},
		{"zero_P", marked(full), q, k, v, 0, kvLen, nH, nKV, hd, 0},
		{"short_q", marked(full), q[:full-1], k, v, P, kvLen, nH, nKV, hd, 0},
		{"short_out", marked(full - 1), q, k, v, P, kvLen, nH, nKV, hd, 0},
		{"short_k", marked(full), q, k[:len(k)-1], v, P, kvLen, nH, nKV, hd, 0},
		{"short_v", marked(full), q, k, v[:len(v)-1], P, kvLen, nH, nKV, hd, 0},
		{"unsupported_hd96", marked(P * nH * 96), q, k, v, P, kvLen, nH, nKV, 96, 0},
		{"nH_not_multiple_of_nKV", marked(P * 6 * hd), q, kWide, vWide, P, kvLen, 6, 4, hd, 0},
		{"zero_nKV", marked(full), q, k, v, P, kvLen, nH, 0, hd, 0},
	} {
		before := slices.Clone(tc.out)
		if _, err := PrefillAttention(tc.out, tc.q, tc.k, tc.v, tc.P, tc.kvLen, tc.nH, tc.nKV, tc.hd, tc.wndw, 0.088); err == nil {
			t.Errorf("%s: PrefillAttention accepted an invalid shape", tc.name)
		}
		if !slices.Equal(tc.out, before) {
			t.Errorf("%s: refused call mutated out", tc.name)
		}
	}
}
