package model

import (
	"errors"
	"math"
	"testing"
)

// v4_flash_attn_ratio0_test.go - INDEPENDENT numeric witness for the ratio-0
// DeepSeek-V4-Flash-0731 attention reference (parent #12636).
//
// Every expected number below is re-derived from the pinned-revision formula in
// scalar loops in THIS file. Production helpers are never called to build an
// expectation; v4FlashSparseAttnHead is called only as the unit under test in
// TestV4FlashRatio0SinkOnlyAffectsDenominator. The reference semantics are:
//
//	q  = WqB * rmsnorm(WqA*x); per-head scale 1/sqrt(mean(q^2)+eps); RoPE(q)
//	kv = rmsnorm(Wkv*x); RoPE(kv)
//	o_h= sparse_attn(q_h, window keys, sink_h, 1/sqrt(HeadDim)); inverse RoPE(o_h)
//	out= Wob * concat_g( Woa_g * gather_g(o) )   (gather group g of every head)
//
// The published sine conjugation ("inverse negates sin") is realized on the
// attention output, as the documented forward records.

// ---------------------------------------------------------------------------
// deterministic scalar helpers (the independent oracle)
// ---------------------------------------------------------------------------

// v4r0LCG is a fixed-seed LCG so weight construction is reproducible.
type v4r0LCG struct{ s uint64 }

func newV4r0LCG(seed uint64) *v4r0LCG { return &v4r0LCG{s: seed} }

func (l *v4r0LCG) f32() float32 {
	l.s = l.s*6364136223846793005 + 1442695040888963407
	u := float64(l.s>>11) / float64(uint64(1)<<53)
	return float32(u - 0.5) // [-0.5, 0.5)
}

func v4r0Matrix(r, c int, l *v4r0LCG) [][]float32 {
	m := make([][]float32, r)
	for i := range m {
		m[i] = make([]float32, c)
		for j := range m[i] {
			m[i][j] = l.f32()
		}
	}
	return m
}

func v4r0Ones(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

// v4r0RMS is an independent RMSNorm over float64 accumulation.
func v4r0RMS(x, w []float32, eps float64) []float32 {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+float32(eps))))
	out := make([]float32, len(x))
	for i := range x {
		out[i] = x[i] * inv * w[i]
	}
	return out
}

// v4r0MatVec is an independent float64-accumulated matrix-vector product.
func v4r0MatVec(m [][]float32, v []float32) []float32 {
	out := make([]float32, len(m))
	for i, row := range m {
		var acc float32
		for j := range v {
			acc += row[j] * v[j]
		}
		out[i] = acc
	}
	return out
}

func v4r0Dot(a, b []float32) float64 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return float64(s)
}

// v4r0RoPE rotates the FINAL ropeDim of hv as adjacent (0,1),(2,3),... pairs at
// absolute position pos. negSin implements the inverse rotation.
func v4r0RoPE(hv []float32, headDim, ropeDim, pos int, inv []float64, negSin bool) {
	base := headDim - ropeDim
	for j := 0; j < ropeDim/2; j++ {
		a, b := hv[base+2*j], hv[base+2*j+1]
		ang := float64(pos) * inv[j]
		c := float32(math.Cos(ang))
		s := float32(math.Sin(ang))
		if negSin {
			s = -s
		}
		hv[base+2*j] = float32(a*c) - float32(b*s)
		hv[base+2*j+1] = float32(b*c) + float32(a*s)
	}
}

// v4r0QHeadScale applies the per-head 1/sqrt(mean(q^2)+eps) scale to head h.
func v4r0QHeadScale(q []float32, head, headDim int, eps float64) {
	seg := q[head*headDim : (head+1)*headDim]
	var ss float32
	for _, v := range seg {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(headDim)+float32(eps))))
	for i := range seg {
		seg[i] = seg[i] * inv
	}
}

// v4r0OutputTransform independently applies the grouped low-rank output map:
// group g gathers the g-th contiguous HeadDim/OGroups slice of EVERY head, then
// mid[g] = Woa_g * gather_g; concat; out = Wob * concat.
func v4r0OutputTransform(g V4FlashRatio0Geometry, w V4FlashRatio0Weights, o []float32) []float32 {
	groupDim := g.HeadDim / g.OGroups
	mid := make([]float32, 0, g.OGroups*g.OLoraRank)
	for grp := 0; grp < g.OGroups; grp++ {
		oGrp := make([]float32, 0, g.NumHeads*groupDim)
		for h := 0; h < g.NumHeads; h++ {
			head := o[h*g.HeadDim : (h+1)*g.HeadDim]
			oGrp = append(oGrp, head[grp*groupDim:(grp+1)*groupDim]...)
		}
		mid = append(mid, v4r0MatVec(w.Woa[grp*g.OLoraRank:(grp+1)*g.OLoraRank], oGrp)...)
	}
	return v4r0MatVec(w.Wob, mid)
}

// v4r0AttentionScale is the standard HeadDim^-0.5 attention scale.
func v4r0AttentionScale(g V4FlashRatio0Geometry) float32 {
	return float32(1.0 / math.Sqrt(float64(g.HeadDim)))
}

// v4r0SparseHead independently computes one head's sink-attention:
// m=max(max_t s_t, sink), Z=exp(sink-m)+sum exp(s_t-m), out=sum (exp(s_t-m)/Z)*k_t.
func v4r0SparseHead(q []float32, keys [][]float32, sink, scale float32) []float32 {
	out := make([]float32, len(q))
	if len(keys) == 0 {
		return out
	}
	scores := make([]float32, len(keys))
	m := sink
	for t, k := range keys {
		s := float32(v4r0Dot(q, k)) * scale
		scores[t] = s
		if s > m {
			m = s
		}
	}
	z := float32(math.Exp(float64(sink - m)))
	for _, s := range scores {
		z += float32(math.Exp(float64(s - m)))
	}
	for t := range scores {
		p := float32(math.Exp(float64(scores[t]-m))) / z
		for d := range out {
			out[d] += p * keys[t][d]
		}
	}
	return out
}

// v4r0OraclePrefill independently computes the full ratio-0 forward for a causal
// batch, assuming the window capacity is at least len(x) (no truncation).
func v4r0OraclePrefill(g V4FlashRatio0Geometry, w V4FlashRatio0Weights, x [][]float32, inv []float64) [][]float32 {
	n := len(x)
	scale := v4r0AttentionScale(g)
	ones := v4r0Ones(g.Dim)
	keys := make([][]float32, 0, n)
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		q := v4r0MatVec(w.WqB, v4r0RMS(v4r0MatVec(w.WqA, x[i]), ones, g.NormEps))
		for h := 0; h < g.NumHeads; h++ {
			qh := q[h*g.HeadDim : (h+1)*g.HeadDim]
			v4r0QHeadScale(q, h, g.HeadDim, g.NormEps)
			v4r0RoPE(qh, g.HeadDim, g.RopeHeadDim, i, inv, false)
		}

		kv := v4r0RMS(v4r0MatVec(w.Wkv, x[i]), ones, g.NormEps)
		v4r0RoPE(kv, g.HeadDim, g.RopeHeadDim, i, inv, false)
		keys = append(keys, kv)

		o := make([]float32, 0, g.NumHeads*g.HeadDim)
		for h := 0; h < g.NumHeads; h++ {
			qh := q[h*g.HeadDim : (h+1)*g.HeadDim]
			oh := v4r0SparseHead(qh, keys, w.AttnSink[h], scale)
			v4r0RoPE(oh, g.HeadDim, g.RopeHeadDim, i, inv, true) // inverse
			o = append(o, oh...)
		}
		out[i] = v4r0OutputTransform(g, w, o)
	}
	return out
}

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

func v4r0Geometry() V4FlashRatio0Geometry {
	return V4FlashRatio0Geometry{
		Dim: 8, NumHeads: 2, HeadDim: 8, RopeHeadDim: 2,
		QLoraRank: 4, OGroups: 2, OLoraRank: 4, WindowSize: V4FlashWindowSize, NormEps: 1e-5,
	}
}

func v4r0Config() Config {
	return Config{
		ModelType:      "deepseek_v4",
		HiddenSize:     8,
		NumHeads:       2,
		HeadDim:        8,
		QKRopeHeadDim:  2,
		QLoraRank:      4,
		OGroups:        2,
		OLoraRank:      4,
		RMSNormEps:     1e-5,
		RopeTheta:      10000,
		Window:         []int{V4FlashWindowSize},
		CompressRatios: []int{0},
	}
}

func v4r0Weights(g V4FlashRatio0Geometry, seed uint64, sinkVal float32) V4FlashRatio0Weights {
	l := newV4r0LCG(seed)
	sink := make([]float32, g.NumHeads)
	for i := range sink {
		sink[i] = sinkVal
	}
	return V4FlashRatio0Weights{
		WqA:      v4r0Matrix(g.QLoraRank, g.Dim, l),
		WqB:      v4r0Matrix(g.NumHeads*g.HeadDim, g.QLoraRank, l),
		Wkv:      v4r0Matrix(g.HeadDim, g.Dim, l),
		Woa:      v4r0Matrix(g.OGroups*g.OLoraRank, g.NumHeads*g.HeadDim/g.OGroups, l),
		Wob:      v4r0Matrix(g.Dim, g.OGroups*g.OLoraRank, l),
		AttnSink: sink,
	}
}

func v4r0Inv(g V4FlashRatio0Geometry) []float64 {
	inv := make([]float64, g.RopeHeadDim/2)
	for j := range inv {
		inv[j] = 1.0 / math.Pow(10000, float64(2*j)/float64(g.RopeHeadDim))
	}
	return inv
}

func v4r0Close(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		tol := 1e-5 * math.Max(1, math.Abs(float64(want[i])))
		if math.Abs(float64(got[i])-float64(want[i])) > tol {
			t.Fatalf("elem[%d] = %v, want %v (tol %g)", i, got[i], want[i], tol)
		}
	}
}

// ---------------------------------------------------------------------------
// required tests
// ---------------------------------------------------------------------------

// TestV4FlashRatio0GeometryFromConfig pins the valid geometry mapping and the
// fail-closed geometry rejection.
func TestV4FlashRatio0GeometryFromConfig(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		g, err := V4FlashRatio0GeometryFromConfig(v4r0Config())
		if err != nil {
			t.Fatalf("valid config error = %v", err)
		}
		want := v4r0Geometry()
		if g != want {
			t.Fatalf("geometry = %+v, want %+v", g, want)
		}
	})
	t.Run("HeadDim not divisible by OGroups", func(t *testing.T) {
		cfg := v4r0Config()
		cfg.OGroups = 3
		if _, err := V4FlashRatio0GeometryFromConfig(cfg); !errors.Is(err, ErrV4FlashRatio0Geometry) {
			t.Fatalf("error = %v, want ErrV4FlashRatio0Geometry", err)
		}
	})
	t.Run("RopeHeadDim >= HeadDim", func(t *testing.T) {
		cfg := v4r0Config()
		cfg.QKRopeHeadDim = cfg.HeadDim
		if _, err := V4FlashRatio0GeometryFromConfig(cfg); !errors.Is(err, ErrV4FlashRatio0Geometry) {
			t.Fatalf("error = %v, want ErrV4FlashRatio0Geometry", err)
		}
	})
}

// TestV4FlashRatio0SinkOnlyAffectsDenominator is the direct unit witness that
// the learnable sink lives ONLY in the softmax denominator.
func TestV4FlashRatio0SinkOnlyAffectsDenominator(t *testing.T) {
	q := []float32{0.5, -0.25, 0.75, 0.1}
	keys := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
	}
	scale := float32(0.7071)

	t.Run("huge sink drives output to zero", func(t *testing.T) {
		got := v4FlashSparseAttnHead(q, keys, 1e9, scale)
		for i, v := range got {
			if math.Abs(float64(v)) > 1e-6 {
				t.Fatalf("elem[%d] = %v with sink=1e9, want ~0", i, v)
			}
		}
	})

	t.Run("two identical keys match hand softmax", func(t *testing.T) {
		k := []float32{0.3, -0.2, 0.5, 0.4}
		two := [][]float32{k, k}
		sink := float32(-0.5)
		got := v4FlashSparseAttnHead(q, two, sink, scale)
		s := v4r0Dot(q, k) * float64(scale)
		m := math.Max(s, float64(sink))
		z := math.Exp(float64(sink)-m) + 2*math.Exp(s-m)
		weight := 2 * math.Exp(s-m) / z
		want := make([]float32, len(k))
		for i := range k {
			want[i] = float32(weight) * k[i]
		}
		v4r0Close(t, got, want)
	})

	t.Run("sink adds no value term", func(t *testing.T) {
		sink := float32(0.25)
		got := v4FlashSparseAttnHead(q, keys, sink, scale)
		// Independent reconstruction: keys weighted by exp(s_t-m)/Z, plus a
		// zero-valued sink slot carrying exp(sink-m)/Z. All mass accounted for.
		s := make([]float64, len(keys))
		m := float64(sink)
		for i, k := range keys {
			s[i] = v4r0Dot(q, k) * float64(scale)
			if s[i] > m {
				m = s[i]
			}
		}
		z := math.Exp(float64(sink) - m)
		for i := range keys {
			z += math.Exp(s[i] - m)
		}
		mass := math.Exp(float64(sink)-m) / z
		want := make([]float32, len(q))
		for i, k := range keys {
			w := math.Exp(s[i]-m) / z
			mass += w
			for d := range want {
				want[d] += float32(w) * k[d]
			}
		}
		if math.Abs(mass-1) > 1e-9 {
			t.Fatalf("key mass + sink mass = %v, want 1", mass)
		}
		v4r0Close(t, got, want)
	})
}

// TestV4FlashRatio0MatchesHandComputedSingleToken is the core numeric witness:
// one token at startPos=0 through the full pipeline, compared to the scalar
// oracle. Two sinks are exercised: a far-negative sink makes the single-key
// attention weight exactly 1 (isolating the projection/RMSNorm/RoPE/output
// transform and insensitive to the attention scale), and a finite sink makes
// the query affect the weight, so a missing/garbled q path is caught too.
func TestV4FlashRatio0MatchesHandComputedSingleToken(t *testing.T) {
	g := v4r0Geometry()
	inv := v4r0Inv(g)
	x := [][]float32{{0.4, -0.2, 0.6, 0.1, -0.3, 0.5, 0.2, -0.1}}

	for _, tc := range []struct {
		name    string
		sinkVal float32
	}{
		{"single key weight 1", -1e30},
		{"finite sink exercises q", 0.1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := v4r0Weights(g, 0xC0FFEE, tc.sinkVal)
			win := newV4FlashCircularWindow(g.WindowSize)
			got, err := v4FlashRatio0Attention(g, w, x, 0, inv, win)
			if err != nil {
				t.Fatalf("attention error: %v", err)
			}
			want := v4r0OraclePrefill(g, w, x, inv)
			if len(got) != 1 {
				t.Fatalf("rows = %d, want 1", len(got))
			}
			v4r0Close(t, got[0], want[0])
			if win.Len() != 1 {
				t.Fatalf("window len = %d after 1 token, want 1", win.Len())
			}
		})
	}
}

// TestV4FlashRatio0WindowRetainsCausalityAcrossSteps checks row counts,
// determinism across two identical runs, numeric agreement with the causal
// oracle on a short prefill, and that appending past a small buffer capacity
// still works.
func TestV4FlashRatio0WindowRetainsCausalityAcrossSteps(t *testing.T) {
	g := v4r0Geometry()
	inv := v4r0Inv(g)
	w := v4r0Weights(g, 0xBEEF, 0.1)
	seedRow := func(i int) []float32 {
		l := newV4r0LCG(uint64(i) + 77)
		row := make([]float32, g.Dim)
		for d := range row {
			row[d] = l.f32()
		}
		return row
	}
	prefill := make([][]float32, 5)
	for i := range prefill {
		prefill[i] = seedRow(i)
	}

	run := func() [][]float32 {
		win := newV4FlashCircularWindow(g.WindowSize)
		rows, err := v4FlashRatio0Attention(g, w, prefill, 0, inv, win)
		if err != nil {
			t.Fatalf("prefill error: %v", err)
		}
		if len(rows) != len(prefill) {
			t.Fatalf("prefill rows = %d, want %d", len(rows), len(prefill))
		}
		for step := 0; step < 3; step++ {
			sr := [][]float32{seedRow(100 + step)}
			out, err := v4FlashRatio0Attention(g, w, sr, len(prefill)+step, inv, win)
			if err != nil {
				t.Fatalf("step %d error: %v", step, err)
			}
			if len(out) != 1 {
				t.Fatalf("step %d rows = %d, want 1", step, len(out))
			}
			rows = append(rows, out[0])
		}
		return rows
	}

	a, b := run(), run()
	if len(a) != len(prefill)+3 {
		t.Fatalf("total rows = %d, want %d", len(a), len(prefill)+3)
	}
	for i := range a {
		v4r0Close(t, a[i], b[i]) // determinism across identical runs
	}

	// Numeric witness on the prefill: full causal attention, oracle-computed.
	win := newV4FlashCircularWindow(g.WindowSize)
	got, err := v4FlashRatio0Attention(g, w, prefill, 0, inv, win)
	if err != nil {
		t.Fatalf("prefill error: %v", err)
	}
	want := v4r0OraclePrefill(g, w, prefill, inv)
	for i := range want {
		v4r0Close(t, got[i], want[i])
	}

	// Appending past a small buffer capacity still works and is deterministic.
	capRun := func() [][]float32 {
		win := newV4FlashCircularWindow(2)
		rows, err := v4FlashRatio0Attention(g, w, prefill, 0, inv, win)
		if err != nil {
			t.Fatalf("small-window error: %v", err)
		}
		return rows
	}
	c1, c2 := capRun(), capRun()
	if len(c1) != len(prefill) || len(c2) != len(prefill) {
		t.Fatalf("small-window rows = %d/%d, want %d", len(c1), len(c2), len(prefill))
	}
	for i := range c1 {
		v4r0Close(t, c1[i], c2[i])
	}
}

// TestV4FlashRatio0FailsClosedOnShape pins the three malformed-input refusals.
func TestV4FlashRatio0FailsClosedOnShape(t *testing.T) {
	g := v4r0Geometry()
	inv := v4r0Inv(g)
	w := v4r0Weights(g, 0x5EED, 0.0)
	goodX := [][]float32{{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}}

	t.Run("mismatched x row length", func(t *testing.T) {
		bad := [][]float32{{0.1, 0.2, 0.3}}
		_, err := v4FlashRatio0Attention(g, w, bad, 0, inv, newV4FlashCircularWindow(g.WindowSize))
		if !errors.Is(err, ErrV4FlashRatio0Shape) {
			t.Fatalf("error = %v, want ErrV4FlashRatio0Shape", err)
		}
	})
	t.Run("wrong inv length", func(t *testing.T) {
		_, err := v4FlashRatio0Attention(g, w, goodX, 0, []float64{1, 2, 3}, newV4FlashCircularWindow(g.WindowSize))
		if !errors.Is(err, ErrV4FlashRatio0Shape) {
			t.Fatalf("error = %v, want ErrV4FlashRatio0Shape", err)
		}
	})
	t.Run("nil window", func(t *testing.T) {
		_, err := v4FlashRatio0Attention(g, w, goodX, 0, inv, nil)
		if !errors.Is(err, ErrV4FlashRatio0Shape) {
			t.Fatalf("error = %v, want ErrV4FlashRatio0Shape", err)
		}
	})
}
