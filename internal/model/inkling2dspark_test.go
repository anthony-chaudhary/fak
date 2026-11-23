package model

import (
	"math"
	"math/rand"
	"testing"
)

type inkling2DSparkTestEmbedding struct {
	width int
	data  []float32
}

func (e inkling2DSparkTestEmbedding) Row(token int, dst []float32) {
	copy(dst, e.data[token*e.width:(token+1)*e.width])
}

func inkling2DSparkTestDraft(t *testing.T, local []bool, window int) *Inkling2DSparkDraft {
	t.Helper()
	rng := rand.New(rand.NewSource(7))
	vec := func(n int, scale float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = scale * float32(rng.NormFloat64())
		}
		return v
	}
	ones := func(n int) []float32 {
		v := vec(n, 0.1)
		for i := range v {
			v[i] += 1
		}
		return v
	}
	mat := func(r, c int) *Inkling2LoopDenseMatrix {
		return &Inkling2LoopDenseMatrix{R: r, C: c, Data: vec(r*c, float32(1/math.Sqrt(float64(c))))}
	}
	hp := Inkling2DSparkHParams{
		Hidden: 16, Layers: len(local), Heads: 4, KVHeads: 2, HeadDim: 8, Intermediate: 24,
		Block: 4, MarkovRank: 4, SlidingWindow: window, VocabUnpadded: 30, Taps: 3,
		LocalLayer: local, NormEps: 1e-6, RopeTheta: 1e4,
	}
	kv := hp.KVHeads * hp.HeadDim
	w := &Inkling2DSparkWeights{
		Norm: ones(hp.Hidden), MaskEmbed: vec(hp.Hidden, 1),
		MarkovW1:      inkling2DSparkTestEmbedding{hp.MarkovRank, vec(32*hp.MarkovRank, 1)},
		RnnProj:       mat(3*hp.MarkovRank, 2*hp.MarkovRank+2*hp.Hidden),
		RnnBias:       vec(3*hp.MarkovRank, 0.1),
		HiddenOut:     mat(hp.Hidden, hp.MarkovRank),
		HiddenOutBias: vec(hp.Hidden, 0.1),
		MarkovNorm:    ones(hp.Hidden),
		TokEmbd:       inkling2DSparkTestEmbedding{hp.Hidden, vec(32*hp.Hidden, 1)},
		TokNorm:       ones(hp.Hidden),
		Output:        mat(32, hp.Hidden),
	}
	for l := 0; l <= hp.Layers; l++ {
		w.FuseWeight = append(w.FuseWeight, vec(hp.Taps, 1))
	}
	for l := 0; l < hp.Layers; l++ {
		w.Layers = append(w.Layers, Inkling2DSparkLayerWeights{
			InputNorm: ones(hp.Hidden), PostAttnNorm: ones(hp.Hidden), DFlareNorm: ones(hp.Hidden),
			QNorm: ones(hp.HeadDim), KNorm: ones(hp.HeadDim), KNormCtx: ones(hp.HeadDim),
			Wq: mat(hp.Heads*hp.HeadDim, hp.Hidden), Wk: mat(kv, hp.Hidden), Wv: mat(kv, hp.Hidden),
			Wo: mat(hp.Hidden, hp.Heads*hp.HeadDim), WkCtx: mat(kv, hp.Hidden), WvCtx: mat(kv, hp.Hidden),
			Gate: mat(hp.Intermediate, hp.Hidden), Up: mat(hp.Intermediate, hp.Hidden), Down: mat(hp.Hidden, hp.Intermediate),
		})
	}
	d, err := NewInkling2DSparkDraft(hp, w)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func inkling2DSparkTestFeatures(n, taps, hidden int, seed int64) [][][]float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([][][]float32, n)
	for i := range out {
		for t := 0; t < taps; t++ {
			f := make([]float32, hidden)
			for j := range f {
				f[j] = float32(rng.NormFloat64())
			}
			out[i] = append(out[i], f)
		}
	}
	return out
}

func inkling2DSparkRun(t *testing.T, d *Inkling2DSparkDraft, features [][][]float32, first int) ([]int, [][]float32) {
	t.Helper()
	for i, f := range features {
		if err := d.Commit(first+i, f); err != nil {
			t.Fatal(err)
		}
	}
	tokens, hidden, err := d.draft(5, first+len(features))
	if err != nil {
		t.Fatal(err)
	}
	return tokens, hidden
}

func inkling2DSparkMaxDiff(a, b [][]float32) float64 {
	var m float64
	for i := range a {
		for j := range a[i] {
			m = math.Max(m, math.Abs(float64(a[i][j]-b[i][j])))
		}
	}
	return m
}

// RoPE makes the draft depend only on relative positions: shifting the whole
// context and block keeps every hidden row and drafted token.
func TestInkling2DSparkRelativePositionInvariant(t *testing.T) {
	features := inkling2DSparkTestFeatures(9, 3, 16, 1)
	base, baseHidden := inkling2DSparkRun(t, inkling2DSparkTestDraft(t, []bool{false, false}, 0), features, 0)
	shifted, shiftedHidden := inkling2DSparkRun(t, inkling2DSparkTestDraft(t, []bool{false, false}, 0), features, 1000)
	if diff := inkling2DSparkMaxDiff(baseHidden, shiftedHidden); diff > 1e-4 {
		t.Fatalf("hidden moved by %g under a position shift", diff)
	}
	for i := range base {
		if base[i] != shifted[i] {
			t.Fatalf("drafts %v vs shifted %v", base, shifted)
		}
	}
	if len(base) != 3 {
		t.Fatalf("drafted %d tokens, want Block-1 = 3", len(base))
	}
}

// Local layers ignore context older than SlidingWindow before the anchor; a
// global layer does not, and Rewind restores the shorter context exactly.
func TestInkling2DSparkLocalWindowAndRewind(t *testing.T) {
	features := inkling2DSparkTestFeatures(8, 3, 16, 2)
	perturbed := inkling2DSparkTestFeatures(8, 3, 16, 2)
	perturbed[0] = inkling2DSparkTestFeatures(1, 3, 16, 99)[0] // position 0, outside a window of 4 before 8

	_, a := inkling2DSparkRun(t, inkling2DSparkTestDraft(t, []bool{true, true}, 4), features, 0)
	_, b := inkling2DSparkRun(t, inkling2DSparkTestDraft(t, []bool{true, true}, 4), perturbed, 0)
	if diff := inkling2DSparkMaxDiff(a, b); diff != 0 {
		t.Fatalf("local layers read context outside the window (diff %g)", diff)
	}
	_, g1 := inkling2DSparkRun(t, inkling2DSparkTestDraft(t, []bool{false, true}, 4), features, 0)
	_, g2 := inkling2DSparkRun(t, inkling2DSparkTestDraft(t, []bool{false, true}, 4), perturbed, 0)
	if inkling2DSparkMaxDiff(g1, g2) == 0 {
		t.Fatal("global layer ignored old context")
	}

	d := inkling2DSparkTestDraft(t, []bool{false, true}, 4)
	inkling2DSparkRun(t, d, features, 0)
	d.Rewind(6)
	if d.Context() != 6 {
		t.Fatalf("context after rewind = %d, want 6", d.Context())
	}
	_, rewound, err := d.draft(5, 6)
	if err != nil {
		t.Fatal(err)
	}
	_, fresh := inkling2DSparkRun(t, inkling2DSparkTestDraft(t, []bool{false, true}, 4), features[:6], 0)
	if diff := inkling2DSparkMaxDiff(rewound, fresh); diff != 0 {
		t.Fatalf("rewound draft differs from a fresh short context (diff %g)", diff)
	}
}
