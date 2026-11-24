package model

import (
	"math"
	"reflect"
	"slices"
	"testing"
)

func TestInkling2LoopConvolveStaticAndDynamic(t *testing.T) {
	h := &Inkling2LoopHost{HP: Inkling2LoopHParams{ConvKernel: 4}, conv: [][]float32{make([]float32, 2*3)}}
	w := []float32{1, 2, 3, 4, 10, 20, 30, 40}
	xs := [][]float32{{1, -1}, {2, -2}, {3, -3}, {4, -4}}
	var got []float32
	for _, x := range xs {
		got = h.convolve(0, x, w, nil, 0, 0)
	}
	// x[t-3..t] = 1,2,3,4 against taps oldest-first 1,2,3,4 plus the residual, over sqrt(5)
	want0 := (4 + (1*1 + 2*2 + 3*3 + 4*4)) / float32(math.Sqrt(5))
	want1 := (-4 + -(1*10 + 2*20 + 3*30 + 4*40)) / float32(math.Sqrt(5))
	if math.Abs(float64(got[0]-want0)) > 1e-5 || math.Abs(float64(got[1]-want1)) > 1e-4 {
		t.Fatalf("static conv = %v, want [%v %v]", got, want0, want1)
	}

	h = &Inkling2LoopHost{HP: Inkling2LoopHParams{ConvKernel: 4}, conv: [][]float32{make([]float32, 2*3)}}
	dyn := []float32{0.5, 0.25, 0, 1} // one head: lag 0..3
	zero := make([]float32, 8)
	for _, x := range xs {
		got = h.convolve(0, x, zero, dyn, 1, 0)
	}
	// lag 0 -> x[t]=4, lag 1 -> 3, lag 3 -> 1
	wantDyn := (4 + 0.5*4 + 0.25*3 + 1*1) / float32(3)
	if math.Abs(float64(got[0]-wantDyn)) > 1e-5 {
		t.Fatalf("dynamic conv = %v, want %v", got[0], wantDyn)
	}
}

func TestInkling2LoopMixWeightsNormalizedStates(t *testing.T) {
	a := []float32{3, 0, 0, 0}
	b := []float32{0, 5, 0, 0}
	q := []float32{2, 0, 0, 0}
	got := inkling2Mix([][]float32{a}, b, q)
	// normalized a = (2,0,0,0), b = (0,2,0,0); scores 4/2 and 0
	wa := math.Exp(2) / (math.Exp(2) + 1)
	want := []float32{float32(2 * wa), float32(2 * (1 - wa)), 0, 0}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("mix = %v, want %v", got, want)
		}
	}
}

func TestInkling2LoopClipCapsNormOnlyAboveThreshold(t *testing.T) {
	x := []float32{3, 4}
	inkling2Clip(x, nil, 10)
	if x[0] != 3 || x[1] != 4 {
		t.Fatalf("clip under threshold changed %v", x)
	}
	inkling2Clip(x, []float32{2, 2}, 5)
	if n := math.Hypot(float64(x[0]), float64(x[1])); math.Abs(n-5) > 1e-4 {
		t.Fatalf("clipped norm %v, want 5", n)
	}
}

func TestInkling2LoopTopKPrefersLowerIndexOnTies(t *testing.T) {
	got := inkling2TopK([]float32{1, 3, 3, 2}, 3)
	if got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Fatalf("topk = %v", got)
	}
}

type inkling2TestEmbedding struct{ dim int }

func (e inkling2TestEmbedding) Row(token int, dst []float32) {
	for i := range dst {
		dst[i] = float32(math.Sin(float64(token*e.dim + i + 1)))
	}
}

type inkling2TestExperts struct {
	n int
	m []*Inkling2LoopDenseMatrix
}

func (e *inkling2TestExperts) Experts() int { return e.n }
func (e *inkling2TestExperts) Rows() int    { return e.m[0].R }
func (e *inkling2TestExperts) Cols() int    { return e.m[0].C }
func (e *inkling2TestExperts) ExpertMatVec(x int, dst, v []float32) {
	e.m[x].MatVec(dst, v)
}

func inkling2TestMat(rows, cols, seed int) *Inkling2LoopDenseMatrix {
	m := &Inkling2LoopDenseMatrix{R: rows, C: cols, Data: make([]float32, rows*cols)}
	for i := range m.Data {
		m.Data[i] = float32(math.Sin(float64(seed*7919+i)*0.37)) * 0.3
	}
	return m
}

func inkling2TestVec(n, seed int, base float32) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = base + 0.1*float32(math.Cos(float64(seed*31+i)))
	}
	return v
}

func inkling2TinyModel() (Inkling2LoopHParams, *Inkling2LoopWeights) {
	hp := Inkling2LoopHParams{
		Hidden: 8, Layers: 6, Heads: 2, KVHeads: 1, HeadDim: 4, VocabUnpadded: 10,
		SlidingWindow: 3, RelLength: 2, RelExtent: 5, ConvKernel: 4, AttnResBlock: 2,
		Experts: 4, ExpertsUsed: 2, ExpertWidth: 3, CoreStart: 2, CoreEnd: 4, Passes: 3, GateRank: 2,
		LocalLayer: []bool{true, false, true, false, true, false},
		NormEps:    1e-6, QKClip: 2, RelClip: 1, LogScaleFloor: 2, LogScaleAlpha: 0.1,
		SituBeta: 4, SituLinearBeta: 25, ExpertWeightsScale: 8, FinalLogitSoftcap: 30,
	}
	w := &Inkling2LoopWeights{
		TokEmbd: inkling2TestEmbedding{dim: hp.Hidden}, TokNorm: inkling2TestVec(8, 1, 1),
		Output: inkling2TestMat(12, 8, 2), OutputNorm: inkling2TestVec(8, 3, 1), OutputResQuery: inkling2TestVec(8, 4, 0),
	}
	kv := hp.KVHeads * hp.HeadDim
	for il := 0; il < hp.Layers; il++ {
		s := 100 * (il + 1)
		lw := Inkling2LoopLayerWeights{
			Wq: inkling2TestMat(8, 8, s+1), Wk: inkling2TestMat(kv, 8, s+2), Wv: inkling2TestMat(kv, 8, s+3),
			Wo: inkling2TestMat(8, 8, s+4), Gate: inkling2TestMat(8, 8, s+5), Rel: inkling2TestMat(4, 8, s+6),
			SinkProj: inkling2TestMat(2, 8, s+7),
			QClip:    inkling2TestVec(8, s+8, 1), RelClip: inkling2TestVec(4, s+9, 1),
			KConv: inkling2TestVec(4*kv, s+10, 0), VConv: inkling2TestVec(4*kv, s+11, 0),
			AttnPostConv: inkling2TestVec(32, s+12, 0), FFNPostConv: inkling2TestVec(32, s+13, 0),
			FFNResQuery: inkling2TestVec(8, s+14, 0), FFNNorm: inkling2TestVec(8, s+15, 1),
			ExpProbsBias: inkling2TestVec(4, s+16, 0), Router: inkling2TestMat(4, 8, s+17),
		}
		extent := hp.RelExtent
		if hp.LocalLayer[il] {
			extent = hp.SlidingWindow
			lw.AttnNorm = inkling2TestVec(8, s+18, 1)
		} else {
			lw.QInpNorm, lw.KInpNorm, lw.VInpNorm = inkling2TestVec(8, s+19, 1), inkling2TestVec(8, s+20, 1), inkling2TestVec(8, s+21, 1)
			lw.KConvProj, lw.VConvProj = inkling2TestMat(4, 8, s+22), inkling2TestMat(4, 8, s+23)
		}
		lw.RelBias = inkling2TestVec(extent*hp.RelLength, s+24, 0)
		if il > 0 {
			n := 3
			if hp.LocalLayer[il] {
				n = 1
			}
			for j := 0; j < n; j++ {
				lw.AttnResQuery = append(lw.AttnResQuery, inkling2TestVec(8, s+25+j, 0))
			}
		}
		gu, dn := &inkling2TestExperts{n: 4}, &inkling2TestExperts{n: 4}
		for e := 0; e < 4; e++ {
			gu.m = append(gu.m, inkling2TestMat(6, 8, s+30+e))
			dn.m = append(dn.m, inkling2TestMat(8, 3, s+40+e))
		}
		lw.GateUp, lw.Down = gu, dn
		w.Layers = append(w.Layers, lw)
	}
	for c := 0; c < hp.CoreEnd-hp.CoreStart; c++ {
		var cw Inkling2LoopControllerWeights
		for j := 0; j < 4; j++ {
			s := 1000*(c+1) + 10*j
			cw.MsgQuery[j] = inkling2TestVec(8, s, 0)
			cw.Down[j] = inkling2TestMat(2, 16, s+1)
			cw.Gain[j] = inkling2TestMat(8, 2, s+2)
			cw.Forget[j] = inkling2TestMat(8, 2, s+3)
		}
		w.Controllers = append(w.Controllers, cw)
	}
	return hp, w
}

func TestInkling2LoopHostTinyForwardIsDeterministicAndUsesEveryPassSlot(t *testing.T) {
	hp, w := inkling2TinyModel()
	run := func() [][]float32 {
		h, err := NewInkling2LoopHost(hp, w)
		if err != nil {
			t.Fatal(err)
		}
		if h.Slots() != 6+2*2 {
			t.Fatalf("slots = %d, want 10", h.Slots())
		}
		var out [][]float32
		for _, tok := range []int{1, 4, 2, 7, 3} {
			out = append(out, append([]float32(nil), h.Step(tok)...))
		}
		for s := range h.keys {
			if len(h.keys[s]) != 5 {
				t.Fatalf("slot %d holds %d keys, want 5", s, len(h.keys[s]))
			}
		}
		return out
	}
	a, b := run(), run()
	for i := range a {
		if len(a[i]) != hp.VocabUnpadded {
			t.Fatalf("logits len %d, want unpadded %d", len(a[i]), hp.VocabUnpadded)
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] || math.IsNaN(float64(a[i][j])) || math.Abs(float64(a[i][j])) > 30 {
				t.Fatalf("step %d logit %d: %v vs %v (softcap 30)", i, j, a[i][j], b[i][j])
			}
		}
	}
}

func TestInkling2LoopHostPassCountChangesOutput(t *testing.T) {
	hp, w := inkling2TinyModel()
	logits := func(passes int) []float32 {
		hp.Passes = passes
		h, err := NewInkling2LoopHost(hp, w)
		if err != nil {
			t.Fatal(err)
		}
		h.Step(1)
		return h.Step(5)
	}
	one, three := logits(1), logits(3)
	same := true
	for i := range one {
		if one[i] != three[i] {
			same = false
		}
	}
	if same {
		t.Fatal("one and three loop passes produced identical logits; the cross-pass messages are not applied")
	}
}

func TestInkling2LoopHostRejectsMisalignedCore(t *testing.T) {
	hp, w := inkling2TinyModel()
	hp.CoreStart = 1
	if _, err := NewInkling2LoopHost(hp, w); err == nil {
		t.Fatal("misaligned loop core accepted")
	}
}

// inkling2ReverseBatcher runs each batch back to front, so a forward that
// leaned on job order or on MatVecs running before the batch returns would
// diverge from the unbatched path.
type inkling2ReverseBatcher struct{ batches, jobs int }

func (b *inkling2ReverseBatcher) MatVecBatch(jobs []Inkling2LoopMatVecJob) {
	b.batches++
	b.jobs += len(jobs)
	for i := len(jobs) - 1; i >= 0; i-- {
		if j := jobs[i]; j.Matrix != nil {
			j.Matrix.MatVec(j.Dst, j.X)
		} else {
			j.Experts.ExpertMatVec(j.Expert, j.Dst, j.X)
		}
	}
}

func TestInkling2LoopHostBatcherMatchesUnbatchedForward(t *testing.T) {
	run := func(b Inkling2LoopBatcher) [][]float32 {
		hp, w := inkling2TinyModel()
		w.Batcher = b
		h, err := NewInkling2LoopHost(hp, w)
		if err != nil {
			t.Fatal(err)
		}
		var out [][]float32
		for _, tok := range []int{1, 5, 2, 7} {
			out = append(out, append([]float32(nil), h.Step(tok)...))
		}
		return out
	}
	want := run(nil)
	b := &inkling2ReverseBatcher{}
	got := run(b)
	if b.batches == 0 || b.jobs <= b.batches {
		t.Fatalf("batcher saw %d batches / %d jobs; the forward did not batch", b.batches, b.jobs)
	}
	for s := range want {
		for i := range want[s] {
			if got[s][i] != want[s][i] {
				t.Fatalf("step %d logit %d: batched %v, unbatched %v", s, i, got[s][i], want[s][i])
			}
		}
	}
}

func TestInkling2LoopPrefillKeepsSlotsAndFeatureTaps(t *testing.T) {
	hp, w := inkling2TinyModel()
	a, err := NewInkling2LoopHost(hp, w)
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewInkling2LoopHost(hp, w)
	if err != nil {
		t.Fatal(err)
	}
	a.FeatureVisits = []int{1, 3, a.Visits()}
	b.FeatureVisits = a.FeatureVisits
	for _, id := range []int{1, 4, 2} {
		a.Step(id)
		if got := b.PrefillToken(id, false); got != nil {
			t.Fatal("non-final prompt row produced logits")
		}
		if !reflect.DeepEqual(a.Features, b.Features) {
			t.Fatal("feature taps differ after token", id)
		}
	}
	if !slices.Equal(a.Step(3), b.Step(3)) {
		t.Fatal("decode logits changed after prompt-only rows")
	}
}
