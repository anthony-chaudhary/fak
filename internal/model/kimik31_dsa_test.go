package model

// kimik31_dsa_test.go — closed-sentinel and closed-form tests for the Kimi-
// K3.1 Gated DSA host oracle (ticket 1611-kimi-k31 leaf L6). Numeric parity
// vs the independent float64 transcription lives in kimik31_dsa_naive_test.go.

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// dsaTinyParams builds a validated tiny-geometry DSA param set: NH=2,
// qlora=3, kvlora=2, nope=2, rope=1, vdim=2, hidden=4, idxHeads=2, idxDim=2,
// topK=2. Weights are deterministic pseudo-random values in [-1, 1).
func dsaTinyParams(seed int64, withGate bool) *KimiK31DSAParams {
	rnd := rand.New(rand.NewSource(seed))
	pick := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = rnd.Float32()*2 - 1
		}
		return out
	}
	p := &KimiK31DSAParams{
		NumHeads: 2, QLoRA: 3, KVLoRA: 2, QkNope: 2, QkRope: 1, VDim: 2, Hidden: 4,
		IdxHeads: 2, IdxDim: 2, TopK: 2,
		RMSEps: 1e-5, IdxEps: 1e-5,
		WQa:       pick(3 * 4),
		WQaNorm:   pick(3),
		WQb:       pick(2 * 3 * 3),
		WKvA:      pick(3 * 4),
		WKvANorm:  pick(2),
		WKb:       pick(2 * 2 * 2),
		WVb:       pick(2 * 2 * 2),
		WO:        pick(4 * 2 * 2),
		WIdxQ:     pick(2 * 2 * 3),
		WIdxK:     pick(2 * 4),
		WIdxProj:  pick(2 * 4),
		IdxKNormW: pick(2),
		IdxKNormB: pick(2),
	}
	if withGate {
		p.WGate = pick(2 * 2 * 4)
	}
	return p
}

func TestKimiK31DSAWeightNames(t *testing.T) {
	n := KimiK31DSAWeightNamesFor(7)
	want := KimiK31DSAWeightNames{
		QaProj:      "model.layers.7.self_attn.q_a_proj.weight",
		QaNorm:      "model.layers.7.self_attn.q_a_layernorm.weight",
		QbProj:      "model.layers.7.self_attn.q_b_proj.weight",
		KvAProj:     "model.layers.7.self_attn.kv_a_proj_with_mqa.weight",
		KvANorm:     "model.layers.7.self_attn.kv_a_layernorm.weight",
		KvBCombined: "model.layers.7.self_attn.kv_b_proj.weight",
		KBProj:      "model.layers.7.self_attn.k_b_proj.weight",
		VBProj:      "model.layers.7.self_attn.v_b_proj.weight",
		OGate:       "model.layers.7.self_attn.o_gate.weight",
		OProj:       "model.layers.7.self_attn.o_proj.weight",
		IdxQb:       "model.layers.7.self_attn.indexer.wq_b.weight",
		IdxK:        "model.layers.7.self_attn.indexer.wk.weight",
		IdxProj:     "model.layers.7.self_attn.indexer.weights_proj.weight",
		IdxKNormW:   "model.layers.7.self_attn.indexer.k_norm.weight",
		IdxKNormB:   "model.layers.7.self_attn.indexer.k_norm.bias",
	}
	if n != want {
		t.Fatalf("weight names mismatch:\n got %+v\nwant %+v", n, want)
	}
}

func TestKimiK31DSAValidate(t *testing.T) {
	base := dsaTinyParams(1, false)
	if err := ValidateKimiK31DSAParams(base); err != nil {
		t.Fatalf("baseline params must validate: %v", err)
	}
	if err := ValidateKimiK31DSAParams(dsaTinyParams(1, true)); err != nil {
		t.Fatalf("gated params must validate: %v", err)
	}

	mutate := func(name string, fn func(p *KimiK31DSAParams)) *KimiK31DSAParams {
		p := *base
		fn(&p)
		return &p
	}
	shorten := func(s *[]float32, d int) { *s = (*s)[:len(*s)-d] }
	lengthen := func(s *[]float32, d int) {
		grow := make([]float32, d)
		*s = append(*s, grow...)
	}
	cases := []struct {
		name string
		p    *KimiK31DSAParams
	}{
		{"nil", nil},
		{"heads0", mutate("heads0", func(p *KimiK31DSAParams) { p.NumHeads = 0 })},
		{"qlora0", mutate("qlora0", func(p *KimiK31DSAParams) { p.QLoRA = 0 })},
		{"kvlora0", mutate("kvlora0", func(p *KimiK31DSAParams) { p.KVLoRA = 0 })},
		{"nope0", mutate("nope0", func(p *KimiK31DSAParams) { p.QkNope = 0 })},
		{"rope0", mutate("rope0", func(p *KimiK31DSAParams) { p.QkRope = 0 })},
		{"vdim0", mutate("vdim0", func(p *KimiK31DSAParams) { p.VDim = 0 })},
		{"hidden0", mutate("hidden0", func(p *KimiK31DSAParams) { p.Hidden = 0 })},
		{"idxheads0", mutate("idxheads0", func(p *KimiK31DSAParams) { p.IdxHeads = 0 })},
		{"idxdim0", mutate("idxdim0", func(p *KimiK31DSAParams) { p.IdxDim = 0 })},
		{"topk0", mutate("topk0", func(p *KimiK31DSAParams) { p.TopK = 0 })},
		{"rmseps0", mutate("rmseps0", func(p *KimiK31DSAParams) { p.RMSEps = 0 })},
		{"idxeps0", mutate("idxeps0", func(p *KimiK31DSAParams) { p.IdxEps = 0 })},
		{"wqa_short", mutate("wqa_short", func(p *KimiK31DSAParams) { shorten(&p.WQa, 1) })},
		{"wqanorm_long", mutate("wqanorm_long", func(p *KimiK31DSAParams) { lengthen(&p.WQaNorm, 1) })},
		{"wqb_short", mutate("wqb_short", func(p *KimiK31DSAParams) { shorten(&p.WQb, 1) })},
		{"wkva_long", mutate("wkva_long", func(p *KimiK31DSAParams) { lengthen(&p.WKvA, 1) })},
		{"wkvanorm_short", mutate("wkvanorm_short", func(p *KimiK31DSAParams) { shorten(&p.WKvANorm, 1) })},
		{"wkb_long", mutate("wkb_long", func(p *KimiK31DSAParams) { lengthen(&p.WKb, 1) })},
		{"wvb_short", mutate("wvb_short", func(p *KimiK31DSAParams) { shorten(&p.WVb, 1) })},
		{"wo_long", mutate("wo_long", func(p *KimiK31DSAParams) { lengthen(&p.WO, 1) })},
		{"widxq_short", mutate("widxq_short", func(p *KimiK31DSAParams) { shorten(&p.WIdxQ, 1) })},
		{"widxk_long", mutate("widxk_long", func(p *KimiK31DSAParams) { lengthen(&p.WIdxK, 1) })},
		{"widxproj_short", mutate("widxproj_short", func(p *KimiK31DSAParams) { shorten(&p.WIdxProj, 1) })},
		{"idxnormw_long", mutate("idxnormw_long", func(p *KimiK31DSAParams) { lengthen(&p.IdxKNormW, 1) })},
		{"idxnormb_short", mutate("idxnormb_short", func(p *KimiK31DSAParams) { shorten(&p.IdxKNormB, 1) })},
		{"gate_badlen", mutate("gate_badlen", func(p *KimiK31DSAParams) { p.WGate = make([]float32, 5) })},
	}
	for _, tc := range cases {
		err := ValidateKimiK31DSAParams(tc.p)
		if !errors.Is(err, ErrKimiK31DSAParamShape) {
			t.Errorf("%s: want ErrKimiK31DSAParamShape, got %v", tc.name, err)
		}
	}
}

func TestKimiK31DSATopKSelection(t *testing.T) {
	finf := float32(math.Inf(-1))
	cases := []struct {
		name   string
		scores []float32
		topK   int
		want   []int
	}{
		{"distinct", []float32{0.5, 3.0, 1.5, 2.0}, 2, []int{1, 3}},
		{"tie_lower_index_wins", []float32{1.0, 2.0, 2.0, 0.5}, 2, []int{1, 2}},
		{"all_tied", []float32{1, 1, 1}, 2, []int{0, 1}},
		{"fewer_cells_than_topk", []float32{0.9, -0.2}, 5, []int{0, 1}},
		{"neg_inf_rank_last", []float32{finf, 0.1, finf, 0.2}, 2, []int{1, 3}},
		{"neg_inf_fills_when_small", []float32{finf, 0.3}, 3, []int{0, 1}},
		{"single_cell", []float32{-4.2}, 1, []int{0}},
	}
	for _, tc := range cases {
		got := SelectKimiK31DSATopK(tc.scores, tc.topK)
		if len(got) != len(tc.want) {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
				break
			}
		}
	}
	if got := SelectKimiK31DSATopK(nil, 3); got != nil {
		t.Errorf("empty scores: want nil, got %v", got)
	}
	if got := SelectKimiK31DSATopK([]float32{1}, 0); got != nil {
		t.Errorf("topK=0: want nil, got %v", got)
	}
	// Determinism: repeated calls on the same input agree exactly.
	a := SelectKimiK31DSATopK([]float32{2.0, 1.0, 2.0, 0.5, 2.0}, 3)
	b := SelectKimiK31DSATopK([]float32{2.0, 1.0, 2.0, 0.5, 2.0}, 3)
	if len(a) != len(b) {
		t.Fatalf("determinism: %v vs %v", a, b)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("determinism: %v vs %v", a, b)
		}
	}
}

// TestKimiK31DSAStepClosedForm hand-derives one step on a degenerate
// geometry (NH=1, qlora=2, kvlora=2, nope=1, rope=1, vdim=1, hidden=2,
// idxHeads=1, idxDim=2, topK=1, x=[1,2]) with identity-ish weights:
//
//	inv_q  = 1/sqrt(2.5+eps)          (rmsnorm of [1,2], mean sq 2.5)
//	qr     = [inv_q, 2*inv_q];  qNope=[inv_q], qPe=[2*inv_q]
//	latent = [inv_q, 2*inv_q, 0]      (k_pe half is 0)
//	ik     = LayerNorm([1,2]) = [-0.5*inv_ln, 0.5*inv_ln],
//	       inv_ln = 1/sqrt(0.25+eps)
//	iq     = [inv_q, 2*inv_q]; iw = [1/sqrt(2)] (pre-scaled)
//	score  = iw0*relu(iq.ik) = inv_q*inv_ln/sqrt(2)  -> sel = [0]
//	qAbs   = WKb^T*qNope = [inv_q, 0]
//	attn   = kqScale*(qAbs.latent + qPe.kpe) = inv_q^2/sqrt(2); w=1
//	vProj  = WVb*latent = [inv_q]
//	final  = WO*out = [inv_q, inv_q]            (WO = [1,1])
func TestKimiK31DSAStepClosedForm(t *testing.T) {
	const eps = 1e-5
	p := &KimiK31DSAParams{
		NumHeads: 1, QLoRA: 2, KVLoRA: 2, QkNope: 1, QkRope: 1, VDim: 1, Hidden: 2,
		IdxHeads: 1, IdxDim: 2, TopK: 1,
		RMSEps: eps, IdxEps: eps,
		WQa:       []float32{1, 0, 0, 1},
		WQaNorm:   []float32{1, 1},
		WQb:       []float32{1, 0, 0, 1},
		WKvA:      []float32{1, 0, 0, 1, 0, 0},
		WKvANorm:  []float32{1, 1},
		WKb:       []float32{1, 0},
		WVb:       []float32{1, 0},
		WO:        []float32{1, 1},
		WIdxQ:     []float32{1, 0, 0, 1},
		WIdxK:     []float32{1, 0, 0, 1},
		WIdxProj:  []float32{1, 0},
		IdxKNormW: []float32{1, 1},
		IdxKNormB: []float32{0, 0},
	}
	if err := ValidateKimiK31DSAParams(p); err != nil {
		t.Fatalf("closed-form params must validate: %v", err)
	}
	x := []float32{1, 2}
	st := NewKimiK31DSALayerState(4, p.KVLoRA+p.QkRope, p.IdxDim)
	out := ForwardKimiK31DSAStep(st, p, x)

	invQ := 1.0 / math.Sqrt(2.5+eps)
	invLN := 1.0 / math.Sqrt(0.25+eps)
	wantLatent := []float64{invQ, 2 * invQ, 0}
	wantIK := []float64{-0.5 * invLN, 0.5 * invLN}
	checkClose := func(name string, got float32, want float64) {
		t.Helper()
		diff := math.Abs(float64(got) - want)
		if diff > 1e-4+1e-4*math.Abs(want) {
			t.Errorf("%s: got %v, want %v (diff %v)", name, got, want, diff)
		}
	}
	// Cache rows.
	if st.Len() != 1 {
		t.Fatalf("state len: got %d, want 1", st.Len())
	}
	for i, w := range wantLatent {
		checkClose("latent", st.Latent[i], w)
	}
	for i, w := range wantIK {
		checkClose("indexk", st.IndexK[i], w)
	}
	// Output: [inv_q, inv_q] (no gate).
	checkClose("out0", out[0], invQ)
	checkClose("out1", out[1], invQ)

	// Gated variant: WGate = [1,0] -> g = sigmoid(x0) = sigmoid(1).
	p.WGate = []float32{1, 0}
	if err := ValidateKimiK31DSAParams(p); err != nil {
		t.Fatalf("gated closed-form params must validate: %v", err)
	}
	st.Reset()
	outG := ForwardKimiK31DSAStep(st, p, x)
	sig1 := 1.0 / (1.0 + math.Exp(-1.0))
	checkClose("gated_out0", outG[0], invQ*sig1)
	checkClose("gated_out1", outG[1], invQ*sig1)
}

func TestKimiK31DSAPrefillSeqEqualsSteps(t *testing.T) {
	p := dsaTinyParams(7, true)
	if err := ValidateKimiK31DSAParams(p); err != nil {
		t.Fatalf("fixture must validate: %v", err)
	}
	rnd := rand.New(rand.NewSource(42))
	T := 4
	xSeq := make([]float32, T*p.Hidden)
	for i := range xSeq {
		xSeq[i] = rnd.Float32()*2 - 1
	}
	stA := NewKimiK31DSALayerState(T, p.KVLoRA+p.QkRope, p.IdxDim)
	seq := ForwardKimiK31DSAPrefillSeq(stA, p, xSeq, T)

	stB := NewKimiK31DSALayerState(T, p.KVLoRA+p.QkRope, p.IdxDim)
	var stepOut []float32
	for t := 0; t < T; t++ {
		stepOut = append(stepOut, ForwardKimiK31DSAStep(stB, p, xSeq[t*p.Hidden:(t+1)*p.Hidden])...)
	}
	if len(seq) != len(stepOut) {
		t.Fatalf("lengths %d vs %d", len(seq), len(stepOut))
	}
	for i := range seq {
		if seq[i] != stepOut[i] {
			t.Fatalf("prefill-seq vs steps diverged at %d: %v vs %v", i, seq[i], stepOut[i])
		}
	}
}

func TestKimiK31DSAStatePlumbing(t *testing.T) {
	st := NewKimiK31DSALayerState(4, 3, 2)
	if st.Len() != 0 || st.ByteSize() != 0 {
		t.Fatalf("fresh state: len %d bytesize %d", st.Len(), st.ByteSize())
	}
	rowL := []float32{1, 2, 3}
	rowK := []float32{4, 5}
	for i := 0; i < 3; i++ {
		st.Append(rowL, rowK)
	}
	if st.Len() != 3 {
		t.Fatalf("len: got %d, want 3", st.Len())
	}
	if st.ByteSize() != 4*(3*3+3*2) {
		t.Fatalf("bytesize: got %d, want %d", st.ByteSize(), 4*(3*3+3*2))
	}
	cl := st.Clone()
	cl.Latent[0] = 99
	cl.IndexK[1] = 99
	if st.Latent[0] == 99 || st.IndexK[1] == 99 {
		t.Fatal("clone must not alias the original")
	}
	if cl.LatentDim != 3 || cl.IdxDim != 2 {
		t.Fatalf("clone geometry: (%d,%d)", cl.LatentDim, cl.IdxDim)
	}
	st.Reset()
	if st.Len() != 0 || st.ByteSize() != 0 {
		t.Fatalf("after reset: len %d bytesize %d", st.Len(), st.ByteSize())
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("append with wrong row length must panic")
			}
		}()
		st.Append([]float32{1, 2}, rowK)
	}()
}

func TestKimiK31DSAStepWithSelection(t *testing.T) {
	// Dense selection (topK >= cells) must reproduce the internal path
	// bit-for-bit, including the shared-indexer entry point.
	p := dsaTinyParams(11, true)
	p.TopK = 5 // >= any cell count below
	if err := ValidateKimiK31DSAParams(p); err != nil {
		t.Fatalf("fixture must validate: %v", err)
	}
	rnd := rand.New(rand.NewSource(13))
	T := 3
	xSeq := make([]float32, T*p.Hidden)
	for i := range xSeq {
		xSeq[i] = rnd.Float32()*2 - 1
	}
	latentDim := p.KVLoRA + p.QkRope

	stA := NewKimiK31DSALayerState(T, latentDim, p.IdxDim)
	ref := ForwardKimiK31DSAPrefillSeq(stA, p, xSeq, T)

	stB := NewKimiK31DSALayerState(T, latentDim, p.IdxDim)
	var got []float32
	for t := 0; t < T; t++ {
		sel := make([]int, t+1)
		for i := range sel {
			sel[i] = i
		}
		got = append(got, ForwardKimiK31DSAStepWithSelection(stB, p, xSeq[t*p.Hidden:(t+1)*p.Hidden], sel)...)
	}
	for i := range ref {
		if ref[i] != got[i] {
			t.Fatalf("dense with-selection diverged at %d: %v vs %v", i, ref[i], got[i])
		}
	}

	// Guard rails.
	stC := NewKimiK31DSALayerState(2, latentDim, p.IdxDim)
	func() {
		defer func() {
			if recover() == nil {
				t.Error("out-of-range sel entry must panic")
			}
		}()
		ForwardKimiK31DSAStepWithSelection(stC, p, xSeq[:p.Hidden], []int{5})
	}()
	func() {
		defer func() {
			if recover() == nil {
				t.Error("empty sel must panic")
			}
		}()
		ForwardKimiK31DSAStepWithSelection(stC, p, xSeq[:p.Hidden], nil)
	}()
}
