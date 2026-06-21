package model

import (
	"math"
	"testing"
)

// denseCfgForMoETest is a small dense (NumExperts==0) config used by the no-op gate.
func denseCfgForMoETest() Config {
	return Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
}

// inlineDenseFFN is the verbatim open-coded dense SwiGLU FFN as it existed inline in
// blockStep before the ffnKind dispatch: g=gate(xn); u=up(xn); g=silu(g)*u;
// delta=down(g). The no-op gate proves ffnFor(dense).apply is Float32bits-identical
// to this reference, so a dense config keeps max|Δ|=0.
func inlineDenseFFN(m *Model, layer int, xn []float32) []float32 {
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	p := func(s string) string { return layerName(layer, s) }
	g := matRows(m.tensor(p("mlp.gate_proj.weight")), xn, I, H)
	u := matRows(m.tensor(p("mlp.up_proj.weight")), xn, I, H)
	for i := 0; i < I; i++ {
		g[i] = silu(g[i]) * u[i]
	}
	return matRows(m.tensor(p("mlp.down_proj.weight")), g, H, I)
}

// TestMoEDenseNoOpIdentical is the load-bearing gate: routing the dense FFN through
// the new ffnKind dispatch must be Float32bits-identical to the inline SwiGLU, so a
// Llama/dense model is bit-for-bit unchanged (max|Δ|=0). Asserted over many random
// post-norm hidden vectors per layer.
func TestMoEDenseNoOpIdentical(t *testing.T) {
	cfg := denseCfgForMoETest()
	if cfg.IsMoE() {
		t.Fatal("dense config must report IsMoE()==false")
	}
	m := NewSynthetic(cfg)
	ffn := ffnFor(cfg)
	if _, ok := ffn.(denseSwiGLU); !ok {
		t.Fatalf("dense config selected %T, want denseSwiGLU", ffn)
	}

	// Deterministic pseudo-random xn vectors (a separate LCG so the inputs are
	// arbitrary, not the synthetic weights themselves).
	seed := uint64(0xDEADBEEFCAFEBABE)
	nextF := func() float32 {
		seed = seed*6364136223846793005 + 1442695040888963407
		return float32(seed>>40)/float32(1<<24)*2 - 1
	}
	for l := 0; l < cfg.NumLayers; l++ {
		for trial := 0; trial < 8; trial++ {
			xn := make([]float32, cfg.HiddenSize)
			for i := range xn {
				xn[i] = nextF()
			}
			want := inlineDenseFFN(m, l, xn)
			got := ffn.apply(m, l, xn, f32Kernel{m})
			assertFloat32BitsEqual(t, "dense FFN delta", want, got)
		}
	}
}

func TestDenseActivationMLPWithBias(t *testing.T) {
	cfg := Config{HiddenSize: 2, IntermediateSize: 3, DenseMLP: true, ActGeluErf: true}
	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{Name: layerName(0, "mlp.gate_proj.weight"), Shape: []int{3, 2}, Data: []float32{
			0.5, -0.25,
			-0.75, 0.125,
			0.25, 0.5,
		}},
		{Name: layerName(0, "mlp.gate_proj.bias"), Shape: []int{3}, Data: []float32{0.1, -0.2, 0.3}},
		{Name: layerName(0, "mlp.down_proj.weight"), Shape: []int{2, 3}, Data: []float32{
			0.2, -0.4, 0.6,
			-0.3, 0.5, 0.7,
		}},
		{Name: layerName(0, "mlp.down_proj.bias"), Shape: []int{2}, Data: []float32{0.01, -0.02}},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	xn := []float32{0.75, -0.5}
	got := ffnFor(cfg).apply(m, 0, xn, f32Kernel{m})

	h0 := geluErf(0.5*xn[0] + -0.25*xn[1] + 0.1)
	h1 := geluErf(-0.75*xn[0] + 0.125*xn[1] - 0.2)
	h2 := geluErf(0.25*xn[0] + 0.5*xn[1] + 0.3)
	want := []float32{
		0.2*h0 + -0.4*h1 + 0.6*h2 + 0.01,
		-0.3*h0 + 0.5*h1 + 0.7*h2 - 0.02,
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("dense activation mlp[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

// moeCfgForTest is a synthetic 4-expert top-2 MoE config (no real Mixtral download).
func moeCfgForTest() Config {
	return Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
		NumExperts:        4,
		NumExpertsPerTok:  2,
		NormTopKProb:      true,
	}
}

// TestMoEWiring drives the synthetic 2-layer MoE end to end through Prefill/Step and
// checks the router -> top-k -> per-expert -> weighted-sum dataflow is wired: the FFN
// selects exactly NumExpertsPerTok experts and produces finite logits.
func TestMoEWiring(t *testing.T) {
	cfg := moeCfgForTest()
	if !cfg.IsMoE() {
		t.Fatal("MoE config must report IsMoE()==true")
	}
	m := NewSyntheticMoE(cfg)
	if _, ok := ffnFor(cfg).(moeFFN); !ok {
		t.Fatalf("MoE config selected %T, want moeFFN", ffnFor(cfg))
	}

	// Router returns exactly K picks of distinct in-range experts, weights in [0,1]
	// summing to 1 under norm_topk_prob.
	xn := make([]float32, cfg.HiddenSize)
	for i := range xn {
		xn[i] = float32(i%5)*0.3 - 0.6
	}
	picks := route(m, 0, xn, f32Kernel{m})
	if len(picks) != cfg.NumExpertsPerTok {
		t.Fatalf("router returned %d picks, want top-k=%d", len(picks), cfg.NumExpertsPerTok)
	}
	seen := map[int]bool{}
	var wsum float32
	for _, pk := range picks {
		if pk.expert < 0 || pk.expert >= cfg.NumExperts {
			t.Fatalf("picked expert %d out of range [0,%d)", pk.expert, cfg.NumExperts)
		}
		if seen[pk.expert] {
			t.Fatalf("expert %d picked twice", pk.expert)
		}
		seen[pk.expert] = true
		wsum += pk.weight
	}
	if math.Abs(float64(wsum)-1) > 1e-5 {
		t.Fatalf("norm_topk_prob gate weights sum to %v, want 1", wsum)
	}

	// End to end: Prefill + a couple of decode steps yield finite logits over vocab.
	s := m.NewSession()
	logits := s.Prefill([]int{3, 17, 5, 23, 41})
	if len(logits) != cfg.VocabSize {
		t.Fatalf("prefill logits len = %d, want vocab %d", len(logits), cfg.VocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("prefill logit[%d] not finite: %v", i, v)
		}
	}
	for _, id := range []int{11, 29} {
		logits = s.Step(id)
		for i, v := range logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("decode logit[%d] not finite: %v", i, v)
			}
		}
	}
}

func TestMoEMixedDenseAndSparseLayerDispatch(t *testing.T) {
	const H, I, E, K = 2, 2, 2, 1
	cfg := Config{
		HiddenSize:       H,
		NumLayers:        2,
		IntermediateSize: I,
		NumExperts:       E,
		NumExpertsPerTok: K,
		NormTopKProb:     true,
	}
	tensors := []NamedTensorF32{
		{Name: layerName(0, "mlp.gate_proj.weight"), Shape: []int{I, H}, Data: []float32{
			0.5, -0.25,
			0.125, 0.75,
		}},
		{Name: layerName(0, "mlp.up_proj.weight"), Shape: []int{I, H}, Data: []float32{
			0.25, 0.5,
			-0.5, 0.125,
		}},
		{Name: layerName(0, "mlp.down_proj.weight"), Shape: []int{H, I}, Data: []float32{
			0.75, -0.25,
			0.5, 0.125,
		}},
		{Name: routerName(1), Shape: []int{E, H}, Data: []float32{
			1, 0,
			0, 1,
		}},
	}
	for e := 0; e < E; e++ {
		base := float32(e + 1)
		tensors = append(tensors,
			NamedTensorF32{Name: expertName(1, e, "gate_proj.weight"), Shape: []int{I, H}, Data: []float32{base, 0, 0, base}},
			NamedTensorF32{Name: expertName(1, e, "up_proj.weight"), Shape: []int{I, H}, Data: []float32{0.5 * base, 0, 0, 0.5 * base}},
			NamedTensorF32{Name: expertName(1, e, "down_proj.weight"), Shape: []int{H, I}, Data: []float32{0.25 * base, 0, 0, 0.25 * base}},
		)
	}
	m, err := NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	if _, ok := m.ffnForLayer(0).(denseSwiGLU); !ok {
		t.Fatalf("layer 0 selected %T, want denseSwiGLU", m.ffnForLayer(0))
	}
	if _, ok := m.ffnForLayer(1).(moeFFN); !ok {
		t.Fatalf("layer 1 selected %T, want moeFFN", m.ffnForLayer(1))
	}
	xn := []float32{0.75, -0.5}
	assertFloat32BitsEqual(t, "mixed layer 0 dense delta",
		denseSwiGLU{}.apply(m, 0, xn, f32Kernel{m}),
		m.ffnForLayer(0).apply(m, 0, xn, f32Kernel{m}))
	assertFloat32BitsEqual(t, "mixed layer 1 moe delta",
		moeFFN{}.apply(m, 1, xn, f32Kernel{m}),
		m.ffnForLayer(1).apply(m, 1, xn, f32Kernel{m}))
}

// TestMoERoutingHandComputed pins the router and weighted sum against a hand-computed
// reference on a tiny single-position MoE FFN: H=2, I=2, E=4, top-2, explicit weights.
// It asserts (a) the two experts chosen are exactly the top-2 post-softmax experts and
// (b) the gate-weighted expert sum equals the reference to f32 tolerance.
func TestMoERoutingHandComputed(t *testing.T) {
	const H, I, E, K = 2, 2, 4, 2
	cfg := Config{
		HiddenSize:       H,
		NumLayers:        1,
		NumHeads:         1,
		NumKVHeads:       1,
		HeadDim:          2,
		IntermediateSize: I,
		VocabSize:        4,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		NumExperts:       E,
		NumExpertsPerTok: K,
		NormTopKProb:     true,
		EOSTokenID:       -1,
	}

	// Router [E,H]: rows chosen so softmax(router @ xn) has a clear top-2.
	// xn = [1, 1]. router rows: e0=[2,0]->2, e1=[0,3]->3, e2=[1,1]->2, e3=[-1,-1]->-2.
	// logits = [2,3,2,-2]; top-2 by prob = e1 (3) then a tie between e0 and e2 (both 2),
	// broken to the lower index e0.
	router := []float32{
		2, 0, // e0
		0, 3, // e1
		1, 1, // e2
		-1, -1, // e3
	}
	xn := []float32{1, 1}

	// Per-expert SwiGLU weights. gate/up are [I,H], down is [H,I]. Keep them distinct
	// per expert so the weighted sum is sensitive to which experts are picked.
	tensors := []NamedTensorF32{
		{Name: "model.embed_tokens.weight", Shape: []int{cfg.VocabSize, H}, Data: make([]float32, cfg.VocabSize*H)},
		{Name: layerName(0, "input_layernorm.weight"), Shape: []int{H}, Data: []float32{1, 1}},
		{Name: layerName(0, "self_attn.q_proj.weight"), Shape: []int{H, H}, Data: []float32{1, 0, 0, 1}},
		{Name: layerName(0, "self_attn.k_proj.weight"), Shape: []int{H, H}, Data: []float32{1, 0, 0, 1}},
		{Name: layerName(0, "self_attn.v_proj.weight"), Shape: []int{H, H}, Data: []float32{1, 0, 0, 1}},
		{Name: layerName(0, "self_attn.o_proj.weight"), Shape: []int{H, H}, Data: []float32{1, 0, 0, 1}},
		{Name: layerName(0, "post_attention_layernorm.weight"), Shape: []int{H}, Data: []float32{1, 1}},
		{Name: routerName(0), Shape: []int{E, H}, Data: router},
		{Name: "model.norm.weight", Shape: []int{H}, Data: []float32{1, 1}},
	}
	for e := 0; e < E; e++ {
		base := float32(e + 1)
		tensors = append(tensors,
			NamedTensorF32{Name: expertName(0, e, "gate_proj.weight"), Shape: []int{I, H}, Data: []float32{0.1 * base, 0, 0, 0.1 * base}},
			NamedTensorF32{Name: expertName(0, e, "up_proj.weight"), Shape: []int{I, H}, Data: []float32{0.2 * base, 0, 0, 0.2 * base}},
			NamedTensorF32{Name: expertName(0, e, "down_proj.weight"), Shape: []int{H, I}, Data: []float32{0.3 * base, 0, 0, 0.3 * base}},
		)
	}
	m, err := NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatalf("build model: %v", err)
	}

	// --- hand-computed reference ---------------------------------------------
	logits := []float64{2, 3, 2, -2}
	mx := 3.0
	var z float64
	exps := make([]float64, E)
	for i, l := range logits {
		exps[i] = math.Exp(l - mx)
		z += exps[i]
	}
	probs := make([]float64, E)
	for i := range probs {
		probs[i] = exps[i] / z
	}
	// top-2: e1 highest; e0 and e2 tie at 2, lower index e0 wins.
	wantExperts := []int{1, 0}
	sumSel := probs[1] + probs[0]
	wExpert := map[int]float64{1: probs[1] / sumSel, 0: probs[0] / sumSel}

	// reference expert SwiGLU on xn=[1,1]: gate=up=down are diagonal*c, so for input
	// [1,1]: g=[0.1b,0.1b], u=[0.2b,0.2b], silu(g)*u, then down scales by 0.3b.
	refExpert := func(e int) []float64 {
		b := float64(e + 1)
		out := make([]float64, H)
		for i := 0; i < I; i++ {
			gi := 0.1 * b
			ui := 0.2 * b
			s := float64(silu(float32(gi))) * ui
			out[i] = 0.3 * b * s
		}
		return out
	}
	wantDelta := make([]float64, H)
	for _, e := range wantExperts {
		ro := refExpert(e)
		for i := 0; i < H; i++ {
			wantDelta[i] += wExpert[e] * ro[i]
		}
	}

	// --- implementation ------------------------------------------------------
	picks := route(m, 0, xn, f32Kernel{m})
	if len(picks) != K {
		t.Fatalf("got %d picks, want %d", len(picks), K)
	}
	if picks[0].expert != wantExperts[0] || picks[1].expert != wantExperts[1] {
		t.Fatalf("router picked experts %d,%d; want %d,%d (top-2 post-softmax, tie to lower index)",
			picks[0].expert, picks[1].expert, wantExperts[0], wantExperts[1])
	}
	for _, pk := range picks {
		if math.Abs(float64(pk.weight)-wExpert[pk.expert]) > 1e-6 {
			t.Fatalf("expert %d gate weight = %v, want %v", pk.expert, pk.weight, wExpert[pk.expert])
		}
	}
	gotDelta := moeFFN{}.apply(m, 0, xn, f32Kernel{m})
	for i := 0; i < H; i++ {
		if math.Abs(float64(gotDelta[i])-wantDelta[i]) > 1e-5 {
			t.Fatalf("MoE weighted-sum delta[%d] = %v, want %v", i, gotDelta[i], wantDelta[i])
		}
	}
}

func TestGPTOSSRouterUsesTopKSoftmaxAndBias(t *testing.T) {
	cfg := Config{
		HiddenSize:       2,
		IntermediateSize: 2,
		NumLayers:        1,
		NumExperts:       3,
		NumExpertsPerTok: 2,
		ModelType:        "gpt_oss",
	}
	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{Name: routerName(0), Shape: []int{3, 2}, Data: []float32{
			0, 0,
			0, 0,
			0, 0,
		}},
		{Name: routerBiasName(0), Shape: []int{3}, Data: []float32{1, 2, 0}},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	picks := route(m, 0, []float32{0, 0}, f32Kernel{m})
	if len(picks) != 2 || picks[0].expert != 1 || picks[1].expert != 0 {
		t.Fatalf("gpt-oss router picks = %+v, want experts [1 0]", picks)
	}
	want0 := float32(math.Exp(2) / (math.Exp(2) + math.Exp(1)))
	want1 := float32(math.Exp(1) / (math.Exp(2) + math.Exp(1)))
	if math.Abs(float64(picks[0].weight-want0)) > 1e-6 || math.Abs(float64(picks[1].weight-want1)) > 1e-6 {
		t.Fatalf("gpt-oss router weights = %v,%v want %v,%v", picks[0].weight, picks[1].weight, want0, want1)
	}
}

func TestGPTOSSExpertActivationWithBias(t *testing.T) {
	cfg := Config{
		HiddenSize:       1,
		IntermediateSize: 2,
		NumLayers:        1,
		NumExperts:       1,
		NumExpertsPerTok: 1,
		ModelType:        "gpt_oss",
	}
	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{Name: routerName(0), Shape: []int{1, 1}, Data: []float32{0}},
		{Name: expertName(0, 0, "gate_proj.weight"), Shape: []int{2, 1}, Data: []float32{5, -1}},
		{Name: expertName(0, 0, "gate_proj.bias"), Shape: []int{2}, Data: []float32{0, 0}},
		{Name: expertName(0, 0, "up_proj.weight"), Shape: []int{2, 1}, Data: []float32{4, -4}},
		{Name: expertName(0, 0, "up_proj.bias"), Shape: []int{2}, Data: []float32{0, 0}},
		{Name: expertName(0, 0, "down_proj.weight"), Shape: []int{1, 2}, Data: []float32{0.5, -0.25}},
		{Name: expertName(0, 0, "down_proj.bias"), Shape: []int{1}, Data: []float32{0.3}},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	got := moeFFN{}.apply(m, 0, []float32{2}, f32Kernel{m})
	g0 := float64(7 * sigmoid(1.702*7) * (7 + 1))
	g1 := float64((-2) * sigmoid(1.702*(-2)) * (-7 + 1))
	want := float32(0.5*g0 - 0.25*g1 + 0.3)
	if len(got) != 1 || math.Abs(float64(got[0]-want)) > 1e-5 {
		t.Fatalf("gpt-oss expert delta = %v want [%v]", got, want)
	}
}

// TestMoEKVOrthogonal proves the MoE swap is KV-orthogonal: the FFN form does not
// touch the KV cache. A single-layer model isolates the property — the one layer's
// K/V/Kraw are computed from the embedding (pre-FFN), so with IDENTICAL attention
// weights a dense-FFN model and an MoE-FFN model must produce byte-identical
// K/V/Kraw, regardless of FFN, because every cache append lives in the attention
// section and the FFN writes only the residual. (With >1 layer the FFN delta of an
// earlier layer legitimately changes the residual feeding the next layer's
// attention, so the bytes differ for a correct reason — that is the hidden state
// changing, not the cache machinery; the single-layer form is the clean witness.)
func TestMoEKVOrthogonal(t *testing.T) {
	cfg := moeCfgForTest()
	cfg.NumLayers = 1 // isolate: no downstream attention to inherit the FFN delta
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV, I, V, E := cfg.NumHeads, cfg.NumKVHeads, cfg.IntermediateSize, cfg.VocabSize, cfg.NumExperts

	// Deterministic shared weight generator.
	seed := uint64(0x1234567890ABCDEF)
	fill := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			seed = seed*6364136223846793005 + 1442695040888963407
			out[i] = (float32(seed>>40)/float32(1<<24)*2 - 1) * 0.1
		}
		return out
	}
	ones := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = 1
		}
		return out
	}

	// The attention + embedding tensors shared by BOTH models, byte-for-byte.
	shared := []NamedTensorF32{
		{Name: "model.embed_tokens.weight", Shape: []int{V, H}, Data: fill(V * H)},
		{Name: "model.norm.weight", Shape: []int{H}, Data: ones(H)},
	}
	for l := 0; l < cfg.NumLayers; l++ {
		shared = append(shared,
			NamedTensorF32{Name: layerName(l, "input_layernorm.weight"), Shape: []int{H}, Data: ones(H)},
			NamedTensorF32{Name: layerName(l, "self_attn.q_proj.weight"), Shape: []int{nH * hd, H}, Data: fill(nH * hd * H)},
			NamedTensorF32{Name: layerName(l, "self_attn.k_proj.weight"), Shape: []int{nKV * hd, H}, Data: fill(nKV * hd * H)},
			NamedTensorF32{Name: layerName(l, "self_attn.v_proj.weight"), Shape: []int{nKV * hd, H}, Data: fill(nKV * hd * H)},
			NamedTensorF32{Name: layerName(l, "self_attn.o_proj.weight"), Shape: []int{H, nH * hd}, Data: fill(H * nH * hd)},
			NamedTensorF32{Name: layerName(l, "post_attention_layernorm.weight"), Shape: []int{H}, Data: ones(H)},
		)
	}

	clone := func(in []NamedTensorF32) []NamedTensorF32 {
		out := make([]NamedTensorF32, len(in))
		copy(out, in)
		return out
	}

	// Dense model: shared attention + one FFN triple per layer.
	denseCfg := cfg
	denseCfg.NumExperts = 0
	denseCfg.NumExpertsPerTok = 0
	denseCfg.NormTopKProb = false
	denseTensors := clone(shared)
	for l := 0; l < cfg.NumLayers; l++ {
		denseTensors = append(denseTensors,
			NamedTensorF32{Name: layerName(l, "mlp.gate_proj.weight"), Shape: []int{I, H}, Data: fill(I * H)},
			NamedTensorF32{Name: layerName(l, "mlp.up_proj.weight"), Shape: []int{I, H}, Data: fill(I * H)},
			NamedTensorF32{Name: layerName(l, "mlp.down_proj.weight"), Shape: []int{H, I}, Data: fill(H * I)},
		)
	}
	dense, err := NewFromF32Tensors(denseCfg, denseTensors)
	if err != nil {
		t.Fatalf("build dense: %v", err)
	}

	// MoE model: SAME shared attention tensors + a router and experts per layer.
	moeTensors := clone(shared)
	for l := 0; l < cfg.NumLayers; l++ {
		moeTensors = append(moeTensors, NamedTensorF32{Name: routerName(l), Shape: []int{E, H}, Data: fill(E * H)})
		for e := 0; e < E; e++ {
			moeTensors = append(moeTensors,
				NamedTensorF32{Name: expertName(l, e, "gate_proj.weight"), Shape: []int{I, H}, Data: fill(I * H)},
				NamedTensorF32{Name: expertName(l, e, "up_proj.weight"), Shape: []int{I, H}, Data: fill(I * H)},
				NamedTensorF32{Name: expertName(l, e, "down_proj.weight"), Shape: []int{H, I}, Data: fill(H * I)},
			)
		}
	}
	moe, err := NewFromF32Tensors(cfg, moeTensors)
	if err != nil {
		t.Fatalf("build moe: %v", err)
	}

	prompt := []int{3, 17, 5, 23, 41, 2, 19}
	ds := dense.NewSession()
	ms := moe.NewSession()
	ds.Prefill(prompt)
	ms.Prefill(prompt)
	assertKVCacheBitsEqual(t, "dense-vs-MoE prefill", ds.Cache, ms.Cache)

	for _, id := range []int{11, 29, 7} {
		ds.Step(id)
		ms.Step(id)
	}
	assertKVCacheBitsEqual(t, "dense-vs-MoE decode", ds.Cache, ms.Cache)
}
