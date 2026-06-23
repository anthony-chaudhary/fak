package model

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// minimax_m3_test.go — host-tractable witnesses for the wired MiniMax-M3 MSA forward.
//
// These prove the STRUCTURE of the cacheless forward on a synthetic MiniMax-M3-shaped
// checkpoint (no HF download, no real weights): the family dispatches through
// layerMiniMax; full_attention layers run dense GQA while minimax_m3_sparse layers run
// the lightning-indexer block-sparse path; the forward is finite end-to-end; the sparse
// path reduces EXACTLY to dense causal GQA when the block budget covers every causal
// block (sparsity is a strict subset, not a different computation); the sparse path is
// non-vacuous (a small budget genuinely changes the output); and the SwiGLU-OAI gate
// matches its closed form. The NUMERIC parity against a real checkpoint is a separate
// (GPU/artifact-node) gate — see TestOptionalMiniMaxM3Oracle* in oracle_test.go.

// newSyntheticMiniMaxM3 builds an in-memory MiniMax-M3 Model: standard GQA q/k/v/o with
// per-head q_norm/k_norm, a lightning indexer (self_attn.indexer.{q_proj,k_proj,q_norm,
// k_norm}) on every minimax_m3_sparse layer, and a SwiGLU-OAI MoE FFN (router + score
// correction bias + per-expert gate/up/down + an always-on shared expert) per layer. Like
// the other synthetic builders the logits are meaningless; what is faithful is the tensor
// layout and the code path it drives. Norm weights init to 1.0, the router correction bias
// to 0.0, all matmul weights to a fixed-seed LCG in [-scale, scale].
func newSyntheticMiniMaxM3(cfg Config) *Model {
	if !cfg.isMiniMaxSparseAttn() {
		panic("model: newSyntheticMiniMaxM3 needs a minimax_m3 sparse-attention cfg")
	}
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	H, I, V, E := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize, cfg.NumExperts
	nIdx, idxDim := cfg.IndexNHeads, cfg.IndexHeadDim
	shared := cfg.SharedIntermediateSize
	if shared == 0 {
		shared = I
	}

	type ts struct {
		name  string
		shape []int
	}
	var tensors []ts
	add := func(name string, shape ...int) { tensors = append(tensors, ts{name, shape}) }
	add("model.embed_tokens.weight", V, H)
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		ap := p + "self_attn."
		add(p+"input_layernorm.weight", H)
		add(ap+"q_proj.weight", nH*hd, H)
		add(ap+"k_proj.weight", nKV*hd, H)
		add(ap+"v_proj.weight", nKV*hd, H)
		add(ap+"o_proj.weight", H, nH*hd)
		add(ap+"q_norm.weight", hd)
		add(ap+"k_norm.weight", hd)
		if cfg.isMSALayer(l) {
			add(ap+"indexer.q_proj.weight", nIdx*idxDim, H)
			add(ap+"indexer.k_proj.weight", idxDim, H)
			add(ap+"indexer.q_norm.weight", idxDim)
			add(ap+"indexer.k_norm.weight", idxDim)
		}
		add(p+"post_attention_layernorm.weight", H)
		add(routerName(l), E, H)
		add(layerName(l, "mlp.gate.e_score_correction_bias"), E)
		for e := 0; e < E; e++ {
			add(expertName(l, e, "gate_proj.weight"), I, H)
			add(expertName(l, e, "up_proj.weight"), I, H)
			add(expertName(l, e, "down_proj.weight"), H, I)
		}
		add(layerName(l, "mlp.shared_experts.gate_proj.weight"), shared, H)
		add(layerName(l, "mlp.shared_experts.up_proj.weight"), shared, H)
		add(layerName(l, "mlp.shared_experts.down_proj.weight"), H, shared)
	}
	add("model.norm.weight", H)

	man := make(map[string]tensorMeta, len(tensors))
	off := 0
	for _, t := range tensors {
		n := 1
		for _, d := range t.shape {
			n *= d
		}
		man[t.name] = tensorMeta{Dtype: "F32", Shape: t.shape, Offset: off, Nbytes: n * 4}
		off += n * 4
	}

	raw := make([]byte, off)
	seed := uint64(0x9E3779B97F4A7C15)
	next := func() float32 {
		seed = seed*6364136223846793005 + 1442695040888963407
		u := float32(seed>>40) / float32(1<<24)
		return u*2 - 1
	}
	for _, t := range tensors {
		mm := man[t.name]
		n := mm.Nbytes / 4
		isOne := strings.HasSuffix(t.name, "layernorm.weight") ||
			strings.HasSuffix(t.name, "q_norm.weight") ||
			strings.HasSuffix(t.name, "k_norm.weight") ||
			t.name == "model.norm.weight"
		isZero := strings.HasSuffix(t.name, "e_score_correction_bias")
		scale := float32(0.1)
		if strings.Contains(t.name, "embed_tokens") {
			scale = 0.2
		}
		for i := 0; i < n; i++ {
			var f float32
			switch {
			case isOne:
				f = 1.0
			case isZero:
				f = 0.0
			default:
				f = next() * scale
			}
			binary.LittleEndian.PutUint32(raw[mm.Offset+i*4:], math.Float32bits(f))
		}
	}
	cfg.TieWordEmbeddings = true
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

func miniMaxM3TestConfig(layerTypes []string) Config {
	return Config{
		HiddenSize:             32,
		NumLayers:              len(layerTypes),
		NumHeads:               8,
		NumKVHeads:             2,
		HeadDim:                4,
		IntermediateSize:       16,
		SharedIntermediateSize: 8,
		VocabSize:              48,
		RMSNormEps:             1e-5,
		RopeTheta:              10000,
		PartialRotaryFactor:    0.5,
		QKNorm:                 true,
		NumExperts:             4,
		NumExpertsPerTok:       2,
		NormTopKProb:           true,
		RoutedScalingFactor:    1.0,
		NSharedExperts:         1,
		SwigluAlpha:            1.702,
		SwigluLimit:            7.0,
		IndexNHeads:            2, // == NumKVHeads: one index head per GQA group
		IndexHeadDim:           4,
		IndexBlockSize:         2,
		IndexTopKBlocks:        1,
		IndexLocalBlocks:       1,
		EOSTokenID:             -1,
		ModelType:              "minimax_m3",
		Architectures:          []string{"MiniMaxM3ForCausalLM"},
		LayerTypes:             layerTypes,
	}
}

func TestMiniMaxM3MSAForwardRunsThroughNativeKernel(t *testing.T) {
	cfg := miniMaxM3TestConfig([]string{"full_attention", "minimax_m3_sparse", "minimax_m3_sparse"})
	m := newSyntheticMiniMaxM3(cfg)
	if !m.Cfg.isMiniMax() || !m.Cfg.isMiniMaxSparseAttn() {
		t.Fatalf("synthetic config not recognized as minimax_m3 sparse (key=%q)", m.Cfg.archFamilyKey())
	}
	if m.Cfg.isMSALayer(0) || !m.Cfg.isMSALayer(1) || !m.Cfg.isMSALayer(2) {
		t.Fatalf("isMSALayer split = %v/%v/%v, want false/true/true",
			m.Cfg.isMSALayer(0), m.Cfg.isMSALayer(1), m.Cfg.isMSALayer(2))
	}
	if _, ok := m.ffnForLayer(1).(minimaxMoeFFN); !ok {
		t.Fatalf("layer 1 FFN = %T, want minimaxMoeFFN", m.ffnForLayer(1))
	}

	// A prompt longer than IndexBlockSize*IndexTopKBlocks so the sparse selection
	// actually drops some causal keys on the sparse layers.
	prompt := []int{3, 17, 5, 23, 11, 7, 41, 2}
	act := m.Forward(prompt)
	if len(act.Hidden) != cfg.NumLayers+1 {
		t.Fatalf("hidden len = %d, want %d", len(act.Hidden), cfg.NumLayers+1)
	}
	if len(act.Logits) != len(prompt) {
		t.Fatalf("logits len = %d, want %d", len(act.Logits), len(prompt))
	}
	for l, h := range act.Hidden {
		for i, v := range h {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("hidden[%d][%d] not finite: %v", l, i, v)
			}
		}
	}
	for pos, row := range act.Logits {
		if len(row) != cfg.VocabSize {
			t.Fatalf("logits[%d] len = %d, want vocab %d", pos, len(row), cfg.VocabSize)
		}
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("logits[%d][%d] not finite: %v", pos, i, v)
			}
		}
	}
}

// TestMiniMaxM3MSASupersetEqualsDenseForward proves MSA is a strict subset of dense
// causal GQA: when the block budget covers every causal block, the sparse layers admit
// every causal key, so the whole forward must be byte-identical to the same model run
// with every layer forced to full_attention (which uses the dense GQA path). The two
// runs share the SAME weights; only the attention key set differs, and with a saturating
// budget it does not differ at all.
func TestMiniMaxM3MSASupersetEqualsDenseForward(t *testing.T) {
	cfg := miniMaxM3TestConfig([]string{"minimax_m3_sparse", "minimax_m3_sparse", "minimax_m3_sparse"})
	cfg.IndexTopKBlocks = 64 // >> number of blocks: every causal block is selected
	cfg.IndexLocalBlocks = 64
	m := newSyntheticMiniMaxM3(cfg)

	prompt := []int{3, 17, 5, 23, 11, 7, 41, 2}
	sparse := m.Forward(prompt)

	// Re-run the identical weights as a pure dense model.
	full := append([]string(nil), cfg.LayerTypes...)
	for i := range full {
		full[i] = "full_attention"
	}
	m.Cfg.LayerTypes = full
	for l := 0; l < m.Cfg.NumLayers; l++ {
		if m.Cfg.isMSALayer(l) {
			t.Fatalf("layer %d still MSA after forcing full_attention", l)
		}
	}
	dense := m.Forward(prompt)

	for pos := range sparse.Logits {
		if d, at := maxAbsDiff(sparse.Logits[pos], dense.Logits[pos]); d != 0 {
			t.Fatalf("saturating-budget MSA != dense GQA at pos %d: max|Δ|=%.3e at %d", pos, d, at)
		}
	}
}

// TestMiniMaxM3MSASparsityIsNonVacuous guards against the indexer/selection being a
// silent no-op: a tight block budget (top-1 block + 1 local block) MUST change the
// forward output versus the saturating-budget (dense-equivalent) run on the same weights.
func TestMiniMaxM3MSASparsityIsNonVacuous(t *testing.T) {
	layers := []string{"minimax_m3_sparse", "minimax_m3_sparse", "minimax_m3_sparse"}
	tight := miniMaxM3TestConfig(layers)
	tight.IndexBlockSize = 1
	tight.IndexTopKBlocks = 1
	tight.IndexLocalBlocks = 1
	mTight := newSyntheticMiniMaxM3(tight)

	wide := tight
	wide.IndexTopKBlocks = 64
	wide.IndexLocalBlocks = 64
	mWide := &Model{Cfg: wide, manifest: mTight.manifest, raw: mTight.raw} // identical weights

	prompt := []int{3, 17, 5, 23, 11, 7, 41, 2}
	a := mTight.Forward(prompt)
	b := mWide.Forward(prompt)
	last := len(prompt) - 1
	if d, _ := maxAbsDiff(a.Logits[last], b.Logits[last]); d == 0 {
		t.Fatalf("tight vs saturating MSA budget produced identical logits — selection is vacuous")
	}
}

// TestMiniMaxM3DenseLayerUsesOAIMLPAtDenseWidth witnesses the first-k DENSE-layer FFN
// path (#495 scope item 2). The real 60-layer MiniMax-M3 config carries
// moe_layer_freq=[0,0,0,1,...] — its first 3 layers are dense OAI MLPs at
// dense_intermediate_size (12288), NOT routed MoE. This builds a one-layer MiniMax model
// with ONLY a dense mlp.{gate,up,down}_proj (no router, no experts) at DenseIntermediateSize
// != IntermediateSize and proves: (a) ffnForLayer dispatches it to minimaxDenseFFN (not the
// generic plain-SiLU denseSwiGLU, which would also read the wrong width); (b) the output
// equals the SwiGLU-OAI closed form at the dense width to f32 bits; and (c) it genuinely
// differs from the plain-SiLU SwiGLU the generic path would have applied.
func TestMiniMaxM3DenseLayerUsesOAIMLPAtDenseWidth(t *testing.T) {
	const H, denseI, moeI = 2, 3, 2 // dense width != routed-expert width on purpose
	cfg := Config{
		HiddenSize:            H,
		NumLayers:             1,
		IntermediateSize:      moeI, // routed-expert width — must NOT be used by the dense layer
		DenseIntermediateSize: denseI,
		VocabSize:             8,
		RMSNormEps:            1e-5,
		NumExperts:            2, // IsMoE() so ffnForLayer enters the MiniMax hybrid branch
		NumExpertsPerTok:      2,
		SwigluAlpha:           1.702,
		SwigluLimit:           7.0,
		ModelType:             "minimax_m3",
		Architectures:         []string{"MiniMaxM3ForCausalLM"},
	}
	tensors := []NamedTensorF32{
		{Name: layerName(0, "mlp.gate_proj.weight"), Shape: []int{denseI, H}, Data: []float32{
			0.5, -0.25,
			0.125, 0.75,
			-0.4, 0.2,
		}},
		{Name: layerName(0, "mlp.up_proj.weight"), Shape: []int{denseI, H}, Data: []float32{
			0.25, 0.5,
			-0.5, 0.125,
			0.3, -0.15,
		}},
		{Name: layerName(0, "mlp.down_proj.weight"), Shape: []int{H, denseI}, Data: []float32{
			0.75, -0.25, 0.5,
			0.5, 0.125, -0.3,
		}},
	}
	m, err := NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	if _, ok := m.ffnForLayer(0).(minimaxDenseFFN); !ok {
		t.Fatalf("dense MiniMax layer FFN = %T, want minimaxDenseFFN", m.ffnForLayer(0))
	}

	mat := f32Kernel{m}
	xn := []float32{0.75, -0.5}
	g := mat.mul(layerName(0, "mlp.gate_proj.weight"), xn, denseI, H)
	u := mat.mul(layerName(0, "mlp.up_proj.weight"), xn, denseI, H)

	// (b) SwiGLU-OAI reference at the dense width.
	gOAI := append([]float32(nil), g...)
	uOAI := append([]float32(nil), u...)
	swigluOAIInPlace(gOAI, uOAI, cfg)
	wantOAI := mat.mul(layerName(0, "mlp.down_proj.weight"), mat.prep(gOAI), H, denseI)
	assertFloat32BitsEqual(t, "dense OAI MLP delta",
		wantOAI, minimaxDenseFFN{}.apply(m, 0, xn, mat))

	// (c) plain-SiLU SwiGLU reference (silu(g)*u) — what the generic denseSwiGLU would
	// have computed — must differ, proving the OAI gate is genuinely wired.
	gSiLU := make([]float32, denseI)
	for i := range gSiLU {
		gSiLU[i] = float32(float64(g[i])/(1+math.Exp(float64(-g[i])))) * u[i]
	}
	wantSiLU := mat.mul(layerName(0, "mlp.down_proj.weight"), mat.prep(gSiLU), H, denseI)
	if d, _ := maxAbsDiff(wantOAI, wantSiLU); d == 0 {
		t.Fatalf("dense OAI MLP is bit-identical to plain-SiLU SwiGLU — OAI gate is vacuous")
	}
}

// TestMiniMaxM3SwigluOAIActivation pins the SwiGLU-OAI gate to its closed form, including
// the gate/up clamps at ±swiglu_limit.
func TestMiniMaxM3SwigluOAIActivation(t *testing.T) {
	cfg := Config{SwigluAlpha: 1.702, SwigluLimit: 7.0}
	// One value inside the clamp, one beyond it (clamped).
	g := []float32{2.0, 9.0}
	u := []float32{1.5, 20.0}
	swigluOAIInPlace(g, u, cfg)

	oai := func(gate, up, alpha, limit float32) float32 {
		if gate > limit {
			gate = limit
		}
		if up > limit {
			up = limit
		} else if up < -limit {
			up = -limit
		}
		glu := gate * float32(1/(1+math.Exp(float64(-gate*alpha))))
		return (up + 1) * glu
	}
	for i, want := range []float32{oai(2.0, 1.5, 1.702, 7.0), oai(9.0, 20.0, 1.702, 7.0)} {
		if math.Abs(float64(g[i]-want)) > 1e-6 {
			t.Fatalf("swigluOAI[%d] = %v, want %v", i, g[i], want)
		}
	}
}
