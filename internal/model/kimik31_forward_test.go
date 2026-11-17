package model

// kimik31_forward_test.go - the L8 composite-dispatch witness for ticket
// 1611-kimi-k31 / issue #14363. It pins the two halves the keystone asks for:
//
//  1. ClassifyForwardPath on a kimi_k31 config returns the K3.1 forward-path
//     class (ForwardKimiK31) - never the generic attnSeq fall-through that
//     would panic on the KDA/DSA weight set (there is no self_attn.q_proj), and
//     never the qwen35 GDN path (K3.1 shares no layer_types token with it).
//  2. A small-tensor forward smoke through the KDA/DSA/trio composition on
//     synthetic weights produces FINITE logits and advances the per-layer
//     recurrent/cell state.
//
// fak-test:runtime fast est=40ms lane=default

import (
	"encoding/json"
	"math/rand"
	"testing"
)

// TestKimiK31ClassifiesToForwardPath is the #14363 keystone witness: the
// published 93-layer K3.1 config classifies to ForwardKimiK31 (the L8 path),
// distinct from both the attnSeq default and the qwen35 GDN hybrid.
func TestKimiK31ClassifiesToForwardPath(t *testing.T) {
	var cfg Config
	if err := json.Unmarshal([]byte(kimiK31ConfigFixture(93)), &cfg); err != nil {
		t.Fatalf("unmarshal K3.1 config: %v", err)
	}
	if !cfg.IsKimiK31() {
		t.Fatalf("precondition: IsKimiK31() = false (family key %q)", cfg.archFamilyKey())
	}
	got, err := ClassifyForwardPath(cfg, nil)
	if err != nil {
		t.Fatalf("recognized K3.1 config must classify, got refusal: %v", err)
	}
	if got != ForwardKimiK31 {
		t.Fatalf("ClassifyForwardPath = %q, want %q", got, ForwardKimiK31)
	}
	if got == ForwardAttnSeqGQA {
		t.Fatal("K3.1 classified to the generic attnSeq path: the KDA/DSA weight set has no self_attn.q_proj and would panic mid-request")
	}
	if got == ForwardQwen35GDN {
		t.Fatal("K3.1 classified to the qwen35 GDN path: K3.1 shares no layer_types token with qwen35")
	}
}

// TestKimiK31ClassifyHonorsLayerKind pins that the classifier's answer and the
// composite's per-layer dispatch read the SAME predicates (IsKimiK31KDALayer /
// IsKimiK31DSALayer): a config the classifier names ForwardKimiK31 has a
// kda/dsa token on every layer the composite will run.
func TestKimiK31ClassifyHonorsLayerKind(t *testing.T) {
	cfg := tinyKimiK31CompositeConfig()
	if got, err := ClassifyForwardPath(cfg, nil); err != nil || got != ForwardKimiK31 {
		t.Fatalf("ClassifyForwardPath = %q (err %v), want %q", got, err, ForwardKimiK31)
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.IsKimiK31KDALayer(l) && !cfg.IsKimiK31DSALayer(l) {
			t.Fatalf("layer %d carries neither kda nor dsa token: %v", l, cfg.LayerTypes)
		}
	}
}

// TestKimiK31ForwardSmokeFiniteLogits is the #14363 small-tensor forward smoke:
// two KDA layers + one Gated-DSA layer run end to end on synthetic weights and
// produce finite logits through a synthetic lm_head, with the DSA cell cache and
// the KDA recurrent state advancing across the sequence.
func TestKimiK31ForwardSmokeFiniteLogits(t *testing.T) {
	cfg := tinyKimiK31CompositeConfig()
	p := tinyKimiK31CompositeParams(cfg)
	T := 2
	st, err := NewKimiK31ForwardState(cfg, p, T)
	if err != nil {
		t.Fatalf("NewKimiK31ForwardState: %v", err)
	}

	rng := rand.New(rand.NewSource(7))
	xSeq := make([]float32, T*p.Hidden)
	for i := range xSeq {
		xSeq[i] = rng.Float32()*2 - 1
	}

	hidden := RunKimiK31Forward(cfg, p, st, xSeq, T)
	if len(hidden) != T*p.Hidden {
		t.Fatalf("ForwardKimiK31 returned %d values, want %d", len(hidden), T*p.Hidden)
	}
	if !allFinite(hidden) {
		t.Fatalf("ForwardKimiK31 produced a non-finite hidden value")
	}

	// Synthetic lm_head (vocab x hidden): the "finite logits" half of the witness.
	vocab := 5
	head := make([]float32, vocab*p.Hidden)
	for i := range head {
		head[i] = rng.Float32()*2 - 1
	}
	for pos := 0; pos < T; pos++ {
		logits := kdaMatVecMul(head, hidden[pos*p.Hidden:(pos+1)*p.Hidden], vocab, p.Hidden)
		if !allFinite(logits) {
			t.Fatalf("position %d: non-finite logits", pos)
		}
	}

	// State advanced: the DSA layer cached T cells; each KDA layer's recurrence
	// moved off zero.
	dsaLayer := cfg.NumLayers - 1
	if st.DSA[dsaLayer].Len() != T {
		t.Fatalf("DSA layer %d cached %d cells, want %d", dsaLayer, st.DSA[dsaLayer].Len(), T)
	}
	for l := 0; l < dsaLayer; l++ {
		kst := st.KDA[l]
		nonzero := false
		for _, v := range kst.S {
			if v != 0 {
				nonzero = true
				break
			}
		}
		if !nonzero {
			t.Fatalf("KDA layer %d recurrent state stayed zero across %d tokens", l, T)
		}
	}
}

// TestKimiK31ForwardDecodeMatchesPrefill pins that a stepwise decode of the
// same ids reproduces the sequence forward bit-for-bit (the same state threading
// the decode loop uses).
func TestKimiK31ForwardDecodeMatchesPrefill(t *testing.T) {
	cfg := tinyKimiK31CompositeConfig()
	p := tinyKimiK31CompositeParams(cfg)
	T := 3
	rng := rand.New(rand.NewSource(11))
	xSeq := make([]float32, T*p.Hidden)
	for i := range xSeq {
		xSeq[i] = rng.Float32()*2 - 1
	}

	prefillState, err := NewKimiK31ForwardState(cfg, p, T)
	if err != nil {
		t.Fatalf("prefill state: %v", err)
	}
	prefill := RunKimiK31Forward(cfg, p, prefillState, xSeq, T)

	decodeState, err := NewKimiK31ForwardState(cfg, p, T)
	if err != nil {
		t.Fatalf("decode state: %v", err)
	}
	decode := make([]float32, T*p.Hidden)
	for tok := 0; tok < T; tok++ {
		o := ForwardKimiK31Token(cfg, p, decodeState, xSeq[tok*p.Hidden:(tok+1)*p.Hidden])
		copy(decode[tok*p.Hidden:(tok+1)*p.Hidden], o)
	}

	for i := range prefill {
		if prefill[i] != decode[i] {
			t.Fatalf("prefill/decode diverge at %d: %v vs %v", i, prefill[i], decode[i])
		}
	}
}

// TestKimiK31ForwardStateRefusesMissingParams proves the composite stays
// fail-closed: a layer the config marks KDA/DSA with no oracle params, and a
// config whose layer carries no kda/dsa token, both refuse rather than running a
// wrong oracle.
func TestKimiK31ForwardStateRefusesMissingParams(t *testing.T) {
	cfg := tinyKimiK31CompositeConfig()

	t.Run("missing DSA params", func(t *testing.T) {
		p := tinyKimiK31CompositeParams(cfg)
		p.DSA[cfg.NumLayers-1] = nil
		if _, err := NewKimiK31ForwardState(cfg, p, 1); err == nil {
			t.Fatal("expected a refusal for a DSA layer with no params")
		}
	})

	t.Run("layer with no kda/dsa token", func(t *testing.T) {
		bad := cfg
		bad.LayerTypes = []string{KimiK31LayerTypeKDA, "full_attention", KimiK31LayerTypeDSA}
		p := tinyKimiK31CompositeParams(cfg)
		if _, err := NewKimiK31ForwardState(bad, p, 1); err == nil {
			t.Fatal("expected a refusal for a layer carrying no kda/dsa token")
		}
	})
}

// tinyKimiK31CompositeConfig is a minimal K3.1 stack: two KDA layers then one
// Gated-DSA layer, hidden 4 - the layer-kind mix the composite dispatches on.
func tinyKimiK31CompositeConfig() Config {
	return Config{
		ModelType:  "kimi_k31",
		NumLayers:  3,
		HiddenSize: 4,
		RMSNormEps: 1e-5,
		LayerTypes: []string{KimiK31LayerTypeKDA, KimiK31LayerTypeKDA, KimiK31LayerTypeDSA},
	}
}

// tinyKimiK31CompositeParams builds synthetic oracle params matching the tiny
// config's geometry (hidden 4; KDA 2 heads x 2 dims, conv 2; DSA hidden 4).
func tinyKimiK31CompositeParams(cfg Config) *KimiK31ForwardParams {
	rng := rand.New(rand.NewSource(3))
	pick := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = rng.Float32()*2 - 1
		}
		return out
	}
	hidden := cfg.HiddenSize
	attnNorm := make([][]float32, cfg.NumLayers)
	for l := range attnNorm {
		attnNorm[l] = make([]float32, hidden)
		for i := range attnNorm[l] {
			attnNorm[l][i] = 1
		}
	}
	finalNorm := make([]float32, hidden)
	for i := range finalNorm {
		finalNorm[i] = 1
	}

	kdaConv := func() *GLM5NextKDAConvFilter {
		f := NewGLM5NextKDAConvFilter(4, 2)
		for i := range f.Weight {
			f.Weight[i] = rng.Float32()*2 - 1
		}
		return f
	}
	kda := make([]*KimiK31KDAParams, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		kda[l] = &KimiK31KDAParams{
			NumHeads: 2, HeadDim: 2, ConvKernel: 2, HiddenSize: hidden,
			Wq: pick(4 * hidden), Wk: pick(4 * hidden), Wv: pick(4 * hidden),
			ConvQ: kdaConv(), ConvK: kdaConv(), ConvV: kdaConv(),
			WFA: pick(2 * hidden), WFB: pick(4 * 2),
			DTBias: pick(4), AFolded: []float32{-0.5, -0.25},
			WBeta: pick(2 * hidden), GateLowerBound: -5.0,
			WGate: pick(4 * hidden), ONorm: pick(2), WO: pick(hidden * 4),
			RMSEps: 1e-5,
		}
	}

	dsa := make([]*KimiK31DSAParams, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		dsa[l] = &KimiK31DSAParams{
			NumHeads: 2, QLoRA: 3, KVLoRA: 2, QkNope: 2, QkRope: 1, VDim: 2, Hidden: hidden,
			IdxHeads: 2, IdxDim: 2, TopK: 2, RMSEps: 1e-5, IdxEps: 1e-5,
			WQa: pick(3 * hidden), WQaNorm: pick(3), WQb: pick(2 * 3 * 3),
			WKvA: pick(3 * hidden), WKvANorm: pick(2), WKb: pick(2 * 2 * 2), WVb: pick(2 * 2 * 2),
			WO: pick(hidden * 2 * 2), WGate: pick(2 * 2 * hidden),
			WIdxQ: pick(2 * 2 * 3), WIdxK: pick(2 * hidden), WIdxProj: pick(2 * hidden),
			IdxKNormW: pick(2), IdxKNormB: pick(2),
		}
	}

	return &KimiK31ForwardParams{
		Hidden:    hidden,
		RMSEps:    1e-5,
		AttnNorm:  attnNorm,
		FinalNorm: finalNorm,
		KDA:       kda,
		DSA:       dsa,
	}
}
