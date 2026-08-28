package model

import (
	"encoding/json"
	"math"
	"testing"
)

// qwen35MoEBaseTensors builds the minimal attention + router + per-expert SwiGLU tensor
// set for a one-layer E-expert MoE model, mirroring the harness in TestMoERoutingHandComputed.
// Callers append the shared-expert tensors (or not) to exercise the qwen3_5_moe branch.
func qwen35MoEBaseTensors(H, I, E int, router []float32) []NamedTensorF32 {
	tensors := []NamedTensorF32{
		{Name: "model.embed_tokens.weight", Shape: []int{4, H}, Data: make([]float32, 4*H)},
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
	return tensors
}

// refRoutedDelta hand-computes the routed-experts-only weighted sum for xn=[1,1] with the
// router rows used below (top-2 = e1 then e0, norm_topk_prob renormalizing the gate weights).
func refRoutedDelta(H, I int) []float64 {
	logits := []float64{2, 3, 2, -2}
	mx := 3.0
	var z float64
	exps := make([]float64, len(logits))
	for i, l := range logits {
		exps[i] = math.Exp(l - mx)
		z += exps[i]
	}
	probs := make([]float64, len(logits))
	for i := range probs {
		probs[i] = exps[i] / z
	}
	sumSel := probs[1] + probs[0]
	wExpert := map[int]float64{1: probs[1] / sumSel, 0: probs[0] / sumSel}
	refExpert := func(e int) []float64 {
		b := float64(e + 1)
		out := make([]float64, H)
		for i := 0; i < I; i++ {
			x := 0.1 * b
			s := (x / (1 + math.Exp(-x))) * (0.2 * b)
			out[i] = 0.3 * b * s
		}
		return out
	}
	delta := make([]float64, H)
	for _, e := range []int{1, 0} {
		ro := refExpert(e)
		for i := 0; i < H; i++ {
			delta[i] += wExpert[e] * ro[i]
		}
	}
	return delta
}

func parsedQwen35MoETestConfig(t *testing.T) Config {
	t.Helper()
	const raw = `{
		"model_type":"qwen3_5_moe",
		"hidden_size":2,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":2,
		"num_hidden_layers":1,
		"vocab_size":4,
		"intermediate_size":2,
		"num_experts":4,
		"num_experts_per_tok":2,
		"moe_intermediate_size":2,
		"shared_expert_intermediate_size":2,
		"rms_norm_eps":0.00001,
		"rope_theta":10000
	}`
	var cfg Config
	if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
		t.Fatalf("unmarshal bounded Qwen3.5-MoE config: %v", err)
	}
	cfg.EOSTokenID = -1
	return cfg
}

// TestQwen35MoEAbsentNormalizationUsesProductionRouting is the issue's named
// acceptance witness for absent config normalization through route and moeFFN.
func TestQwen35MoEAbsentNormalizationUsesProductionRouting(t *testing.T) {
	testQwen35MoEProductionRoutingAndSharedExpert(t)
}

// TestQwen35MoESharedExpertAdded keeps the shared-expert acceptance surface explicit.
func TestQwen35MoESharedExpertAdded(t *testing.T) {
	testQwen35MoEProductionRoutingAndSharedExpert(t)
}

// testQwen35MoEProductionRoutingAndSharedExpert is a bounded synthetic production-path
// regression: a parsed qwen3_5_moe config must normalize routed weights and add its
// always-on, sigmoid-gated shared expert. It is not a full-checkpoint or external-oracle witness.
func testQwen35MoEProductionRoutingAndSharedExpert(t *testing.T) {
	t.Helper()
	const H, I, E, K, SI = 2, 2, 4, 2, 2
	cfg := parsedQwen35MoETestConfig(t)
	if cfg.NumExperts != E || cfg.NumExpertsPerTok != K || cfg.SharedIntermediateSize != SI {
		t.Fatalf("parsed bounded axes = experts:%d top-k:%d shared:%d, want %d/%d/%d",
			cfg.NumExperts, cfg.NumExpertsPerTok, cfg.SharedIntermediateSize, E, K, SI)
	}
	router := []float32{2, 0, 0, 3, 1, 1, -1, -1}
	xn := []float32{1, 1}

	baseTensors := qwen35MoEBaseTensors(H, I, E, router)
	routedOnly, err := NewFromF32Tensors(cfg, baseTensors)
	if err != nil {
		t.Fatalf("build routed-only model: %v", err)
	}
	picks := route(routedOnly, 0, xn, f32Kernel{routedOnly})
	if len(picks) != K || picks[0].expert != 1 || picks[1].expert != 0 {
		t.Fatalf("route picks = %+v, want experts [1 0]", picks)
	}
	if sum := picks[0].weight + picks[1].weight; math.Abs(float64(sum)-1) > 1e-6 {
		t.Fatalf("normalized routed weight sum = %v, want 1", sum)
	}
	routedGot := moeFFN{}.apply(routedOnly, 0, xn, f32Kernel{routedOnly})

	tensors := append([]NamedTensorF32(nil), baseTensors...)
	// Shared expert (SwiGLU at width SI) + scalar hidden->1 sigmoid gate.
	tensors = append(tensors,
		NamedTensorF32{Name: qwen35SharedExpertName(0, "gate_proj.weight"), Shape: []int{SI, H}, Data: []float32{0.5, 0, 0, 0.5}},
		NamedTensorF32{Name: qwen35SharedExpertName(0, "up_proj.weight"), Shape: []int{SI, H}, Data: []float32{0.4, 0, 0, 0.4}},
		NamedTensorF32{Name: qwen35SharedExpertName(0, "down_proj.weight"), Shape: []int{H, SI}, Data: []float32{0.6, 0, 0, 0.6}},
		NamedTensorF32{Name: qwen35SharedExpertName(0, "gate.weight"), Shape: []int{1, H}, Data: []float32{0.25, 0.25}},
	)
	m, err := NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatalf("build model: %v", err)
	}

	// Hand-computed shared-expert reference on xn=[1,1]:
	//   g = 0.5, u = 0.4 (per channel); silu(g)*u; down scales by 0.6.
	//   gate scalar = sigmoid(0.25*1 + 0.25*1) = sigmoid(0.5).
	gate := 1.0 / (1.0 + math.Exp(-0.5))
	sharedCh := 0.6 * ((0.5 / (1 + math.Exp(-0.5))) * 0.4)
	want := refRoutedDelta(H, I)
	for i := 0; i < H; i++ {
		want[i] += gate * sharedCh
	}

	got := moeFFN{}.apply(m, 0, xn, f32Kernel{m})
	for i := 0; i < H; i++ {
		if math.Abs(float64(got[i])-want[i]) > 1e-5 {
			t.Fatalf("delta[%d] = %v, want %v (routed + gated shared expert)", i, got[i], want[i])
		}
		if math.Abs(float64(got[i]-routedGot[i])-gate*sharedCh) > 1e-5 {
			t.Fatalf("shared additive delta[%d] = %v, want %v", i, got[i]-routedGot[i], gate*sharedCh)
		}
	}
	// Shared contribution must be non-zero (else the test would pass even if the branch were dead).
	if gate*sharedCh < 1e-6 {
		t.Fatalf("reference shared contribution is ~0 (%v); test would be vacuous", gate*sharedCh)
	}
}

// TestQwen35MoENoSharedExpertUnchanged proves the shared-expert branch is presence-guarded:
// a checkpoint WITHOUT the singular shared_expert_gate tensor returns the routed-only sum,
// so Mixtral / Qwen3-MoE stay byte-for-byte unchanged.
func TestQwen35MoENoSharedExpertUnchanged(t *testing.T) {
	const H, I, E, K = 2, 2, 4, 2
	cfg := Config{
		HiddenSize: H, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 2,
		IntermediateSize: I, VocabSize: 4, RMSNormEps: 1e-5, RopeTheta: 10000,
		NumExperts: E, NumExpertsPerTok: K, NormTopKProb: true, EOSTokenID: -1,
	}
	router := []float32{2, 0, 0, 3, 1, 1, -1, -1}
	xn := []float32{1, 1}

	m, err := NewFromF32Tensors(cfg, qwen35MoEBaseTensors(H, I, E, router))
	if err != nil {
		t.Fatalf("build model: %v", err)
	}
	want := refRoutedDelta(H, I)
	got := moeFFN{}.apply(m, 0, xn, f32Kernel{m})
	for i := 0; i < H; i++ {
		if math.Abs(float64(got[i])-want[i]) > 1e-5 {
			t.Fatalf("delta[%d] = %v, want %v (routed-only, no shared expert)", i, got[i], want[i])
		}
	}
}
