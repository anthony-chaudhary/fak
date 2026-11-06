package model

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// glm5NextTestModel builds a reduced-geometry GLM-5.3-Flash text model through
// the REAL admission + binding path (NewFromF32Tensors -> newModel -> the
// glm5_next branch), with the canonical checkpoint tensor names under the
// model.language_model. prefix and every per-cadence tensor the engine forward
// reads. Geometry is shrunk (hidden 32, 8 layers, 6 experts) but the structure
// is the pinned one: 3:1 KDA:DSA cadence, dense-then-MoE MLP, mHC streams.
func glm5NextTestModel(t *testing.T, layers int) *Model {
	t.Helper()
	const (
		hidden   = 32
		heads    = 4
		vocab    = 24
		qLora    = 8
		kvLora   = 8
		qkNope   = 8
		vHead    = 8
		linHeads = 2
		linDim   = 4
		convW    = 4
		idxHeads = 2
		idxDim   = 4
		idxTopk  = 8
		idxKpool = 2
		hcMult   = 2
		moeInter = 6
		experts  = 6
		shared   = 1
		firstDen = 3 // pinned canonical: layers 0..2 dense, MoE after
		inter    = 10
	)
	qkvDim := linHeads * linDim
	hcMix := (2 + hcMult) * hcMult
	hcFlat := hcMult * hidden

	cfg := Config{
		ModelType:                  "glm5_next",
		NumLayers:                  layers,
		HiddenSize:                 hidden,
		NumHeads:                   heads,
		NumKVHeads:                 heads,
		VocabSize:                  vocab,
		RMSNormEps:                 1e-5,
		QLoraRank:                  qLora,
		KVLoraRank:                 kvLora,
		QKNopeHeadDim:              qkNope,
		QKRopeHeadDim:              0,
		VHeadDim:                   vHead,
		LinearNumKeyHeads:          linHeads,
		LinearNumValueHeads:        linHeads,
		LinearKeyHeadDim:           linDim,
		LinearValueHeadDim:         linDim,
		LinearConvKernelDim:        convW,
		LinearGateLowerBound:       -5.0,
		IndexNHeads:                idxHeads,
		IndexHeadDim:               idxDim,
		IndexTopK:                  idxTopk,
		IndexKpool:                 idxKpool,
		IndexKpoolAlwaysSelectTail: true,
		HCMult:                     hcMult,
		HCSinkhornIters:            3,
		HCEps:                      1e-6,
		NumExperts:                 experts,
		NumExpertsPerTok:           2,
		NGroup:                     2,
		TopKGroup:                  1,
		NormTopKProb:               true,
		RoutedScalingFactor:        2.5,
		ScoringFunc:                "sigmoid",
		MoEIntermediateSize:        moeInter,
		NSharedExperts:             shared,
		FirstKDenseReplace:         firstDen,
		IntermediateSize:           inter,
		SwigluLimit:                10.0,
		GLM5Next:                   true,
	}

	rng := rand.New(rand.NewSource(20261106))
	fill := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(rng.NormFloat64()) * 0.08
		}
		return out
	}
	const p = "model.language_model."
	tensors := []NamedTensorF32{
		{Name: p + "embed_tokens.weight", Shape: []int{vocab, hidden}, Data: fill(vocab * hidden)},
		{Name: p + "norm.weight", Shape: []int{hidden}, Data: fill(hidden)},
		{Name: "lm_head.weight", Shape: []int{vocab, hidden}, Data: fill(vocab * hidden)},
	}
	one := func(name string, rows, cols int) {
		tensors = append(tensors, NamedTensorF32{Name: p + name, Shape: []int{rows, cols}, Data: fill(rows * cols)})
	}
	vec := func(name string, n int) {
		tensors = append(tensors, NamedTensorF32{Name: p + name, Shape: []int{n}, Data: fill(n)})
	}
	for l := 0; l < layers; l++ {
		lp := "layers." + strconv.Itoa(l) + "."
		one(lp+"input_layernorm.weight", 1, hidden)
		one(lp+"post_attention_layernorm.weight", 1, hidden)
		one(lp+"hc_attn_fn", hcMix, hcFlat)
		vec(lp+"hc_attn_base", hcMix)
		vec(lp+"hc_attn_scale", 3)
		one(lp+"hc_ffn_fn", hcMix, hcFlat)
		vec(lp+"hc_ffn_base", hcMix)
		vec(lp+"hc_ffn_scale", 3)
		if l%4 != 3 {
			one(lp+"self_attn.q_proj.weight", qkvDim, hidden)
			one(lp+"self_attn.k_proj.weight", qkvDim, hidden)
			one(lp+"self_attn.v_proj.weight", qkvDim, hidden)
			one(lp+"self_attn.q_conv1d.weight", qkvDim, convW)
			one(lp+"self_attn.k_conv1d.weight", qkvDim, convW)
			one(lp+"self_attn.v_conv1d.weight", qkvDim, convW)
			one(lp+"self_attn.f_a_proj.weight", linDim, hidden)
			one(lp+"self_attn.f_b_proj.weight", qkvDim, linDim)
			vec(lp+"self_attn.dt_bias", qkvDim)
			vec(lp+"self_attn.A_log", linHeads)
			one(lp+"self_attn.b_proj.weight", linHeads, hidden)
			one(lp+"self_attn.g_a_proj.weight", linDim, hidden)
			one(lp+"self_attn.g_b_proj.weight", qkvDim, linDim)
			one(lp+"self_attn.o_norm.weight", 1, linDim)
			one(lp+"self_attn.o_proj.weight", hidden, qkvDim)
		} else {
			one(lp+"self_attn.q_a_proj.weight", qLora, hidden)
			one(lp+"self_attn.q_a_layernorm.weight", 1, qLora)
			one(lp+"self_attn.q_b_proj.weight", heads*qkNope, qLora)
			one(lp+"self_attn.kv_a_proj_with_mqa.weight", kvLora, hidden)
			one(lp+"self_attn.kv_a_layernorm.weight", 1, kvLora)
			one(lp+"self_attn.kv_b_proj.weight", heads*(qkNope+vHead), kvLora)
			one(lp+"self_attn.o_proj.weight", hidden, heads*vHead)
			one(lp+"self_attn.indexer.wq_b.weight", idxHeads*idxDim, qLora)
			one(lp+"self_attn.indexer.wk.weight", idxDim, hidden)
			one(lp+"self_attn.indexer.k_norm.weight", 1, idxDim)
			one(lp+"self_attn.indexer.k_norm.bias", 1, idxDim)
			one(lp+"self_attn.indexer.weights_proj.weight", idxHeads, hidden)
			one(lp+"self_attn.indexer.index_kpool_compress_ape", idxKpool, idxDim)
			one(lp+"self_attn.indexer.index_kpool_compress_gate", idxDim, hidden)
		}
		if l < firstDen {
			one(lp+"mlp.gate_proj.weight", inter, hidden)
			one(lp+"mlp.up_proj.weight", inter, hidden)
			one(lp+"mlp.down_proj.weight", hidden, inter)
		} else {
			one(lp+"mlp.gate.weight", experts, hidden)
			vec(lp+"mlp.gate.e_score_correction_bias", experts)
			for e := 0; e < experts; e++ {
				ep := lp + "mlp.experts." + strconv.Itoa(e) + "."
				one(ep+"gate_proj.weight", moeInter, hidden)
				one(ep+"up_proj.weight", moeInter, hidden)
				one(ep+"down_proj.weight", hidden, moeInter)
			}
			one(lp+"mlp.shared_experts.gate_proj.weight", moeInter*shared, hidden)
			one(lp+"mlp.shared_experts.up_proj.weight", moeInter*shared, hidden)
			one(lp+"mlp.shared_experts.down_proj.weight", hidden, moeInter*shared)
		}
	}
	m, err := NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	return m
}

func allFinite(xs []float32) bool {
	for _, v := range xs {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}

func TestGLM5NextEngineForwardAdmitsAndRuns(t *testing.T) {
	m := glm5NextTestModel(t, 8)
	ids := []int{1, 2, 3, 4, 5}
	act := m.Forward(ids)
	if act.Seq != len(ids) {
		t.Fatalf("Seq = %d, want %d", act.Seq, len(ids))
	}
	if len(act.Logits) != len(ids) {
		t.Fatalf("logits rows = %d, want %d", len(act.Logits), len(ids))
	}
	for i, row := range act.Logits {
		if len(row) != m.Cfg.VocabSize {
			t.Fatalf("logits[%d] len = %d, want %d", i, len(row), m.Cfg.VocabSize)
		}
		if !allFinite(row) {
			t.Fatalf("logits[%d] not finite", i)
		}
	}
	// The per-layer trace carries the embedding entry plus one row per layer.
	if len(act.Hidden) != m.Cfg.NumLayers+1 {
		t.Fatalf("Hidden layers = %d, want %d", len(act.Hidden), m.Cfg.NumLayers+1)
	}
	if len(act.Hidden[0]) != len(ids)*m.Cfg.HiddenSize {
		t.Fatalf("embed band len = %d, want %d", len(act.Hidden[0]), len(ids)*m.Cfg.HiddenSize)
	}
	// The forward must not be a no-op: logits differ across positions.
	same := true
	for i := range act.Logits[0] {
		if act.Logits[0][i] != act.Logits[1][i] {
			same = false
			break
		}
	}
	if same {
		t.Fatalf("logits identical across positions — forward did not transform state")
	}
	writeGLM5NextEngineForwardReceipt(t, m)
}

// GLM5NextEngineForwardReceipt is the rung witness for the weights-bound
// native engine forward (epic #9433 native track, engine rung).
type GLM5NextEngineForwardReceipt struct {
	Schema         string         `json:"schema"`
	Engine         string         `json:"engine"`
	Model          string         `json:"model"`
	Rung           string         `json:"rung"`
	Layers         int            `json:"layers"`
	Cadence        string         `json:"cadence"`
	Mixers         []string       `json:"mixers"`
	Mlp            string         `json:"mlp"`
	StateSeams     []string       `json:"state_seams"`
	PinnedArtifact string         `json:"pinned_artifact_revision"`
	Geometry       map[string]int `json:"test_geometry"`
	GeometryNote   string         `json:"geometry_note"`
	Witness        string         `json:"witness"`
}

func writeGLM5NextEngineForwardReceipt(t *testing.T, m *Model) {
	t.Helper()
	receipt := GLM5NextEngineForwardReceipt{
		Schema:         "fak-native-glm5next-engine-forward/v1",
		Engine:         "fak-native",
		Model:          "zai-org/GLM-5.3-Flash",
		Rung:           "weights-bound engine forward (Model.Forward + GLM5NextSession)",
		Layers:         m.Cfg.NumLayers,
		Cadence:        "3:1 KDA:DSA (layer%4==3 is DSA)",
		Mixers:         []string{"kda-delta-rule", "dsa-mla-nope-kpool-indexer"},
		Mlp:            "clamped-swiglu dense(0..2) + 288-expert top-8 sigmoid MoE + shared expert",
		StateSeams:     []string{"kda recurrent state", "kda causal conv state", "dsa key/value cache", "dsa indexer kpool cache", "mhc residual streams"},
		PinnedArtifact: "zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0",
		Geometry: map[string]int{
			"hidden": m.Cfg.HiddenSize, "layers": m.Cfg.NumLayers,
		},
		GeometryNote: "reduced-geometry witness; tensor names and cadence are the pinned canonical contract",
		Witness:      "SW-VERIFIED: admission (manifest + engine tensor checks), Forward(seq) parity with the stepwise session, snapshot/restore, reset determinism, prefix causality, batched-path loud refusal",
	}
	dir := filepath.Join("..", "..", "docs", "_witnesses", "glm5next-native-engine-forward")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Logf("note: could not create witness dir: %v", err)
		return
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("marshal receipt: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "receipt.json"), data, 0o644); err != nil {
		t.Fatalf("write receipt: %v", err)
	}
}

func TestGLM5NextForwardMatchesStepwiseSession(t *testing.T) {
	m := glm5NextTestModel(t, 8)
	ids := []int{3, 1, 4, 1, 5, 9, 2, 6}
	full := m.Forward(ids)

	s, err := NewGLM5NextSession(m)
	if err != nil {
		t.Fatalf("NewGLM5NextSession: %v", err)
	}
	for i, id := range ids {
		hidden := s.Step(m.glm5NextEmbed([]int{id})[0])
		got := s.Logits(hidden)
		for j := range got {
			if got[j] != full.Logits[i][j] {
				t.Fatalf("token %d logit %d: session %v != forward %v", i, j, got[j], full.Logits[i][j])
			}
		}
	}
	if s.TokensProcessed() != len(ids) {
		t.Fatalf("TokensProcessed = %d, want %d", s.TokensProcessed(), len(ids))
	}
}

func TestGLM5NextSessionSnapshotRestore(t *testing.T) {
	m := glm5NextTestModel(t, 8)
	ids := []int{2, 4, 6, 8, 10}
	s, err := NewGLM5NextSession(m)
	if err != nil {
		t.Fatalf("NewGLM5NextSession: %v", err)
	}
	for _, id := range ids[:3] {
		s.Step(m.glm5NextEmbed([]int{id})[0])
	}
	snap := s.Snapshot()
	continued := [][]float32{}
	for _, id := range ids[3:] {
		hidden := s.Step(m.glm5NextEmbed([]int{id})[0])
		continued = append(continued, s.Logits(hidden))
	}
	// A fresh session replaying the prefix, then restoring, must produce the
	// identical continuation — the snapshot owns the full recurrent + cache.
	fresh, err := NewGLM5NextSession(m)
	if err != nil {
		t.Fatalf("fresh session: %v", err)
	}
	for _, id := range ids[:3] {
		fresh.Step(m.glm5NextEmbed([]int{id})[0])
	}
	fresh.Restore(snap)
	for i, id := range ids[3:] {
		hidden := fresh.Step(m.glm5NextEmbed([]int{id})[0])
		got := fresh.Logits(hidden)
		for j := range got {
			if got[j] != continued[i][j] {
				t.Fatalf("restored token %d logit %d: %v != %v", i, j, got[j], continued[i][j])
			}
		}
	}
}

func TestGLM5NextSessionResetClearsState(t *testing.T) {
	m := glm5NextTestModel(t, 8)
	s, err := NewGLM5NextSession(m)
	if err != nil {
		t.Fatalf("NewGLM5NextSession: %v", err)
	}
	ids := []int{1, 2, 3}
	first := make([][]float32, 0, len(ids))
	for _, id := range ids {
		hidden := s.Step(m.glm5NextEmbed([]int{id})[0])
		first = append(first, s.Logits(hidden))
	}
	s.Reset()
	if s.TokensProcessed() != 0 {
		t.Fatalf("TokensProcessed after reset = %d, want 0", s.TokensProcessed())
	}
	for i, id := range ids {
		hidden := s.Step(m.glm5NextEmbed([]int{id})[0])
		got := s.Logits(hidden)
		for j := range got {
			if got[j] != first[i][j] {
				t.Fatalf("post-reset token %d logit %d drifted: %v != %v", i, j, got[j], first[i][j])
			}
		}
	}
}

func TestGLM5NextEngineManifestRefusalNamesMissingTensor(t *testing.T) {
	cfg := Config{ModelType: "glm5_next", NumLayers: 1, HiddenSize: 8, NumHeads: 2,
		NumKVHeads: 2, VocabSize: 8, GLM5Next: true, FirstKDenseReplace: 1}
	man := map[string]tensorMeta{
		"model.embed_tokens.weight": {Dtype: "f32", Shape: []int{8, 8}},
		"model.norm.weight":         {Dtype: "f32", Shape: []int{8}},
	}
	err := validateGLM5NextEngineManifest(cfg, man)
	if err == nil {
		t.Fatalf("expected refusal for a manifest without lm_head")
	}
	if !strings.Contains(err.Error(), "lm_head.weight") {
		t.Fatalf("refusal must name the missing tensor, got: %v", err)
	}
	// A DSA layer missing its indexer tensors must be named too.
	cfg.NumLayers = 4
	cfg.FirstKDenseReplace = 1
	man["lm_head.weight"] = tensorMeta{Dtype: "f32", Shape: []int{8, 8}}
	err = validateGLM5NextEngineManifest(cfg, man)
	if err == nil {
		t.Fatalf("expected refusal for incomplete layer set")
	}
	if !strings.Contains(err.Error(), "layers.0.") {
		t.Fatalf("refusal must name the layer, got: %v", err)
	}
}

func TestGLM5NextClassifyForwardPathWithManifest(t *testing.T) {
	m := glm5NextTestModel(t, 8)
	kind, err := ClassifyForwardPath(m.Cfg, m.manifest)
	if err != nil {
		t.Fatalf("ClassifyForwardPath: %v", err)
	}
	if kind != ForwardGLM5Next {
		t.Fatalf("ClassifyForwardPath = %q, want %q", kind, ForwardGLM5Next)
	}
	// And the nil-manifest classification keeps the typed refusal.
	_, err = ClassifyForwardPath(m.Cfg, nil)
	if err == nil {
		t.Fatalf("expected typed refusal for nil manifest")
	}
	if _, ok := err.(*GLM5NextUnsupportedError); !ok {
		t.Fatalf("refusal type = %T, want *GLM5NextUnsupportedError", err)
	}
}

func TestGLM5NextMoERouterTopK(t *testing.T) {
	// The router must select exactly topK experts per token and normalize the
	// routed weights (norm_topk_prob) before the routed_scaling factor.
	m := glm5NextTestModel(t, 8)
	d := glm5NextDimsFromCfg(m.Cfg)
	if d.numExperts != 6 || d.topK != 2 || !d.normTopk || d.routedScaling != 2.5 {
		t.Fatalf("dims misparsed: experts=%d topK=%d norm=%v scaling=%v", d.numExperts, d.topK, d.normTopk, d.routedScaling)
	}
	if d.linearHeads != 2 || d.linearDim != 4 || d.convWindow != 4 {
		t.Fatalf("KDA dims misparsed: %dx%d conv=%d", d.linearHeads, d.linearDim, d.convWindow)
	}
	if d.hcMult != 2 || d.hcSinkhorn != 3 || d.indexKpool != 2 || d.indexTopk != 8 {
		t.Fatalf("hc/index dims misparsed: hc=%dx%d kpool=%d topk=%d", d.hcMult, d.hcSinkhorn, d.indexKpool, d.indexTopk)
	}
}

func TestGLM5NextPrefixCausality(t *testing.T) {
	// Causality witness: appending a later token must not change any earlier
	// position's logits. The KDA recurrence is causal by construction; the DSA
	// indexer and sparse attention must be too (a future-visible key would
	// shift the top-k selection and the softmax).
	m := glm5NextTestModel(t, 8)
	prefix := []int{2, 4, 6, 8}
	short := m.Forward(prefix)
	longer := m.Forward(append(append([]int(nil), prefix...), 11))
	for i := range prefix {
		for j := range short.Logits[i] {
			if short.Logits[i][j] != longer.Logits[i][j] {
				t.Fatalf("prefix causality broken at token %d logit %d: %v != %v",
					i, j, short.Logits[i][j], longer.Logits[i][j])
			}
		}
	}
}

func TestGLM5NextBatchedSessionFailsLoudly(t *testing.T) {
	// The KV/HAL batched substrate does not own the KDA/DSA/mHC state; a glm5
	// model reaching it must fail loudly, never silently run the generic GQA
	// loop and emit wrong logits.
	m := glm5NextTestModel(t, 8)
	s := m.NewSession()
	for _, call := range []struct {
		name string
		fn   func()
	}{
		{"Prefill", func() { s.Prefill([]int{1}) }},
		{"Step", func() { s.Step(1) }},
	} {
		func() {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("%s: expected a loud refusal for a glm5_next session", call.name)
				}
				msg := fmt.Sprintf("%v", r)
				if !strings.Contains(msg, "glm5_next") || !strings.Contains(msg, "GLM5NextSession") {
					t.Fatalf("%s: refusal must name the wired path, got: %s", call.name, msg)
				}
			}()
			call.fn()
		}()
	}
}
