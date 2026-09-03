package model

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

func TestQwen35MTPFuseDeterministicTinyTensor(t *testing.T) {
	m := qwen35MTPForwardTestModel(t, map[string]testMTPWeight{
		"mtp.pre_fc_norm_hidden.weight":    {shape: []int{2}, data: []float32{1, 1}},
		"mtp.pre_fc_norm_embedding.weight": {shape: []int{2}, data: []float32{1, 1}},
		"mtp.fc.weight":                    {shape: []int{2, 4}, data: []float32{1, 0, 0, 0, 0, 0, 0, 1}},
	})

	got, err := m.Qwen35MTPFuse([]float32{3, 4}, []float32{0, 5})
	if err != nil {
		t.Fatalf("fuse MTP decoder input: %v", err)
	}
	// Embedding-first: fusedInput is [normed_emb[0], normed_emb[1], normed_hid[0], normed_hid[1]]
	// Row 0 picks normed_emb[0] = 0; Row 1 picks normed_hid[1] = 4 / sqrt(12.5) = 1.13137085
	want := []float32{0.0, 1.13137085}
	if len(got) != len(want) {
		t.Fatalf("fused shape = [%d], want [%d]", len(got), len(want))
	}
	for i := range want {
		if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > 1e-6 {
			t.Fatalf("fused[%d] = %.8f, want %.8f (diff %.8g)", i, got[i], want[i], diff)
		}
	}
}

func TestQwen35MTPFuseRejectsInputShape(t *testing.T) {
	m := qwen35MTPForwardTestModel(t, map[string]testMTPWeight{
		"mtp.pre_fc_norm_hidden.weight":    {shape: []int{2}, data: []float32{1, 1}},
		"mtp.pre_fc_norm_embedding.weight": {shape: []int{2}, data: []float32{1, 1}},
		"mtp.fc.weight":                    {shape: []int{2, 4}, data: make([]float32, 8)},
	})

	_, err := m.Qwen35MTPFuse([]float32{1}, []float32{1, 2})
	var forwardErr *Qwen35MTPForwardError
	if !errors.As(err, &forwardErr) {
		t.Fatalf("shape error = %v, want *Qwen35MTPForwardError", err)
	}
	if forwardErr.Stage != "prior hidden shape" || forwardErr.Want != "[2]" || forwardErr.Got != "[1]" {
		t.Fatalf("shape error = %+v, want explicit prior-hidden [1] -> [2] refusal", forwardErr)
	}
}

func TestQwen35MTPFuseRejectsCheckpointTensorShape(t *testing.T) {
	m := qwen35MTPForwardTestModel(t, map[string]testMTPWeight{
		"mtp.pre_fc_norm_hidden.weight":    {shape: []int{2}, data: []float32{1, 1}},
		"mtp.pre_fc_norm_embedding.weight": {shape: []int{2}, data: []float32{1, 1}},
		"mtp.fc.weight":                    {shape: []int{4, 2}, data: make([]float32, 8)},
	})

	_, err := m.Qwen35MTPFuse([]float32{1, 2}, []float32{3, 4})
	var forwardErr *Qwen35MTPForwardError
	if !errors.As(err, &forwardErr) {
		t.Fatalf("tensor shape error = %v, want *Qwen35MTPForwardError", err)
	}
	if forwardErr.Stage != "weight shape" || forwardErr.Tensor != "mtp.fc.weight" || forwardErr.Want != "[2 4]" || forwardErr.Got != "[4 2]" {
		t.Fatalf("tensor shape error = %+v, want explicit mtp.fc [4 2] -> [2 4] refusal", forwardErr)
	}
}

func TestQwen35MTPFuseRejectsCheckpointTensorDtype(t *testing.T) {
	m := qwen35MTPForwardTestModel(t, map[string]testMTPWeight{
		"mtp.pre_fc_norm_hidden.weight":    {shape: []int{2}, data: []float32{1, 1}},
		"mtp.pre_fc_norm_embedding.weight": {shape: []int{2}, data: []float32{1, 1}},
		"mtp.fc.weight":                    {shape: []int{2, 4}, data: make([]float32, 8)},
	})
	meta := m.manifest["mtp.fc.weight"]
	meta.Dtype = "BF16"
	m.manifest["mtp.fc.weight"] = meta

	_, err := m.Qwen35MTPFuse([]float32{1, 2}, []float32{3, 4})
	var forwardErr *Qwen35MTPForwardError
	if !errors.As(err, &forwardErr) {
		t.Fatalf("tensor dtype error = %v, want *Qwen35MTPForwardError", err)
	}
	if forwardErr.Stage != "weight dtype" || forwardErr.Tensor != "mtp.fc.weight" || forwardErr.Want != "F32" || forwardErr.Got != "BF16" {
		t.Fatalf("tensor dtype error = %+v, want explicit mtp.fc BF16 -> F32 refusal", forwardErr)
	}
}

type testMTPWeight struct {
	shape []int
	data  []float32
}

func qwen35MTPForwardTestModel(t *testing.T, weights map[string]testMTPWeight) *Model {
	t.Helper()
	manifest := completeQwen35MTPManifest()
	raw := make([]byte, 0)
	for name, weight := range weights {
		offset := len(raw)
		for _, v := range weight.data {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			raw = append(raw, buf[:]...)
		}
		manifest[name] = tensorMeta{Dtype: "F32", Shape: weight.shape, Offset: offset, Nbytes: 4 * len(weight.data)}
	}
	cfg := qwen35MTPTestConfig()
	cfg.HiddenSize = 2
	cfg.RMSNormEps = 0
	return &Model{Cfg: cfg, manifest: manifest, raw: raw}
}

func TestQwen35MTPForwardDeterministicTinyTensor(t *testing.T) {
	m := qwen35MTPTinyForwardModel(t)
	forward, err := m.NewQwen35MTPForward()
	if err != nil {
		t.Fatalf("construct MTP forward: %v", err)
	}
	t.Cleanup(forward.Close)

	got, err := forward.Forward(0, []float32{3, 4}, []float32{0, 5})
	if err != nil {
		t.Fatalf("execute MTP forward: %v", err)
	}
	if forward.draft.Cache.Len() != 1 {
		t.Fatalf("draft cache length = %d, want 1", forward.draft.Cache.Len())
	}

	// mtp.fc selects the separately-normalized prior hidden. Attention has zero
	// projections; the retained dense decoder MLP is identity gate/up/down, so
	// its non-zero SwiGLU residual proves mtp.layers.0 executed before mtp.norm.
	x := []float32{3 / float32(math.Sqrt(12.5)), 4 / float32(math.Sqrt(12.5))}
	for i := range x {
		x[i] += x[i] * x[i] / (1 + float32(math.Exp(float64(-x[i]))))
	}
	ss := (x[0]*x[0] + x[1]*x[1]) / 2
	inv := 1 / float32(math.Sqrt(float64(ss)))
	want := []float32{x[0] * inv, x[1] * inv, (x[0] + x[1]) * inv}
	if len(got) != len(want) {
		t.Fatalf("logit shape = [%d], want [%d]", len(got), len(want))
	}
	for i := range want {
		if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > 1e-6 {
			t.Fatalf("logits[%d] = %.8f, want %.8f (diff %.8g)", i, got[i], want[i], diff)
		}
	}
}

func TestQwen35MTPForwardKeepsF32MechanismOnF32Layout(t *testing.T) {
	forward, err := qwen35MTPTinyForwardModel(t).NewQwen35MTPForward()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(forward.Close)
	if forward.tensorFormat != Qwen38MTPFormatF32 {
		t.Fatalf("tensor format=%q, want F32", forward.tensorFormat)
	}
	if _, ok := forward.mat.(f32Kernel); !ok {
		t.Fatalf("F32 MTP layout selected kernel %T", forward.mat)
	}
	if forward.draft.Q4K || forward.draft.MetalQ4K {
		t.Fatalf("F32 MTP layout unexpectedly enabled Q4_K/Metal: q4k=%v metal=%v", forward.draft.Q4K, forward.draft.MetalQ4K)
	}
}

func TestQwen35MTPForwardRejectsNonMonotonicPosition(t *testing.T) {
	forward, err := qwen35MTPTinyForwardModel(t).NewQwen35MTPForward()
	if err != nil {
		t.Fatalf("construct MTP forward: %v", err)
	}
	t.Cleanup(forward.Close)
	if _, err := forward.Forward(0, []float32{3, 4}, []float32{0, 5}); err != nil {
		t.Fatalf("first MTP forward: %v", err)
	}
	_, err = forward.Forward(0, []float32{3, 4}, []float32{0, 5})
	var forwardErr *Qwen35MTPForwardError
	if !errors.As(err, &forwardErr) || forwardErr.Stage != "position" {
		t.Fatalf("repeated position error = %v, want typed position refusal", err)
	}
}

func qwen35MTPTinyForwardModel(t *testing.T) *Model {
	t.Helper()
	zero := func(n int) []float32 { return make([]float32, n) }
	weights := map[string]testMTPWeight{
		"mtp.pre_fc_norm_hidden.weight":                {shape: []int{2}, data: []float32{1, 1}},
		"mtp.pre_fc_norm_embedding.weight":             {shape: []int{2}, data: []float32{1, 1}},
		"mtp.fc.weight":                                {shape: []int{2, 4}, data: []float32{0, 0, 1, 0, 0, 0, 0, 1}},
		"mtp.norm.weight":                              {shape: []int{2}, data: []float32{1, 1}},
		"mtp.layers.0.input_layernorm.weight":          {shape: []int{2}, data: []float32{1, 1}},
		"mtp.layers.0.post_attention_layernorm.weight": {shape: []int{2}, data: []float32{1, 1}},
		"mtp.layers.0.self_attn.q_norm.weight":         {shape: []int{1}, data: []float32{1}},
		"mtp.layers.0.self_attn.k_norm.weight":         {shape: []int{1}, data: []float32{1}},
		"mtp.layers.0.self_attn.q_proj.weight":         {shape: []int{4, 2}, data: zero(8)},
		"mtp.layers.0.self_attn.k_proj.weight":         {shape: []int{1, 2}, data: zero(2)},
		"mtp.layers.0.self_attn.v_proj.weight":         {shape: []int{1, 2}, data: zero(2)},
		"mtp.layers.0.self_attn.o_proj.weight":         {shape: []int{2, 2}, data: zero(4)},
		"mtp.layers.0.mlp.gate_proj.weight":            {shape: []int{2, 2}, data: []float32{1, 0, 0, 1}},
		"mtp.layers.0.mlp.up_proj.weight":              {shape: []int{2, 2}, data: []float32{1, 0, 0, 1}},
		"mtp.layers.0.mlp.down_proj.weight":            {shape: []int{2, 2}, data: []float32{1, 0, 0, 1}},
		"lm_head.weight":                               {shape: []int{3, 2}, data: []float32{1, 0, 0, 1, 1, 1}},
	}
	manifest := make(map[string]tensorMeta, len(weights))
	raw := make([]byte, 0)
	for name, weight := range weights {
		offset := len(raw)
		for _, v := range weight.data {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			raw = append(raw, buf[:]...)
		}
		manifest[name] = tensorMeta{Dtype: "F32", Shape: weight.shape, Offset: offset, Nbytes: 4 * len(weight.data)}
	}
	cfg := qwen35MTPTestConfig()
	cfg.HiddenSize = 2
	cfg.NumLayers = 1
	cfg.NumHeads = 2
	cfg.NumKVHeads = 1
	cfg.HeadDim = 1
	cfg.IntermediateSize = 2
	cfg.VocabSize = 3
	cfg.RMSNormEps = 0
	cfg.RopeTheta = 10000
	cfg.AttnOutputGate = true
	cfg.QKNorm = true
	return &Model{Cfg: cfg, manifest: manifest, raw: raw}
}

// TestQwen35MTPFuseEmbeddingFirstWitness (#9960):
// First witness: set fc=[I|0], vary hidden while fixing embedding, and prove output is invariant.
func TestQwen35MTPFuseEmbeddingFirstWitness(t *testing.T) {
	// h=2, fc is [2, 4]: [I_2 | 0_2]
	// First 2 columns are identity [1 0; 0 1], second 2 columns are [0 0; 0 0]
	fcData := []float32{
		1, 0, 0, 0, // row 0: selects embedding[0]
		0, 1, 0, 0, // row 1: selects embedding[1]
	}
	m := qwen35MTPForwardTestModel(t, map[string]testMTPWeight{
		"mtp.pre_fc_norm_hidden.weight":    {shape: []int{2}, data: []float32{1, 1}},
		"mtp.pre_fc_norm_embedding.weight": {shape: []int{2}, data: []float32{1, 1}},
		"mtp.fc.weight":                    {shape: []int{2, 4}, data: fcData},
	})

	fixedEmbedding := []float32{2.0, 3.0}

	// Baseline with hidden = [1.0, 1.0]
	baseOut, err := m.Qwen35MTPFuse([]float32{1.0, 1.0}, fixedEmbedding)
	if err != nil {
		t.Fatalf("baseline fuse failed: %v", err)
	}

	// Vary hidden across multiple vectors: output must remain identical to baseOut
	hiddenVariations := [][]float32{
		{10.0, -5.0},
		{0.0, 100.0},
		{-20.0, -30.0},
		{55.5, 12.3},
	}

	for i, hidden := range hiddenVariations {
		got, err := m.Qwen35MTPFuse(hidden, fixedEmbedding)
		if err != nil {
			t.Fatalf("variation %d fuse failed: %v", i, err)
		}
		for dim := range baseOut {
			if math.Abs(float64(got[dim]-baseOut[dim])) > 1e-6 {
				t.Fatalf("variation %d dim %d: got %v, want %v (output not invariant to hidden under [I|0])",
					i, dim, got[dim], baseOut[dim])
			}
		}
	}
}
