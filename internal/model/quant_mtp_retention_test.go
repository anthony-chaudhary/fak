package model

import (
	"reflect"
	"testing"
)

func TestQuantBuilderMTPRetentionIsPerBuild(t *testing.T) {
	globalBefore := RetainMTP
	cfg := Config{
		ModelType: "qwen35", HiddenSize: 256, VocabSize: 1, NumLayers: 1,
		NumHeads: 4, NumKVHeads: 2, HeadDim: 64,
		NumNextNPredictLayers: 1, LayerTypes: []string{"linear_attention"},
	}
	builders := []*QuantBuilder{NewQuantBuilder(cfg, false), NewQuantBuilder(cfg, false), NewQuantBuilder(cfg, false)}
	wantRetention := []bool{true, false, globalBefore}
	for i, builder := range builders[:2] {
		// The legacy fallback lets the pre-option source reach the tensor-store
		// assertion, so RED demonstrates the global-only behavior, not a compile error.
		if configurable, ok := any(builder).(interface{ SetMTPRetention(bool) error }); ok {
			if err := configurable.SetMTPRetention(wantRetention[i]); err != nil {
				t.Fatal(err)
			}
		}
	}

	const norm = "mtp.norm.weight"
	const q8 = "mtp.layers.0.mlp.gate_proj.weight"
	const q4 = "mtp.layers.0.self_attn.o_proj.weight"
	const q6 = "mtp.layers.0.mlp.down_proj.weight"
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(i+1) / 256
	}
	// Interleave identical inputs after all policies are configured. A setter
	// implemented by changing the global flag cannot isolate these builders.
	for _, builder := range builders {
		if err := builder.AddF32Tensor("lm_head.weight", []int{1, 256}, values); err != nil {
			t.Fatal(err)
		}
		if err := builder.AddF32Tensor(norm, []int{256}, values); err != nil {
			t.Fatal(err)
		}
		if err := builder.AddF32Tensor(q8, []int{1, 256}, values); err != nil {
			t.Fatal(err)
		}
		if err := builder.AddResidentQ4K(q4, []int{1, 256}, makeTestQ4KRaw(1, 256)); err != nil {
			t.Fatal(err)
		}
		if err := builder.AddResidentQ6K(q6, []int{1, 256}, make([]byte, kindQ6K.blockBytes())); err != nil {
			t.Fatal(err)
		}
	}
	models := make([]*Model, len(builders))
	for i, builder := range builders {
		m, err := builder.Build()
		if err != nil {
			t.Fatal(err)
		}
		models[i] = m
		t.Cleanup(func() { _ = m.CloseWeights() })
		_, hasNorm := m.manifest[norm]
		_, hasQ8 := m.q8w[q8]
		_, hasQ4 := m.q4kw[q4]
		_, hasQ6 := m.kqw[q6]
		for format, present := range map[string]bool{"F32": hasNorm, "Q8": hasQ8, "Q4_K": hasQ4, "Q6_K": hasQ6} {
			if present != wantRetention[i] {
				t.Errorf("builder %d retained %s MTP tensor=%t want=%t (unchanged global=%t)", i, format, present, wantRetention[i], globalBefore)
			}
		}
		if m.q8w["lm_head.weight"] == nil {
			t.Errorf("builder %d lost target head", i)
		}
	}
	for i := 1; i < len(models); i++ {
		if !reflect.DeepEqual(models[0].q8w["lm_head.weight"], models[i].q8w["lm_head.weight"]) {
			t.Errorf("builder %d changed target weights under MTP policy", i)
		}
	}
	if RetainMTP != globalBefore {
		t.Errorf("builder policy changed global RetainMTP: got=%t want=%t", RetainMTP, globalBefore)
	}
}
