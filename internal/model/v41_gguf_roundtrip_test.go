package model

import (
	"encoding/json"
	"errors"
	"testing"
)

// Canonical GGUF transport is separate from official checkpoint admission.
// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=10ms lane=default
func TestDeepSeekV41GGUFJSONIdentityBoundary(t *testing.T) {
	raw, _ := readDeepSeekV41Config(t)
	for _, mutate := range []struct {
		name string
		edit func(map[string]any)
	}{
		{"GGUF root with official text", func(d map[string]any) { d["model_type"] = "deepseek41" }},
		{"official root with GGUF text", func(d map[string]any) { d["text_config"].(map[string]any)["model_type"] = "deepseek41" }},
		{"official pair with GGUF quantization", func(d map[string]any) {
			d["quantization_config"] = map[string]any{"quant_method": "gguf", "gguf_type": "Q2_K"}
		}},
		{"official pair without Engram", func(d map[string]any) { delete(d["text_config"].(map[string]any), "engram_layer_ids") }},
	} {
		t.Run(mutate.name, func(t *testing.T) {
			var doc map[string]any
			if err := json.Unmarshal(raw, &doc); err != nil {
				t.Fatal(err)
			}
			mutate.edit(doc)
			encoded, err := json.Marshal(doc)
			if err != nil {
				t.Fatal(err)
			}
			var cfg Config
			if err := json.Unmarshal(encoded, &cfg); !errors.Is(err, ErrV41ConfigAdmission) {
				t.Fatalf("error=%v, want unchanged official admission refusal", err)
			}
		})
	}
	t.Run("legacy nil envelope", func(t *testing.T) {
		var cfg Config
		if err := json.Unmarshal([]byte(`{"model_type":"deepseek41","hidden_size":32,"num_hidden_layers":3,"num_attention_heads":1,"num_key_value_heads":1}`), &cfg); err != nil {
			t.Fatal(err)
		}
		if cfg.DeepSeekV41 != nil || !cfg.IsDeepSeekV41() {
			t.Fatal("legacy canonical GGUF changed its nil-envelope contract")
		}
	})
	t.Run("unset candidate and complete transport", func(t *testing.T) {
		cfg := Config{ModelType: "deepseek41", NumLayers: 3, HiddenSize: 32, NumHeads: 1, NumKVHeads: 1, NumNextNPredictLayers: 2,
			CompressRatios: []int{0, 1, 1}, IndexerTypes: []string{"dense", "full", "shared"}}
		var err error
		cfg.DeepSeekV41, err = NewDeepSeekV41GGUFConfig(cfg, V41GGUFTopology{})
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(cfg)
		if err != nil {
			t.Fatal(err)
		}
		var got Config
		if err := json.Unmarshal(encoded, &got); err != nil || got.DeepSeekV41 == nil || got.DeepSeekV41.CandidateSourceLayerID != -1 {
			t.Fatalf("unset candidate did not survive: cfg=%+v err=%v", got.DeepSeekV41, err)
		}
		if got.NumNextNPredictLayers != 2 || got.DeepSeekV41.DSpark == nil || got.DeepSeekV41.DSpark.NextNPredictLayers != 2 {
			t.Fatal("flat next-token prediction marker was lost")
		}
		for _, null := range []bool{false, true} {
			var doc map[string]json.RawMessage
			if err := json.Unmarshal(encoded, &doc); err != nil {
				t.Fatal(err)
			}
			if null {
				doc["candidate_source_layer_id"] = json.RawMessage("null")
			} else {
				delete(doc, "candidate_source_layer_id")
			}
			bad, err := json.Marshal(doc)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(bad, &got); !errors.Is(err, ErrV41ConfigAdmission) {
				t.Fatalf("missing/null candidate silently became layer zero: %v", err)
			}
		}
		cfg.DeepSeekV41.WrapperModelType = "deepseek_v41"
		if _, err := json.Marshal(cfg); !errors.Is(err, ErrV41ConfigAdmission) {
			t.Fatalf("hybrid in-memory identity admitted: %v", err)
		}
	})
}
