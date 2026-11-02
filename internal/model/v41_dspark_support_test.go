package model

import (
	"encoding/json"
	"errors"
	"testing"
)

func dsparkSupportHeader(t *testing.T) V41DSparkGGUFHeader {
	t.Helper()
	raw, cfg := readDeepSeekV41Config(t)
	d := cfg.DeepSeekV41.DSpark
	h := V41DSparkGGUFHeader{Architecture: "deepseek41-dspark", TargetArchitecture: "deepseek41", SourceConfig: string(raw), WeightEncoding: "native MXFP4; lossless F16/F32 expansion", TargetLayerIDs: append([]int(nil), d.TargetLayerIDs...), Parameters: map[string]int{"block_size": d.BlockSize, "markov_rank": d.MarkovRank, "noise_token_id": d.NoiseTokenID, "n_routed_experts": d.NumRoutedExperts, "num_experts_per_tok": d.NumExpertsPerTok}}
	inv, err := DeepSeekV41DSparkInventory(cfg)
	if err != nil {
		t.Fatal(err)
	}
	specs, err := dsparkGGUFSpecs(inv, d.NumRoutedExperts)
	if err != nil {
		t.Fatal(err)
	}
	for name, spec := range specs {
		dtype := "F32"
		if spec.AllowedDtypes[0] == "MXFP4" {
			dtype = "MXFP4"
		}
		h.Tensors = append(h.Tensors, V41TensorRef{Name: name, Shape: spec.Shape, Dtype: dtype})
	}
	return h
}

// fak-test:justify why=regression when=changed:internal/model/v41_dspark_support.go
// fak-test:runtime fast est=50ms lane=default
func TestDeepSeekV41DSparkSupportMetadata(t *testing.T) {
	// Full independently spelled destination inventory is exercised through
	// ggufload.File.Config; this table isolates config and metadata invariants.
	h := dsparkSupportHeader(t)
	cfg, err := ValidateV41DSparkGGUFHeader(h)
	if err != nil {
		t.Fatal(err)
	}
	if !errors.Is(cfg.RefuseDeepSeekV41DSpark(), ErrV41DSparkUnsupported) {
		t.Fatal("header validation enabled execution")
	}
	for _, tc := range []struct {
		name   string
		change func(*V41DSparkGGUFHeader)
	}{
		{"architecture", func(h *V41DSparkGGUFHeader) { h.Architecture = "deepseek41" }},
		{"target", func(h *V41DSparkGGUFHeader) { h.TargetArchitecture = "deepseek2" }},
		{"encoding", func(h *V41DSparkGGUFHeader) { h.WeightEncoding = "F16" }},
		{"missing scalar", func(h *V41DSparkGGUFHeader) { delete(h.Parameters, "noise_token_id") }},
		{"extra scalar", func(h *V41DSparkGGUFHeader) { h.Parameters["claimed_lossless"] = 1 }},
		{"scalar disagreement", func(h *V41DSparkGGUFHeader) { h.Parameters["markov_rank"]++ }},
		{"repeated tap", func(h *V41DSparkGGUFHeader) { h.TargetLayerIDs[1] = h.TargetLayerIDs[0] }},
		{"source syntax", func(h *V41DSparkGGUFHeader) { h.SourceConfig = "{" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := dsparkSupportHeader(t)
			tc.change(&h)
			if _, err := ValidateV41DSparkGGUFHeader(h); !errors.Is(err, ErrV41DSparkArtifact) {
				t.Fatalf("invalid header accepted: %v", err)
			}
		})
	}
	for _, tc := range []struct {
		name  string
		key   string
		value any
		top   bool
	}{
		{"huge layer allocation", "num_hidden_layers", 1 << 30, false},
		{"top-level layer override", "num_hidden_layers", 1 << 30, true},
		{"top-level hidden override", "hidden_size", 1024, true},
		{"wrong stage count", "num_nextn_predict_layers", 2, false},
		{"wrong block", "dspark_block_size", 6, false},
		{"window too small", "sliding_window", 4, false},
		{"out of range noise", "dspark_noise_token_id", 129280, false},
		{"expert inequality", "dspark_num_experts_per_tok", 129, false},
		{"duplicate source taps", "dspark_target_layer_ids", []int{37, 37, 39}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := dsparkSupportHeader(t)
			var source map[string]any
			if err := json.Unmarshal([]byte(h.SourceConfig), &source); err != nil {
				t.Fatal(err)
			}
			target := source
			if !tc.top {
				target = source["text_config"].(map[string]any)
			}
			target[tc.key] = tc.value
			raw, err := json.Marshal(source)
			if err != nil {
				t.Fatal(err)
			}
			h.SourceConfig = string(raw)
			if _, err := ValidateV41DSparkGGUFHeader(h); !errors.Is(err, ErrV41DSparkArtifact) {
				t.Fatalf("invalid source config accepted: %v", err)
			}
		})
	}
}
