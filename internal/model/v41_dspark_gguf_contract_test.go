package model

import (
	"errors"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/model/v41_dspark.go
// fak-test:runtime fast est=50ms lane=default

// TestDSparkGGUFContractAdmission pins the fail-closed admission contract for
// a V4.1 config that requests the DSpark decoder while carrying a
// support-GGUF artifact declaration: with no declaration the refusal is
// byte-for-byte unchanged; a valid declaration still refuses execution with
// the unsupported sentinel; a violated declaration (dspark.* key set, strict
// dimension validation, unclaimed-tensor refusal) is refused with the typed
// artifact sentinel instead. No arm admits speculative execution.
func TestDSparkGGUFContractAdmission(t *testing.T) {
	t.Run("no_declaration_unchanged", func(t *testing.T) {
		_, cfg := readDeepSeekV41Config(t)
		err := cfg.RefuseDeepSeekV41DSpark()
		if !errors.Is(err, ErrV41DSparkUnsupported) || errors.Is(err, ErrV41DSparkArtifact) {
			t.Fatalf("refusal = %v, want ErrV41DSparkUnsupported only", err)
		}
	})

	t.Run("valid_artifact_still_refused", func(t *testing.T) {
		_, cfg := readDeepSeekV41Config(t)
		h := dsparkSupportHeader(t)
		cfg.DeepSeekV41.DSpark.GGUF = &h
		err := cfg.RefuseDeepSeekV41DSpark()
		if !errors.Is(err, ErrV41DSparkUnsupported) || errors.Is(err, ErrV41DSparkArtifact) {
			t.Fatalf("refusal = %v, want ErrV41DSparkUnsupported only", err)
		}
	})

	for _, tc := range []struct {
		name   string
		change func(*V41DSparkGGUFHeader)
	}{
		{"architecture", func(h *V41DSparkGGUFHeader) { h.Architecture = "deepseek41" }},
		{"target_architecture", func(h *V41DSparkGGUFHeader) { h.TargetArchitecture = "deepseek2" }},
		{"weight_encoding", func(h *V41DSparkGGUFHeader) { h.WeightEncoding = "F16" }},
		{"missing_scalar", func(h *V41DSparkGGUFHeader) { delete(h.Parameters, "noise_token_id") }},
		{"unknown_scalar", func(h *V41DSparkGGUFHeader) { h.Parameters["claimed_lossless"] = 1 }},
		{"scalar_disagreement", func(h *V41DSparkGGUFHeader) { h.Parameters["markov_rank"]++ }},
		{"repeated_tap", func(h *V41DSparkGGUFHeader) { h.TargetLayerIDs[1] = h.TargetLayerIDs[0] }},
		{"source_syntax", func(h *V41DSparkGGUFHeader) { h.SourceConfig = "{" }},
		{"unclaimed_tensor", func(h *V41DSparkGGUFHeader) {
			h.Tensors = append(h.Tensors, V41TensorRef{Name: "mtp.9.unclaimed.weight", Shape: []int{1}, Dtype: "F32"})
		}},
		{"duplicate_tensor", func(h *V41DSparkGGUFHeader) {
			h.Tensors = append(h.Tensors, h.Tensors[len(h.Tensors)-1])
		}},
		{"missing_tensor", func(h *V41DSparkGGUFHeader) {
			h.Tensors = h.Tensors[:len(h.Tensors)-1]
		}},
		{"wrong_dtype", func(h *V41DSparkGGUFHeader) {
			for i := range h.Tensors {
				if h.Tensors[i].Dtype == "MXFP4" {
					h.Tensors[i].Dtype = "F32"
					return
				}
			}
			t.Fatal("fixture lost its MXFP4 expert tensors")
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, cfg := readDeepSeekV41Config(t)
			h := dsparkSupportHeader(t)
			tc.change(&h)
			cfg.DeepSeekV41.DSpark.GGUF = &h
			err := cfg.RefuseDeepSeekV41DSpark()
			if !errors.Is(err, ErrV41DSparkArtifact) || errors.Is(err, ErrV41DSparkUnsupported) {
				t.Fatalf("refusal = %v, want ErrV41DSparkArtifact only", err)
			}
		})
	}
}
