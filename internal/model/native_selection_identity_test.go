package model

import (
	"encoding/json"
	"regexp"
	"testing"
)

func nativeSelectionIdentityFixture() NativeSelectionIdentity {
	return NativeSelectionIdentity{
		Schema:              NativeSelectionIdentitySchemaV1,
		ModelRef:            "qwen3.8-27b-q4_k_m",
		Backend:             "metal",
		ForwardPath:         "metal/session-forward",
		Quantization:        NativeSelectionQuantizationQ4K,
		PrefillChunkTokens:  4096,
		CPUOffloadExperts:   8,
		Q4KGateUpOutputSlab: true,
	}
}

func TestNativeSelectionIdentityDeterministic(t *testing.T) {
	want := nativeSelectionIdentityFixture()
	firstJSON, err := want.CanonicalJSON()
	if err != nil {
		t.Fatal(err)
	}
	firstDigest, err := want.Digest()
	if err != nil {
		t.Fatal(err)
	}
	for range 32 {
		gotJSON, err := (NativeSelectionIdentity{
			Schema: want.Schema, ModelRef: want.ModelRef, Backend: want.Backend,
			ForwardPath: want.ForwardPath, Quantization: want.Quantization,
			PrefillChunkTokens: want.PrefillChunkTokens, CPUOffloadExperts: want.CPUOffloadExperts,
			Q4KGateUpOutputSlab: want.Q4KGateUpOutputSlab,
		}).CanonicalJSON()
		if err != nil {
			t.Fatal(err)
		}
		gotDigest, err := want.Digest()
		if err != nil {
			t.Fatal(err)
		}
		if string(gotJSON) != string(firstJSON) || gotDigest != firstDigest {
			t.Fatalf("canonical identity drifted: json=%s digest=%s", gotJSON, gotDigest)
		}
	}
}

func TestNativeSelectionIdentityGoldenV1(t *testing.T) {
	id := nativeSelectionIdentityFixture()
	gotJSON, err := id.CanonicalJSON()
	if err != nil {
		t.Fatal(err)
	}
	const wantJSON = `{"schema":"fak.kernel-selection/v1","model_ref":"qwen3.8-27b-q4_k_m","backend":"metal","forward_path":"metal/session-forward","quantization":"Q4_K","prefill_chunk_tokens":4096,"cpu_offload_experts":8,"q4k_gate_up_output_slab":true}`
	if string(gotJSON) != wantJSON {
		t.Fatalf("canonical JSON = %s, want %s", gotJSON, wantJSON)
	}
	gotDigest, err := id.Digest()
	if err != nil {
		t.Fatal(err)
	}
	const wantDigest = "sha256:fb8478bc011bc12d9606d30602f23c4ebc0610b92a89e436058e02c0e944c9bd"
	if gotDigest != wantDigest {
		t.Fatalf("digest = %q, want %q", gotDigest, wantDigest)
	}
}

func TestNativeSelectionIdentityFunctionalAxisMutation(t *testing.T) {
	base := nativeSelectionIdentityFixture()
	baseDigest, err := base.Digest()
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		mutate func(*NativeSelectionIdentity)
	}{
		{"model_ref", func(id *NativeSelectionIdentity) { id.ModelRef = "qwen3.8-27b-q4_k_s" }},
		{"backend", func(id *NativeSelectionIdentity) { id.Backend = "cuda" }},
		{"forward_path", func(id *NativeSelectionIdentity) { id.ForwardPath = "device/generic" }},
		{"quantization", func(id *NativeSelectionIdentity) {
			id.Quantization = NativeSelectionQuantizationQ8_0
			id.PrefillChunkTokens = 0
			id.Q4KGateUpOutputSlab = false
		}},
		{"prefill_chunk_tokens", func(id *NativeSelectionIdentity) { id.PrefillChunkTokens = 2048 }},
		{"cpu_offload_experts", func(id *NativeSelectionIdentity) { id.CPUOffloadExperts = 9 }},
		{"q4k_gate_up_output_slab", func(id *NativeSelectionIdentity) { id.Q4KGateUpOutputSlab = false }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mutated := base
			tt.mutate(&mutated)
			before, _ := json.Marshal(base)
			after, _ := json.Marshal(mutated)
			if string(before) == string(after) {
				t.Fatal("mutation was vacuous")
			}
			got, err := mutated.Digest()
			if err != nil {
				t.Fatal(err)
			}
			if got == baseDigest {
				t.Fatalf("mutation retained digest %q", got)
			}
		})
	}
}

func TestNativeSelectionIdentityRejectsIncompleteRecord(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*NativeSelectionIdentity)
	}{
		{"unknown schema", func(id *NativeSelectionIdentity) { id.Schema = "v2" }},
		{"empty model", func(id *NativeSelectionIdentity) { id.ModelRef = "" }},
		{"empty backend", func(id *NativeSelectionIdentity) { id.Backend = "" }},
		{"empty path", func(id *NativeSelectionIdentity) { id.ForwardPath = "" }},
		{"unknown quant", func(id *NativeSelectionIdentity) { id.Quantization = "INT3" }},
		{"negative chunk", func(id *NativeSelectionIdentity) { id.PrefillChunkTokens = -1 }},
		{"negative offload", func(id *NativeSelectionIdentity) { id.CPUOffloadExperts = -1 }},
		{"non-q4k chunk", func(id *NativeSelectionIdentity) { id.Quantization = NativeSelectionQuantizationQ8_0 }},
		{"non-q4k slab", func(id *NativeSelectionIdentity) {
			id.Quantization = NativeSelectionQuantizationF32
			id.PrefillChunkTokens = 0
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			id := nativeSelectionIdentityFixture()
			tt.mutate(&id)
			if digest, err := id.Digest(); err == nil || digest != "" {
				t.Fatalf("Digest() = %q, %v; want empty digest and error", digest, err)
			}
		})
	}
}

func TestNativeSelectionIdentityDigestFormat(t *testing.T) {
	digest, err := nativeSelectionIdentityFixture().Digest()
	if err != nil {
		t.Fatal(err)
	}
	if !regexp.MustCompile(`^sha256:[0-9a-f]{64}$`).MatchString(digest) {
		t.Fatalf("digest = %q, want canonical lowercase SHA-256", digest)
	}
}

func TestNativeSelectionIdentityIgnoresReceiptPresentation(t *testing.T) {
	id := nativeSelectionIdentityFixture()
	first := NativeInferenceReceipt{NativeSelection: id, PrefillSeconds: 1, TokenIDs: []int{1}, TokenLogprobs: []float64{-0.1}}
	second := NativeInferenceReceipt{NativeSelection: id, PrefillSeconds: 99, DecodeSeconds: 12, TokenIDs: []int{9, 8}, TokenLogprobs: []float64{-9, -8}}
	firstDigest, err := first.NativeSelection.Digest()
	if err != nil {
		t.Fatal(err)
	}
	secondDigest, err := second.NativeSelection.Digest()
	if err != nil {
		t.Fatal(err)
	}
	if firstDigest != secondDigest {
		t.Fatalf("receipt presentation changed selection identity: %s != %s", firstDigest, secondDigest)
	}
}
