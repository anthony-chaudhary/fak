package bitnetmeta

import (
	"testing"
)

// Invariant: Capabilities must retain declared schema versions and not permit empty activation definitions.
// Guard: descriptor validation rejects empty capabilities and mismatched schemas.

func TestBitnetMetaLifecycle(t *testing.T) {
	t.Parallel()

	caps := Capabilities{
		Schemas:     []string{SchemaV1},
		Formats:     []string{"safetensors@1"},
		Activations: []string{"integer/8"},
		Packings:    []string{"bitplane-lsb"},
		Recipes:     []string{"native-bitnet@1"},
		Runtimes:    []string{"bitnet.cpp@2026.08"},
		Hardware:    []string{"cpu/x86-64-avx2"},
	}

	raw := []byte(`{
		"schema": "bitnetmeta/v1",
		"artifact": {
			"id": "test-model-v1",
			"format": "safetensors",
			"version": "1",
			"origin": "native-trained"
		},
		"weights": {
			"semantic": "native-ternary-1.58bit",
			"label": "1.58-bit",
			"levels": [-1, 0, 1]
		},
		"activation": {
			"format": "integer",
			"bits": 8
		},
		"packing": {
			"scheme": "bitplane-lsb",
			"storage_bits": 8,
			"values_per_unit": 4
		},
		"recipe": {
			"id": "native-bitnet",
			"version": "1",
			"kind": "native-training"
		},
		"runtime": {
			"id": "bitnet.cpp",
			"version": "2026.08"
		},
		"hardware": {
			"measured": false
		}
	}`)

	adjudication := ParseAndAdjudicate(raw, caps)
	if adjudication.Outcome != OutcomeAccept {
		t.Fatalf("expected accept outcome, got %s: %s (%s)", adjudication.Outcome, adjudication.Reason, adjudication.Detail)
	}
	if adjudication.Descriptor == nil {
		t.Fatal("expected non-nil descriptor on accept")
	}
	if adjudication.Descriptor.Weights.Semantic != WeightNativeTernary {
		t.Fatalf("expected WeightNativeTernary, got %s", adjudication.Descriptor.Weights.Semantic)
	}
}
