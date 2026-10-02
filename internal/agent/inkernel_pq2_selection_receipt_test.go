package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// The resident session flag is shared by Q4_K and Q2_0. The wire selection
// must identify the model's actual weights and distinguish Prism's signed,
// rotated PQ2_0 contract from an ordinary unrotated Q2_0 resident tensor.
func TestNativeSelectionReceiptUsesModelOwnedPQ2Identity(t *testing.T) {
	const weight = "model.layers.0.mlp.up_proj.weight"
	newQ2Model := func(prism bool) *model.Model {
		t.Helper()
		b := model.NewQuantBuilder(model.Config{ModelType: "qwen35", HiddenSize: 128, IntermediateSize: 128, NumLayers: 1}, false)
		if err := b.AddResidentQ2(weight, []int{1, 128}, make([]byte, 34)); err != nil {
			t.Fatal(err)
		}
		m, err := b.Build()
		if err != nil {
			t.Fatal(err)
		}
		if m.Q2Count() != 1 || m.Q4KCount() != 0 {
			t.Fatalf("Q2 fixture counts = %d Q2, %d Q4K", m.Q2Count(), m.Q4KCount())
		}
		if prism {
			signs := make([]int, 128)
			for i := range signs {
				signs[i] = 1
			}
			if err := m.SetPrismHadamard(model.PrismHadamardSpec{
				BlockSize: 4, SignWidths: []int{128}, SignValues: signs,
				WeightNames: []string{weight},
			}); err != nil {
				t.Fatal(err)
			}
		}
		return m
	}
	newQ4KModel := func() *model.Model {
		t.Helper()
		b := model.NewQuantBuilder(model.Config{ModelType: "qwen35", HiddenSize: 256, IntermediateSize: 256, NumLayers: 1}, false)
		if err := b.AddResidentQ4K(weight, []int{1, 256}, make([]byte, 144)); err != nil {
			t.Fatal(err)
		}
		m, err := b.Build()
		if err != nil {
			t.Fatal(err)
		}
		if m.Q2Count() != 0 || m.Q4KCount() != 1 {
			t.Fatalf("Q4K fixture counts = %d Q2, %d Q4K", m.Q2Count(), m.Q4KCount())
		}
		return m
	}

	cases := []struct {
		name  string
		m     *model.Model
		quant string
	}{
		{"prism PQ2", newQ2Model(true), "PQ2_0"},
		{"plain Q2", newQ2Model(false), "Q2_0"},
		{"Q4K control", newQ4KModel(), "Q4_K"},
	}
	digests := make(map[string]string, len(cases))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := NewInKernelPlanner(tc.m, nil, "same-model-ref", true, nil, false)
			receipt := p.buildNativeInferenceReceipt(&nativeInferenceMeasurement{
				tokenIDs: []int{7}, logprobs: []float64{-0.25},
			}, 0, 0)
			if got := receipt.NativeSelection.Quantization; got != tc.quant {
				t.Fatalf("kernel_selection.quantization = %q, want %q", got, tc.quant)
			}
			if err := receipt.NativeSelection.Validate(); err != nil {
				t.Fatalf("selection identity invalid: %v", err)
			}
			canonical, err := receipt.NativeSelection.CanonicalJSON()
			if err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(canonical)
			wantDigest := "sha256:" + hex.EncodeToString(sum[:])
			if receipt.NativeSelectionDigest != wantDigest {
				t.Fatalf("selection digest = %q, want %q", receipt.NativeSelectionDigest, wantDigest)
			}
			raw, err := json.Marshal(receipt)
			if err != nil {
				t.Fatal(err)
			}
			var wire struct {
				Selection model.NativeSelectionIdentity `json:"kernel_selection"`
				Digest    string                        `json:"kernel_selection_digest"`
			}
			if err := json.Unmarshal(raw, &wire); err != nil {
				t.Fatal(err)
			}
			if wire.Selection.Quantization != tc.quant || wire.Digest != wantDigest {
				t.Fatalf("wire selection = %q / %q, want %q / %q", wire.Selection.Quantization, wire.Digest, tc.quant, wantDigest)
			}
			digests[tc.quant] = wantDigest
		})
	}
	if len(digests) != len(cases) {
		t.Fatalf("distinct quantization axes produced %d digests, want %d", len(digests), len(cases))
	}
}
