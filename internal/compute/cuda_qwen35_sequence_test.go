//go:build cuda

package compute

import (
	"errors"
	"testing"
)

func qwen35SequenceGeometryFixture(layers int) Qwen35SequencePrefillRequest {
	req := Qwen35SequencePrefillRequest{
		Path: Qwen35SequencePrefillPath, TokenIDs: []int{1, 2},
		Hidden: Qwen35DenseHidden, Intermediate: Qwen35DenseIntermediate,
		NumHeads: Qwen35DenseQueryHeads, NumKVHeads: Qwen35DenseKVHeads,
		HeadDim: Qwen35DenseHeadDim, RotaryDim: Qwen35DenseHeadDim / 4,
		NumKeyHeads: Qwen35DenseGDNGroups, NumValueHeads: Qwen35DenseGDNRank,
		KeyHeadDim: Qwen35DenseGDNState, ValueHeadDim: Qwen35DenseGDNState,
		ConvKernel: Qwen35DenseGDNConv, RMSNormEpsilon: 1e-6,
		Layers: make([]Qwen35SequenceLayer, layers), States: make([]Qwen35SequenceState, layers),
		RoPEThetaForLayer: make([]float64, layers),
	}
	for layer := range req.Layers {
		req.Layers[layer].Linear = (layer+1)%4 != 0
		req.RoPEThetaForLayer[layer] = 1e7
	}
	return req
}

func TestQwen35SequenceProductionGeometryContract(t *testing.T) {
	req := qwen35SequenceGeometryFixture(Qwen35DenseMainLayers)
	if err := validateQwen35SequenceGeometry(req); err != nil {
		t.Fatalf("exact dense Qwen3.8 geometry refused: %v", err)
	}
	for name, mutate := range map[string]func(*Qwen35SequencePrefillRequest){
		"metadata layer included": func(r *Qwen35SequencePrefillRequest) {
			r.Layers = append(r.Layers, Qwen35SequenceLayer{})
			r.States = append(r.States, Qwen35SequenceState{})
			r.RoPEThetaForLayer = append(r.RoPEThetaForLayer, 1e7)
		},
		"wrong full-attention cadence": func(r *Qwen35SequencePrefillRequest) { r.Layers[2].Linear = false },
		"wrong hidden size":            func(r *Qwen35SequencePrefillRequest) { r.Hidden-- },
		"wrong GDN rank":               func(r *Qwen35SequencePrefillRequest) { r.NumValueHeads-- },
	} {
		t.Run(name, func(t *testing.T) {
			bad := qwen35SequenceGeometryFixture(Qwen35DenseMainLayers)
			mutate(&bad)
			var contractErr *Qwen35SequenceError
			if err := validateQwen35SequenceGeometry(bad); err == nil || !errors.As(err, &contractErr) {
				t.Fatalf("malformed production geometry error = %v, want typed refusal", err)
			}
		})
	}
}

func TestQwen35SequenceBoundedFixtureGeometry(t *testing.T) {
	req := qwen35SequenceGeometryFixture(4)
	req.Hidden, req.Intermediate = 32, 64
	req.NumHeads, req.NumKVHeads, req.HeadDim, req.RotaryDim = 4, 2, 8, 8
	req.NumKeyHeads, req.NumValueHeads = 2, 4
	req.KeyHeadDim, req.ValueHeadDim, req.ConvKernel = 8, 8, 3
	if err := validateQwen35SequenceGeometry(req); err != nil {
		t.Fatalf("bounded parity geometry refused: %v", err)
	}
}
