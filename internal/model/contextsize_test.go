package model

import (
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func qwen35ContextPlanConfig() Config {
	return Config{
		NumLayers:             4,
		NumHeads:              4,
		NumKVHeads:            2,
		HeadDim:               8,
		HiddenSize:            32,
		IntermediateSize:      64,
		VocabSize:             128,
		MaxPositionEmbeddings: 65536,
		LayerTypes:            []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim:   3,
		LinearNumKeyHeads:     2,
		LinearKeyHeadDim:      4,
		LinearNumValueHeads:   3,
		LinearValueHeadDim:    4,
	}
}

func TestContextSizeConfigQwen35ChargesCompactKVAndFixedState(t *testing.T) {
	cfg := qwen35ContextPlanConfig()
	got := cfg.ContextSizeConfig()
	if got.KV.NumLayers != 1 {
		t.Fatalf("planned token-indexed KV layers = %d, want one full-attention layer", got.KV.NumLayers)
	}

	// Per recurrent layer: 3*4*4 recurrent floats plus two conv rows of
	// (2*(2*4) + 3*4) floats. Three recurrent layers, four bytes per float.
	const wantFixed = int64(3 * (3*4*4 + 2*(2*(2*4)+3*4)) * 4)
	if len(got.SessionState) != 1 {
		t.Fatalf("session-state plan = %#v, want one fixed recurrent/conv demand", got.SessionState)
	}
	d := got.SessionState[0]
	if d.Class != compute.MemoryKVCache || d.Bytes != wantFixed || d.Detail != "qwen35-gdn-recurrent-conv-state" || d.DType != compute.F32.String() {
		t.Fatalf("session-state demand = %#v, want %d-byte f32 Qwen recurrent/conv state", d, wantFixed)
	}

	const tokens = 65536
	const wantKV = int64(tokens * 1 * 2 * 8 * 3 * 4)
	plan := got.PerContextMemoryPlan(tokens)
	if kv := plan.ByClass()[compute.MemoryKVCache]; kv != wantKV+wantFixed {
		t.Fatalf("Qwen KV-class bytes = %d, want compact K/Kraw/V %d + fixed state %d", kv, wantKV, wantFixed)
	}
	uniform := int64(tokens * cfg.NumLayers * cfg.NumKVHeads * cfg.HeadDim * 3 * 4)
	if plan.ByClass()[compute.MemoryKVCache] >= uniform {
		t.Fatalf("hybrid plan still charges uniform token KV: got %d, old uniform %d", plan.ByClass()[compute.MemoryKVCache], uniform)
	}
}

func TestContextSizeConfigDenseModelRemainsIdentical(t *testing.T) {
	cfg := Config{
		NumLayers: 4, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		HiddenSize: 32, IntermediateSize: 64, VocabSize: 128,
		MaxPositionEmbeddings: 4096, RopeTheta: 10000,
	}
	want := compute.ContextSizeConfig{
		KV: compute.KVConfig{NumLayers: 4, NumKVHeads: 2, HeadDim: 8, RopeTheta: 10000},
		Scratch: compute.TransformerScratchConfig{
			HiddenSize: 32, IntermediateSize: 64, VocabSize: 128,
			NumLayers: 4, NumHeads: 4, NumKVHeads: 2, HeadDim: 8, IncludeLogits: true,
		},
		MaxContext: 4096,
	}
	if got := cfg.ContextSizeConfig(); !reflect.DeepEqual(got, want) {
		t.Fatalf("dense projection changed:\n got %#v\nwant %#v", got, want)
	}
}

func TestContextSizeConfigQwen35OverflowStaysUniform(t *testing.T) {
	cfg := qwen35ContextPlanConfig()
	cfg.LinearNumValueHeads = math.MaxInt
	got := cfg.ContextSizeConfig()
	if got.KV.NumLayers != cfg.NumLayers || got.SessionState != nil {
		t.Fatalf("unprovable fixed state received a discount: KV layers=%d state=%#v, want uniform %d and no fixed demand", got.KV.NumLayers, got.SessionState, cfg.NumLayers)
	}
}
