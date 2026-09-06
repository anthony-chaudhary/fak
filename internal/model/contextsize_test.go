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

func qwen38ContextPlanConfig() Config {
	types := make([]string, 64)
	for i := range types {
		if i%4 == 3 {
			types[i] = "full_attention"
		} else {
			types[i] = "linear_attention"
		}
	}
	return Config{
		ModelType:             "qwen3_5_text",
		NumLayers:             64,
		NumHeads:              24,
		NumKVHeads:            4,
		HeadDim:               256,
		HiddenSize:            5120,
		IntermediateSize:      17408,
		VocabSize:             256,
		MaxPositionEmbeddings: 20000,
		LayerTypes:            types,
		LinearConvKernelDim:   4,
		LinearNumKeyHeads:     16,
		LinearKeyHeadDim:      128,
		LinearNumValueHeads:   48,
		LinearValueHeadDim:    128,
	}
}

func TestContextSizeConfigQwen38_20kContext(t *testing.T) {
	cfg := qwen38ContextPlanConfig()
	got := cfg.ContextSizeConfig()

	// 1. MaxPositionEmbeddings = 20000
	if got.MaxContext != 20000 {
		t.Fatalf("MaxContext = %d, want 20000", got.MaxContext)
	}

	// 2. KV layer count is exactly 16 (not 64)
	const wantFullLayers = 16
	if got.KV.NumLayers != wantFullLayers {
		t.Fatalf("planned token-indexed KV layers = %d, want %d (16 full_attention layers out of 64)", got.KV.NumLayers, wantFullLayers)
	}

	// 3. Recurrent state is fixed O(1) size (independent of context length)
	const wantLinearLayers = 48
	const wantRecurrent = int64(wantLinearLayers * 48 * 128 * 128 * 4) // 150,994,944 bytes
	const convDim = 2*(16*128) + (48*128)                              // 10,240 floats
	const wantConv = int64(wantLinearLayers * (4 - 1) * convDim * 4)  // 5,898,240 bytes
	const wantFixed = wantRecurrent + wantConv                         // 156,893,184 bytes

	if len(got.SessionState) != 1 {
		t.Fatalf("session-state plan = %#v, want one fixed recurrent/conv demand", got.SessionState)
	}
	d := got.SessionState[0]
	if d.Class != compute.MemoryKVCache || d.Bytes != wantFixed || d.Detail != "qwen35-gdn-recurrent-conv-state" || d.DType != compute.F32.String() {
		t.Fatalf("session-state demand = %#v, want %d-byte f32 Qwen recurrent/conv state", d, wantFixed)
	}

	// Verify O(1) scaling: recurrent/conv session state bytes do not change across token counts.
	for _, tok := range []int{0, 100, 1000, 10000, 20000} {
		p := got.PerContextMemoryPlan(tok)
		var foundState bool
		for _, dem := range p {
			if dem.Detail == "qwen35-gdn-recurrent-conv-state" {
				if dem.Bytes != wantFixed {
					t.Fatalf("tokens=%d: recurrent state bytes = %d, want fixed %d", tok, dem.Bytes, wantFixed)
				}
				foundState = true
			}
		}
		if !foundState {
			t.Fatalf("tokens=%d: missing fixed recurrent state in plan", tok)
		}
	}

	// 4. Total memory plan at 20,000 tokens correctly projects KV bytes and recurrent state bytes.
	const tokens = 20000
	const wantKV = int64(tokens * wantFullLayers * 4 * 256 * 3 * 4) // 3,932,160,000 bytes
	plan := got.PerContextMemoryPlan(tokens)
	kvClassBytes := plan.ByClass()[compute.MemoryKVCache]
	if kvClassBytes != wantKV+wantFixed {
		t.Fatalf("KV-class bytes at %d tokens = %d, want compact KV %d + fixed %d = %d", tokens, kvClassBytes, wantKV, wantFixed, wantKV+wantFixed)
	}

	uniformKV := int64(tokens * cfg.NumLayers * cfg.NumKVHeads * cfg.HeadDim * 3 * 4) // 15,728,640,000 bytes
	if kvClassBytes >= uniformKV {
		t.Fatalf("hybrid plan did not discount KV: got %d, uniform %d", kvClassBytes, uniformKV)
	}
}

