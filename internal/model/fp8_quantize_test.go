package model

import (
	"math"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
// TestFP8TiledQuantLoadMatchesExpanded holds the actual safetensors adapter to
// the pre-existing decoder and Q8 quantizer. Runtime estimate is unmeasured.
func TestFP8TiledQuantLoadMatchesExpanded(t *testing.T) {
	t.Run("lookup", func(t *testing.T) {
		for code := 0; code < 256; code++ {
			got, want := fp8E4M3Lookup(byte(code)), fp8E4M3ToF32(byte(code))
			if math.IsNaN(float64(want)) {
				if !math.IsNaN(float64(got)) {
					t.Fatalf("E4M3FN 0x%02x = %v, want NaN", code, got)
				}
			} else if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("E4M3FN 0x%02x = %08x, want %08x", code, math.Float32bits(got), math.Float32bits(want))
			}
		}
	})

	// Both scale dimensions cross 128; the last row and column tiles are ragged.
	const out, in = 129, 160
	weight := make([]byte, out*in)
	for i := range weight {
		weight[i] = byte(i % 256)
		if weight[i]&0x7f == 0x7f {
			weight[i]-- // The numeric fixture is finite; lookup tests both NaN codes.
		}
	}
	scales := []float32{0.125, 1.3, 0, 2.75}
	expanded, err := decodeFP8BlockScale("oracle", out, in, weight, scales)
	if err != nil {
		t.Fatal(err)
	}
	want := quantizeQ8(expanded, out, in)
	for _, tc := range []struct {
		label, name, canonical string
		cfg                    Config
		keepF32                bool
	}{
		{
			label: "projection", name: "model.layers.0.self_attn.q_proj.weight",
			canonical: "model.layers.0.self_attn.q_proj.weight",
		},
		{
			label: "canonicalized", name: "model.language_model.layers.0.self_attn.q_proj.weight",
			canonical: "model.layers.0.self_attn.q_proj.weight",
			cfg:       Config{ModelType: "qwen3_5", HiddenSize: in},
		},
		{
			label: "routed expert", name: "model.layers.0.mlp.experts.0.down_proj.weight",
			canonical: "model.layers.0.mlp.experts.0.down_proj.weight",
			cfg:       Config{NumLayers: 1, NumExperts: 1, HiddenSize: in},
		},
		{
			label: "tied embedding fallback", name: "model.embed_tokens.weight",
			canonical: "model.embed_tokens.weight", keepF32: true,
		},
	} {
		t.Run(tc.label, func(t *testing.T) {
			scaleName := tc.name + "_scale_inv"
			path := writeTinySafetensors(t, map[string]tinySTTensor{
				tc.name:   {dtype: "F8_E4M3", shape: []int{out, in}, data: weight},
				scaleName: {dtype: "F32", shape: []int{2, 2}, data: f32Bytes(scales)},
			})
			m, err := LoadSafetensorsQuant(path, tc.cfg)
			if err != nil {
				t.Fatal(err)
			}
			got := m.q8w[tc.canonical]
			if got == nil || got.out != out || got.in != in || got.nblk != want.nblk {
				t.Fatalf("missing or wrong Q8 shape for %s", tc.canonical)
			}
			for i, q := range want.q {
				if got.q[i] != q {
					t.Fatalf("Q8 code[%d] = %d, want %d", i, got.q[i], q)
				}
			}
			for i, d := range want.d {
				if math.Float32bits(got.d[i]) != math.Float32bits(d) {
					t.Fatalf("Q8 scale[%d] = %08x, want %08x", i, math.Float32bits(got.d[i]), math.Float32bits(d))
				}
			}
			if _, present := m.manifest[scaleName]; present {
				t.Fatal("scale companion survived the loader")
			}
			if tc.keepF32 {
				if !m.has(tc.canonical) || len(m.raw) != out*in*4 {
					t.Fatal("tied embedding lost its existing float32 row-gather representation")
				}
			} else if m.has(tc.canonical) || len(m.raw) != 0 {
				t.Fatal("quant-only projection retained float32 data")
			}
		})
	}

	t.Run("source MoE malformed rank", func(t *testing.T) {
		const name = "model.layers.0.mlp.experts.down_proj.weight"
		cfg := Config{NumLayers: 1, NumExperts: 1, HiddenSize: in, IntermediateSize: 64}
		if _, direct := fp8DirectQ8Name(cfg, name, []int{out, in}); direct {
			t.Fatal("source MoE transform was bypassed")
		}
		path := writeTinySafetensors(t, map[string]tinySTTensor{
			name:                {dtype: "F8_E4M3", shape: []int{out, in}, data: weight},
			name + "_scale_inv": {dtype: "F32", shape: []int{2, 2}, data: f32Bytes(scales)},
		})
		if _, err := LoadSafetensorsQuant(path, cfg); err == nil || !strings.Contains(err.Error(), "shape") {
			t.Fatalf("malformed source MoE shape error = %v", err)
		}
	})

	t.Run("payload guards", func(t *testing.T) {
		for _, tc := range []struct {
			shape  []int
			weight []byte
			scales []byte
		}{
			{[]int{out, in}, weight[:len(weight)-1], f32Bytes(scales)},
			{[]int{out, in}, weight, f32Bytes(scales[:3])},
			{[]int{1, 33}, make([]byte, 33), f32Bytes([]float32{1})},
			{[]int{0, in}, nil, nil},
		} {
			if _, err := quantizeFP8BlockScaleQ8("bad", tc.shape, tc.weight, tc.scales); err == nil {
				t.Fatalf("accepted malformed FP8/Q8 payload with shape %v", tc.shape)
			}
		}
	})
}
