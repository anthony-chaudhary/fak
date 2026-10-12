package ggufload

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestDeepSeek41GGUFHeaderAxesMatchHF loads the published 58d8ac86 shard-1
// hyperparameter keys and requires the exported Config JSON to carry the same
// compress_rope_theta, sliding window, YaRN scaling and SwiGLU clamp as the
// pinned HF config.json. The strix3 09f98b106 probe panicked on theta=0.
func TestDeepSeek41GGUFHeaderAxesMatchHF(t *testing.T) {
	p := "deepseek41."
	clamp := make([]Value, 43)
	for i := range clamp {
		clamp[i] = Value{Type: TypeFloat32, Value: float32(10)}
	}
	meta := vcruzQ2KReceiptMeta()
	meta[p+"attention.compress_rope_freq_base"] = Value{Type: TypeFloat32, Value: float32(160000)}
	meta[p+"attention.sliding_window"] = Value{Type: TypeUint32, Value: uint32(128)}
	meta[p+"rope.scaling.type"] = Value{Type: TypeString, Value: "yarn"}
	meta[p+"rope.scaling.factor"] = Value{Type: TypeFloat32, Value: float32(16)}
	meta[p+"rope.scaling.original_context_length"] = Value{Type: TypeUint32, Value: uint32(65536)}
	meta[p+"rope.scaling.yarn_beta_fast"] = Value{Type: TypeFloat32, Value: float32(32)}
	meta[p+"rope.scaling.yarn_beta_slow"] = Value{Type: TypeFloat32, Value: float32(1)}
	meta[p+"swiglu_clamp_exp"] = Value{Type: TypeArray, Value: clamp}
	meta[p+"swiglu_clamp_shexp"] = Value{Type: TypeArray, Value: clamp}

	cfg, err := (&File{Metadata: meta}).Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}

	hfRaw, err := os.ReadFile(filepath.Join("..", "model", "testdata", "deepseek_v41_flash_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var root struct {
		Text struct {
			CompressRopeTheta float64 `json:"compress_rope_theta"`
			SlidingWindow     float64 `json:"sliding_window"`
			SwigluLimit       float64 `json:"swiglu_limit"`
			RopeScaling       struct {
				Factor   float64 `json:"factor"`
				Orig     float64 `json:"original_max_position_embeddings"`
				BetaFast float64 `json:"beta_fast"`
				BetaSlow float64 `json:"beta_slow"`
			} `json:"rope_scaling"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(hfRaw, &root); err != nil {
		t.Fatal(err)
	}
	hf := root.Text
	wantTheta := hf.CompressRopeTheta
	if wantTheta == 0 || hf.SlidingWindow == 0 || hf.SwigluLimit == 0 || hf.RopeScaling.Factor == 0 {
		t.Fatalf("HF fixture lacks an axis: theta=%g window=%g swiglu=%g yarn=%+v", wantTheta, hf.SlidingWindow, hf.SwigluLimit, hf.RopeScaling)
	}

	if v, _ := got["compress_rope_theta"].(float64); v != wantTheta {
		t.Errorf("compress_rope_theta = %v, want HF %g", got["compress_rope_theta"], wantTheta)
	}
	if v, _ := got["swiglu_limit"].(float64); v != hf.SwigluLimit {
		t.Errorf("swiglu_limit = %v, want HF %g", got["swiglu_limit"], hf.SwigluLimit)
	}
	if v, _ := got["rope_scaling_type"].(string); v != "yarn" {
		t.Errorf("rope_scaling_type = %v, want yarn", got["rope_scaling_type"])
	}
	if v, _ := got["rope_scaling_factor"].(float64); v != hf.RopeScaling.Factor {
		t.Errorf("rope_scaling_factor = %v, want HF %g", got["rope_scaling_factor"], hf.RopeScaling.Factor)
	}
	if v, _ := got["rope_scaling_original_max_position_embeddings"].(float64); v != hf.RopeScaling.Orig {
		t.Errorf("rope orig ctx = %v, want HF %g", got["rope_scaling_original_max_position_embeddings"], hf.RopeScaling.Orig)
	}
	if cfg.LongRope == nil || cfg.LongRope.BetaFast != hf.RopeScaling.BetaFast || cfg.LongRope.BetaSlow != hf.RopeScaling.BetaSlow {
		t.Errorf("LongRope = %+v, want beta %g/%g", cfg.LongRope, hf.RopeScaling.BetaFast, hf.RopeScaling.BetaSlow)
	}
	if len(cfg.Window) != cfg.NumLayers {
		t.Fatalf("Window len = %d, want %d layers", len(cfg.Window), cfg.NumLayers)
	}
	for l, w := range cfg.Window {
		if float64(w) != hf.SlidingWindow {
			t.Fatalf("Window[%d] = %d, want HF %g", l, w, hf.SlidingWindow)
		}
	}
}
