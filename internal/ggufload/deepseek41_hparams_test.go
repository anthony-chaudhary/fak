package ggufload

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func vcruzF32Array(values ...float32) Value {
	items := make([]Value, len(values))
	for i, v := range values {
		items[i] = Value{Type: TypeFloat32, Value: v}
	}
	return Value{Type: TypeArray, Value: items}
}

// vcruzHParamMeta adds the hyperparameter keys read off the published
// vcruz305/DeepSeek-V4.1-Flash-GGUF@58d8ac86 shard-1 header on strix3.
func vcruzHParamMeta() map[string]Value {
	meta := vcruzQ2KReceiptMeta()
	p := "deepseek41."
	clamp := make([]float32, 43)
	for i := range clamp {
		clamp[i] = 10
	}
	meta[p+"attention.compress_rope_freq_base"] = Value{Type: TypeFloat32, Value: float32(160000)}
	meta[p+"attention.sliding_window"] = Value{Type: TypeUint32, Value: uint32(128)}
	meta[p+"rope.scaling.type"] = Value{Type: TypeString, Value: "yarn"}
	meta[p+"rope.scaling.factor"] = Value{Type: TypeFloat32, Value: float32(16)}
	meta[p+"rope.scaling.original_context_length"] = Value{Type: TypeUint32, Value: uint32(65536)}
	meta[p+"rope.scaling.yarn_beta_fast"] = Value{Type: TypeFloat32, Value: float32(32)}
	meta[p+"rope.scaling.yarn_beta_slow"] = Value{Type: TypeFloat32, Value: float32(1)}
	meta[p+"swiglu_clamp_exp"] = vcruzF32Array(clamp...)
	meta[p+"swiglu_clamp_shexp"] = vcruzF32Array(clamp...)
	return meta
}

func hfDeepSeekV41Config(t *testing.T) model.Config {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "model", "testdata", "deepseek_v41_flash_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg model.Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestDeepSeek41GGUFHParamsMatchHFConfig(t *testing.T) {
	hf := hfDeepSeekV41Config(t)
	f := &File{Metadata: vcruzHParamMeta()}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if hf.DeepSeekV41 == nil || cfg.CompressRopeTheta != hf.DeepSeekV41.CompressRopeTheta {
		t.Errorf("CompressRopeTheta = %g, want HF %v", cfg.CompressRopeTheta, hf.DeepSeekV41)
	}
	if len(cfg.Window) != len(hf.Window) {
		t.Fatalf("Window len = %d, want HF %d", len(cfg.Window), len(hf.Window))
	}
	for l := range hf.Window {
		if cfg.Window[l] != hf.Window[l] {
			t.Fatalf("Window[%d] = %d, want HF %d", l, cfg.Window[l], hf.Window[l])
		}
	}
	if cfg.RopeScaling != hf.RopeScaling || cfg.RopeFactor != hf.RopeFactor || cfg.RopeOrigContext != hf.RopeOrigContext {
		t.Errorf("rope scaling = %q/%g/%d, want HF %q/%g/%d", cfg.RopeScaling, cfg.RopeFactor, cfg.RopeOrigContext, hf.RopeScaling, hf.RopeFactor, hf.RopeOrigContext)
	}
	if cfg.LongRope == nil || hf.LongRope == nil ||
		cfg.LongRope.BetaFast != hf.LongRope.BetaFast || cfg.LongRope.BetaSlow != hf.LongRope.BetaSlow ||
		cfg.LongRope.Factor != hf.LongRope.Factor || cfg.LongRope.OriginalMaxPositionEmbeddings != hf.LongRope.OriginalMaxPositionEmbeddings {
		t.Errorf("LongRope = %+v, want HF %+v", cfg.LongRope, hf.LongRope)
	}
	if cfg.SwigluLimit != hf.SwigluLimit {
		t.Errorf("SwigluLimit = %g, want HF %g", cfg.SwigluLimit, hf.SwigluLimit)
	}
}

func TestDeepSeek41GGUFHParamsAbsentLeaveZero(t *testing.T) {
	cfg, err := (&File{Metadata: vcruzQ2KReceiptMeta()}).Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.CompressRopeTheta != 0 || cfg.Window != nil || cfg.RopeScaling != "" || cfg.LongRope != nil || cfg.SwigluLimit != 0 {
		t.Errorf("absent keys produced theta=%g window=%v scaling=%q longrope=%v swiglu=%g",
			cfg.CompressRopeTheta, cfg.Window, cfg.RopeScaling, cfg.LongRope, cfg.SwigluLimit)
	}
}

func TestDeepSeek41GGUFHParamsMalformedRefuse(t *testing.T) {
	p := "deepseek41."
	cases := map[string]func(map[string]Value){
		"zero compress base": func(m map[string]Value) {
			m[p+"attention.compress_rope_freq_base"] = Value{Type: TypeFloat32, Value: float32(0)}
		},
		"integer compress base": func(m map[string]Value) {
			m[p+"attention.compress_rope_freq_base"] = Value{Type: TypeUint32, Value: uint32(160000)}
		},
		"zero sliding window": func(m map[string]Value) {
			m[p+"attention.sliding_window"] = Value{Type: TypeUint32, Value: uint32(0)}
		},
		"unmapped rope scaling": func(m map[string]Value) {
			m[p+"rope.scaling.type"] = Value{Type: TypeString, Value: "longrope"}
		},
		"yarn without factor": func(m map[string]Value) { delete(m, p+"rope.scaling.factor") },
		"per-layer swiglu clamp": func(m map[string]Value) {
			m[p+"swiglu_clamp_shexp"] = vcruzF32Array(10, 7)
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			meta := vcruzHParamMeta()
			mutate(meta)
			if _, err := (&File{Metadata: meta}).Config(); !errors.Is(err, ErrDeepSeek41HParam) {
				t.Fatalf("err = %v, want ErrDeepSeek41HParam", err)
			}
		})
	}
}
