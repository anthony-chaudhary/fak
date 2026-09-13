package model

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
)

func pinnedV4Config() Config {
	return Config{ModelType: "deepseek_v4", NumLayers: 61, HiddenSize: 7168, NumExperts: 384, NumExpertsPerTok: 6, MoEIntermediateSize: 3072, NSharedExperts: 1, ExpertDtype: "fp4", NormTopKProb: true, RoutedScalingFactor: 2.5, ScoringFunc: "sqrtsoftplus", TopKMethod: "noaux_tc", SwigluLimit: 10}
}

func TestAdmitDeepSeekV4ConfigPinnedArtifact(t *testing.T) {
	const raw = `{"model_type":"deepseek_v4","num_hidden_layers":61,"hidden_size":7168,"n_routed_experts":384,"num_experts_per_tok":6,"moe_intermediate_size":3072,"n_shared_experts":1,"expert_dtype":"fp4","norm_topk_prob":true,"routed_scaling_factor":2.5,"scoring_func":"sqrtsoftplus","topk_method":"noaux_tc","swiglu_limit":10}`
	var cfg Config
	if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.ExpertDtype != "fp4" {
		t.Fatalf("expert_dtype=%q", cfg.ExpertDtype)
	}
	if err := AdmitDeepSeekV4Config(cfg); err != nil {
		t.Fatal(err)
	}
	if !cfg.IsDeepSeekV4() {
		t.Fatal("identity predicate false")
	}
}

func TestAdmitDeepSeekV4ConfigFailsClosed(t *testing.T) {
	cases := map[string]func(*Config){
		"architecture": func(c *Config) { c.ModelType = "deepseek2" },
		"layers":       func(c *Config) { c.NumLayers = 60 }, "hidden": func(c *Config) { c.HiddenSize = 4096 },
		"experts": func(c *Config) { c.NumExperts = 256 }, "topk": func(c *Config) { c.NumExpertsPerTok = 8 },
		"moe_intermediate": func(c *Config) { c.MoEIntermediateSize = 2048 }, "shared": func(c *Config) { c.NSharedExperts = 0 },
		"dtype": func(c *Config) { c.ExpertDtype = "mxfp4" }, "normalize": func(c *Config) { c.NormTopKProb = false },
		"scale": func(c *Config) { c.RoutedScalingFactor = 1 }, "score": func(c *Config) { c.ScoringFunc = "softmax" },
		"method": func(c *Config) { c.TopKMethod = "greedy" }, "limit_negative": func(c *Config) { c.SwigluLimit = -1 },
		"limit_nan": func(c *Config) { c.SwigluLimit = math.NaN() }, "limit_inf": func(c *Config) { c.SwigluLimit = math.Inf(1) },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := pinnedV4Config()
			mutate(&cfg)
			err := AdmitDeepSeekV4Config(cfg)
			if !errors.Is(err, ErrV4ConfigAdmission) {
				t.Fatalf("err=%v", err)
			}
		})
	}
}

// TestAdmitDeepSeekV4ConfigMegaMoELayout pins the SGLang v0.5.19 W4A4 MegaMoE
// descriptor: a DeepSeek-V4 config that declares the MegaMoE layout must carry a
// coherent (format, block-scale) pair, and every malformed/unknown descriptor must
// fail closed through ErrV4ConfigAdmission. Without this check a V4 config could
// name an unimplemented W4A4 layout and reach runtime weight I/O.
func TestAdmitDeepSeekV4ConfigMegaMoELayout(t *testing.T) {
	// A declared MegaMoE layout with a coherent MXFP4 (block 32 / E8M0) pair admits.
	ok := pinnedV4Config()
	ok.MoELayout = "w4a4_megamoe"
	ok.MoEWeightFormat = "mxfp4"
	ok.MoEBlockScaleElements = 32
	ok.MoEBlockScaleEncoding = "e8m0"
	if err := AdmitDeepSeekV4Config(ok); err != nil {
		t.Fatalf("coherent megamoe descriptor rejected: %v", err)
	}

	// A config that names NO MegaMoE layout stays on the legacy path: the descriptor
	// fields are ignored, so a pinned non-MegaMoE V4 config still admits unchanged.
	legacy := pinnedV4Config()
	if err := AdmitDeepSeekV4Config(legacy); err != nil {
		t.Fatalf("legacy layout rejected: %v", err)
	}

	cases := map[string]func(*Config){
		"unknown_layout":      func(c *Config) { c.MoELayout = "w4a4_mystery" },
		"layout_blank_format": func(c *Config) { c.MoEWeightFormat = "" },
		"format_not_mxfp4":    func(c *Config) { c.MoEWeightFormat = "nvfp4" },
		"format_unknown":      func(c *Config) { c.MoEWeightFormat = "fp4" },
		"scale_elements_0":    func(c *Config) { c.MoEBlockScaleElements = 0 },
		"scale_elements_16":   func(c *Config) { c.MoEBlockScaleElements = 16 },
		"scale_encoding_e4m3": func(c *Config) { c.MoEBlockScaleEncoding = "e4m3" },
		"scale_encoding_blank": func(c *Config) {
			c.MoEBlockScaleEncoding = ""
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := ok
			mutate(&cfg)
			if err := AdmitDeepSeekV4Config(cfg); !errors.Is(err, ErrV4ConfigAdmission) {
				t.Fatalf("err=%v, want ErrV4ConfigAdmission", err)
			}
		})
	}
}

func TestDeepSeekV4IdentityDoesNotAliasOtherArchitectures(t *testing.T) {
	for _, kind := range []string{"deepseek2", "glm_moe_dsa", "gpt_oss", ""} {
		cfg := Config{ModelType: kind}
		if cfg.IsDeepSeekV4() {
			t.Fatalf("admitted %q", kind)
		}
	}
}
