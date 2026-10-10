package model

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=10ms lane=default
func TestValidateBackendForwardConfigRejectsConfiguredRoPE(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	be := compute.Pick("cpu-ref")
	if be == nil {
		t.Fatal("cpu-ref backend missing")
	}
	base := Config{HiddenSize: 16, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 4, IntermediateSize: 32, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1}
	checkRefusal := func(t *testing.T, err error, axis string) {
		t.Helper()
		var unsupported *UnsupportedBackendForwardError
		if !errors.As(err, &unsupported) || unsupported.Forward != forwardGenericHAL || unsupported.Backend != be.Name() || unsupported.IntendedPath != "compute HAL" {
			t.Fatalf("wrong typed RoPE refusal: %T %v", err, err)
		}
		if !strings.Contains(unsupported.Reason, axis) || !strings.Contains(unsupported.Reason, "issue #12608") || strings.Contains(err.Error(), "issue #4714") {
			t.Fatalf("wrong generic RoPE diagnostic: %v", err)
		}
	}
	for _, tc := range []struct {
		name string
		edit func(*Config)
		axis string
	}{
		{name: "theta-only", edit: func(*Config) {}},
		{name: "explicit none", edit: func(c *Config) { c.RopeScaling, c.RopeFactor = "none", 8 }},
		{name: "llama3", edit: func(c *Config) {
			c.RopeScaling, c.RopeFactor, c.RopeLowFreqFactor, c.RopeHighFreqFactor, c.RopeOrigContext = "llama3", 8, 1, 4, 8192
		}, axis: "scaling"},
		{name: "yarn", edit: func(c *Config) { c.RopeScaling, c.RopeFactor, c.RopeOrigContext = "yarn", 16, 65536 }, axis: "scaling"},
		{name: "named scaling with identity factor", edit: func(c *Config) { c.RopeScaling, c.RopeFactor = "yarn", 1 }, axis: "scaling"},
		{name: "partial width", edit: func(c *Config) { c.PartialRotaryFactor = 0.5 }, axis: "partial rotary width"},
		{name: "explicit full width", edit: func(c *Config) { c.PartialRotaryFactor = 1 }},
		{name: "later theta", edit: func(c *Config) { c.RopeThetaPerLayer = []float64{0, 500000} }, axis: "per-layer RoPE theta"},
		{name: "identity theta overrides", edit: func(c *Config) { c.RopeThetaPerLayer = []float64{0, c.RopeTheta, 500000} }},
		{name: "longrope short regime", edit: func(c *Config) {
			c.LongRope = &RopeScaling{Type: "longrope", OriginalMaxPositionEmbeddings: 4096, ShortFactor: []float64{1, 2}, LongFactor: []float64{1, 1}}
		}, axis: "longrope factor"},
		{name: "longrope long regime", edit: func(c *Config) {
			c.MaxPositionEmbeddings = 8192
			c.LongRope = &RopeScaling{Type: "longrope", OriginalMaxPositionEmbeddings: 4096, ShortFactor: []float64{1, 1}, LongFactor: []float64{1, 2}}
		}, axis: "longrope factor"},
		{name: "longrope effective identity", edit: func(c *Config) {
			c.MaxPositionEmbeddings = 8192
			c.LongRope = &RopeScaling{Type: "longrope", OriginalMaxPositionEmbeddings: 4096, ShortFactor: []float64{2, 3}, LongFactor: []float64{1, 1}}
		}},
		{name: "longrope ignored zero and surplus", edit: func(c *Config) { c.LongRope = &RopeScaling{Type: "longrope", ShortFactor: []float64{0, 1, 2}} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			tc.edit(&cfg)
			err := ValidateBackendForwardConfig(cfg, be)
			if tc.axis != "" {
				checkRefusal(t, err, tc.axis)
			} else if err != nil {
				t.Fatalf("effective theta-only control refused: %v", err)
			}
			if err := ValidateBackendForwardConfig(cfg, nil); err != nil {
				t.Fatalf("legacy reference admission changed: %v", err)
			}
		})
	}

	// Dedicated routes retain their own contracts. These are admission controls,
	// not execution or numerical qualification for synthetic model configurations.
	for _, modelType := range []string{"deepseek41", "deepseek2", "glm_moe_dsa", "minimax_m3", "gemma4"} {
		cfg := base
		cfg.ModelType, cfg.RopeScaling = modelType, "yarn"
		if err := ValidateBackendForwardConfig(cfg, be); err != nil {
			t.Fatalf("dedicated route %q changed: %v", modelType, err)
		}
	}
	hybrid := qwen35HybridTestCfg()
	hybrid.RopeScaling = "yarn"
	accepted := &pathQwen35Backend{recordingQwen35Backend: &recordingQwen35Backend{}, path: Qwen35GDNCUDAPath}
	if err := ValidateBackendForwardConfig(hybrid, accepted); err != nil {
		t.Fatalf("hybrid structural admission changed: %v", err)
	}
	var unsupportedHybrid *UnsupportedBackendForwardError
	if err := ValidateBackendForwardConfig(hybrid, be); !errors.As(err, &unsupportedHybrid) || unsupportedHybrid.Forward != ForwardQwen35GDN {
		t.Fatalf("hybrid refusal changed: %v", err)
	}

	cfg := base
	cfg.RopeScaling, cfg.RopeFactor, cfg.RopeLowFreqFactor, cfg.RopeHighFreqFactor, cfg.RopeOrigContext = "llama3", 8, 1, 4, 8192
	m := NewSynthetic(cfg)
	defer m.CloseWeights()
	spy := &layerNormAdmissionBackend{Backend: be}
	state := m.ensureWeightCloser()
	state.mu.Lock()
	before := state.sessions
	state.mu.Unlock()
	s, err := m.NewBackendSessionChecked(spy)
	if s != nil {
		s.Close()
		t.Fatal("theta-only HAL returned a configured-RoPE session")
	}
	checkRefusal(t, err, "scaling")
	state.mu.Lock()
	after := state.sessions
	state.mu.Unlock()
	if before != after || spy.newKV != 0 || spy.resets != 0 || spy.uploads != 0 {
		t.Fatalf("refusal acquired resources: holds=%d->%d kv=%d reset=%d uploads=%d", before, after, spy.newKV, spy.resets, spy.uploads)
	}
}
