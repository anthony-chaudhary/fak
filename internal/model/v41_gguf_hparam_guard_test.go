package model

import (
	"errors"
	"testing"
)

func v41PublishedHParamFixture() Config {
	return Config{
		ModelType:       "deepseek41",
		NumLayers:       40,
		CompressRatios:  append([]int(nil), v41PublishedCompressRatios...),
		HCMult:          4,
		HCSinkhornIters: 20,
		HCEps:           1e-6,
		SwigluLimit:     10,
		NormTopKProb:    true,
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=10ms lane=default
func TestNewDeepSeekV41GGUFConfigRefusesUnpublishedHyperparams(t *testing.T) {
	admitted := map[string]Config{"published": v41PublishedHParamFixture()}
	widened := v41PublishedHParamFixture()
	widened.HCEps = float64(float32(1e-6))
	admitted["f32-widened eps"] = widened
	admitted["reduced schedule, zero hparams"] = Config{ModelType: "deepseek41", NumLayers: 3, CompressRatios: []int{0, 1, 1}}
	for name, c := range admitted {
		if _, err := NewDeepSeekV41GGUFConfig(c, V41GGUFTopology{}); err != nil {
			t.Errorf("%s: err = %v, want admitted", name, err)
		}
	}

	for _, tc := range []struct {
		name   string
		mutate func(*Config)
	}{
		{"iters 0", func(c *Config) { c.HCSinkhornIters = 0 }},
		{"iters 1", func(c *Config) { c.HCSinkhornIters = 1 }},
		{"iters 19", func(c *Config) { c.HCSinkhornIters = 19 }},
		{"mult 0", func(c *Config) { c.HCMult = 0 }},
		{"mult 3", func(c *Config) { c.HCMult = 3 }},
		{"mult 8", func(c *Config) { c.HCMult = 8 }},
		{"eps 0", func(c *Config) { c.HCEps = 0 }},
		{"eps 1e-5", func(c *Config) { c.HCEps = 1e-5 }},
		{"clamp 0", func(c *Config) { c.SwigluLimit = 0 }},
		{"clamp 7", func(c *Config) { c.SwigluLimit = 7 }},
		{"norm false", func(c *Config) { c.NormTopKProb = false }},
	} {
		c := v41PublishedHParamFixture()
		tc.mutate(&c)
		d41, err := NewDeepSeekV41GGUFConfig(c, V41GGUFTopology{})
		if !errors.Is(err, ErrV41ConfigAdmission) || d41 != nil {
			t.Errorf("%s: (cfg=%v, err=%v), want (nil, ErrV41ConfigAdmission)", tc.name, d41 != nil, err)
		}
	}
}
