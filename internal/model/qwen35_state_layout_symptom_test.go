package model

import (
	"fmt"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestQwen35StateLayoutHostRejectsInvalidBeforeAllocation(t *testing.T) {
	base := qwen35HybridTestCfg()
	for _, tc := range []struct {
		name string
		edit func(*Config)
	}{
		{"zero-value-heads", func(c *Config) { c.LinearNumValueHeads = 0 }},
		{"negative-key-dimension", func(c *Config) { c.LinearKeyHeadDim = -1 }},
		{"overflow-recurrent-row", func(c *Config) { c.LinearKeyHeadDim = int(^uint(0) >> 1); c.LinearValueHeadDim = 2 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			tc.edit(&cfg)
			var caught any
			func() {
				defer func() { caught = recover() }()
				_ = newLinearAttnLayerState(cfg)
			}()
			if caught == nil || !strings.Contains(fmt.Sprint(caught), "qwen35 state layout") {
				t.Fatalf("want checked layout refusal before allocation, got %v", caught)
			}
		})
	}
}
