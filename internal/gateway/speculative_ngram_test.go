package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type ngramGatewayBackend struct {
	compute.Backend
	name string
	path string
}

func (b ngramGatewayBackend) Name() string { return b.name }

func (b ngramGatewayBackend) Qwen35SequenceAllLogitsPath() string { return b.path }

func gatewayNGramTestModel() *model.Model {
	return model.NewSynthetic(model.Config{
		ModelType:  "qwen3_5_text",
		LayerTypes: []string{"linear_attention", "full_attention"},
	})
}

func gatewayNGramEligibleConfig() Config {
	return Config{
		InKernelModel: gatewayNGramTestModel(),
		InKernelQ4K:   true,
		Backend: ngramGatewayBackend{
			Backend: compute.Default(),
			name:    "vulkan",
			path:    compute.Qwen35SequenceAllLogitsPath,
		},
	}
}

func TestNGramSpeculativeRequiresExplicitEligibleVulkanSelection(t *testing.T) {
	for _, env := range []string{"FAK_SPECULATIVE", "SPECULATIVE"} {
		t.Run(env, func(t *testing.T) {
			t.Setenv("FAK_SPECULATIVE", "")
			t.Setenv("SPECULATIVE", "")
			t.Setenv(env, " NGRAM ")

			cfg := gatewayNGramEligibleConfig()
			if !shouldEnableNGramSpeculative(cfg) {
				t.Fatalf("%s=ngram did not admit the eligible Vulkan Qwen hybrid route", env)
			}
			planner := newInKernelChatPlanner(cfg, "qwen38-vulkan", t.Logf).(*agent.InKernelPlanner)
			engine := planner.SpeculativeEngine()
			if engine == nil || engine.PrimaryGenerator() == nil {
				t.Fatal("eligible route did not install a proposal generator")
			}
			if got := engine.PrimaryGenerator().Name(); got != "ngram" {
				t.Fatalf("proposal generator = %q, want ngram", got)
			}
			if got := engine.Config().MaxDraft; got != 4 {
				t.Fatalf("maximum draft = %d, want 4", got)
			}
		})
	}
}

// TestNGramSpeculativeAdmitsCUDAWithAllLogitsPath pins the CUDA arm: a CUDA
// backend that advertises the all-row logits contract admits prompt n-gram
// speculation exactly like Vulkan; one without the marker stays ordinary.
//
// fak-test:runtime fast est=5ms lane=default
func TestNGramSpeculativeAdmitsCUDAWithAllLogitsPath(t *testing.T) {
	t.Setenv("FAK_SPECULATIVE", "ngram")
	t.Setenv("SPECULATIVE", "")
	cfg := gatewayNGramEligibleConfig()
	cfg.Backend = ngramGatewayBackend{Backend: compute.Default(), name: "cuda", path: compute.Qwen35SequenceAllLogitsPath}
	if !shouldEnableNGramSpeculative(cfg) {
		t.Fatal("eligible CUDA Qwen hybrid route advertising all-row logits was not admitted")
	}
	planner := newInKernelChatPlanner(cfg, "qwen38-cuda", t.Logf).(*agent.InKernelPlanner)
	engine := planner.SpeculativeEngine()
	if engine == nil || engine.PrimaryGenerator() == nil || engine.PrimaryGenerator().Name() != "ngram" {
		t.Fatal("eligible CUDA route did not install the n-gram proposal generator")
	}
	for name, backend := range map[string]compute.Backend{
		"cuda without marker": ngramGatewayBackend{Backend: compute.Default(), name: "cuda", path: ""},
		"cuda stale identity": ngramGatewayBackend{Backend: compute.Default(), name: "cuda", path: "stale-contract"},
		"metal with marker":   ngramGatewayBackend{Backend: compute.Default(), name: "metal", path: compute.Qwen35SequenceAllLogitsPath},
		"unknown gpu backend": ngramGatewayBackend{Backend: compute.Default(), name: "rocm", path: compute.Qwen35SequenceAllLogitsPath},
	} {
		t.Run(name, func(t *testing.T) {
			cfg := gatewayNGramEligibleConfig()
			cfg.Backend = backend
			if shouldEnableNGramSpeculative(cfg) {
				t.Fatal("ineligible backend admitted n-gram speculation")
			}
		})
	}
}

func TestNGramSpeculativeLeavesUnsupportedAndUnselectedRoutesOrdinary(t *testing.T) {
	base := gatewayNGramEligibleConfig()
	tests := []struct {
		name string
		env  string
		edit func(*Config)
	}{
		{name: "not selected", env: ""},
		{name: "different mode", env: "mtp"},
		{name: "missing model", env: "ngram", edit: func(c *Config) { c.InKernelModel = nil }},
		{name: "missing backend", env: "ngram", edit: func(c *Config) { c.Backend = nil }},
		{name: "non Q4K load", env: "ngram", edit: func(c *Config) { c.InKernelQ4K = false }},
		{name: "Metal route", env: "ngram", edit: func(c *Config) { c.Metal = true }},
		{name: "non hybrid model", env: "ngram", edit: func(c *Config) {
			c.InKernelModel = model.NewSynthetic(model.Config{ModelType: "qwen3_5_text", LayerTypes: []string{"full_attention"}})
		}},
		{name: "non GPU backend", env: "ngram", edit: func(c *Config) {
			c.Backend = ngramGatewayBackend{Backend: compute.Default(), name: "cpu-ref", path: compute.Qwen35SequenceAllLogitsPath}
		}},
		{name: "missing all logits marker", env: "ngram", edit: func(c *Config) { c.Backend = compute.Default() }},
		{name: "wrong all logits identity", env: "ngram", edit: func(c *Config) {
			c.Backend = ngramGatewayBackend{Backend: compute.Default(), name: "vulkan", path: "stale-contract"}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("FAK_SPECULATIVE", tt.env)
			t.Setenv("SPECULATIVE", "")
			cfg := base
			if tt.edit != nil {
				tt.edit(&cfg)
			}
			if shouldEnableNGramSpeculative(cfg) {
				t.Fatal("unsupported or unselected route admitted n-gram speculation")
			}
			planner := newInKernelChatPlanner(cfg, "ordinary", t.Logf).(*agent.InKernelPlanner)
			if planner.SpeculativeEngine() != nil {
				t.Fatal("unsupported or unselected route installed a speculative engine")
			}
		})
	}
}
