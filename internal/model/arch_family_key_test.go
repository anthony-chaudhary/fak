package model

import (
	"strings"
	"testing"
)

func TestArchFamilyHasMatchesKey(t *testing.T) {
	cfgs := []Config{
		{ModelType: "llama"},
		{ModelType: "glm_moe_dsa", Architectures: []string{"GlmMoeDsaForCausalLM"}},
		{ModelType: "MiniMax-M3", Architectures: []string{"MiniMaxM3 ForCausalLM"}},
		{ModelType: "gemma4_text", Architectures: []string{"Gemma4ForCausalLM"}},
		{ModelType: "qwenİx", Architectures: []string{"Qwen"}},
	}
	for _, c := range cfgs {
		want := c.archFamilyKeyUnicode()
		if got := c.archFamilyKey(); got != want {
			t.Fatalf("archFamilyKey(%+v)=%q want %q", c, got, want)
		}
		for _, sub := range []string{"glm", "minimax", "gemma4", "llama", "qwen", "dsa", "m3"} {
			if got, want := c.archFamilyHas(sub), strings.Contains(want, sub); got != want {
				t.Fatalf("archFamilyHas(%q) on %+v = %v want %v", sub, c, got, want)
			}
		}
	}
	c := Config{ModelType: "glm_moe_dsa", Architectures: []string{"GlmMoeDsaForCausalLM"}}
	if n := testing.AllocsPerRun(100, func() { _ = c.archFamilyHas("glm") }); n != 0 {
		t.Fatalf("archFamilyHas allocates %v per call; want 0", n)
	}
	if n := testing.AllocsPerRun(100, func() { _ = layerName(3, "self_attn.q_proj.weight") }); n != 0 {
		t.Fatalf("layerName allocates %v per warm call; want 0", n)
	}
}
