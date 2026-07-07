package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestFormatConfigLineEmitsActiveSetAxis witnesses that the -plan-only config line surfaces the
// full MoE active-set axis Lane F (#3074) needs — experts_used (K) and expert_ffn_len — not just
// the total expert count. Regression guard for the emit-K increment: the loader already reads
// these into cfg, so the only failure mode is the printer dropping them again.
func TestFormatConfigLineEmitsActiveSetAxis(t *testing.T) {
	cfg := model.Config{
		ModelType:           "glm_moe_dsa",
		NumLayers:           79,
		HiddenSize:          6144,
		NumExperts:          256,
		NumExpertsPerTok:    8,
		MoEIntermediateSize: 1536,
	}
	got := formatConfigLine(cfg)
	for _, want := range []string{
		"model_type=glm_moe_dsa",
		"layers=79",
		"hidden=6144",
		"experts=256",
		"experts_used=8",     // K — the highest-leverage unread scalar (Lane F)
		"expert_ffn_len=1536", // expert FFN length, read while the header is open
	} {
		if !strings.Contains(got, want) {
			t.Errorf("config line missing %q\n  got: %s", want, got)
		}
	}
}
