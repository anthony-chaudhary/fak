package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
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
		"experts_used=8",      // K — the highest-leverage unread scalar (Lane F)
		"expert_ffn_len=1536", // expert FFN length, read while the header is open
	} {
		if !strings.Contains(got, want) {
			t.Errorf("config line missing %q\n  got: %s", want, got)
		}
	}
}

// TestFormatActiveSetLineDivisors witnesses that the -plan-only active-set line emits BOTH roofline
// divisors Lane F (#3074) derives — active-bytes/token (K×per-expert + non-expert stream) and
// active-params/token — when K is known, and flags them PENDING(K) rather than guessing when it is
// not. The field names are the ceiling-doc / quant-sweep table's contract, so a rename must break
// this test.
func TestFormatActiveSetLineDivisors(t *testing.T) {
	known := formatActiveSetLine(ggufload.RoutedExpertActiveSet{
		NumExperts: 256, ExpertsUsed: 8,
		RoutedResident: 1 << 30, PerExpert: 1 << 20, NonExpertResident: 2 << 30,
		ActivePerToken: 4 << 20, ActiveBytesPerToken: 6 << 20, ActiveParamsPerToken: 42_000_000_000,
	})
	for _, want := range []string{"non_expert_resident=", "active_bytes_per_tok=", "active_params_per_tok=42.00B", "DERIVED"} {
		if !strings.Contains(known, want) {
			t.Errorf("K-known active-set line missing %q\n  got: %s", want, known)
		}
	}
	pending := formatActiveSetLine(ggufload.RoutedExpertActiveSet{
		NumExperts: 256, ExpertsUsed: 0, RoutedResident: 1 << 30, PerExpert: 1 << 20, NonExpertResident: 2 << 30,
	})
	for _, want := range []string{"active_bytes_per_tok=PENDING(K)", "active_params_per_tok=PENDING(K)"} {
		if !strings.Contains(pending, want) {
			t.Errorf("K-unread active-set line missing %q\n  got: %s", want, pending)
		}
	}
}
