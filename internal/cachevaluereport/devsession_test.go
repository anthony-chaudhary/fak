package cachevaluereport

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

func devSessionTestNow() time.Time {
	return time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC)
}

// TestFoldDevSessionBenefitPricesKnownModel asserts the Track-3 fold reproduces the exact
// NewSavingsRows economics (the same formula Track 2 uses) over one priced session/model, by
// hand-computing the expected $/token-equiv figures independently.
func TestFoldDevSessionBenefitPricesKnownModel(t *testing.T) {
	sessions := []sessionaudit.Session{
		{
			Session: "s1",
			PerModel: map[string]sessionaudit.ModelCounts{
				"claude-opus-4-8-20260101": {Turns: 3, Input: 100_000, Output: 20_000, CacheRead: 1_000_000, CacheCreate: 200_000},
			},
		},
	}

	rep := FoldDevSessionBenefit(sessions, devSessionTestNow())

	if rep.Sessions != 1 || rep.PricedSessions != 1 {
		t.Fatalf("sessions=%d priced=%d, want 1/1", rep.Sessions, rep.PricedSessions)
	}
	if rep.ModelTiers["opus"] != 1 {
		t.Fatalf("model_tiers[opus] = %d, want 1: %+v", rep.ModelTiers["opus"], rep.ModelTiers)
	}
	if rep.InputTokens != 100_000 || rep.OutputTokens != 20_000 || rep.CacheReadTokens != 1_000_000 || rep.CacheCreationTokens != 200_000 {
		t.Fatalf("token axes not passed through: %+v", rep)
	}
	const wantSaved = 850_000.0
	if diff := rep.SavedTokenEquiv - wantSaved; diff > 1e-6 || diff < -1e-6 {
		t.Fatalf("saved_token_equiv = %.4f, want %.4f", rep.SavedTokenEquiv, wantSaved)
	}
	const wantSpend = 8.25
	if diff := rep.ObservedActualSpendUSD - wantSpend; diff > 1e-6 || diff < -1e-6 {
		t.Fatalf("observed_actual_spend_usd = %.4f, want %.4f", rep.ObservedActualSpendUSD, wantSpend)
	}
	const wantAvoided = 12.75
	if diff := rep.ObservedAPICostAvoidedUSD - wantAvoided; diff > 1e-6 || diff < -1e-6 {
		t.Fatalf("observed_api_cost_avoided_usd = %.4f, want %.4f", rep.ObservedAPICostAvoidedUSD, wantAvoided)
	}
	const wantCounterfactual = 21.0
	if diff := rep.ObservedCounterfactualUSD - wantCounterfactual; diff > 1e-6 || diff < -1e-6 {
		t.Fatalf("observed_counterfactual_usd = %.4f, want %.4f", rep.ObservedCounterfactualUSD, wantCounterfactual)
	}
	if rep.ObservedAPICostReductionPct == nil {
		t.Fatal("reduction pct not set")
	}
	const wantReduction = 100 * 12.75 / 21.0
	if diff := *rep.ObservedAPICostReductionPct - wantReduction; diff > 1e-6 || diff < -1e-6 {
		t.Fatalf("reduction pct = %.6f, want %.6f", *rep.ObservedAPICostReductionPct, wantReduction)
	}
	if rep.Provenance == "" {
		t.Fatal("provenance must be set")
	}
}

// TestFoldDevSessionBenefitSkipsUnpricedModelsAndErroredSessions asserts an unpriced model
// (e.g. a non-Anthropic model sessionaudit's Pricing table has no rate for) contributes to
// UnpricedModels but not to any token/dollar axis, and a session with a discovery/parse Error
// is excluded from the count entirely rather than silently zero-priced.
func TestFoldDevSessionBenefitSkipsUnpricedModelsAndErroredSessions(t *testing.T) {
	sessions := []sessionaudit.Session{
		{
			Session: "s-unpriced",
			PerModel: map[string]sessionaudit.ModelCounts{
				"gpt-4o": {Turns: 1, Input: 500, Output: 100},
			},
		},
		{
			Session: "s-errored",
			Error:   "read: permission denied",
			PerModel: map[string]sessionaudit.ModelCounts{
				"claude-opus-4-8": {Turns: 1, Input: 500, Output: 100, CacheRead: 1000},
			},
		},
	}

	rep := FoldDevSessionBenefit(sessions, devSessionTestNow())

	if rep.Sessions != 1 {
		t.Fatalf("sessions = %d, want 1 (errored session excluded)", rep.Sessions)
	}
	if rep.PricedSessions != 0 {
		t.Fatalf("priced_sessions = %d, want 0 (only model is unpriced)", rep.PricedSessions)
	}
	if rep.UnpricedModels["unpriced"] != 1 {
		t.Fatalf("unpriced_models[unpriced] = %d, want 1: %+v", rep.UnpricedModels["unpriced"], rep.UnpricedModels)
	}
	if rep.InputTokens != 0 || rep.CacheReadTokens != 0 || rep.SavedTokenEquiv != 0 {
		t.Fatalf("unpriced/errored sessions must not contribute token/dollar axes: %+v", rep)
	}
}

func TestFoldDevSessionBenefitEmptyIsHonestZero(t *testing.T) {
	rep := FoldDevSessionBenefit(nil, devSessionTestNow())
	if rep.Sessions != 0 || rep.SavedTokenEquiv != 0 || rep.ObservedAPICostReductionPct != nil {
		t.Fatalf("empty input must fold to honest zero, got %+v", rep)
	}
	if rep.Finding != "no real dev-session transcripts discovered" {
		t.Fatalf("finding = %q", rep.Finding)
	}
}

func TestRenderDevSessionBenefitIncludesProvenanceAndFigures(t *testing.T) {
	rep := FoldDevSessionBenefit([]sessionaudit.Session{
		{
			Session: "s1",
			PerModel: map[string]sessionaudit.ModelCounts{
				"claude-sonnet-4-5": {Turns: 2, Input: 1000, Output: 200, CacheRead: 50_000, CacheCreate: 5_000},
			},
		},
	}, devSessionTestNow())

	out := RenderDevSessionBenefit(rep)
	for _, want := range []string{"Dev-session lens", "1 session(s) discovered", "provenance:", "MAY OVERLAP"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
}

func TestRenderDevSessionBenefitZeroSessionsStillShowsProvenance(t *testing.T) {
	out := RenderDevSessionBenefit(FoldDevSessionBenefit(nil, devSessionTestNow()))
	if !strings.Contains(out, "no real dev-session transcripts discovered") || !strings.Contains(out, "provenance:") {
		t.Fatalf("zero-session render should still show finding + provenance:\n%s", out)
	}
}

// TestTwoTrackReportOmitsDevSessionBenefitWhenNil confirms the JSON contract the CLI recompute
// test relies on: DevSessionBenefit is only present when the caller (the CLI layer) explicitly
// sets it, so the pure FoldTwoTrack/FoldTwoTrackWithUsage path is unaffected.
func TestTwoTrackReportOmitsDevSessionBenefitWhenNil(t *testing.T) {
	rep := FoldTwoTrack(nil, nil, devSessionTestNow())
	if rep.DevSessionBenefit != nil {
		t.Fatalf("DevSessionBenefit must be nil unless explicitly set: %+v", rep.DevSessionBenefit)
	}
}
