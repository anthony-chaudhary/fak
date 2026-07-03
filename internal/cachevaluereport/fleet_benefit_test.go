package cachevaluereport

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

func TestFoldFleetBenefitAggregatesUsageSavingsAndExtension(t *testing.T) {
	track1 := []cachevalueledger.Row{
		{Date: "2026-06-22", SessionType: "run", Turns: 10, PromptTokens: 1000, ReusedTokens: 800},
	}
	track2 := []SavingsRow{
		{
			Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			CacheReadTokens: 2000, CacheCreationTokens: 100,
			SavedTokenEquiv: 1800, NetSavedTokenEquiv: 1800,
			RebateUSD: 0.009, WritePremiumUSD: 0.001, SpendUSD: 0.020,
		},
		{
			Date: "2026-06-22", Provider: "fak", Mechanism: "compaction_shed",
			CompactionShedTokens: 3000, SavedTokenEquiv: 3000, NetSavedTokenEquiv: 3000,
			CompactionSavedUSD: 0.015,
		},
	}
	usage := []gatewayusageledger.Row{
		gatewayusageledger.NewRow("exit", "guard", "claude", "g-1", 90*time.Second, gatewayusageledger.Counters{
			Total:                12,
			Allowed:              8,
			Denied:               2,
			Transformed:          1,
			Quarantined:          1,
			InputTokens:          1000,
			OutputTokens:         50,
			CachedPromptTokens:   2000,
			CacheCreationTokens:  100,
			CachedTurns:          2,
			CompactionFired:      1,
			CompactionShedTokens: 3000,
		}, time.Date(2026, 6, 22, 10, 0, 0, 0, time.UTC)),
	}

	rep := FoldFleetBenefit(track1, track2, usage, FleetBenefitOptions{ContextBudgetTokens: 15000})
	if rep.UsageRows != 1 || rep.ExitSessions != 1 || rep.SessionTypes["guard"] != 1 {
		t.Fatalf("usage/session aggregate = rows %d exits %d types %v, want 1/1 guard=1",
			rep.UsageRows, rep.ExitSessions, rep.SessionTypes)
	}
	if rep.KernelDecisions != 12 || rep.Allowed != 8 || rep.Denied != 2 || rep.Transformed != 1 || rep.Quarantined != 1 {
		t.Fatalf("decision aggregate mismatch: %+v", rep)
	}
	if rep.CacheReadTokens != 2000 || rep.CacheCreationTokens != 100 || rep.CachedTurns != 2 {
		t.Fatalf("cache usage aggregate mismatch: %+v", rep)
	}
	if rep.ProviderPromptCacheTokenEq != 1800 || rep.FakKVPrefixReusedTokens != 800 || rep.FakCompactionTokenEq != 3000 {
		t.Fatalf("owner token-equiv aggregate mismatch: %+v", rep)
	}
	if rep.FakAuthoredTokenEq != 3800 || rep.TotalSavedTokenEq != 5600 {
		t.Fatalf("saved token-equiv total = fak %.0f total %.0f, want 3800/5600", rep.FakAuthoredTokenEq, rep.TotalSavedTokenEq)
	}
	if rep.FakSharePct == nil || !approxTrack2(*rep.FakSharePct, 67.85714285714286) {
		t.Fatalf("fak share = %v, want 67.857142857%%", rep.FakSharePct)
	}
	if !approxTrack2(rep.ObservedActualSpendUSD, 0.020) ||
		!approxTrack2(rep.ObservedAPICostAvoidedUSD, 0.023) ||
		!approxTrack2(rep.ObservedCounterfactualUSD, 0.043) {
		t.Fatalf("API cost aggregate mismatch: actual %.6f avoided %.6f counterfactual %.6f",
			rep.ObservedActualSpendUSD, rep.ObservedAPICostAvoidedUSD, rep.ObservedCounterfactualUSD)
	}
	if rep.ObservedAPICostReductionPct == nil || !approxTrack2(*rep.ObservedAPICostReductionPct, 53.48837209302325) {
		t.Fatalf("API reduction pct = %v, want 53.488372%%", rep.ObservedAPICostReductionPct)
	}
	if rep.ContextExtensionTokens != 3000 || rep.EquivalentContextWindow == nil || !approxTrack2(*rep.EquivalentContextWindow, 0.2) ||
		rep.ContextExtensionPct == nil || !approxTrack2(*rep.ContextExtensionPct, 20) {
		t.Fatalf("context extension mismatch: %+v", rep)
	}
	// Owner-split of the avoided dollars: provider = 0.009 − 0.001 = 0.008; fak =
	// 0.015; and the split must sum EXACTLY to the blended total (0.023).
	if !approxTrack2(rep.ProviderAPICostAvoidedUSD, 0.008) || !approxTrack2(rep.FakAPICostAvoidedUSD, 0.015) {
		t.Fatalf("dollar split = provider %.6f fak %.6f, want 0.008/0.015", rep.ProviderAPICostAvoidedUSD, rep.FakAPICostAvoidedUSD)
	}
	if !approxTrack2(rep.ProviderAPICostAvoidedUSD+rep.FakAPICostAvoidedUSD, rep.ObservedAPICostAvoidedUSD) {
		t.Fatalf("split does not sum to blended: %.6f + %.6f != %.6f",
			rep.ProviderAPICostAvoidedUSD, rep.FakAPICostAvoidedUSD, rep.ObservedAPICostAvoidedUSD)
	}
	// Display cache_read is sourced from Track-2 (the provider row's 2000), never the
	// usage-ledger CacheReadTokens, so the two stay distinct and are not summed.
	if rep.ProviderCacheReadTokens != 2000 {
		t.Fatalf("provider cache_read display = %d, want 2000 (Track-2 source)", rep.ProviderCacheReadTokens)
	}

	out := RenderFleetBenefit(rep)
	for _, want := range []string{
		"Fleet aggregate",
		"saved token-equiv: provider=1800 fak=3800 total=5600",
		"cache_read=2000 (OBSERVED, Track-2)",
		"avoided=$0.0230 (provider $0.0080 + fak $0.0150) reduction=53.49%",
		"session extension: 3000 WITNESSED context token(s) shed = 20.00% of a 15000-token budget",
		"provider prompt-cache dollars are OBSERVED/provider-relayed projections",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
}

// TestFoldFleetBenefitRunRateSplitAndProvisional pins the long-horizon run-rate:
// dollars/tokens normalized over the SAVINGS-row span, split provider vs fak, and
// flagged PROVISIONAL under the thin-window floor.
func TestFoldFleetBenefitRunRateSplitAndProvisional(t *testing.T) {
	// Two provider rows exactly 2 days apart → span 2.0d (< 3d floor → PROVISIONAL).
	// $ avoided per row = rebate 10 − premium 0 = 10; total 20 over 2d = $10/day.
	track2 := []SavingsRow{
		{Date: "2026-06-20", GeneratedAt: "2026-06-20T00:00:00Z", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			CacheReadTokens: 1_000_000, SavedTokenEquiv: 900_000, NetSavedTokenEquiv: 900_000, RebateUSD: 10, SpendUSD: 5},
		{Date: "2026-06-22", GeneratedAt: "2026-06-22T00:00:00Z", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			CacheReadTokens: 1_000_000, SavedTokenEquiv: 900_000, NetSavedTokenEquiv: 900_000, RebateUSD: 10, SpendUSD: 5},
	}
	rep := FoldFleetBenefit(nil, track2, nil, FleetBenefitOptions{})

	if !approxTrack2(rep.SpanDays, 2.0) {
		t.Fatalf("span days = %.6f, want 2.0", rep.SpanDays)
	}
	if !rep.RateProvisional {
		t.Fatalf("2.0d span must be PROVISIONAL (< %.0fd floor)", minHonestSpanDays)
	}
	if !approxTrack2(rep.ProviderUSDAvoidedPerDay, 10) || !approxTrack2(rep.FakUSDAvoidedPerDay, 0) {
		t.Fatalf("per-day split = provider $%.4f fak $%.4f, want 10/0", rep.ProviderUSDAvoidedPerDay, rep.FakUSDAvoidedPerDay)
	}
	if !approxTrack2(rep.USDAvoidedPerDay, 10) || !approxTrack2(rep.USDAvoidedPerWeek, 70) {
		t.Fatalf("blended rate = $%.4f/day $%.4f/week, want 10/70", rep.USDAvoidedPerDay, rep.USDAvoidedPerWeek)
	}
	// Provider-only corpus → fak dollars, fak token-rate, and fak share are all zero.
	if rep.FakAPICostAvoidedUSD != 0 || rep.FakTokenEqPerDay != 0 {
		t.Fatalf("provider-only corpus leaked a fak value: $%.6f/day-$? tok %.4f", rep.FakAPICostAvoidedUSD, rep.FakTokenEqPerDay)
	}
	if rep.FakSharePct == nil || *rep.FakSharePct != 0 {
		t.Fatalf("provider-only fak_share = %v, want 0", rep.FakSharePct)
	}
	out := RenderFleetBenefit(rep)
	for _, want := range []string{
		"run-rate (OBSERVED provider-cache economics, over 2.00d 2026-06-20..2026-06-22)",
		"provider $10.00 + fak $0.00 = $10.00",
		"PROVISIONAL: span < 3d",
		"projection (OBSERVED, straight-line at current rate): 30d ~= provider $300 + fak $0",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("run-rate render missing %q:\n%s", want, out)
		}
	}
}

// TestFoldFleetBenefitRunRateSettledSpan checks the PROVISIONAL flag clears once
// the savings span reaches the honest floor.
func TestFoldFleetBenefitRunRateSettledSpan(t *testing.T) {
	track2 := []SavingsRow{
		{Date: "2026-06-20", GeneratedAt: "2026-06-20T00:00:00Z", Provider: "anthropic", Mechanism: "provider_prompt_cache", RebateUSD: 10, SpendUSD: 5},
		{Date: "2026-06-24", GeneratedAt: "2026-06-24T00:00:00Z", Provider: "anthropic", Mechanism: "provider_prompt_cache", RebateUSD: 10, SpendUSD: 5},
	}
	rep := FoldFleetBenefit(nil, track2, nil, FleetBenefitOptions{})
	if !approxTrack2(rep.SpanDays, 4.0) {
		t.Fatalf("span days = %.6f, want 4.0", rep.SpanDays)
	}
	if rep.RateProvisional {
		t.Fatalf("4.0d span (>= %.0fd floor) must NOT be PROVISIONAL", minHonestSpanDays)
	}
	if strings.Contains(RenderFleetBenefit(rep), "PROVISIONAL") {
		t.Fatalf("settled-span render must not carry PROVISIONAL:\n%s", RenderFleetBenefit(rep))
	}
}

func TestFoldFleetBenefitFallsBackToSavingsRowsForExtensionWhenUsageMissing(t *testing.T) {
	rep := FoldFleetBenefit(nil, []SavingsRow{
		{Date: "2026-06-22", Provider: "fak", Mechanism: "compaction_shed", CompactionShedTokens: 100},
		{Date: "2026-06-23", Provider: "fak", Mechanism: "compaction_shed", CompactionShedTokens: 200},
	}, nil, FleetBenefitOptions{})
	if rep.ContextExtensionTokens != 300 {
		t.Fatalf("fallback extension tokens = %d, want both compaction rows summed to 300", rep.ContextExtensionTokens)
	}
}
