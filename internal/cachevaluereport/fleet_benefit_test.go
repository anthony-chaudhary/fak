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

	out := RenderFleetBenefit(rep)
	for _, want := range []string{
		"Fleet aggregate",
		"saved token-equiv: provider=1800 fak=3800 total=5600",
		"API cost: observed_spend=$0.0200 counterfactual=$0.0430 avoided=$0.0230 reduction=53.49%",
		"session extension: 3000 WITNESSED context token(s) shed = 20.00% of a 15000-token budget",
		"provider prompt-cache dollars are OBSERVED/provider-relayed projections",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
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
