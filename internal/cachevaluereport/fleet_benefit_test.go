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
		gatewayusageledger.NewRow("exit", "guard", "claude", "g-1", 90*time.Second, nil, gatewayusageledger.Counters{
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
	// The usage ledger carried the shed, so the citation counts usage rows (#2711).
	if rep.ShedRows != 1 {
		t.Fatalf("shed rows = %d, want 1 (the one usage row with a nonzero shed)", rep.ShedRows)
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
		"; cited from 1 ledgered shed session(s)",
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

// TestFoldFleetBenefitFakShareBasisSweep witnesses the #2807 basis sweep: the SAME
// fleet fak_share recomputed with the fak compaction shed valued at 1.0x (gross),
// 0.1x (marginal), and the honest per-row warm/cold blend (net = FakSharePct). A
// blended shed (warm+cold) makes all three distinct and ordered marginal <= net <=
// gross, and the gross-net gap is the overstatement the report makes visible.
func TestFoldFleetBenefitFakShareBasisSweep(t *testing.T) {
	track2 := []SavingsRow{
		{
			Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			CacheReadTokens: 400, SavedTokenEquiv: 360, NetSavedTokenEquiv: 360,
		},
		{
			// shed 1000 with a warm witness of 400 → net = 400*0.1 + 600 = 640 (blended),
			// so gross (1000), net (640), and marginal (100) are all distinct.
			Date: "2026-06-22", Provider: "fak", Mechanism: "compaction_shed",
			CompactionShedTokens: 1000, CompactionCacheReadTokens: 400,
			SavedTokenEquiv: 640, NetSavedTokenEquiv: 640,
		},
	}
	rep := FoldFleetBenefit(nil, track2, nil, FleetBenefitOptions{})

	// Net (honest blend) is the existing headline: fak 640 / (provider 360 + fak 640) = 64%.
	if rep.FakSharePct == nil || !approxTrack2(*rep.FakSharePct, 64.0) {
		t.Fatalf("net fak_share = %v, want 64.0%%", rep.FakSharePct)
	}
	// Raw shed captured from the savings row (distinct from the absent usage count).
	if rep.FakCompactionShedTokensSavings != 1000 {
		t.Fatalf("savings-row shed = %d, want 1000", rep.FakCompactionShedTokensSavings)
	}
	// Gross books the whole shed at 1.0x: fak 1000 / (360 + 1000).
	if rep.FakShareGrossPct == nil || !approxTrack2(*rep.FakShareGrossPct, 100.0*1000.0/1360.0) {
		t.Fatalf("gross fak_share = %v, want %.6f%%", rep.FakShareGrossPct, 100.0*1000.0/1360.0)
	}
	// Marginal books it at 0.1x: fak 100 / (360 + 100).
	if rep.FakShareMarginalPct == nil || !approxTrack2(*rep.FakShareMarginalPct, 100.0*100.0/460.0) {
		t.Fatalf("marginal fak_share = %v, want %.6f%%", rep.FakShareMarginalPct, 100.0*100.0/460.0)
	}
	// The sweep is ordered marginal <= net <= gross by construction (#2807/#2798).
	if !(*rep.FakShareMarginalPct <= *rep.FakSharePct && *rep.FakSharePct <= *rep.FakShareGrossPct) {
		t.Fatalf("sweep not ordered marginal<=net<=gross: %.4f / %.4f / %.4f",
			*rep.FakShareMarginalPct, *rep.FakSharePct, *rep.FakShareGrossPct)
	}
	out := RenderFleetBenefit(rep)
	for _, want := range []string{
		"fak_share basis sweep (shed valued 3 ways; #2807)",
		"gross(1.0x)=73.5294%",
		"marginal(0.1x)=21.7391%",
		"net(observed)=64.0000%",
		"(overstatement)",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("basis-sweep render missing %q:\n%s", want, out)
		}
	}
}

// TestFoldFleetBenefitFakShareBasisSweepAbsentWithoutShed pins that a provider-only
// corpus (no fak compaction shed) leaves the sweep nil and prints no sweep line, so
// the extra observability never fabricates a gap where there is nothing to reprice.
func TestFoldFleetBenefitFakShareBasisSweepAbsentWithoutShed(t *testing.T) {
	rep := FoldFleetBenefit(nil, []SavingsRow{
		{Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			CacheReadTokens: 400, SavedTokenEquiv: 360, NetSavedTokenEquiv: 360},
	}, nil, FleetBenefitOptions{})
	if rep.FakCompactionShedTokensSavings != 0 || rep.FakShareGrossPct != nil || rep.FakShareMarginalPct != nil {
		t.Fatalf("no-shed corpus must leave the sweep empty: shed=%d gross=%v marginal=%v",
			rep.FakCompactionShedTokensSavings, rep.FakShareGrossPct, rep.FakShareMarginalPct)
	}
	if strings.Contains(RenderFleetBenefit(rep), "basis sweep") {
		t.Fatalf("no-shed render must not print a basis sweep line:\n%s", RenderFleetBenefit(rep))
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
	// Fallback source ⇒ the citation counts the contributing savings rows (#2711).
	if rep.ShedRows != 2 {
		t.Fatalf("fallback shed rows = %d, want 2 (both compaction savings rows)", rep.ShedRows)
	}
}

// TestFleetBenefitSessionExtensionHonestZero pins the #2711 done condition: a corpus
// whose sessions never shed renders an explicit "still zero because X" session-extension
// line naming the unmet precondition, never a bare 0 or a silently omitted line.
func TestFleetBenefitSessionExtensionHonestZero(t *testing.T) {
	usage := []gatewayusageledger.Row{
		gatewayusageledger.NewRow("exit", "guard", "claude", "g-1", 30*time.Second, nil, gatewayusageledger.Counters{
			Total: 3, Allowed: 3, InputTokens: 500, OutputTokens: 40,
		}, time.Date(2026, 6, 22, 10, 0, 0, 0, time.UTC)),
	}
	rep := FoldFleetBenefit(nil, nil, usage, FleetBenefitOptions{ContextBudgetTokens: 15000})
	if rep.ContextExtensionTokens != 0 || rep.ShedRows != 0 {
		t.Fatalf("no-shed corpus folded extension=%d rows=%d, want 0/0", rep.ContextExtensionTokens, rep.ShedRows)
	}
	out := RenderFleetBenefit(rep)
	want := "session extension: still zero — no WITNESSED compaction-shed tokens in any of the 1 recorded usage row(s)"
	if !strings.Contains(out, want) {
		t.Fatalf("honest-zero render missing %q:\n%s", want, out)
	}
	if strings.Contains(out, "0 WITNESSED context token(s) shed") {
		t.Fatalf("honest-zero render must not show a bare zero figure:\n%s", out)
	}

	// Same corpus WITHOUT a budget flag: the line must still appear (previously it was
	// silently omitted), carrying the same reason.
	repNoBudget := FoldFleetBenefit(nil, nil, usage, FleetBenefitOptions{})
	if !strings.Contains(RenderFleetBenefit(repNoBudget), "session extension: still zero") {
		t.Fatalf("budget-less honest-zero line missing:\n%s", RenderFleetBenefit(repNoBudget))
	}
}
