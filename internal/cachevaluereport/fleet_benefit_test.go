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

// Live fleet aggregate quoted in #3662, reproduced here so the per-work fold is pinned
// against the real corpus it was designed from rather than a toy fixture.
const (
	liveSpendUSD     = 2187.29
	liveAvoidedUSD   = 7994.70
	liveDecisions    = 419702
	liveExitSessions = 2159
)

// TestFleetBenefitCostPerUnitOfAgenticWork pins the #3662 fold: the cumulative totals
// divided by the WORK done (kernel decisions, exit sessions, multi-turn turns) rather
// than by the calendar, reproduced from the LIVE aggregate. It also pins the honesty
// fence that makes the block publishable — the dollar axis renders OBSERVED and carries
// the thin-window PROVISIONAL flag, the token axis renders WITNESSED and does NOT, and
// the two are never blended into one figure.
func TestFleetBenefitCostPerUnitOfAgenticWork(t *testing.T) {
	// Multi-turn corpus (turns >= 2): 20 turns over 2000 prompt / 1600 reused tokens
	// → 100 prompt and 80 reused per turn, a 0.8 reuse ratio. The third row is
	// SINGLE-turn with a huge prompt: it must be excluded from every per-turn
	// denominator (it had no previous turn to reuse from), so if the turns >= 2 gate
	// ever regresses, prompt/turn explodes and this test fails loudly.
	track1 := []cachevalueledger.Row{
		{Date: "2026-06-22", SessionType: "run", Turns: 10, PromptTokens: 1000, ReusedTokens: 800},
		{Date: "2026-06-23", SessionType: "run", Turns: 10, PromptTokens: 1000, ReusedTokens: 800},
		{Date: "2026-06-23", SessionType: "run", Turns: 1, PromptTokens: 999999, ReusedTokens: 0},
	}
	// Savings rows dated exactly 2 days apart → span 2.0d, under the 3d floor, so the
	// dollar-per-work numbers must inherit PROVISIONAL (acceptance criterion 4).
	// Provider rebate 6000.00 + fak compaction 1994.70 − 0 premium = 7994.70 avoided.
	track2 := []SavingsRow{
		{
			Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			CacheReadTokens: 2_000_000, SavedTokenEquiv: 1_000_000, NetSavedTokenEquiv: 1_000_000,
			RebateUSD: 6000.00, WritePremiumUSD: 0, SpendUSD: liveSpendUSD,
		},
		{
			Date: "2026-06-24", Provider: "fak", Mechanism: "compaction_shed",
			CompactionShedTokens: 50_000, SavedTokenEquiv: 50_000, NetSavedTokenEquiv: 50_000,
			CompactionSavedUSD: 1994.70,
		},
	}
	// liveExitSessions exit rows carrying liveDecisions kernel decisions between them;
	// the remainder rides on the first row so the fleet total is EXACTLY the live figure.
	base := uint64(liveDecisions / liveExitSessions)
	usage := make([]gatewayusageledger.Row, 0, liveExitSessions)
	for i := 0; i < liveExitSessions; i++ {
		c := gatewayusageledger.Counters{Total: base, Allowed: base}
		if i == 0 {
			c.Total += uint64(liveDecisions) - base*uint64(liveExitSessions)
			c.Allowed = c.Total
			c.CompactionFired = 1
			c.CompactionShedTokens = 4000 // → 4000/20 = 200 shed tok/turn
		}
		usage = append(usage, gatewayusageledger.NewRow("exit", "guard", "claude", "g-1",
			90*time.Second, nil, c, time.Date(2026, 6, 22, 10, 0, 0, 0, time.UTC)))
	}

	rep := FoldFleetBenefit(track1, track2, usage, FleetBenefitOptions{})

	// Preconditions: the fixture really does reproduce the live aggregate.
	if rep.KernelDecisions != liveDecisions || rep.ExitSessions != liveExitSessions {
		t.Fatalf("fixture denominators = %d decisions / %d exit sessions, want %d/%d",
			rep.KernelDecisions, rep.ExitSessions, liveDecisions, liveExitSessions)
	}
	if !approxTrack2(rep.ObservedActualSpendUSD, liveSpendUSD) || !approxTrack2(rep.ObservedAPICostAvoidedUSD, liveAvoidedUSD) {
		t.Fatalf("fixture dollars = spend %.2f avoided %.2f, want %.2f/%.2f",
			rep.ObservedActualSpendUSD, rep.ObservedAPICostAvoidedUSD, liveSpendUSD, liveAvoidedUSD)
	}
	// The single-turn row is excluded from every per-turn denominator.
	if rep.MultiTurnTurns != 20 || rep.MultiTurnPromptTokens != 2000 || rep.MultiTurnReusedTokens != 1600 {
		t.Fatalf("multi-turn denominators = %d turns / %d prompt / %d reused, want 20/2000/1600 (single-turn row must be excluded)",
			rep.MultiTurnTurns, rep.MultiTurnPromptTokens, rep.MultiTurnReusedTokens)
	}

	// Dollar axis (OBSERVED): avoided / spend / counterfactual per unit of work.
	if rep.AvoidedUSDPerDecision == nil || !approxTrack2(*rep.AvoidedUSDPerDecision, liveAvoidedUSD/liveDecisions) {
		t.Fatalf("avoided $/decision = %v, want %.12f", rep.AvoidedUSDPerDecision, liveAvoidedUSD/liveDecisions)
	}
	if rep.SpendUSDPerDecision == nil || !approxTrack2(*rep.SpendUSDPerDecision, liveSpendUSD/liveDecisions) {
		t.Fatalf("spend $/decision = %v, want %.12f", rep.SpendUSDPerDecision, liveSpendUSD/liveDecisions)
	}
	if rep.CounterfactualUSDPerDecision == nil || !approxTrack2(*rep.CounterfactualUSDPerDecision, (liveSpendUSD+liveAvoidedUSD)/liveDecisions) {
		t.Fatalf("counterfactual $/decision = %v, want %.12f", rep.CounterfactualUSDPerDecision, (liveSpendUSD+liveAvoidedUSD)/liveDecisions)
	}
	// The counterfactual pair must close: spend + avoided == counterfactual, per decision.
	if !approxTrack2(*rep.SpendUSDPerDecision+*rep.AvoidedUSDPerDecision, *rep.CounterfactualUSDPerDecision) {
		t.Fatalf("per-decision counterfactual pair does not close: %.12f + %.12f != %.12f",
			*rep.SpendUSDPerDecision, *rep.AvoidedUSDPerDecision, *rep.CounterfactualUSDPerDecision)
	}
	if rep.AvoidedUSDPerExitSession == nil || !approxTrack2(*rep.AvoidedUSDPerExitSession, liveAvoidedUSD/liveExitSessions) {
		t.Fatalf("avoided $/exit-session = %v, want %.12f", rep.AvoidedUSDPerExitSession, liveAvoidedUSD/liveExitSessions)
	}

	// Token axis (WITNESSED): saved token-equiv and work size per multi-turn turn.
	if rep.SavedTokenEqPerTurn == nil || !approxTrack2(*rep.SavedTokenEqPerTurn, rep.TotalSavedTokenEq/20) {
		t.Fatalf("saved-tok-eq/turn = %v, want %.12f", rep.SavedTokenEqPerTurn, rep.TotalSavedTokenEq/20)
	}
	if rep.TotalSavedTokenEq <= 0 {
		t.Fatalf("fixture must fold a positive saved token-equiv, got %.2f", rep.TotalSavedTokenEq)
	}
	if rep.PromptTokensPerTurn == nil || !approxTrack2(*rep.PromptTokensPerTurn, 100) ||
		rep.ReusedTokensPerTurn == nil || !approxTrack2(*rep.ReusedTokensPerTurn, 80) ||
		rep.ShedTokensPerTurn == nil || !approxTrack2(*rep.ShedTokensPerTurn, 200) {
		t.Fatalf("work per turn = prompt %v reused %v shed %v, want 100/80/200",
			rep.PromptTokensPerTurn, rep.ReusedTokensPerTurn, rep.ShedTokensPerTurn)
	}

	// A 2.0d savings span is under the 3d floor, so the dollar-per-work numbers are
	// PROVISIONAL (acceptance criterion 4).
	if !rep.RateProvisional {
		t.Fatalf("span %.2fd should be PROVISIONAL under the %.0fd floor", rep.SpanDays, minHonestSpanDays)
	}

	out := RenderFleetBenefit(rep)
	for _, want := range []string{
		"cost per unit of agentic work:",
		// Dollar lines: OBSERVED, list-priced, and flagged PROVISIONAL.
		"$/decision (OBSERVED, list-priced projection; over 419702 decision(s)): spend $0.005212 vs counterfactual $0.024260 = avoided $0.019049 [PROVISIONAL: span < 3d]",
		"$/exit-session (OBSERVED, list-priced projection; over 2159 session(s)): avoided $3.702964 [PROVISIONAL: span < 3d]",
		// Token lines: WITNESSED, and the explicit ratio x size x count decomposition.
		"saved-tok-eq/turn (WITNESSED; over 20 multi-turn turn(s)):",
		"work per turn (WITNESSED): prompt=100.00 reused_prefix=80.00 shed=200.00 token(s)",
		"three-factor decomposition (WITNESSED): reuse_ratio 0.8000 x 100.00 prompt tok/turn x 20 turn(s) = 1600 reused token(s)",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("per-work render missing %q:\n%s", want, out)
		}
	}
	// The fence must not blend: the WITNESSED token lines carry no dollar sign and no
	// PROVISIONAL flag (that flag belongs to the OBSERVED dollar projection alone).
	for _, line := range strings.Split(out, "\n") {
		if !strings.Contains(line, "(WITNESSED") {
			continue
		}
		if strings.Contains(line, "PROVISIONAL") || strings.Contains(line, "$") {
			t.Fatalf("WITNESSED per-work line blended an OBSERVED dollar/PROVISIONAL marker: %q", line)
		}
	}
}

// TestFleetBenefitPerWorkUndefinedWithoutDenominators pins the "nil means UNDEFINED, not
// zero" contract: a corpus with no decisions, no exit sessions and no multi-turn turns
// must omit the per-work block entirely rather than render a 0.00 that would read as a
// measured floor.
func TestFleetBenefitPerWorkUndefinedWithoutDenominators(t *testing.T) {
	// One SINGLE-turn track1 row: real work, but no turn that could have reused a prefix.
	track1 := []cachevalueledger.Row{
		{Date: "2026-06-22", SessionType: "run", Turns: 1, PromptTokens: 5000, ReusedTokens: 0},
	}
	rep := FoldFleetBenefit(track1, nil, nil, FleetBenefitOptions{})
	if rep.AvoidedUSDPerDecision != nil || rep.SpendUSDPerDecision != nil ||
		rep.CounterfactualUSDPerDecision != nil || rep.AvoidedUSDPerExitSession != nil ||
		rep.SavedTokenEqPerTurn != nil || rep.PromptTokensPerTurn != nil ||
		rep.ReusedTokensPerTurn != nil || rep.ShedTokensPerTurn != nil {
		t.Fatalf("zero-denominator corpus must leave every per-work ratio nil (UNDEFINED): %+v", rep)
	}
	if out := RenderFleetBenefit(rep); strings.Contains(out, "cost per unit of agentic work") {
		t.Fatalf("per-work block must be omitted when every denominator is zero:\n%s", out)
	}
}
