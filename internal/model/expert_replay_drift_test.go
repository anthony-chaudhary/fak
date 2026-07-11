package model

import "testing"

// buildEvents appends one same-sized touch per (layer 0, expert) id in order.
func driftEvents(bytes int64, experts ...int) []ExpertAccessTraceEvent {
	out := make([]ExpertAccessTraceEvent, 0, len(experts))
	for _, e := range experts {
		out = append(out, ExpertAccessTraceEvent{Layer: 0, Expert: e, WeightBytes: bytes})
	}
	return out
}

// TestExpertReplayDriftExposesLateHorizonCollapse is the observed answer for the long-horizon
// sibling of D1. The trace runs two back-to-back regimes at one 3-expert byte budget:
//
//	regime A (events 0..11): a working set of exactly 3 experts with varied reuse distances
//	  -- everything the offline oracle keeps, pagedRing-LRU also keeps (GoodDecisionRatio=1),
//	  and observed recency strictly beats the layer-sequential null (locality is real).
//	regime B (events 12..23): a 4-expert cyclic scan one wider than the budget -- the classic
//	  LRU-thrash pattern where the oracle preserves hits pagedRing-LRU throws away
//	  (GoodDecisionRatio=0), and recency no longer beats the null (the prior is dead).
//
// THE OBSERVED ANSWER: the whole-trace GoodDecisionRatio sits strictly between 0 and 1 -- it
// reads "LRU is somewhat suboptimal" and conceals that the entire second half of the horizon
// has collapsed to pure thrashing. The per-window drift makes that collapse a named window.
func TestExpertReplayDriftExposesLateHorizonCollapse(t *testing.T) {
	const bytes = int64(144)
	events := driftEvents(bytes,
		// regime A: working set {0,1,2}, hot expert 0 (distance 2), warm 1 and 2 (distance 4).
		0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2,
		// regime B: cyclic scan {3,4,5,6}, one wider than the 3-expert budget.
		3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6,
	)
	trace := ExpertAccessTrace{
		Schema: ExpertReplayTraceSchema, Name: "horizon-drift-witness", Source: "deterministic-unit",
		BudgetBytes: 3 * bytes, Events: events,
	}

	report, err := ReplayExpertAccessTraceDrift(trace, 2)
	if err != nil {
		t.Fatalf("ReplayExpertAccessTraceDrift: %v", err)
	}
	if len(report.Windows) != 2 {
		t.Fatalf("windows=%d, want 2", len(report.Windows))
	}
	w0, w1 := report.Windows[0], report.Windows[1]

	// Regime A window: LRU is offline-optimal and its locality prior is real.
	if w0.LRUGoodDecision != 1 {
		t.Fatalf("regime-A window GoodDecisionRatio=%v, want 1", w0.LRUGoodDecision)
	}
	if w0.LocalityInverted {
		t.Fatalf("regime-A window locality inverted; want recency to beat the null (corr=%v null=%v)",
			w0.Locality.RecencyNextUseCorrelation, w0.Locality.LayerSequentialNullCorrelation)
	}
	// Regime B window: LRU recovers none of the hits the oracle preserves, and the prior is dead.
	if w1.LRUGoodDecision != 0 {
		t.Fatalf("regime-B window GoodDecisionRatio=%v, want 0", w1.LRUGoodDecision)
	}
	if w1.OracleHitBytes <= w1.LRUHitBytes {
		t.Fatalf("regime-B oracle hits=%d must exceed LRU hits=%d to be genuine regret", w1.OracleHitBytes, w1.LRUHitBytes)
	}
	if !w1.LocalityInverted {
		t.Fatalf("regime-B window locality not inverted; corr=%v null=%v samples=%d",
			w1.Locality.RecencyNextUseCorrelation, w1.Locality.LayerSequentialNullCorrelation, w1.Locality.Samples)
	}
	if !w0.OracleExact || !w1.OracleExact {
		t.Fatalf("small windows unexpectedly used the approximate oracle: w0=%t w1=%t", w0.OracleExact, w1.OracleExact)
	}

	// The drift the whole-trace scalar averages away.
	if report.GoodDecisionMin != 0 || report.GoodDecisionMax != 1 || report.GoodDecisionSpread != 1 {
		t.Fatalf("drift min/max/spread=%v/%v/%v, want 0/1/1",
			report.GoodDecisionMin, report.GoodDecisionMax, report.GoodDecisionSpread)
	}
	if report.WorstWindow != 1 {
		t.Fatalf("worst window=%d, want 1 (the late-horizon collapse)", report.WorstWindow)
	}
	if report.GoodDecisionTrendSlope >= 0 {
		t.Fatalf("trend slope=%v, want negative (regret worsens over the horizon)", report.GoodDecisionTrendSlope)
	}
	if report.LocalityInversions != 1 {
		t.Fatalf("locality inversions=%d, want exactly 1 (the late window only)", report.LocalityInversions)
	}

	// The concealment itself: the single scalar D1 reports is strictly inside (worst, best),
	// so on its own it can neither reveal nor locate the fully-thrashing late window.
	if !(report.WholeTraceGoodDecision > report.GoodDecisionMin && report.WholeTraceGoodDecision < report.GoodDecisionMax) {
		t.Fatalf("whole-trace GoodDecisionRatio=%v, want strictly between min=%v and max=%v",
			report.WholeTraceGoodDecision, report.GoodDecisionMin, report.GoodDecisionMax)
	}
	t.Logf("horizon drift: whole=%.4f windows=[A %.4f, B %.4f] spread=%.4f slope=%.4f inversions=%d worst=%d",
		report.WholeTraceGoodDecision, w0.LRUGoodDecision, w1.LRUGoodDecision,
		report.GoodDecisionSpread, report.GoodDecisionTrendSlope, report.LocalityInversions, report.WorstWindow)
}

func TestExpertReplayDriftGuardsWindowCount(t *testing.T) {
	const bytes = int64(144)
	trace := ExpertAccessTrace{
		Schema: ExpertReplayTraceSchema, Name: "guard", Source: "deterministic-unit",
		BudgetBytes: 2 * bytes, Events: driftEvents(bytes, 0, 1, 0),
	}
	if _, err := ReplayExpertAccessTraceDrift(trace, 0); err == nil {
		t.Fatal("accepted a zero window count")
	}
	if _, err := ReplayExpertAccessTraceDrift(trace, len(trace.Events)+1); err == nil {
		t.Fatal("accepted more windows than sized events")
	}
	// A single window must reproduce the whole-trace gauge exactly and report no drift.
	report, err := ReplayExpertAccessTraceDrift(trace, 1)
	if err != nil {
		t.Fatalf("single-window drift: %v", err)
	}
	if report.GoodDecisionSpread != 0 || report.GoodDecisionTrendSlope != 0 {
		t.Fatalf("single window reported drift spread=%v slope=%v, want 0/0", report.GoodDecisionSpread, report.GoodDecisionTrendSlope)
	}
	if report.Windows[0].LRUGoodDecision != report.WholeTraceGoodDecision {
		t.Fatalf("single window GDR=%v disagrees with whole-trace GDR=%v",
			report.Windows[0].LRUGoodDecision, report.WholeTraceGoodDecision)
	}
}
