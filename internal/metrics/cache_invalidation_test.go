package metrics

import (
	"encoding/json"
	"testing"
)

// TestParseInvalidationModeDefaultsDeferred is the core contract: a state-mutating
// operation inherits the deferred default and only pays the cold rebuild when the
// operator explicitly opts in with --now (#2895).
func TestParseInvalidationModeDefaultsDeferred(t *testing.T) {
	if got := ParseInvalidationMode(false); got != InvalidationDeferred {
		t.Fatalf("ParseInvalidationMode(false) = %q, want deferred (the default, no --now)", got)
	}
	if got := ParseInvalidationMode(true); got != InvalidationNow {
		t.Fatalf("ParseInvalidationMode(true) = %q, want now (--now opt-in)", got)
	}
	if !InvalidationNow.IsNow() {
		t.Fatalf("InvalidationNow.IsNow() = false, want true")
	}
	if InvalidationDeferred.IsNow() {
		t.Fatalf("InvalidationDeferred.IsNow() = true, want false")
	}
	// A zero/unknown mode is treated as the deferred default: effective next session.
	if !InvalidationDeferred.EffectiveNextSession() {
		t.Fatalf("deferred should be effective next session")
	}
	if InvalidationNow.EffectiveNextSession() {
		t.Fatalf("now should be effective this turn, not next session")
	}
	var zero InvalidationMode
	if !zero.EffectiveNextSession() {
		t.Fatalf("zero mode should default to deferred (effective next session)")
	}
}

// TestWitnessInvalidationPricesEachMode witnesses the trade each way: deferred avoids
// the cold rebuild now (saving == cost) at the price of a staleness window; now pays
// the rebuild immediately with no saving and no staleness.
func TestWitnessInvalidationPricesEachMode(t *testing.T) {
	deferred := WitnessInvalidation(InvalidationDeferred, 1800, 4)
	if deferred.TokensSavedByDeferring != 1800 {
		t.Fatalf("deferred TokensSavedByDeferring = %d, want 1800 (== cold rebuild avoided now)", deferred.TokensSavedByDeferring)
	}
	if deferred.StalenessTurns != 4 {
		t.Fatalf("deferred StalenessTurns = %d, want 4", deferred.StalenessTurns)
	}
	if !deferred.EffectiveNextSession {
		t.Fatalf("deferred witness should be effective next session")
	}

	now := WitnessInvalidation(InvalidationNow, 1800, 4)
	if now.TokensSavedByDeferring != 0 {
		t.Fatalf("now TokensSavedByDeferring = %d, want 0 (rebuild paid, nothing deferred)", now.TokensSavedByDeferring)
	}
	if now.StalenessTurns != 0 {
		t.Fatalf("now StalenessTurns = %d, want 0 (change in effect this turn)", now.StalenessTurns)
	}
	if now.EffectiveNextSession {
		t.Fatalf("now witness should NOT be effective next session")
	}
	if now.ColdRebuildTokens != 1800 {
		t.Fatalf("now ColdRebuildTokens = %d, want 1800 (the cost paid)", now.ColdRebuildTokens)
	}
}

// TestWitnessInvalidationClampsNegative guards the caller-supplied cost inputs.
func TestWitnessInvalidationClampsNegative(t *testing.T) {
	w := WitnessInvalidation(InvalidationDeferred, -5, -3)
	if w.ColdRebuildTokens != 0 || w.TokensSavedByDeferring != 0 || w.StalenessTurns != 0 {
		t.Fatalf("negative inputs must clamp to zero, got %+v", w)
	}
}

// TestWitnessInvalidationTradeoffPricesBothWays confirms the tradeoff view prices the
// same mutation under both modes so a command can surface the choice.
func TestWitnessInvalidationTradeoffPricesBothWays(t *testing.T) {
	tr := WitnessInvalidationTradeoff(1200, 3)
	if tr.Deferred.Mode != InvalidationDeferred || tr.Now.Mode != InvalidationNow {
		t.Fatalf("tradeoff modes = %q/%q, want deferred/now", tr.Deferred.Mode, tr.Now.Mode)
	}
	if tr.Deferred.TokensSavedByDeferring != 1200 {
		t.Fatalf("deferred side should save 1200, got %d", tr.Deferred.TokensSavedByDeferring)
	}
	if tr.Now.TokensSavedByDeferring != 0 {
		t.Fatalf("now side should save 0, got %d", tr.Now.TokensSavedByDeferring)
	}
}

// TestRecommendMode exercises the decision support: a free rebuild is never worth a
// staleness window (Now); a staleness window beyond tolerance forces Now; otherwise the
// deferred default wins.
func TestRecommendMode(t *testing.T) {
	cases := []struct {
		name              string
		coldRebuildTokens int64
		stalenessTurns    int
		maxStaleness      int
		want              InvalidationMode
	}{
		{"free rebuild recommends now", 0, 4, 10, InvalidationNow},
		{"costly rebuild within tolerance defers", 1500, 2, 5, InvalidationDeferred},
		{"costly rebuild beyond tolerance forces now", 1500, 9, 5, InvalidationNow},
		{"costly rebuild exactly at tolerance defers", 900, 5, 5, InvalidationDeferred},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tr := WitnessInvalidationTradeoff(tc.coldRebuildTokens, tc.stalenessTurns)
			if got := tr.RecommendMode(tc.maxStaleness); got != tc.want {
				t.Fatalf("RecommendMode(%d) = %q, want %q", tc.maxStaleness, got, tc.want)
			}
		})
	}
}

// TestFoldCacheInvalidationReport is the witnessed report the issue asks for: a session's
// mutation witnesses folded into the total cold-rebuild cost AVOIDED by deferring vs the
// cost PAID by the opt-in immediate mutations, plus the worst staleness window opened.
func TestFoldCacheInvalidationReport(t *testing.T) {
	report := FoldCacheInvalidation([]CacheInvalidationWitness{
		WitnessInvalidation(InvalidationDeferred, 1000, 3),
		WitnessInvalidation(InvalidationDeferred, 400, 7),
		WitnessInvalidation(InvalidationNow, 2500, 0),
	})

	if report.Mutations != 3 {
		t.Fatalf("Mutations = %d, want 3", report.Mutations)
	}
	if report.Deferred != 2 || report.Immediate != 1 {
		t.Fatalf("deferred/immediate = %d/%d, want 2/1", report.Deferred, report.Immediate)
	}
	if report.TokensSavedByDeferring != 1400 {
		t.Fatalf("TokensSavedByDeferring = %d, want 1400 (1000+400 avoided)", report.TokensSavedByDeferring)
	}
	if report.ColdRebuildTokensPaid != 2500 {
		t.Fatalf("ColdRebuildTokensPaid = %d, want 2500 (the --now mutation)", report.ColdRebuildTokensPaid)
	}
	if report.MaxStalenessTurns != 7 {
		t.Fatalf("MaxStalenessTurns = %d, want 7 (worst deferral window)", report.MaxStalenessTurns)
	}

	// The report is an operator readout: it must carry the priced-trade keys through JSON.
	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var shaped map[string]any
	if err := json.Unmarshal(raw, &shaped); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	for _, key := range []string{
		"mutations", "deferred", "immediate",
		"tokens_saved_by_deferring", "cold_rebuild_tokens_paid", "max_staleness_turns",
	} {
		if _, ok := shaped[key]; !ok {
			t.Fatalf("cache-invalidation report JSON missing %q: %s", key, raw)
		}
	}
}

// TestFoldCacheInvalidationEmpty is the zero case: no mutations, an all-zero readout.
func TestFoldCacheInvalidationEmpty(t *testing.T) {
	report := FoldCacheInvalidation(nil)
	if report.Mutations != 0 || report.TokensSavedByDeferring != 0 || report.MaxStalenessTurns != 0 {
		t.Fatalf("empty fold should be all-zero, got %+v", report)
	}
}
