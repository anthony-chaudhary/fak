package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// TestBuildSpendRollupEveryFigureLabeled is #2903's done condition at the fold:
// every figure the roster fold emits carries a valuation basis and a
// WITNESSED/OBSERVED label, so the gate passes on real output.
func TestBuildSpendRollupEveryFigureLabeled(t *testing.T) {
	throttled := true
	live := 3
	avail := true
	rows := []fleetaccounts.Account{
		{Tag: "alpha", Kind: fleetaccounts.KindWorker, Available: &avail, LiveSessions: &live},
		{Tag: "beta", Kind: fleetaccounts.KindWorker, Available: &avail, Throttled: &throttled},
		{Tag: "skipped-no-runtime-row", Kind: fleetaccounts.KindWorker},
		{Tag: "skipped-non-worker", Kind: fleetaccounts.KindExcluded, Available: &avail},
	}
	rollup := buildSpendRollup(rows)
	if defects := metrics.GateSpendLabeled(rollup); len(defects) != 0 {
		t.Fatalf("roster fold emitted unlabeled figures: %v", defects)
	}
	// 2 accounts x 2 figures + 2 fleet totals; the rows without a runtime fold
	// contribute nothing rather than a guessed zero.
	if got := len(rollup.Figures); got != 6 {
		t.Fatalf("figure count = %d, want 6: %+v", got, rollup.Figures)
	}
}

// TestBuildSpendRollupProvenanceSides locks the confusion risk the issue names:
// the session count is fak-authored (WITNESSED) and the throttle state is the
// provider's relayed number (OBSERVED) — never swapped, never upgraded in totals.
func TestBuildSpendRollupProvenanceSides(t *testing.T) {
	avail := true
	rows := []fleetaccounts.Account{{Tag: "alpha", Kind: fleetaccounts.KindWorker, Available: &avail}}
	for _, f := range buildSpendRollup(rows).Figures {
		switch f.Metric {
		case "live_sessions", "fleet_live_sessions":
			if f.Provenance != metrics.SpendWitnessed {
				t.Errorf("%s provenance = %q, want WITNESSED", f.Metric, f.Provenance)
			}
		case "usage_throttled", "fleet_usage_throttled":
			if f.Provenance != metrics.SpendObserved {
				t.Errorf("%s provenance = %q, want OBSERVED", f.Metric, f.Provenance)
			}
		default:
			t.Errorf("unexpected metric %q", f.Metric)
		}
	}
}
