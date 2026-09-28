package main

// dispatch_tick_preflight_share_test.go — the payload-to-gate ratio
// (gateshare/fak/lesson-4).
//
// The claim being pinned: admission time is the product, not a fixed cost. On the
// measured ledger the most recent tick spent 31625ms of a 52378ms tick in preflight
// (60.3%), loop-wide preflight mean 10425ms against a 21828ms tick mean (47.8%), and
// the same tick carried preflight_cap 6 against max_workers 30. A tick that spends
// more wall-clock DECIDING to work than working has an inverted ladder, and above
// 300permille the correct response is to delete a gate rather than tune one.
//
// The metric is an integer permille rather than a float ratio because the ledger's
// metrics map is map[string]int64: a fold must not be able to round two different
// ticks onto the same value at the alarm boundary.

import "testing"

func TestDispatchTickPreflightSharePermille(t *testing.T) {
	cases := []struct {
		name      string
		timings   map[string]int64
		wantKey   bool
		wantValue int64
	}{
		{
			// The measured shape: 31625 / 52378 -> 603permille.
			name:      "measured tick is over the alarm boundary",
			timings:   map[string]int64{"preflight": 31625, "total": 52378},
			wantKey:   true,
			wantValue: 603,
		},
		{
			name:      "healthy tick stays well under the boundary",
			timings:   map[string]int64{"preflight": 1000, "total": 20000},
			wantKey:   true,
			wantValue: 50,
		},
		{
			// Absent is "not measured", which must never render as 0permille — 0 would
			// claim the gate was free.
			name:    "no preflight duration means no key, not zero",
			timings: map[string]int64{"total": 20000},
			wantKey: false,
		},
		{
			name:    "no total means the ratio is undefined, not zero",
			timings: map[string]int64{"preflight": 900},
			wantKey: false,
		},
		{
			name:    "zero total is a divide-by-zero guard",
			timings: map[string]int64{"preflight": 900, "total": 0},
			wantKey: false,
		},
		{
			name:    "empty timings emit nothing",
			timings: map[string]int64{},
			wantKey: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			metrics := dispatchTickLoopMetrics(map[string]any{"timings_ms": tc.timings})
			got, ok := metrics[dispatchTickPreflightPermilleKey]
			if ok != tc.wantKey {
				t.Fatalf("key present = %v, want %v (metrics=%v)", ok, tc.wantKey, metrics)
			}
			if tc.wantKey && got != tc.wantValue {
				t.Fatalf("%s = %d, want %d", dispatchTickPreflightPermilleKey, got, tc.wantValue)
			}
		})
	}
}

// The measured tick must actually breach the documented alarm boundary, otherwise the
// "delete a gate, don't tune one" guidance is advice about a condition that never fires.
func TestDispatchTickPreflightShareBreachesAlarmBoundary(t *testing.T) {
	metrics := dispatchTickLoopMetrics(map[string]any{
		"timings_ms": map[string]int64{"preflight": 31625, "total": 52378},
	})
	got := metrics[dispatchTickPreflightPermilleKey]
	if got < dispatchTickPreflightAlarmPermille {
		t.Fatalf("measured tick share = %dpermille, want >= alarm boundary %d",
			got, dispatchTickPreflightAlarmPermille)
	}
}

// The ratio is only auditable if the two durations it is derived from are ALSO on the
// ledger. A reader must be able to recompute 603 from the row, not trust a summary.
func TestDispatchTickPreflightShareCarriesItsConstituents(t *testing.T) {
	metrics := dispatchTickLoopMetrics(map[string]any{
		"preflight":   map[string]any{"live": 0, "cap": 6},
		"max_workers": 30,
		"timings_ms":  map[string]int64{"preflight": 31625, "total": 52378},
	})
	for _, k := range []string{"preflight_ms", "tick_total_ms", dispatchTickPreflightPermilleKey, "preflight_cap", "max_workers"} {
		if _, ok := metrics[k]; !ok {
			t.Fatalf("metric %q missing from the ledger row: %v", k, metrics)
		}
	}
	// Recompute from the row: the summary must be checkable by hand.
	pf, total := metrics["preflight_ms"], metrics["tick_total_ms"]
	if got := (pf * 1000) / total; got != metrics[dispatchTickPreflightPermilleKey] {
		t.Fatalf("summary %d is not recomputable from its constituents (%d/%d -> %d)",
			metrics[dispatchTickPreflightPermilleKey], pf, total, got)
	}
}
