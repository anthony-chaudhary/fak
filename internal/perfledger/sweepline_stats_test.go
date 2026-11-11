// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
package perfledger

// Adversarial contract tests for the ai-dynamo/aiperf sweepline_stats port
// (upstream pin ab610add4e8876fca223197596c728e17365f381, Apache-2.0).
//
// Expectations are derived from the pinned spec:
//   - Time-weighted: step function clipped to [windowStart, windowEnd]; the
//     segment straddling windowStart contributes the value held from before
//     the window (value at the last event <= windowStart; 0 if none); the
//     segment straddling windowEnd is sliced at windowEnd. Avg/Std normalize
//     by the window length; percentiles read off the duration-weighted CDF
//     (first cumulative fraction >= q, searchsorted-left, clamped to last).
//   - Active-weighted: rate curve restricted to segments where the mask curve
//     is STRICTLY positive; boundary grid is the sorted deduped union of the
//     window edges and all rate/mask event timestamps strictly inside the
//     window; Avg/Std normalize by ACTIVE duration; percentiles normalize by
//     total active duration. Idle time outside active segments must not shift
//     active percentiles or avg.

import (
	"math"
	"testing"
)

const sweepRelEps = 1e-9

// expectSweepLineStats asserts all eight fields. Zero-valued want fields are
// compared absolutely (relative comparison is meaningless at zero); non-zero
// want fields use a 1e-9 relative tolerance, relaxed to absTol when the
// caller passes one (e.g. the upstream 3.2 std case).
func expectSweepLineStats(t *testing.T, name string, got, want SweepLineStats, absTol float64) {
	t.Helper()
	for _, f := range []struct {
		field string
		g, w  float64
	}{
		{"Avg", got.Avg, want.Avg},
		{"Min", got.Min, want.Min},
		{"Max", got.Max, want.Max},
		{"P50", got.P50, want.P50},
		{"P90", got.P90, want.P90},
		{"P95", got.P95, want.P95},
		{"P99", got.P99, want.P99},
		{"Std", got.Std, want.Std},
	} {
		if sweepFloatClose(f.g, f.w, absTol) {
			continue
		}
		t.Errorf("%s: %s = %.17g, want %.17g (absTol=%g)", name, f.field, f.g, f.w, absTol)
	}
}

func sweepFloatClose(g, w, absTol float64) bool {
	if g == w {
		return true
	}
	diff := math.Abs(g - w)
	if absTol > 0 && diff <= absTol {
		return true
	}
	scale := math.Abs(g)
	if aw := math.Abs(w); aw > scale {
		scale = aw
	}
	return diff/scale <= sweepRelEps
}

// expectZeroSweepLineStats asserts the exact all-zero struct (degenerate input).
func expectZeroSweepLineStats(t *testing.T, name string, got SweepLineStats) {
	t.Helper()
	if got != ZeroSweepLineStats {
		t.Errorf("%s: got %+v, want exact zero struct %+v", name, got, ZeroSweepLineStats)
	}
}

// --- Time-weighted: upstream reference cases (aiperf test_sweep.py) ---

func TestTimeWeightedStatsConstantValue(t *testing.T) {
	t.Run("constant_over_full_window", func(t *testing.T) {
		// Upstream case 1: ts=[0,100] vals=[5,0], window [0,100].
		got := ComputeTimeWeightedStats([]float64{0, 100}, []float64{5, 0}, 0, 100)
		expectSweepLineStats(t, "constant", got,
			SweepLineStats{Avg: 5, Min: 5, Max: 5, P50: 5, P90: 5, P95: 5, P99: 5, Std: 0}, 0)
	})
	t.Run("single_event_held_across_window", func(t *testing.T) {
		// Upstream case 6: ts=[0] vals=[3], window [0,100] — the value is
		// held from the single event across the whole window.
		got := ComputeTimeWeightedStats([]float64{0}, []float64{3}, 0, 100)
		expectSweepLineStats(t, "single-event", got,
			SweepLineStats{Avg: 3, Min: 3, Max: 3, P50: 3, P90: 3, P95: 3, P99: 3, Std: 0}, 0)
	})
}

func TestTimeWeightedStatsTwoSegments(t *testing.T) {
	// Upstream case 2: ts=[0,80,100] vals=[2,10,0], window [0,100].
	// avg = (2*80 + 10*20)/100 = 3.6; std = sqrt(10.24) = 3.2.
	// Percentiles from the duration CDF: (2,d=80) frac 0.8, (10,d=20) frac 1.0.
	got := ComputeTimeWeightedStats([]float64{0, 80, 100}, []float64{2, 10, 0}, 0, 100)
	expectSweepLineStats(t, "two-segments", got, SweepLineStats{
		Avg: 3.6, Min: 2, Max: 10, P50: 2, P90: 10, P95: 10, P99: 10, Std: 3.2,
	}, 0.01)
}

func TestTimeWeightedStatsPercentileDurationWeighting(t *testing.T) {
	// Upstream case 3: ts=[0,900,1000] vals=[1,100,0], window [0,1000].
	// Segments (1,d=900) frac 0.9 and (100,d=100) frac 1.0: p50/p90 land on 1,
	// p95/p99 land on 100. avg = 10.9; std = sqrt(882.09) = 29.7.
	got := ComputeTimeWeightedStats([]float64{0, 900, 1000}, []float64{1, 100, 0}, 0, 1000)
	expectSweepLineStats(t, "unequal-durations", got, SweepLineStats{
		Avg: 10.9, Min: 1, Max: 100, P50: 1, P90: 1, P95: 100, P99: 100, Std: math.Sqrt(882.09),
	}, 0)
}

func TestTimeWeightedStatsWindowClipping(t *testing.T) {
	t.Run("event_exactly_at_window_start", func(t *testing.T) {
		// Upstream case 4: ts=[0,50,100] vals=[1,5,0], window [50,100].
		// The event AT windowStart takes effect; the pre-window value 1 is
		// excluded entirely.
		got := ComputeTimeWeightedStats([]float64{0, 50, 100}, []float64{1, 5, 0}, 50, 100)
		expectSweepLineStats(t, "clip", got,
			SweepLineStats{Avg: 5, Min: 5, Max: 5, P50: 5, P90: 5, P95: 5, P99: 5, Std: 0}, 0)
	})
	t.Run("partial_segment_clip_both_edges", func(t *testing.T) {
		// Upstream case 5: ts=[0,100] vals=[2,0], window [25,75] — the
		// straddling segment contributes the held value 2, sliced to the window.
		got := ComputeTimeWeightedStats([]float64{0, 100}, []float64{2, 0}, 25, 75)
		expectSweepLineStats(t, "partial-clip", got,
			SweepLineStats{Avg: 2, Min: 2, Max: 2, P50: 2, P90: 2, P95: 2, P99: 2, Std: 0}, 0)
	})
	t.Run("window_starts_before_first_event_zero_prefix", func(t *testing.T) {
		// Adversarial: the implicit 0 segment before the first event must be
		// part of the window. ts=[50,100] vals=[3,0], window [0,100]:
		// [0,50)->0, [50,100)->3. avg=1.5, std=1.5, p50=0 (frac 0.5 at 0).
		got := ComputeTimeWeightedStats([]float64{50, 100}, []float64{3, 0}, 0, 100)
		expectSweepLineStats(t, "zero-prefix", got, SweepLineStats{
			Avg: 1.5, Min: 0, Max: 3, P50: 0, P90: 3, P95: 3, P99: 3, Std: 1.5,
		}, 0)
	})
	t.Run("event_exactly_at_window_end_is_zero_length", func(t *testing.T) {
		// Adversarial: the event AT windowEnd opens a zero-length segment that
		// must not contribute. ts=[0,50,100] vals=[4,6,2], window [0,100]:
		// [0,50)->4, [50,100)->6; the trailing 2 has duration 0.
		got := ComputeTimeWeightedStats([]float64{0, 50, 100}, []float64{4, 6, 2}, 0, 100)
		expectSweepLineStats(t, "end-event", got, SweepLineStats{
			Avg: 5, Min: 4, Max: 6, P50: 4, P90: 6, P95: 6, P99: 6, Std: 1,
		}, 0)
	})
	t.Run("event_exactly_at_window_start_mid_curve", func(t *testing.T) {
		// Adversarial: event exactly at windowStart on a curve that extends
		// beyond windowEnd. ts=[10,60,100] vals=[2,8,0], window [10,100]:
		// [10,60)->2 (d=50), [60,100)->8 (d=40). avg=14/3; std=sqrt(80/9).
		// Percentile fracs: 50/90 at 2, 1.0 at 8 -> p50=2, p90+=8.
		got := ComputeTimeWeightedStats([]float64{10, 60, 100}, []float64{2, 8, 0}, 10, 100)
		expectSweepLineStats(t, "start-event", got, SweepLineStats{
			Avg: 14.0 / 3.0, Min: 2, Max: 8, P50: 2, P90: 8, P95: 8, P99: 8, Std: math.Sqrt(80.0 / 9.0),
		}, 0)
	})
}

func TestTimeWeightedStatsDegenerateInputs(t *testing.T) {
	cases := []struct {
		name string
		ts   []float64
		vals []float64
		ws   float64
		we   float64
	}{
		{"empty_step_function", nil, nil, 0, 100},
		{"empty_value_slice", []float64{0, 100}, nil, 0, 100},
		{"zero_duration_window", []float64{0, 100}, []float64{5, 0}, 50, 50},
		{"negative_window", []float64{0, 100}, []float64{5, 0}, 100, 0},
		{"window_entirely_before_first_event", []float64{50}, []float64{3}, 0, 10},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ComputeTimeWeightedStats(tc.ts, tc.vals, tc.ws, tc.we)
			expectZeroSweepLineStats(t, tc.name, got)
		})
	}
}

func TestTimeWeightedStatsDuplicateTimestamps(t *testing.T) {
	t.Run("zero_duration_segment_dropped", func(t *testing.T) {
		// Adversarial: duplicate ts create a zero-duration segment whose value
		// must not leak into Min (or any stat). ts=[0,0,100] vals=[1,5,0]:
		// [0,0)->1 has duration 0 and is dropped; [0,100)->5 carries the window.
		got := ComputeTimeWeightedStats([]float64{0, 0, 100}, []float64{1, 5, 0}, 0, 100)
		expectSweepLineStats(t, "zero-dur-dropped", got,
			SweepLineStats{Avg: 5, Min: 5, Max: 5, P50: 5, P90: 5, P95: 5, P99: 5, Std: 0}, 0)
	})
	t.Run("last_duplicate_wins_for_held_value", func(t *testing.T) {
		// Adversarial: with duplicate timestamps the LAST value at that
		// timestamp is the one held forward (values[i] is held until
		// sortedTS[i+1]; only the final duplicate has positive duration).
		// ts=[0,50,50,100] vals=[10,20,30,0]: [0,50)->10, [50,50)->20 (d=0),
		// [50,100)->30. avg=20, std=10; CDF (10,d=50) frac 0.5, (30,d=50) frac 1.
		got := ComputeTimeWeightedStats([]float64{0, 50, 50, 100}, []float64{10, 20, 30, 0}, 0, 100)
		expectSweepLineStats(t, "last-dup-wins", got, SweepLineStats{
			Avg: 20, Min: 10, Max: 30, P50: 10, P90: 30, P95: 30, P99: 30, Std: 10,
		}, 0)
	})
}

// --- Active-weighted: upstream reference cases ---

func TestTimeWeightedStatsActiveOnlyEmptyInputs(t *testing.T) {
	cases := []struct {
		name     string
		rateTS   []float64
		rateVals []float64
		maskTS   []float64
		maskVals []float64
		ws, we   float64
	}{
		{"all_inputs_empty", nil, nil, nil, nil, 0, 100},
		{"zero_window", []float64{0, 100}, []float64{5, 0}, []float64{0, 100}, []float64{1, 0}, 50, 50},
		{"empty_mask_never_active", []float64{0, 100}, []float64{5, 0}, nil, nil, 0, 100},
		{"empty_rate_curve", nil, nil, []float64{0, 100}, []float64{1, 0}, 0, 100},
		{"mask_all_zero_no_active_segments", []float64{0, 100}, []float64{5, 0}, []float64{0, 50, 100}, []float64{0, 0, 0}, 0, 100},
		{"negative_mask_not_strictly_positive", []float64{0, 100}, []float64{5, 0}, []float64{0, 100}, []float64{-1, 0}, 0, 100},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ComputeActiveWeightedStats(tc.rateTS, tc.rateVals, tc.maskTS, tc.maskVals, tc.ws, tc.we)
			expectZeroSweepLineStats(t, tc.name, got)
		})
	}
}

func TestTimeWeightedStatsActiveOnlyExcludesIdle(t *testing.T) {
	// Upstream case 12: rate ts=[0,50] vals=[100,0]; mask ts=[0,50] vals=[1,0];
	// window [0,100]. Time-weighted avg is diluted to 50 by the idle half;
	// active avg must be 100.
	rateTS := []float64{0, 50}
	rateVals := []float64{100, 0}
	maskTS := []float64{0, 50}
	maskVals := []float64{1, 0}

	tw := ComputeTimeWeightedStats(rateTS, rateVals, 0, 100)
	expectSweepLineStats(t, "time-weighted-contrast", tw, SweepLineStats{
		Avg: 50, Min: 0, Max: 100, P50: 0, P90: 100, P95: 100, P99: 100, Std: 50,
	}, 0)

	got := ComputeActiveWeightedStats(rateTS, rateVals, maskTS, maskVals, 0, 100)
	expectSweepLineStats(t, "active", got,
		SweepLineStats{Avg: 100, Min: 100, Max: 100, P50: 100, P90: 100, P95: 100, P99: 100, Std: 0}, 0)
}

func TestTimeWeightedStatsActiveOnlyPercentilesIgnoreIdle(t *testing.T) {
	// Upstream case 13: rate ts=[0,10,20,100] vals=[50,150,0,0];
	// mask ts=[0,20] vals=[1,0]; window [0,1000]. Active segments are
	// [0,10)@50 and [10,20)@150 (active dur 20); the idle [20,1000) must not
	// dilute avg (100) or shift percentiles (p50=50, p99=150).
	got := ComputeActiveWeightedStats(
		[]float64{0, 10, 20, 100}, []float64{50, 150, 0, 0},
		[]float64{0, 20}, []float64{1, 0}, 0, 1000)
	expectSweepLineStats(t, "active-percentiles", got, SweepLineStats{
		Avg: 100, Min: 50, Max: 150, P50: 50, P90: 150, P95: 150, P99: 150, Std: 50,
	}, 0)
}

func TestTimeWeightedStatsActiveOnlyPartialOverlap(t *testing.T) {
	// Upstream case 14: rate ts=[0,100] vals=[200,0]; mask ts=[0,100] vals=[1,0];
	// window [50,80]. Neither curve has an event inside the window; the held
	// values (200, 1) must still be resolved on the edge-only grid {50,80}.
	got := ComputeActiveWeightedStats(
		[]float64{0, 100}, []float64{200, 0},
		[]float64{0, 100}, []float64{1, 0}, 50, 80)
	expectSweepLineStats(t, "partial-overlap", got,
		SweepLineStats{Avg: 200, Min: 200, Max: 200, P50: 200, P90: 200, P95: 200, P99: 200, Std: 0}, 0)
}

func TestTimeWeightedStatsActiveOnlyWindowBoundaries(t *testing.T) {
	t.Run("step_state_preserved_across_window", func(t *testing.T) {
		// Upstream case 15: rate ts=[0,10,20] vals=[5,10,999];
		// mask ts=[0,15,20] vals=[1,0,1]; window [10,20]. The mask is 0 on
		// [15,20), so only [10,15) at rate 10 is active; the 999 event sits
		// exactly at windowEnd (zero length) and must not leak in.
		got := ComputeActiveWeightedStats(
			[]float64{0, 10, 20}, []float64{5, 10, 999},
			[]float64{0, 15, 20}, []float64{1, 0, 1}, 10, 20)
		expectSweepLineStats(t, "boundary-state", got,
			SweepLineStats{Avg: 10, Min: 10, Max: 10, P50: 10, P90: 10, P95: 10, P99: 10, Std: 0}, 0)
	})
	t.Run("mask_turns_on_exactly_at_window_end", func(t *testing.T) {
		// Adversarial: the mask event at windowEnd is on the boundary grid, so
		// the [90,100) sliver IS active. rate ts=[0,100] vals=[5,0];
		// mask ts=[0,90] vals=[0,1]; window [0,100].
		got := ComputeActiveWeightedStats(
			[]float64{0, 100}, []float64{5, 0},
			[]float64{0, 90}, []float64{0, 1}, 0, 100)
		expectSweepLineStats(t, "mask-on-at-end", got,
			SweepLineStats{Avg: 5, Min: 5, Max: 5, P50: 5, P90: 5, P95: 5, P99: 5, Std: 0}, 0)
	})
	t.Run("mask_event_exactly_at_window_start_turns_on", func(t *testing.T) {
		// Adversarial: a mask event exactly AT windowStart takes effect at the
		// window edge (step lookup is last event <= grid point, not <).
		// rate ts=[0,100] vals=[5,0]; mask ts=[10,40] vals=[1,0]; window [10,90].
		got := ComputeActiveWeightedStats(
			[]float64{0, 100}, []float64{5, 0},
			[]float64{10, 40}, []float64{1, 0}, 10, 90)
		expectSweepLineStats(t, "mask-on-at-start", got,
			SweepLineStats{Avg: 5, Min: 5, Max: 5, P50: 5, P90: 5, P95: 5, P99: 5, Std: 0}, 0)
	})
}

func TestTimeWeightedStatsActiveOnlyAdversarial(t *testing.T) {
	t.Run("mask_event_splits_active_segment", func(t *testing.T) {
		// Adversarial: mask events strictly inside the window split rate
		// segments and vice versa. rate ts=[0,20,60,100] vals=[10,40,80,0];
		// mask ts=[0,30,70,100] vals=[0,1,0,1]; window [0,100].
		// Grid {0,20,30,60,70,100}; active segments [30,60)@40 (d=30) and
		// [60,70)@80 (d=10); active dur 40. avg=50; std=sqrt(300).
		// CDF: (40,d=30) frac 0.75, (80,d=10) frac 1.0 -> p50=40, p90+=80.
		got := ComputeActiveWeightedStats(
			[]float64{0, 20, 60, 100}, []float64{10, 40, 80, 0},
			[]float64{0, 30, 70, 100}, []float64{0, 1, 0, 1}, 0, 100)
		expectSweepLineStats(t, "split-segments", got, SweepLineStats{
			Avg: 50, Min: 40, Max: 80, P50: 40, P90: 80, P95: 80, P99: 80, Std: math.Sqrt(300),
		}, 0)
	})
	t.Run("rate_event_outside_window_held_value_resolved", func(t *testing.T) {
		// Adversarial: the rate curve has NO event inside the window; its held
		// value must still be resolved on the mask-driven grid.
		// rate ts=[0] vals=[6]; mask ts=[0,20,50,100] vals=[0,1,0,1];
		// window [10,90]. Grid {10,20,50,90}; active [20,50)@6 (d=30).
		got := ComputeActiveWeightedStats(
			[]float64{0}, []float64{6},
			[]float64{0, 20, 50, 100}, []float64{0, 1, 0, 1}, 10, 90)
		expectSweepLineStats(t, "held-rate", got,
			SweepLineStats{Avg: 6, Min: 6, Max: 6, P50: 6, P90: 6, P95: 6, P99: 6, Std: 0}, 0)
	})
	t.Run("rate_never_fired_active_value_is_zero", func(t *testing.T) {
		// Adversarial: mask is positive on [20,50) but the rate's first event
		// (500) is after the window, so the resolved rate is 0 before first
		// event. Active duration is 30 (not degenerate) but every stat is 0.
		got := ComputeActiveWeightedStats(
			[]float64{500}, []float64{6},
			[]float64{0, 20, 50}, []float64{0, 1, 0}, 0, 100)
		expectSweepLineStats(t, "zero-rate-active", got, SweepLineStats{}, 0)
	})
	t.Run("duplicate_mask_timestamps_zero_length_blip", func(t *testing.T) {
		// Adversarial: duplicate mask timestamps make the active blip at t=50
		// zero-length; with last-duplicate-wins step lookup the mask is 0 on
		// [50,100) and there are NO active segments -> zeros.
		got := ComputeActiveWeightedStats(
			[]float64{0, 100}, []float64{5, 0},
			[]float64{50, 50}, []float64{1, 0}, 0, 100)
		expectZeroSweepLineStats(t, "mask-blip", got)
	})
}
