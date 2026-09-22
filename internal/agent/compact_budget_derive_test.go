package agent

import "testing"

// TestDeriveCompactHistoryBudget pins the native shed-line policy that makes a
// ~150k agent context usable. The rules under test are the ones the serve and
// turnkey constructors both depend on, so a drift here is a drift in production.
func TestDeriveCompactHistoryBudget(t *testing.T) {
	tests := []struct {
		name      string
		window    int
		requested int
		want      int
	}{
		// An explicit operator budget is never second-guessed.
		{name: "explicit wins", window: 150000, requested: 12345, want: 12345},
		{name: "explicit wins over tiny window", window: 100, requested: 7, want: 7},

		// No resolved window => no honest shed-line; preserve the historical
		// no-compaction default rather than inventing a bound.
		{name: "unknown window stays unwired", window: 0, want: 0},
		{name: "negative window stays unwired", window: -1, want: 0},

		// A window at or below the reserve has no resident slack to spend. A
		// positive budget here would shed the live task itself.
		{name: "window equals reserve", window: DefaultNativeOutputReserveTokens, want: 0},
		{name: "window below reserve", window: 8000, want: 0},

		// The load-bearing 150k case: 150k - 32k = 118k usable, 60% => 70.8k.
		// This is materially ABOVE the 48k/96k provider-shaped defaults, which is
		// the whole point: a 1M-declared native model must not be flattened.
		{name: "150k window", window: 150000, want: 70800},

		// A larger window keeps scaling rather than saturating at a constant.
		{name: "1M declared window", window: 1048576, want: 609945},

		// Small-but-real windows get the reserve-sized floor, never a degenerate
		// line that would shed everything.
		{name: "small window floors at reserve", window: 40000, want: DefaultNativeOutputReserveTokens},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DeriveCompactHistoryBudget(tt.window, tt.requested); got != tt.want {
				t.Fatalf("DeriveCompactHistoryBudget(%d, %d) = %d, want %d", tt.window, tt.requested, got, tt.want)
			}
		})
	}
}

// TestDeriveCompactHistoryBudgetIsMonotone is the anti-regression shape guard: the
// shed-line must never SHRINK as the window grows. A non-monotone derivation would
// silently punish an operator for enabling a larger context.
func TestDeriveCompactHistoryBudgetIsMonotone(t *testing.T) {
	prev := 0
	for window := 0; window <= 2_000_000; window += 5000 {
		got := DeriveCompactHistoryBudget(window, 0)
		if got < prev {
			t.Fatalf("budget regressed as the window grew: window=%d budget=%d < previous=%d", window, got, prev)
		}
		prev = got
	}
}
