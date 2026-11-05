package attemptbudget

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestDecideCooldownTimestampBounds(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name             string
		at, backoff, now int64
		wantUntil        int64
		wantStatus       Status
	}{
		{"ordinary", 100, 10, 105, 110, StatusCoolingDown},
		{"expired_boundary", 100, 10, 110, 110, StatusDispatchable},
		{"zero_backoff", 100, 0, 1, 100, StatusCoolingDown},
		{"positive_overflow", math.MaxInt64, 1, 1, math.MaxInt64, StatusCoolingDown},
		{"overflow_at_last_clock", math.MaxInt64, 1, math.MaxInt64, math.MaxInt64, StatusCoolingDown},
		{"exact_upper_boundary", math.MaxInt64 - 1, 1, math.MaxInt64, math.MaxInt64, StatusDispatchable},
		{"negative_underflow", math.MinInt64, -1, 1, math.MinInt64, StatusDispatchable},
		{"exact_lower_boundary", math.MinInt64 + 1, -1, 1, math.MinInt64, StatusDispatchable},
		{"negative_override", 100, -10, 80, 90, StatusCoolingDown},
		{"zero_clock", math.MaxInt64, 1, 0, math.MaxInt64, StatusDispatchable},
		{"negative_clock", math.MaxInt64, 1, -1, math.MaxInt64, StatusDispatchable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := Decide(Input{
				IssueID: "clock", Budget: 2, NowUnix: tc.now,
				Attempts: []Attempt{{FailureClass: "other", AtUnix: tc.at}},
				Backoff:  map[FailureClass]int64{FailureClassOther: tc.backoff},
			})
			if got.CooldownUntilUnix != tc.wantUntil || got.Status != tc.wantStatus || got.BackoffSeconds != tc.backoff {
				t.Fatalf("decision = %+v; want until=%d status=%s backoff=%d", got, tc.wantUntil, tc.wantStatus, tc.backoff)
			}
			if got.Verdict != "" || got.BlockReason != "" || got.Route != "" {
				t.Fatalf("under-budget decision acquired a block: %+v", got)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDecideOverflowPreservesBudgetPrecedence(t *testing.T) {
	t.Parallel()
	in := Input{
		IssueID: "budget", Budget: 1, NowUnix: math.MaxInt64,
		Attempts: []Attempt{{FailureClass: "other", AtUnix: math.MaxInt64}},
		Backoff:  map[FailureClass]int64{FailureClassOther: 1, FailureClassRateLimit: 1},
	}
	if got := Decide(in); got.Status != StatusHeld || got.Verdict != VerdictStructuralBlock {
		t.Fatalf("structural budget must still hold: %+v", got)
	}
	in.Attempts[0].FailureClass = "rate_limit"
	if got := Decide(in); got.Status != StatusCoolingDown || got.Verdict != VerdictTransientRetry {
		t.Fatalf("transient retry must retain overflowed cooldown: %+v", got)
	}
	in.TransientBudget = 1
	if got := Decide(in); got.Status != StatusHeld || got.Verdict != VerdictTransientExhausted {
		t.Fatalf("exhausted transient budget must still hold: %+v", got)
	}
}
