package main

import (
	"context"
	"testing"
	"time"
)

// lateClock is a deterministic fake clock whose every wake-up is
// systematically late by a fixed overshoot: sleepFn lands the clock at
// now + requested + overshoot. This models the scheduler jitter / slow-loop
// wake-up that an absolute-deadline pacer must not let accumulate.
type lateClock struct {
	now       time.Time
	overshoot time.Duration
	slept     []time.Duration
}

func (c *lateClock) nowFn() time.Time { return c.now }

func (c *lateClock) sleepFn(_ context.Context, d time.Duration) bool {
	c.slept = append(c.slept, d)
	c.now = c.now.Add(d + c.overshoot)
	return true
}

// runFixedInterval is the naive contrast pacer: it sleeps a constant period
// before every request, so each wake-up's overshoot is added to the next
// target and the error grows with the request count.
func runFixedInterval(ctx context.Context, n int, period time.Duration, sleep func(context.Context, time.Duration) bool) int {
	issued := 0
	for i := 0; i < n; i++ {
		if ctx.Err() != nil {
			break
		}
		if !sleep(ctx, period) {
			break
		}
		issued++
	}
	return issued
}

// TestPacerPrecision pins the non-accumulating property of the absolute-deadline
// pacer against a fake clock whose wake-ups are systematically late. The
// absolute-deadline pacer's total error stays within one wake-up step of the
// ideal N/rate, while a fixed-interval pacer's error grows with N; a deadline
// already in the past yields no (never a negative) sleep; and the reported
// achieved rate matches the requested rate within tolerance. The test fails if
// the pacer accumulates drift.
func TestPacerPrecision(t *testing.T) {
	const (
		period    = 10 * time.Millisecond
		overshoot = 1 * time.Millisecond
		requests  = 50
	)
	origin := time.Date(2026, 11, 17, 12, 0, 0, 0, time.UTC)
	ideal := time.Duration(requests) * period

	t.Run("absoluteDeadlineDoesNotAccumulate", func(t *testing.T) {
		clk := &lateClock{now: origin, overshoot: overshoot}
		p := newDeadlinePacer(origin, period, clk.nowFn, clk.sleepFn)
		issued := p.run(context.Background(), requests, func() bool { return true })
		if issued != requests {
			t.Fatalf("issued=%d want %d", issued, requests)
		}
		for _, d := range clk.slept {
			if d <= 0 {
				t.Fatalf("non-positive sleep %s (a past deadline must sleep zero, never negative)", d)
			}
		}
		elapsed := clk.now.Sub(origin)
		drift := elapsed - ideal
		// The absolute-deadline pacer lands every request at its deadline plus
		// one wake-up overshoot, so the total error is a single step regardless
		// of N: it must not accumulate.
		if drift < 0 || drift > overshoot {
			t.Fatalf("absolute-deadline drift=%s want within [0,%s] (elapsed=%s ideal=%s)", drift, overshoot, elapsed, ideal)
		}
	})

	t.Run("fixedIntervalAccumulates", func(t *testing.T) {
		clk := &lateClock{now: origin, overshoot: overshoot}
		issued := runFixedInterval(context.Background(), requests, period, clk.sleepFn)
		if issued != requests {
			t.Fatalf("issued=%d want %d", issued, requests)
		}
		elapsed := clk.now.Sub(origin)
		drift := elapsed - ideal
		// The fixed-interval pacer adds each overshoot to the next target, so
		// the error grows linearly with N.
		want := time.Duration(requests) * overshoot
		if drift != want {
			t.Fatalf("fixed-interval drift=%s want %s (elapsed=%s ideal=%s)", drift, want, elapsed, ideal)
		}
		if drift <= overshoot {
			t.Fatalf("fixed-interval drift=%s should exceed one step %s", drift, overshoot)
		}
	})

	t.Run("pastDeadlineSleepsZeroNotNegative", func(t *testing.T) {
		// Origin far in the past: every deadline is already behind now, so the
		// pacer must issue immediately with no sleep (never a negative sleep).
		clk := &lateClock{now: origin.Add(10 * ideal), overshoot: overshoot}
		p := newDeadlinePacer(origin, period, clk.nowFn, clk.sleepFn)
		issued := p.run(context.Background(), requests, func() bool { return true })
		if issued != requests {
			t.Fatalf("issued=%d want %d", issued, requests)
		}
		if len(clk.slept) != 0 {
			t.Fatalf("slept=%v want no sleeps for past deadlines", clk.slept)
		}
	})

	t.Run("achievedRateMatchesRequested", func(t *testing.T) {
		clk := &lateClock{now: origin, overshoot: overshoot}
		p := newDeadlinePacer(origin, period, clk.nowFn, clk.sleepFn)
		issued := p.run(context.Background(), requests, func() bool { return true })
		rep := p.report(issued, clk.now)
		wantRate := float64(time.Second) / float64(period) // 100/s
		if rep.Rate != wantRate {
			t.Fatalf("requested rate=%v want %v", rep.Rate, wantRate)
		}
		if tol := 0.01 * wantRate; rep.Achieved < wantRate-tol || rep.Achieved > wantRate+tol {
			t.Fatalf("achieved rate=%v want within 1%% of %v", rep.Achieved, wantRate)
		}
		if rep.Drift < 0 || rep.Drift > overshoot {
			t.Fatalf("reported drift=%s want within [0,%s]", rep.Drift, overshoot)
		}
	})
}
