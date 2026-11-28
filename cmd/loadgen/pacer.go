// Absolute-deadline request pacing.
//
// (src/aiperf/timing/high_res_timer.py), pinned at commit
//
// Upstream arms a timerfd with CLOCK_MONOTONIC + TFD_TIMER_ABSTIME so every
// wake-up targets an ABSOLUTE instant rather than a relative interval: a late
// wake-up (scheduler jitter, a slow loop body) does not push the next target
// out, so per-request error does not accumulate over a run. Go has no timerfd,
// but the same idea is expressible with the stdlib: compute each request's
// absolute deadline from the run origin and the requested rate, and sleep only
// the remaining time to that deadline. This file implements that idea in
// loadgen's own idiom (an injectable clock and sleep, so the schedule is
// deterministic and testable); no upstream code is copied.
package main

import (
	"context"
	"time"
)

// deadlinePacer schedules request N at the absolute target time
// origin + N/rate, so wake-up error does not accumulate: a late wake-up only
// shortens the next sleep, it never shifts the schedule. It is the
// non-accumulating counterpart to a fixed-interval pacer, which sleeps a
// constant gap between requests and therefore adds each wake-up's overshoot to
// the next target.
type deadlinePacer struct {
	origin time.Time
	period time.Duration
	now    func() time.Time
	sleep  func(context.Context, time.Duration) bool
}

// newDeadlinePacer builds a pacer that targets origin + N*period for the N-th
// request (N is one-based). A non-positive period degenerates to "issue
// immediately" (every deadline is at or before now), which is the correct
// behavior for an unbounded rate.
func newDeadlinePacer(origin time.Time, period time.Duration, now func() time.Time, sleep func(context.Context, time.Duration) bool) *deadlinePacer {
	if now == nil {
		now = time.Now
	}
	if sleep == nil {
		sleep = deadlinePacerSleep
	}
	return &deadlinePacer{origin: origin, period: period, now: now, sleep: sleep}
}

// deadlinePacerSleep blocks until d elapses or ctx finishes; false means the
// context finished first. It is the default sleep for a deadlinePacer when the
// caller does not inject one.
func deadlinePacerSleep(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

// deadline returns the absolute target time for the n-th request (n is
// one-based): origin + n*period.
func (p *deadlinePacer) deadline(n int) time.Time {
	return p.origin.Add(time.Duration(n) * p.period)
}

// run issues up to n requests, each at its absolute deadline, and returns the
// number actually issued. It sleeps only the remaining time to each deadline
// and never a negative duration: a deadline already in the past yields no sleep
// at all, so a stalled run catches up immediately instead of sleeping. It stops
// early when the context finishes or issue() reports the request was not
// accepted.
func (p *deadlinePacer) run(ctx context.Context, n int, issue func() bool) int {
	issued := 0
	for i := 1; i <= n; i++ {
		if ctx.Err() != nil {
			break
		}
		remaining := p.deadline(i).Sub(p.now())
		if remaining > 0 {
			if !p.sleep(ctx, remaining) {
				break
			}
		}
		if !issue() {
			break
		}
		issued++
	}
	return issued
}

// pacingReport summarizes a completed pacing run: the requested rate, the
// achieved rate, and the drift between the ideal and actual elapsed time.
type pacingReport struct {
	Requests int
	Elapsed  time.Duration
	Rate     float64       // requested requests/second
	Achieved float64       // Requests / Elapsed, requests/second
	Drift    time.Duration // Elapsed - ideal (Requests/rate); positive = behind
}

// report computes the achieved rate and drift for a run that issued `issued`
// requests between origin and end. A zero or negative elapsed time reports a
// zero achieved rate rather than dividing by zero.
func (p *deadlinePacer) report(issued int, end time.Time) pacingReport {
	elapsed := end.Sub(p.origin)
	rep := pacingReport{Requests: issued, Elapsed: elapsed}
	if p.period > 0 {
		rep.Rate = float64(time.Second) / float64(p.period)
		rep.Drift = elapsed - time.Duration(issued)*p.period
	}
	if elapsed > 0 {
		rep.Achieved = float64(issued) / elapsed.Seconds()
	}
	return rep
}
