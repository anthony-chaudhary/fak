package main

// Adversarial pacing tests for the aiperf-lifecycle-2 direct port
// (absolute-deadline pacing with a bounded catch-up window). This file is
// additive: it complements TestPacingCatchUp in loadgen_test.go without
// modifying that gating suite, and probes the catch-up window boundary the
// gating cases do not.

import (
	"context"
	"testing"
	"time"
)

// advStep scripts one pacer sleep: when the sleep resolves, the scripted clock
// jumps to wake; ok=false simulates the context finishing during the sleep.
type advStep struct {
	wake time.Time
	ok   bool
}

// advSim is a scripted clock + sleep + issue recorder wired into pacer.run via
// its injectable now/sleep fields.
type advSim struct {
	base     time.Time
	cur      time.Time
	steps    []advStep
	stepIdx  int
	slept    []time.Duration
	issuedAt []time.Time
}

func newAdvSim(base time.Time) *advSim {
	return &advSim{base: base, cur: base}
}

func (s *advSim) nowFn() time.Time {
	return s.cur
}

func (s *advSim) sleepFn(ctx context.Context, d time.Duration) bool {
	s.slept = append(s.slept, d)
	if s.stepIdx < len(s.steps) {
		st := s.steps[s.stepIdx]
		s.stepIdx++
		s.cur = st.wake
		return st.ok
	}
	s.cur = s.cur.Add(d) // unscripted tail: natural advance
	return true
}

func (s *advSim) issueFn() bool {
	s.issuedAt = append(s.issuedAt, s.cur)
	return true
}

func advAssertDurations(t *testing.T, label string, got, want []time.Duration) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %v (n=%d) want %v (n=%d)", label, got, len(got), want, len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]=%v want %v (full: got %v want %v)", label, i, got[i], want[i], got, want)
		}
	}
}

func advAssertTimes(t *testing.T, label string, got, want []time.Time) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %v (n=%d) want %v (n=%d)", label, got, len(got), want, len(want))
	}
	for i := range want {
		if !got[i].Equal(want[i]) {
			t.Fatalf("%s[%d]=%v want %v (full: got %v want %v)", label, i, got[i], want[i], got, want)
		}
	}
}

// Exactly maxCatchUp late is INSIDE the window: the strict
// next.Add(maxCatchUp).Before(now) check must not re-anchor, so the original
// cumulative schedule holds (the 80ms second sleep proves the deadline stayed
// at 200ms rather than re-anchoring to the 120ms wake).
func TestPacingAdvBoundaryExactMaxCatchUpLateStaysOnSchedule(t *testing.T) {
	const (
		interval = 100 * time.Millisecond
		catchUp  = 20 * time.Millisecond
	)
	start := time.Date(2026, 11, 11, 9, 0, 0, 0, time.UTC)

	sim := newAdvSim(start)
	sim.steps = []advStep{
		{wake: start.Add(120 * time.Millisecond), ok: true}, // exactly 20ms late
		{wake: start.Add(200 * time.Millisecond), ok: true},
		{wake: start.Add(300 * time.Millisecond), ok: true},
	}
	p := newPacer(interval, catchUp, sim.nowFn, sim.sleepFn)
	if got := p.run(context.Background(), 3, sim.issueFn); got != 3 {
		t.Fatalf("issued=%d want 3", got)
	}
	advAssertDurations(t, "slept", sim.slept,
		[]time.Duration{100 * time.Millisecond, 80 * time.Millisecond, 100 * time.Millisecond})
	advAssertTimes(t, "issuedAt", sim.issuedAt, []time.Time{
		start.Add(120 * time.Millisecond),
		start.Add(200 * time.Millisecond),
		start.Add(300 * time.Millisecond),
	})
}

// Strictly MORE than maxCatchUp late re-anchors to now: the missed tick issues
// immediately with no sleep (no burst — one issue at the wake), and the next
// deadline is a full interval measured from the re-anchored now (421ms, not
// 400ms).
func TestPacingAdvReanchorStrictlyPastWindowNoBurst(t *testing.T) {
	const (
		interval = 100 * time.Millisecond
		catchUp  = 20 * time.Millisecond
	)
	start := time.Date(2026, 11, 11, 9, 0, 0, 0, time.UTC)

	sim := newAdvSim(start)
	sim.steps = []advStep{
		{wake: start.Add(100 * time.Millisecond), ok: true},
		{wake: start.Add(321 * time.Millisecond), ok: true}, // 21ms past the 300ms deadline: re-anchor
		{wake: start.Add(421 * time.Millisecond), ok: true},
	}
	p := newPacer(interval, catchUp, sim.nowFn, sim.sleepFn)
	if got := p.run(context.Background(), 4, sim.issueFn); got != 4 {
		t.Fatalf("issued=%d want 4", got)
	}
	advAssertDurations(t, "slept", sim.slept,
		[]time.Duration{100 * time.Millisecond, 100 * time.Millisecond, 100 * time.Millisecond})
	advAssertTimes(t, "issuedAt", sim.issuedAt, []time.Time{
		start.Add(100 * time.Millisecond),
		start.Add(321 * time.Millisecond),
		start.Add(321 * time.Millisecond),
		start.Add(421 * time.Millisecond),
	})
}

// Exactly AT the window edge (now == next + maxCatchUp) does NOT re-anchor:
// the strict Before check keeps the cumulative absolute schedule, so the next
// deadline stays at 400ms (not 420ms) and the pacer closes the gap with a
// short 80ms sleep instead of forfeiting the tick.
func TestPacingAdvReanchorNotFiredExactlyAtWindowEdge(t *testing.T) {
	const (
		interval = 100 * time.Millisecond
		catchUp  = 20 * time.Millisecond
	)
	start := time.Date(2026, 11, 11, 9, 0, 0, 0, time.UTC)

	sim := newAdvSim(start)
	sim.steps = []advStep{
		{wake: start.Add(100 * time.Millisecond), ok: true},
		{wake: start.Add(320 * time.Millisecond), ok: true}, // exactly 20ms past the 300ms deadline
		{wake: start.Add(400 * time.Millisecond), ok: true},
	}
	p := newPacer(interval, catchUp, sim.nowFn, sim.sleepFn)
	if got := p.run(context.Background(), 4, sim.issueFn); got != 4 {
		t.Fatalf("issued=%d want 4", got)
	}
	advAssertDurations(t, "slept", sim.slept,
		[]time.Duration{100 * time.Millisecond, 100 * time.Millisecond, 80 * time.Millisecond})
	advAssertTimes(t, "issuedAt", sim.issuedAt, []time.Time{
		start.Add(100 * time.Millisecond),
		start.Add(320 * time.Millisecond),
		start.Add(320 * time.Millisecond),
		start.Add(400 * time.Millisecond),
	})
}
