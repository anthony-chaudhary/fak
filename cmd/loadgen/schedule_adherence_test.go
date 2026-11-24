package main

import (
	"context"
	"testing"
	"time"
)

// TestPacerScheduleAdherence pins that the pacer reports how far issuance fell
// behind the labelled schedule. Re-anchoring after a stall keeps the rate but
// delays every later request, so a point can issue all its requests and still
// have offered less load than its label; that point must read ScheduleKept
// false rather than pass silently.
func TestPacerScheduleAdherence(t *testing.T) {
	const (
		interval = 10 * time.Millisecond
		catchUp  = 10 * time.Millisecond
	)
	base := time.Date(2026, 11, 23, 12, 0, 0, 0, time.UTC)

	t.Run("on time keeps the schedule", func(t *testing.T) {
		sim := newAdvSim(base)
		p := newPacer(interval, catchUp, sim.nowFn, sim.sleepFn)
		if got := p.run(context.Background(), 200, sim.issueFn); got != 200 {
			t.Fatalf("issued=%d want 200", got)
		}
		var pr PointResult
		p.adherence.apply(&pr)
		if pr.ScheduleKept == nil || !*pr.ScheduleKept {
			t.Fatalf("on-time run: ScheduleKept=%v want true (%+v)", pr.ScheduleKept, pr)
		}
		if pr.PlannedRequests != 200 || pr.ScheduleReanchors != 0 || pr.ScheduleLagMaxMS != 0 {
			t.Fatalf("on-time run: planned=%d reanchors=%d maxLag=%v, want 200/0/0",
				pr.PlannedRequests, pr.ScheduleReanchors, pr.ScheduleLagMaxMS)
		}
	})

	t.Run("a stall past catch-up loses the schedule", func(t *testing.T) {
		sim := newAdvSim(base)
		// The first sleep wakes 300 ms late; every later deadline is re-anchored
		// from there, so each remaining request trails the ideal schedule.
		sim.steps = []advStep{{wake: base.Add(interval + 300*time.Millisecond), ok: true}}
		p := newPacer(interval, catchUp, sim.nowFn, sim.sleepFn)
		if got := p.run(context.Background(), 100, sim.issueFn); got != 100 {
			t.Fatalf("issued=%d want 100: every request is still issued", got)
		}
		var pr PointResult
		p.adherence.apply(&pr)
		if pr.ScheduleReanchors != 1 {
			t.Errorf("reanchors=%d want 1", pr.ScheduleReanchors)
		}
		if pr.ScheduleLagP99MS < 290 {
			t.Errorf("lag p99=%.1fms want about 300ms behind the ideal schedule", pr.ScheduleLagP99MS)
		}
		if pr.ScheduleKept == nil || *pr.ScheduleKept {
			t.Fatalf("stalled run: ScheduleKept=%v want false (%+v)", pr.ScheduleKept, pr)
		}
	})

	t.Run("blocking feed counts as lag", func(t *testing.T) {
		// Every feed waits 900 ms for a free worker. The pacer then never needs
		// to sleep late, but each request still leaves well after its deadline.
		sim := newAdvSim(base)
		p := newPacer(time.Second, catchUp, sim.nowFn, sim.sleepFn)
		got := p.run(context.Background(), 20, func() bool {
			sim.cur = sim.cur.Add(900 * time.Millisecond)
			return sim.issueFn()
		})
		if got != 20 {
			t.Fatalf("issued=%d want 20", got)
		}
		var pr PointResult
		p.adherence.apply(&pr)
		if pr.ScheduleLagP99MS < 899 {
			t.Errorf("lag p99=%.1fms want about 900ms: back-pressure delays acceptance", pr.ScheduleLagP99MS)
		}
		if pr.ScheduleKept == nil || *pr.ScheduleKept {
			t.Fatalf("blocked feed: ScheduleKept=%v want false", pr.ScheduleKept)
		}
	})

	t.Run("too few issued loses the schedule", func(t *testing.T) {
		sim := newAdvSim(base)
		issuedN := 0
		p := newPacer(interval, catchUp, sim.nowFn, sim.sleepFn)
		got := p.run(context.Background(), 100, func() bool {
			if issuedN == 98 {
				return false
			}
			issuedN++
			return true
		})
		if got != 98 {
			t.Fatalf("issued=%d want 98", got)
		}
		var pr PointResult
		p.adherence.apply(&pr)
		if pr.ScheduleKept == nil || *pr.ScheduleKept {
			t.Fatalf("98/100 issued: ScheduleKept=%v want false", pr.ScheduleKept)
		}
	})
}

// TestRunReportsAdherenceOnlyForScheduledPoints pins which points carry the
// adherence fields: a rate-shaped arrival does, while a zero-gap burst and an
// unpaced sweep have no schedule to keep and leave them out.
// fak-test:runtime fast est=100ms
func TestRunReportsAdherenceOnlyForScheduledPoints(t *testing.T) {
	t.Parallel()
	point := func(arrival ArrivalPattern, rate float64) PointResult {
		mr := Run(t.Context(), &Config{
			URL: "http://127.0.0.1:1/v1/chat/completions", Model: "m",
			Concurrencies: []int{1}, RequestsPer: 3,
			Arrival: arrival, ArrivalRate: rate, ArrivalSeed: 7,
		})
		if len(mr.Points) != 1 {
			t.Fatalf("%q: points=%d want 1", arrival, len(mr.Points))
		}
		return mr.Points[0]
	}
	if pr := point(arrivalPoisson, 1000); pr.ScheduleKept == nil || pr.PlannedRequests != 3 {
		t.Errorf("poisson point: kept=%v planned=%d, want adherence reported for 3 planned", pr.ScheduleKept, pr.PlannedRequests)
	}
	if pr := point(arrivalBurst, 0); pr.ScheduleKept != nil || pr.PlannedRequests != 0 {
		t.Errorf("burst point: kept=%v planned=%d, want no adherence fields", pr.ScheduleKept, pr.PlannedRequests)
	}
	if pr := point("", 0); pr.ScheduleKept != nil {
		t.Errorf("unpaced point: kept=%v, want no adherence fields", pr.ScheduleKept)
	}
}
