package main

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func TestSchedulerCancelAll(t *testing.T) {
	s := NewLoopScheduler()
	var ran atomic.Int32
	started := make(chan struct{})
	cancelled := make(chan struct{})

	s.ScheduleAt(time.Now(), func(ctx context.Context) {
		ran.Add(1)
		close(started)
		<-ctx.Done()
		close(cancelled)
	})
	for i := 0; i < 5; i++ {
		s.ScheduleAt(time.Now().Add(time.Hour), func(context.Context) { ran.Add(1) })
	}
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("immediate job never started")
	}
	if p, r := s.Counts(); p != 5 || r != 1 {
		t.Fatalf("counts pending=%d running=%d, want 5/1", p, r)
	}

	if dropped := s.CancelAll(); dropped != 5 {
		t.Fatalf("dropped %d pending, want 5", dropped)
	}
	select {
	case <-cancelled:
	case <-time.After(2 * time.Second):
		t.Fatal("running job context was not cancelled")
	}
	s.Wait()
	if got := ran.Load(); got != 1 {
		t.Fatalf("%d jobs ran, want only the already-running one", got)
	}
	if p, r := s.Counts(); p != 0 || r != 0 {
		t.Fatalf("counts after cancel pending=%d running=%d", p, r)
	}
	if s.ScheduleAt(time.Now(), func(context.Context) {}) {
		t.Fatal("scheduling after CancelAll must be refused")
	}
}
