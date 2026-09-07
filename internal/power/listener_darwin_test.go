//go:build darwin

package power

import (
	"context"
	"testing"
	"time"
)

func TestDarwinPureGoListener(t *testing.T) {
	b := NewPowerBroadcaster()
	obs := newTestObserver()
	cancelObs := b.RegisterObserver(obs)
	defer cancelObs()

	l := &darwinPureGoListener{
		broadcaster: b,
		checkPeriod: 10 * time.Millisecond,
		jumpRatio:   50 * time.Millisecond,
		done:        make(chan struct{}),
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := l.Start(ctx); err != nil {
		t.Fatalf("Start pure-go listener: %v", err)
	}

	// Wait a moment for listener to run normally without jumps
	select {
	case <-time.After(30 * time.Millisecond):
	}

	if obs.SuspendCount() != 0 || obs.ResumeCount() != 0 {
		t.Fatalf("expected 0 events during normal ticks, got suspend=%d resume=%d",
			obs.SuspendCount(), obs.ResumeCount())
	}

	// Now stop the listener cleanly
	if err := l.Stop(); err != nil {
		t.Fatalf("Stop listener: %v", err)
	}

	// Duplicate stop should be safe
	if err := l.Stop(); err != nil {
		t.Fatalf("second Stop: %v", err)
	}
}

func TestDarwinSleepListenerCreation(t *testing.T) {
	b := NewPowerBroadcaster()

	// 1. Pure-go forced listener test
	SetDarwinForcePureGoListenerForTesting(true)
	pureGoL := NewSleepListener(b)
	SetDarwinForcePureGoListenerForTesting(false)

	if pureGoL == nil {
		t.Fatal("expected non-nil pureGo listener")
	}

	ctx, cancel := context.WithCancel(context.Background())
	if err := pureGoL.Start(ctx); err != nil {
		t.Fatalf("pureGoL.Start: %v", err)
	}
	cancel()
	_ = pureGoL.Stop()

	// 2. Default listener (CGO IOKit if cgo enabled, else pure-go)
	defaultL := NewSleepListener(b)
	if defaultL == nil {
		t.Fatal("expected non-nil default listener")
	}

	ctx2, cancel2 := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel2()

	if err := defaultL.Start(ctx2); err != nil {
		t.Fatalf("defaultL.Start: %v", err)
	}

	select {
	case <-time.After(50 * time.Millisecond):
	}
	if err := defaultL.Stop(); err != nil {
		t.Fatalf("defaultL.Stop: %v", err)
	}
}
