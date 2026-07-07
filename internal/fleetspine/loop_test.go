package fleetspine

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

// TestRunListenerFeedsRegistryThenStops: heartbeats pushed onto a fake transport land in the
// registry, and the listener returns promptly when ctx is cancelled.
func TestRunListenerFeedsRegistryThenStops(t *testing.T) {
	tr := newChanTransport(8)
	reg := NewRegistry(RegistryConfig{SelfID: "me"})
	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan struct{})
	go func() { RunListener(ctx, tr, reg, nil); close(done) }()

	t0 := time.Now()
	_ = tr.Advertise(ctx, mkHB("alpha", t0))
	_ = tr.Advertise(ctx, mkHB("beta", t0))

	// Poll until both peers are visible (the listener drains asynchronously).
	deadline := time.After(2 * time.Second)
	for {
		if len(reg.Snapshot(time.Now())) == 2 {
			break
		}
		select {
		case <-deadline:
			t.Fatalf("registry never reached 2 peers, got %d", len(reg.Snapshot(time.Now())))
		case <-time.After(10 * time.Millisecond):
		}
	}

	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("RunListener did not return after cancel")
	}
}

// TestRunAdvertiserEmitsThenStops: the advertiser sends at least once (the immediate announce)
// and returns on cancel.
func TestRunAdvertiserEmitsThenStops(t *testing.T) {
	tr := newChanTransport(16)
	ctx, cancel := context.WithCancel(context.Background())

	var count int32
	// Drain the transport so Advertise never blocks on a full channel.
	go func() { _ = tr.Listen(ctx, func(Heartbeat) { atomic.AddInt32(&count, 1) }) }()

	done := make(chan struct{})
	snap := func() Heartbeat { return mkHB("me", time.Now()) }
	go func() { RunAdvertiser(ctx, tr, 20*time.Millisecond, snap, nil); close(done) }()

	// Wait for at least the immediate announce plus a tick or two.
	deadline := time.After(2 * time.Second)
	for atomic.LoadInt32(&count) < 2 {
		select {
		case <-deadline:
			t.Fatalf("advertiser emitted only %d heartbeats", atomic.LoadInt32(&count))
		case <-time.After(10 * time.Millisecond):
		}
	}

	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("RunAdvertiser did not return after cancel")
	}
}

// TestSpineRunStopsCleanly: the composed Spine.Run starts all three loops and returns once ctx
// is cancelled (no goroutine left blocking).
func TestSpineRunStopsCleanly(t *testing.T) {
	tr := newChanTransport(16)
	reg := NewRegistry(RegistryConfig{SelfID: "me", MissWindow: time.Minute})
	s := &Spine{
		Transport: tr,
		Registry:  reg,
		Interval:  20 * time.Millisecond,
		Snapshot:  func() Heartbeat { return mkHB("me", time.Now()) },
	}
	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan struct{})
	go func() { s.Run(ctx); close(done) }()

	time.Sleep(60 * time.Millisecond) // let the loops spin up
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Spine.Run did not return after cancel (a loop is stuck)")
	}
}
