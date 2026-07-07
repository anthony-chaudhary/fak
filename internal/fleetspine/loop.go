package fleetspine

import (
	"context"
	"time"
)

// Logf is the optional one-line logger the runners use for transient, non-fatal transport
// errors (a failed multicast send, a listen error we are about to return). nil is a no-op, so
// a caller that wants silence passes nothing.
type Logf func(format string, args ...any)

func (l Logf) printf(format string, args ...any) {
	if l != nil {
		l(format, args...)
	}
}

// RunAdvertiser broadcasts THIS machine's heartbeat on every tick of interval until ctx is
// cancelled. snapshot builds the current compact heartbeat (id/host/verdict/version/stamp) —
// it must NOT shell out; the whole point is a cheap, subprocess-free announce. A transient
// send error is logged once and the loop continues (a flapping NIC must never stop the guard).
// It advertises once immediately so a freshly-started peer appears without waiting a full tick.
func RunAdvertiser(ctx context.Context, t Transport, interval time.Duration, snapshot func() Heartbeat, logf Logf) {
	if interval <= 0 {
		interval = 10 * time.Second
	}
	advertise := func() {
		if err := t.Advertise(ctx, snapshot()); err != nil && ctx.Err() == nil {
			logf.printf("fleetspine: advertise: %v", err)
		}
	}
	advertise()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			advertise()
		}
	}
}

// RunListener receives peers' heartbeats and folds each into reg until ctx is cancelled. It
// stamps every heartbeat with the registry clock at arrival, so expiry is measured from when
// WE saw it, not a timestamp the peer controls. A listen error (other than cancellation) is
// logged once and the loop returns — the guard keeps running on the disk-only path.
func RunListener(ctx context.Context, t Transport, reg *Registry, logf Logf) {
	err := t.Listen(ctx, func(hb Heartbeat) {
		reg.Ingest(hb, reg.clock())
	})
	if err != nil && ctx.Err() == nil {
		logf.printf("fleetspine: listen: %v", err)
	}
}

// RunExpiry periodically drops hard-expired peers so a machine that has left the network
// disappears from the view even when no new heartbeats arrive to trigger the read-time prune.
// interval defaults to the registry miss-window when not positive.
func RunExpiry(ctx context.Context, reg *Registry, interval time.Duration) {
	if interval <= 0 {
		interval = reg.missWindow
	}
	if interval <= 0 {
		interval = 15 * time.Minute
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			reg.Expire(reg.clock())
		}
	}
}

// Spine bundles a transport and a registry with the machine's own heartbeat source, and runs
// all three loops under one context. It is the single object the guard builds, starts, and
// reads peers from (via the registry's MachineMaps). Run returns when ctx is cancelled.
type Spine struct {
	Transport Transport
	Registry  *Registry
	Interval  time.Duration
	Snapshot  func() Heartbeat
	Logf      Logf
}

// Run starts the advertiser, listener, and expiry loops and blocks until ctx is cancelled,
// after which all three have returned. Intended to be launched in its own goroutine bound to
// the guard-lifetime context.
func (s *Spine) Run(ctx context.Context) {
	done := make(chan struct{}, 3)
	go func() { RunAdvertiser(ctx, s.Transport, s.Interval, s.Snapshot, s.Logf); done <- struct{}{} }()
	go func() { RunListener(ctx, s.Transport, s.Registry, s.Logf); done <- struct{}{} }()
	go func() { RunExpiry(ctx, s.Registry, s.Interval); done <- struct{}{} }()
	for i := 0; i < 3; i++ {
		<-done
	}
}
