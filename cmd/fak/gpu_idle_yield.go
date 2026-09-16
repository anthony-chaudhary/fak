package main

import (
	"fmt"
	"sync"
	"time"
)

// gpuIdleExitGovernor stops a resident turnkey server once it has served no
// request for a bounded idle window, so the machine-wide GPU lease (and the
// model residency it protects) is RELEASED instead of being pinned for the
// process lifetime.
//
// Motivating defect (#13135): `fak up` / `fak-native up --headless --model 27B`
// acquired /tmp/fak-gpu.lease at startup and held it until exit, so a paired
// modelbench that QUEUED behind the resident server waited without bound and the
// #11582 baseline/candidate receipt was uncapturable without killing serving by
// hand. The strix appliance treats leases as SHORT and reconciled against live
// activity; this brings the same discipline to the Mac turnkey server: while a
// request is in flight the lease is held, and an idle holder STOPS rather than
// squatting.
//
// The coupling to residency is deliberate and is the safety invariant, not an
// incidental detail: releasing the GPU lease while the weights stay resident
// would let a second GPU-heavy process stack residency on the same unified
// memory pool — the 2026-06-18 jetsam cascade the lease exists to prevent. So the
// bounded idle transition is a full stop (residency + lease released together),
// and the launcher/supervisor restarts on demand.
//
// The transition is operator-visible (a stderr line naming the idle window) and
// reversible (--gpu-idle-exit 0 restores the process-lifetime holder).
type gpuIdleExitGovernor struct {
	mu       sync.Mutex
	idle     time.Duration
	active   int
	timer    *time.Timer
	closed   bool
	shutdown func()
	logf     func(format string, args ...any)
	// now/after are test seams; nil uses the real clock.
	now   func() time.Time
	after func(time.Duration) <-chan time.Time
}

// defaultGPUIdleExit is the default idle window for the Mac turnkey server.
// Short enough that a queued benchmark proceeds promptly (the #13135 need),
// long enough that an interactive agent's between-turn think-time does not
// churn the model load.
const defaultGPUIdleExit = 2 * time.Minute

// newGPUIdleExitGovernor builds an idle-exit governor. A non-positive idle
// disables it and returns nil, preserving the historical process-lifetime holder
// byte-for-byte.
func newGPUIdleExitGovernor(idle time.Duration, shutdown func(), logf func(format string, args ...any)) *gpuIdleExitGovernor {
	if idle <= 0 || shutdown == nil {
		return nil
	}
	if logf == nil {
		logf = func(format string, args ...any) { fmt.Printf(format+"\n", args...) }
	}
	return &gpuIdleExitGovernor{idle: idle, shutdown: shutdown, logf: logf}
}

// requestBegan arms the held window: any pending idle timer is cancelled.
func (g *gpuIdleExitGovernor) requestBegan() {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return
	}
	g.active++
	if g.timer != nil {
		g.timer.Stop()
		g.timer = nil
	}
}

// requestEnded removes one in-flight request; the last one arms the idle timer.
func (g *gpuIdleExitGovernor) requestEnded() {
	if g == nil {
		return
	}
	g.mu.Lock()
	if g.closed {
		g.mu.Unlock()
		return
	}
	if g.active > 0 {
		g.active--
	}
	if g.active == 0 && g.timer == nil {
		g.timer = time.AfterFunc(g.idle, g.exitIfIdle)
	}
	g.mu.Unlock()
}

// exitIfIdle stops the server when the idle window has elapsed with no request
// in flight. It is the timer callback.
func (g *gpuIdleExitGovernor) exitIfIdle() {
	if g == nil {
		return
	}
	g.mu.Lock()
	if g.closed || g.active != 0 {
		g.timer = nil
		g.mu.Unlock()
		return
	}
	g.closed = true
	g.timer = nil
	idle := g.idle
	shutdown := g.shutdown
	logf := g.logf
	g.mu.Unlock()
	logf("fak up: no request for %s; stopping the resident server so the GPU lease and model residency are released (#13135)", idle)
	shutdown()
}

// close stops the governor without triggering a shutdown (an operator-initiated
// or signal-initiated stop is already in progress).
func (g *gpuIdleExitGovernor) close() {
	if g == nil {
		return
	}
	g.mu.Lock()
	g.closed = true
	if g.timer != nil {
		g.timer.Stop()
		g.timer = nil
	}
	g.mu.Unlock()
}

// activeRequests reports the in-flight request count (observability/tests).
func (g *gpuIdleExitGovernor) activeRequests() int {
	if g == nil {
		return 0
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.active
}
