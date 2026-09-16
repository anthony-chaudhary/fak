package main

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// waitFor polls until cond is true or the deadline elapses.
func waitFor(t *testing.T, deadline time.Duration, cond func() bool) bool {
	t.Helper()
	end := time.Now().Add(deadline)
	for time.Now().Before(end) {
		if cond() {
			return true
		}
		time.Sleep(2 * time.Millisecond)
	}
	return cond()
}

// TestGPUIdleExitDisabledIsNil proves the historical process-lifetime holder is
// preserved byte-for-byte when the window is zero/negative: no governor, no stop.
func TestGPUIdleExitDisabledIsNil(t *testing.T) {
	for _, idle := range []time.Duration{0, -time.Second} {
		if g := newGPUIdleExitGovernor(idle, func() {}, nil); g != nil {
			t.Fatalf("idle=%s: expected nil governor (historical lifetime holder), got %#v", idle, g)
		}
	}
	// A nil governor must be inert on every method (the field is optional).
	var nilG *gpuIdleExitGovernor
	nilG.requestBegan()
	nilG.requestEnded()
	nilG.close()
	if got := nilG.activeRequests(); got != 0 {
		t.Fatalf("nil governor activeRequests=%d, want 0", got)
	}
}

// TestGPUIdleExitFiresWhenIdle is the #13135 core: after the idle window the
// governor stops the server exactly once so the GPU lease is released.
func TestGPUIdleExitFiresWhenIdle(t *testing.T) {
	var stopped atomic.Int32
	g := newGPUIdleExitGovernor(10*time.Millisecond, func() { stopped.Add(1) }, func(string, ...any) {})

	g.requestBegan()
	g.requestEnded()
	if !waitFor(t, 2*time.Second, func() bool { return stopped.Load() == 1 }) {
		t.Fatalf("expected exactly one idle-stop, got %d", stopped.Load())
	}
	// A second timer must not fire a second stop.
	time.Sleep(50 * time.Millisecond)
	if got := stopped.Load(); got != 1 {
		t.Fatalf("idle-stop fired %d times, want 1", got)
	}
}

// TestGPUIdleExitArmedOnlyOnLastRequest proves a request in flight cancels the
// pending idle stop, and the stop re-arms only after the last request drains.
func TestGPUIdleExitArmedOnlyOnLastRequest(t *testing.T) {
	var stopped atomic.Int32
	g := newGPUIdleExitGovernor(30*time.Millisecond, func() { stopped.Add(1) }, func(string, ...any) {})

	g.requestBegan()
	g.requestEnded()
	// A new request inside the window must cancel the pending stop.
	time.Sleep(5 * time.Millisecond)
	g.requestBegan()
	if waitFor(t, 200*time.Millisecond, func() bool { return stopped.Load() > 0 }) {
		t.Fatal("idle stop fired while a request was in flight")
	}
	g.requestEnded()
	if !waitFor(t, 2*time.Second, func() bool { return stopped.Load() == 1 }) {
		t.Fatalf("expected the idle stop after the last request drained, got %d", stopped.Load())
	}
}

// TestGPUIdleExitCloseSuppressesStop proves an operator/signal stop quiesces the
// governor without a spurious second shutdown.
func TestGPUIdleExitCloseSuppressesStop(t *testing.T) {
	var stopped atomic.Int32
	g := newGPUIdleExitGovernor(10*time.Millisecond, func() { stopped.Add(1) }, func(string, ...any) {})
	g.close()
	time.Sleep(80 * time.Millisecond)
	if got := stopped.Load(); got != 0 {
		t.Fatalf("closed governor fired %d stops, want 0", got)
	}
	// After close, a begin/end pair must not re-arm either.
	g.requestBegan()
	g.requestEnded()
	time.Sleep(60 * time.Millisecond)
	if got := stopped.Load(); got != 0 {
		t.Fatalf("closed governor fired %d stops after close, want 0", got)
	}
}

// TestGPUIdleExitConcurrentRequests proves the active count is race-free: many
// overlapping requests keep the holder alive until the last one ends.
func TestGPUIdleExitConcurrentRequests(t *testing.T) {
	var stopped atomic.Int32
	g := newGPUIdleExitGovernor(40*time.Millisecond, func() { stopped.Add(1) }, func(string, ...any) {})

	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			g.requestBegan()
			time.Sleep(10 * time.Millisecond)
			g.requestEnded()
		}()
	}
	// While the requests are in flight (well inside the 40ms window) no stop may
	// fire. Sampling at 25ms stays below the window even as the cohort drains.
	time.Sleep(25 * time.Millisecond)
	if stopped.Load() > 0 {
		t.Fatal("idle stop fired while concurrent requests were in flight")
	}
	wg.Wait()
	if !waitFor(t, 2*time.Second, func() bool { return stopped.Load() == 1 }) {
		t.Fatalf("expected one idle stop after concurrency drained, got %d", stopped.Load())
	}
}

// TestTurnkeyServerArmGPUIdleExitDisabled proves the server-level arm is a no-op
// for a non-positive window and for a server with no stop wired.
func TestTurnkeyServerArmGPUIdleExitDisabled(t *testing.T) {
	s := &turnkeyServer{}
	s.armGPUIdleExit(0)
	if s.idleExit != nil {
		t.Fatal("armGPUIdleExit(0) installed a governor; want nil (disabled)")
	}
	s.armGPUIdleExit(defaultGPUIdleExit)
	if s.idleExit != nil {
		t.Fatal("armGPUIdleExit on a server without stop installed a governor; want nil")
	}
}

// TestTurnkeyServerIdleExitIntegration proves the server's own begin/end hooks
// drive the governor to a stop, which is the exact path the resident `fak up`
// server takes to release the machine-wide GPU lease (#13135).
func TestTurnkeyServerIdleExitIntegration(t *testing.T) {
	var stopped atomic.Int32
	s := &turnkeyServer{}
	s.stop = func() { stopped.Add(1) }
	s.armGPUIdleExit(15 * time.Millisecond)
	if s.idleExit == nil {
		t.Fatal("expected an armed idle governor")
	}
	if !s.beginChatRequest() {
		t.Fatal("beginChatRequest refused a fresh server")
	}
	if waitFor(t, 120*time.Millisecond, func() bool { return stopped.Load() > 0 }) {
		t.Fatal("idle stop fired with a request in flight")
	}
	s.endChatRequest()
	if !waitFor(t, 2*time.Second, func() bool { return stopped.Load() == 1 }) {
		t.Fatalf("expected the server to stop once idle, got %d stops", stopped.Load())
	}
}
