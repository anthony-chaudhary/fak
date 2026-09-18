package main

import (
	"strings"
	"sync"
	"testing"
	"time"
)

// TestMemGuardSampleSustainedBreach drives sample() directly with an injected
// clock and RSS so the sustain rule is deterministic and the timer goroutine
// never races the assertions.
func TestMemGuardSampleSustainedBreach(t *testing.T) {
	var (
		mu      sync.Mutex
		stopped bool
		now     = time.Unix(1000, 0)
		rss     = uint64(100)
	)
	g := newMemGuardGovernor(50, time.Hour, 10*time.Second,
		func() uint64 { mu.Lock(); defer mu.Unlock(); return rss },
		func() { mu.Lock(); stopped = true; mu.Unlock() },
		func(string, ...any) {})
	g.now = func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
	g.mu.Lock()
	g.closed = false
	g.mu.Unlock()

	g.sample() // over the ceiling: starts the over-window, no stop yet
	if isStopped(&mu, &stopped) {
		t.Fatal("guard stopped on first over-ceiling sample; want sustain wait")
	}
	mu.Lock()
	now = now.Add(5 * time.Second)
	mu.Unlock()
	g.sample() // still inside the sustain window
	if isStopped(&mu, &stopped) {
		t.Fatal("guard stopped before the sustain window elapsed")
	}
	mu.Lock()
	now = now.Add(6 * time.Second) // now 11s over > 10s sustain
	mu.Unlock()
	g.sample()
	if !isStopped(&mu, &stopped) {
		t.Fatal("guard did not stop after a sustained over-ceiling window")
	}
}

// TestMemGuardClearsWhenUnderCeiling pins the transient-high-water tolerance:
// dropping back under the ceiling must reset the over-window.
func TestMemGuardClearsWhenUnderCeiling(t *testing.T) {
	var (
		mu      sync.Mutex
		stopped bool
		now     = time.Unix(1000, 0)
		rss     = uint64(100)
	)
	g := newMemGuardGovernor(50, time.Hour, 10*time.Second,
		func() uint64 { mu.Lock(); defer mu.Unlock(); return rss },
		func() { mu.Lock(); stopped = true; mu.Unlock() },
		func(string, ...any) {})
	g.now = func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
	g.mu.Lock()
	g.closed = false
	g.mu.Unlock()

	g.sample() // over
	mu.Lock()
	now = now.Add(5 * time.Second)
	rss = 40 // under
	mu.Unlock()
	g.sample()
	mu.Lock()
	now = now.Add(20 * time.Second)
	mu.Unlock()
	g.sample() // still under: never over long enough
	if isStopped(&mu, &stopped) {
		t.Fatal("guard stopped after the breach cleared; want reset")
	}
}

// TestNewMemGuardGovernorDisabled pins that a zero limit yields no governor, so
// --max-rss 0 is a true opt-out.
func TestNewMemGuardGovernorDisabled(t *testing.T) {
	if g := newMemGuardGovernor(0, time.Second, time.Second, nil, func() {}, nil); g != nil {
		t.Fatal("zero limit must disable the guard")
	}
	if g := newMemGuardGovernor(1, time.Second, time.Second, nil, nil, nil); g != nil {
		t.Fatal("nil shutdown must disable the guard")
	}
}

// TestFormatBytes covers the operator-visible log rendering.
func TestFormatBytes(t *testing.T) {
	cases := map[uint64]string{
		0:            "0B",
		512:          "512B",
		2048:         "2.0KB",
		5 * 1 << 30:  "5.0GB",
		27 * 1 << 30: "27.0GB",
	}
	for in, want := range cases {
		if got := formatBytes(in); got != want {
			t.Errorf("formatBytes(%d) = %q, want %q", in, got, want)
		}
	}
	if !strings.Contains(formatBytes(1<<30), "GB") {
		t.Fatal("formatBytes(1GiB) must render GB units")
	}
}

func isStopped(mu *sync.Mutex, stopped *bool) bool {
	mu.Lock()
	defer mu.Unlock()
	return *stopped
}
