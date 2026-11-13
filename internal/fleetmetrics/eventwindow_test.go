package fleetmetrics

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// The windowed-counting contract, adapted from NVIDIA/dcgm-exporter's windowed
// XID counting (internal/pkg/collector/xid_collector.go:32,
// internal/pkg/collector/expcollector.go, count-window config in
// internal/pkg/appconfig/count_window.go) pinned at
// 4e663a526c4624ad0e18ad816f09b7be2c5348a7.
//
// Every assertion below is a closed contract: an exact count, an exact code set,
// or a sentinel error via errors.Is. No assertion depends on error prose.

// TestFleetMetricsCountWindowCollapsesBurstToCount is the load-bearing witness:
// five samples of the SAME code inside the window are ONE series reading 5, not
// five gauge rows each reading 1.
func TestFleetMetricsCountWindowCollapsesBurstToCount(t *testing.T) {
	now := time.Date(2026, 11, 12, 12, 0, 0, 0, time.UTC)
	events := []Event{
		{Code: 79, At: now.Add(-4 * time.Minute)},
		{Code: 79, At: now.Add(-3 * time.Minute)},
		{Code: 79, At: now.Add(-2 * time.Minute)},
		{Code: 79, At: now.Add(-90 * time.Second)},
		{Code: 79, At: now.Add(-time.Second)},
	}

	got, err := CountWindow(events, now, 5*time.Minute)
	if err != nil {
		t.Fatalf("CountWindow: %v", err)
	}
	if len(got.Counts) != 1 {
		t.Fatalf("burst of one code produced %d rows, want 1: %+v", len(got.Counts), got.Counts)
	}
	if got.Counts[0].Code != 79 || got.Counts[0].Count != 5 {
		t.Fatalf("row = %+v, want code 79 count 5 (a burst is a count, not N gauge flips)", got.Counts[0])
	}
	if got.ObservedEvents != 5 {
		t.Fatalf("ObservedEvents = %d, want 5", got.ObservedEvents)
	}
	if !got.Observed.Equal(now.Add(-time.Second)) {
		t.Fatalf("Observed = %v, want the newest in-window sample %v", got.Observed, now.Add(-time.Second))
	}
	if got.Window != 5*time.Minute {
		t.Fatalf("Window = %v, want 5m", got.Window)
	}
}

// TestFleetMetricsCountWindowQuietCodeStaysVisibleAsZero pins the distinction a
// gauge cannot make: a code observed only OUTSIDE the window is retained with
// count 0 (quiet), not dropped (indistinguishable from "no longer reported").
func TestFleetMetricsCountWindowQuietCodeStaysVisibleAsZero(t *testing.T) {
	now := time.Date(2026, 11, 12, 12, 0, 0, 0, time.UTC)
	events := []Event{
		{Code: 48, At: now.Add(-30 * time.Minute)}, // long expired
		{Code: 79, At: now.Add(-time.Minute)},      // live
	}

	got, err := CountWindow(events, now, 5*time.Minute)
	if err != nil {
		t.Fatalf("CountWindow: %v", err)
	}
	if len(got.Counts) != 2 {
		t.Fatalf("got %d rows, want both codes retained (one quiet): %+v", len(got.Counts), got.Counts)
	}
	// Documented order: a live code precedes a quiet one, so 79 comes first even
	// though 48 sorts lower.
	if got.Counts[0] != (EventCount{Code: 79, Count: 1}) {
		t.Fatalf("live code row = %+v, want {79 1}", got.Counts[0])
	}
	if got.Counts[1] != (EventCount{Code: 48, Count: 0}) {
		t.Fatalf("expired code row = %+v, want {48 0} (retained at zero, not dropped)", got.Counts[1])
	}
	if got.CodesInWindow != 1 {
		t.Fatalf("CodesInWindow = %d, want 1 (only 79 fired inside)", got.CodesInWindow)
	}
	if got.ObservedEvents != 1 {
		t.Fatalf("ObservedEvents = %d, want 1", got.ObservedEvents)
	}
}

// TestFleetMetricsCountWindowRefusesNonPositiveWindow: a zero or negative window
// is a caller bug, refused with the sentinel rather than silently folded as a
// zero-width window that reports every code as quiet.
func TestFleetMetricsCountWindowRefusesNonPositiveWindow(t *testing.T) {
	now := time.Date(2026, 11, 12, 12, 0, 0, 0, time.UTC)
	events := []Event{{Code: 79, At: now}}

	for _, window := range []time.Duration{0, -time.Second} {
		got, err := CountWindow(events, now, window)
		if !errors.Is(err, ErrEventWindowNotPositive) {
			t.Fatalf("window %v: err = %v, want ErrEventWindowNotPositive", window, err)
		}
		if len(got.Counts) != 0 {
			t.Fatalf("window %v: refused fold returned rows: %+v", window, got.Counts)
		}
	}
}

// TestFleetMetricsCountWindowBoundaryAndFutureSamples: the window is
// [now-window, now]. A sample exactly on the cutoff is inside (inclusive); a
// sample stamped after now cannot be inside it.
func TestFleetMetricsCountWindowBoundaryAndFutureSamples(t *testing.T) {
	now := time.Date(2026, 11, 12, 12, 0, 0, 0, time.UTC)
	events := []Event{
		{Code: 79, At: now.Add(-5 * time.Minute)}, // exactly the cutoff: inside
		{Code: 79, At: now.Add(-5*time.Minute - time.Nanosecond)},
		{Code: 63, At: now.Add(time.Minute)}, // future stamp: outside
	}

	got, err := CountWindow(events, now, 5*time.Minute)
	if err != nil {
		t.Fatalf("CountWindow: %v", err)
	}
	want := []EventCount{{Code: 79, Count: 1}, {Code: 63, Count: 0}}
	if len(got.Counts) != len(want) {
		t.Fatalf("rows = %+v, want %+v", got.Counts, want)
	}
	for i := range want {
		if got.Counts[i] != want[i] {
			t.Fatalf("row %d = %+v, want %+v", i, got.Counts[i], want[i])
		}
	}
}

// TestFleetMetricsCountWindowRenderPutsWorstFirst: the operator line ordering is
// worst-count-first, then ascending code, so the burst is the first thing read.
func TestFleetMetricsCountWindowRenderPutsWorstFirst(t *testing.T) {
	now := time.Date(2026, 11, 12, 12, 0, 0, 0, time.UTC)
	events := []Event{
		{Code: 79, At: now.Add(-time.Minute)},
		{Code: 79, At: now.Add(-2 * time.Minute)},
		{Code: 79, At: now.Add(-3 * time.Minute)},
		{Code: 48, At: now.Add(-time.Minute)},
		{Code: 63, At: now.Add(-20 * time.Minute)}, // quiet
	}

	got, err := CountWindow(events, now, 5*time.Minute)
	if err != nil {
		t.Fatalf("CountWindow: %v", err)
	}
	out := got.Render("last 5m")
	// Four samples fell inside the window (three 79s + one 48); 63 is quiet.
	for _, want := range []string{"79=3", "48=1", "63=0", "n=4"} {
		if !strings.Contains(out, want) {
			t.Fatalf("Render missing %q: %q", want, out)
		}
	}
	if strings.Index(out, "79=3") > strings.Index(out, "48=1") {
		t.Fatalf("Render is not worst-count-first: %q", out)
	}

	if empty := (EventWindow{Window: time.Minute}).Render("last 1m"); !strings.Contains(empty, "no events sampled") {
		t.Fatalf("empty Render = %q, want an explicit no-events line", empty)
	}
}
