// Windowed GPU-event counting, adapted from the windowed XID counter in
// NVIDIA/dcgm-exporter (Apache-2.0): internal/pkg/collector/xid_collector.go:32
// and the shared windowed collector in
// internal/pkg/collector/expcollector.go, with its count-window configuration in
// internal/pkg/appconfig/count_window.go, pinned at
// 4e663a526c4624ad0e18ad816f09b7be2c5348a7:
// https://github.com/NVIDIA/dcgm-exporter/blob/4e663a526c4624ad0e18ad816f09b7be2c5348a7/internal/pkg/collector/xid_collector.go
//
// The idea adapted here is the one the upstream window exists to fix: the
// exporter samples a LAST-XID gauge, so a burst of the same code arrives as N
// samples in the window and a naive reader reports N gauge flips (the newer
// sample overwriting the older one) instead of one count. fak re-expresses that
// as a pure fold over events fak already stamps: a count per event code inside
// the window, plus the quiet codes retained at zero so "fired zero times" stays
// distinguishable from "no longer observed at all".
//
// This is an ADAPT verdict, not a direct port: no upstream code is copied, and
// the fold is re-expressed over fak's own Event record in idiomatic Go. The
// upstream field/label vocabulary is deliberately NOT mirrored - fak's receipts
// (internal/hostfault, pkg/hwutil) already own that closed vocabulary.

package fleetmetrics

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// ErrEventWindowNotPositive is returned by CountWindow when window <= 0. A
// zero-width window would report every code as quiet, which is never the caller's
// intent, so it is refused instead of folded.
var ErrEventWindowNotPositive = errors.New("fleetmetrics: event window must be positive")

// Event is one sampled fleet GPU event: a numeric code (an NVIDIA XID or SXID,
// or fak's own event code) and when it was observed. Code is intentionally
// numeric-only: operator text is not part of this contract.
type Event struct {
	Code int64
	At   time.Time
}

// EventCount is one row of a window fold: the code and how many samples of it
// fell inside the window. A zero Count is a code that was observed before but not
// within the window - retained so the row is not silently dropped.
type EventCount struct {
	Code  int64
	Count int
}

// EventWindow is the folded result of CountWindow. It carries its own window so
// a receipt cannot be misread against a different one, and Observed so a caller
// can tell live telemetry from a stale replay.
type EventWindow struct {
	// Window is the count window the fold used.
	Window time.Duration
	// Counts is one row per distinct code, worst count first, then ascending
	// code. A code with Count 0 fired only before the window.
	Counts []EventCount
	// Observed is the newest sample accepted into the fold - the "last event"
	// stamp. Zero when nothing was accepted.
	Observed time.Time
	// ObservedEvents is how many samples fell inside the window.
	ObservedEvents int
	// CodesInWindow is how many DISTINCT codes fired inside the window.
	CodesInWindow int
}

// CountWindow folds events into one count per code observed in the half-open
// window [now-window, now]. A burst of the same code inside the window is ONE
// row carrying the burst size, not N single-sample rows - that collapse is the
// whole point of the fold.
//
// Every distinct code in events is retained: one that fired only outside the
// window keeps its row at count 0, so a quiet code stays visible. A sample
// stamped after now cannot be inside the window and is ignored.
//
// It is a pure function: events is neither reordered nor mutated, and it reads no
// clock or global state.
func CountWindow(events []Event, now time.Time, window time.Duration) (EventWindow, error) {
	if window <= 0 {
		return EventWindow{}, fmt.Errorf("%w: got %v", ErrEventWindowNotPositive, window)
	}

	out := EventWindow{Window: window}
	cutoff := now.Add(-window)

	counts := make(map[int64]int, len(events))
	for _, e := range events {
		// Record the code first: a code seen only before the window must stay
		// visible at zero rather than vanish.
		if _, seen := counts[e.Code]; !seen {
			counts[e.Code] = 0
		}
		if e.At.After(now) || e.At.Before(cutoff) {
			continue
		}
		counts[e.Code]++
		out.ObservedEvents++
		if e.At.After(out.Observed) {
			out.Observed = e.At
		}
	}

	out.Counts = make([]EventCount, 0, len(counts))
	for code, count := range counts {
		out.Counts = append(out.Counts, EventCount{Code: code, Count: count})
		if count > 0 {
			out.CodesInWindow++
		}
	}
	// Worst non-zero count first, then ascending code, then quiet codes. Sorting
	// here (not in the caller) is what makes two runs over the same events
	// byte-identical.
	sort.Slice(out.Counts, func(i, j int) bool {
		a, b := out.Counts[i], out.Counts[j]
		if (a.Count > 0) != (b.Count > 0) {
			return a.Count > 0
		}
		if a.Count != b.Count {
			return a.Count > b.Count
		}
		return a.Code < b.Code
	})
	return out, nil
}

// Render formats a window fold as one operator-readable line, worst count first,
// e.g.:
//
//	gpu event window (last 5m, n=4, codes=2): 79=3 48=1 63=0
//
// n is the number of samples inside the window and codes is how many distinct
// codes fired. Quiet codes (count 0) are shown last so a burst is the first thing
// read. An empty fold is reported explicitly rather than as a bare empty list.
func (w EventWindow) Render(label string) string {
	if w.ObservedEvents == 0 {
		return fmt.Sprintf("gpu event window (%s, n=0): no events sampled", label)
	}
	parts := make([]string, 0, len(w.Counts))
	for _, c := range w.Counts {
		parts = append(parts, fmt.Sprintf("%d=%d", c.Code, c.Count))
	}
	return fmt.Sprintf("gpu event window (%s, n=%d, codes=%d): %s",
		label, w.ObservedEvents, w.CodesInWindow, strings.Join(parts, " "))
}
