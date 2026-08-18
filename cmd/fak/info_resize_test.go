package main

import (
	"bytes"
	"errors"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Tests for the pane-geometry refresh the live `fak info` overlay depends on. The bug they pin:
// the interactive frame is padded to exactly the measured pane height and redrawn in place, so a
// measured height LARGER than the real pane overflows it — the pane scrolls and what is left on
// screen is the padded tail, i.e. blank rows. The overlay used to re-measure ONLY when a focus-in
// or a SIGWINCH latched a repaint, and on Windows neither can fire for a `fak guard --split` pane
// (no SIGWINCH; the split hands focus back to the agent pane), so one bad startup measure stood
// for the whole session and the 20% strip just looked empty.

// pollTestSize is a pinnable infoTermSize stand-in whose reported size a test can change between
// polls, plus a call counter so a test can witness that the loop looked at all.
type pollTestSize struct {
	w, h  atomic.Int64
	calls atomic.Int64
	fail  atomic.Bool
}

func (p *pollTestSize) get() (int, int, error) {
	p.calls.Add(1)
	if p.fail.Load() {
		return 0, 0, errors.New("no console")
	}
	return int(p.w.Load()), int(p.h.Load()), nil
}

func (p *pollTestSize) set(w, h int) {
	p.w.Store(int64(w))
	p.h.Store(int64(h))
}

// TestWatchInfoTermSizeSeedMismatchRepaints is the core regression: the watcher is seeded with the
// geometry the overlay is about to PAINT at, not with its own first sample. A --split pane whose
// real size never matches that startup measure — and never changes again afterwards — must still
// get a repaint, because "the size I am drawing at is wrong" is the failure, not "the size
// changed". A watcher baselined off its own first read would see a steady pane and stay silent
// forever, which is exactly the empty-looking strip.
func TestWatchInfoTermSizeSeedMismatchRepaints(t *testing.T) {
	size := &pollTestSize{}
	size.set(118, 10) // the REAL pane: a 20% strip
	tick := make(chan time.Time)
	done := make(chan struct{})
	defer close(done)

	// Seeded with the stale startup measure (a full-window height), which never changes again.
	out := watchInfoTermSize(118, 48, tick, size.get, done)

	tick <- time.Time{}
	select {
	case <-out:
	case <-time.After(2 * time.Second):
		t.Fatal("a pane whose real size differs from the seeded paint geometry must repaint on the first poll")
	}

	// Steady state after the correction: no further repaints for the same numbers.
	tick <- time.Time{}
	tick <- time.Time{}
	select {
	case <-out:
		t.Fatal("an unchanged pane must not keep forcing repaints")
	default:
	}
}

// TestWatchInfoTermSizeChangeAndFailure covers the two remaining arms: a real resize signals, and
// a failed/zero read is skipped rather than reported as a resize to 0 — a pane can report nothing
// mid-teardown, and 0 means "size unknown" downstream, which would blow the layout up.
func TestWatchInfoTermSizeChangeAndFailure(t *testing.T) {
	size := &pollTestSize{}
	size.set(118, 10)
	tick := make(chan time.Time)
	done := make(chan struct{})
	defer close(done)
	out := watchInfoTermSize(118, 10, tick, size.get, done)

	tick <- time.Time{} // same as the seed
	select {
	case <-out:
		t.Fatal("no resize yet: an unchanged pane must stay silent")
	default:
	}

	size.set(118, 22) // the operator dragged the split
	tick <- time.Time{}
	select {
	case <-out:
	case <-time.After(2 * time.Second):
		t.Fatal("a real resize must signal a repaint")
	}

	size.fail.Store(true)
	tick <- time.Time{}
	tick <- time.Time{}
	select {
	case <-out:
		t.Fatal("a failed size read must be skipped, never reported as a resize")
	default:
	}
}

// TestWatchInfoTermSizeCoalesces proves a drag-resize burst collapses to ONE pending repaint
// instead of queueing one per poll: the overlay only ever needs to redraw at the latest geometry.
func TestWatchInfoTermSizeCoalesces(t *testing.T) {
	size := &pollTestSize{}
	size.set(80, 10)
	tick := make(chan time.Time)
	done := make(chan struct{})
	defer close(done)
	out := watchInfoTermSize(80, 10, tick, size.get, done)

	for h := 11; h <= 20; h++ {
		size.set(80, h)
		tick <- time.Time{}
	}
	select {
	case <-out:
	case <-time.After(2 * time.Second):
		t.Fatal("a resize burst must leave a repaint pending")
	}
	select {
	case <-out:
		t.Fatal("a resize burst must coalesce to a single pending repaint")
	default:
	}
}

// TestNewInfoResizeChanStopIsIdempotent: the loop defers the stop func, and a panic path can reach
// it after an explicit call. Closing twice would panic on the done channel.
func TestNewInfoResizeChanStopIsIdempotent(t *testing.T) {
	_, stop := newInfoResizeChan(80, 24)
	stop()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("second stop panicked: %v", r)
		}
	}()
	stop()
}

// TestRunInfoOverlayRemeasuresEveryFrame is the end-to-end witness for the fix. It runs the real
// overlay with a pinned pane size that is SHORTER than the height the overlay was started with —
// the shape of a pane measured before its host finished laying out the split, or resized on a
// platform with no resize signal. Every painted frame must be sized to the live measurement, not
// to the stale startup height: before the fix nothing re-measured unless a focus-in or SIGWINCH
// latched a repaint, so the frame stayed at the startup height and overflowed the pane.
func TestRunInfoOverlayRemeasuresEveryFrame(t *testing.T) {
	const realHeight = 8
	size := &pollTestSize{}
	size.set(80, realHeight)
	restore := infoTermSize
	infoTermSize = size.get
	t.Cleanup(func() { infoTermSize = restore })

	c := healthyThenGoneClient(t, 2)
	var stdout, stderr bytes.Buffer
	// Start at a height four times the real pane — the stale startup measure.
	code := runGuardInfoOverlay(&stdout, &stderr, c, time.Millisecond, false, true /*tty*/, 80, realHeight*4, "visual", "never")
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	if size.calls.Load() == 0 {
		t.Fatal("the overlay never measured the pane: a stale startup height can survive the whole session")
	}
	for _, frame := range paintedInfoFrames(stdout.String()) {
		if rows := strings.Count(frame, "\n") + 1; rows > realHeight {
			t.Fatalf("painted %d rows into a %d-row pane — the frame overflows and the pane shows its blank tail:\n%q",
				rows, realHeight, frame)
		}
	}
}

// infoCursorUpSuffix matches the cursor-up (ESC[<n>A) that opens the NEXT repaint, so it can be
// trimmed off the end of the frame it follows.
var infoCursorUpSuffix = regexp.MustCompile(`\x1b\[\d*A$`)

// paintedInfoFrames splits the overlay's captured stdout into the blocks it actually drew.
// writeGuardInfoFrame emits ESC[<n>A then "\r" ESC[J before every repaint after the first, so
// "\r\033[J" is the frame boundary; the one-time intro line and the closing note are not frames.
func paintedInfoFrames(out string) []string {
	if i := strings.Index(out, "fak info: gateway closed"); i >= 0 {
		out = out[:i]
	}
	if i := strings.IndexByte(out, '\n'); i >= 0 { // drop the one-time intro line
		out = out[i+1:]
	}
	var frames []string
	for _, chunk := range strings.Split(out, "\r\033[J") {
		chunk = infoCursorUpSuffix.ReplaceAllString(strings.TrimRight(chunk, "\r\n"), "")
		if strings.TrimSpace(chunk) != "" {
			frames = append(frames, chunk)
		}
	}
	return frames
}
