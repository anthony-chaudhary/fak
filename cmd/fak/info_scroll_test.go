package main

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

// TestFitGuardInfoStatus pins the width cap that stops the live `fak info` status line from
// wrapping a narrow pane: width <= 0 leaves the line whole (no known pane size), a wide pane
// leaves a short line untouched, and a narrow pane trims the line so the indent + text never
// exceed the pane width (the wrap is the scroll corruptor).
func TestFitGuardInfoStatus(t *testing.T) {
	long := "cache: saving money — reused 95% of text, ×2.50 cheaper, +12,345 tokens · safety: blocked 3, fixed 1 · replies 42 · busy with 1 · running 3h"

	// width 0 -> no cap: the whole line survives behind the two-space indent.
	if got := fitGuardInfoStatus(long, 0); got != "  "+long {
		t.Errorf("width 0 must not cap; got %q", got)
	}
	// A wide pane leaves a line that already fits untouched.
	if got := fitGuardInfoStatus("cache: nothing yet · safety: nothing blocked", 200); got != "  cache: nothing yet · safety: nothing blocked" {
		t.Errorf("wide pane must not alter a fitting line; got %q", got)
	}
	// A narrow pane caps the rendered width to the pane (indent included).
	for _, w := range []int{20, 30, 40} {
		got := fitGuardInfoStatus(long, w)
		if !strings.HasPrefix(got, "  ") {
			t.Errorf("width %d: status must keep the two-space indent; got %q", w, got)
		}
		if dw := dispWidthTUI(got); dw > w {
			t.Errorf("width %d: capped status is %d cells wide, must be <= %d: %q", w, dw, w, got)
		}
	}
}

// TestGuardInfoStartupHeaderNarrowVsWide pins the legend sizing: a narrow pane gets the single
// compact legend line (no 4-line block to wrap), a wide/unknown pane keeps the full legend.
func TestGuardInfoStartupHeaderNarrowVsWide(t *testing.T) {
	narrow := guardInfoStartupHeader("http://127.0.0.1:8080", 2*time.Second, 40)
	if strings.Contains(narrow, "fak re-uses text it already sent") {
		t.Errorf("narrow pane must not print the verbose guide block:\n%s", narrow)
	}
	if !strings.Contains(narrow, "what this means: cache") {
		t.Errorf("narrow pane must print the compact one-line guide:\n%s", narrow)
	}
	if n := strings.Count(narrow, "\n"); n > 2 {
		t.Errorf("narrow header should be header + one compact guide line (<=2 newlines), got %d:\n%s", n, narrow)
	}
	for _, w := range []int{0, 120} {
		full := guardInfoStartupHeader("http://127.0.0.1:8080", 2*time.Second, w)
		if !strings.Contains(full, "fak re-uses text it already sent") {
			t.Errorf("width %d must keep the full guide:\n%s", w, full)
		}
	}
}

// TestRunInfoOverlayNarrowTTYNeverWraps proves the end-to-end scroll fix: in a narrow TTY pane
// every redrawn status row stays within the pane width, so a tick can never wrap onto a second
// row that the next \r cannot clear.
func TestRunInfoOverlayNarrowTTYNeverWraps(t *testing.T) {
	const width = 30
	c := healthyThenGoneClient(t, 2)
	var stdout, stderr bytes.Buffer
	code := runGuardInfoOverlay(&stdout, &stderr, c, time.Millisecond, false /*once*/, true /*tty*/, width)
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	// Each in-place redraw is "\r\033[K<status>"; the status text runs until the next \r or \n.
	segments := strings.Split(out, "\r\033[K")
	if len(segments) < 2 {
		t.Fatalf("expected at least one in-place redraw, got:\n%q", out)
	}
	for _, seg := range segments[1:] {
		status := seg
		if i := strings.IndexAny(status, "\r\n"); i >= 0 {
			status = status[:i]
		}
		if dw := dispWidthTUI(status); dw > width {
			t.Errorf("redrawn status row is %d cells wide, must be <= %d: %q", dw, width, status)
		}
	}
}
