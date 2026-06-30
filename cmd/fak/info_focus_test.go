package main

import (
	"bytes"
	"testing"
	"time"
)

// feedFocusScanner runs a byte slice through one focusScanner (as a raw stream would arrive)
// and returns every non-none event it completed, in order — the resumable behavior under test.
func feedFocusScanner(s *focusScanner, b []byte) []focusEvent {
	var got []focusEvent
	for _, c := range b {
		if ev := s.step(c); ev != focusNone {
			got = append(got, ev)
		}
	}
	return got
}

func eventsEqual(a, b []focusEvent) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestFocusScannerSequences pins the byte-stream parser: clean focus reports, reports split
// across reads, back-to-back reports, an interleaved CSI sequence that is NOT focus (a cursor
// position report) which must yield NO focus event, a raw-mode quit byte, and ESC re-sync.
func TestFocusScannerSequences(t *testing.T) {
	cases := []struct {
		name string
		in   []byte
		want []focusEvent
	}{
		{"clean focus-in", []byte("\x1b[I"), []focusEvent{focusIn}},
		{"clean focus-out", []byte("\x1b[O"), []focusEvent{focusOut}},
		{"back to back in then out", []byte("\x1b[I\x1b[O"), []focusEvent{focusIn, focusOut}},
		// A cursor-position report ESC [ 12;7 R — its params route through focusCSIParams and the
		// 'R' final byte must NOT be read as a focus event.
		{"cursor report is not focus", []byte("\x1b[12;7R"), nil},
		// A focus-in immediately AFTER a swallowed cursor report still decodes.
		{"focus after cursor report", []byte("\x1b[12;7R\x1b[I"), []focusEvent{focusIn}},
		// Ordinary typed text never produces a focus event.
		{"plain text inert", []byte("hello world"), nil},
		// A DECSET reply ESC [ ? 1004 ; ... $ y style sequence — params + intermediate, no focus.
		{"mode reply is not focus", []byte("\x1b[?1004;1$y"), nil},
		{"ctrl-c is quit", []byte{0x03}, []focusEvent{focusQuit}},
		{"ctrl-d is quit", []byte{0x04}, []focusEvent{focusQuit}},
		// A bare ESC then a fresh ESC re-syncs: the second ESC restarts, so [I decodes.
		{"double ESC re-syncs", []byte("\x1b\x1b[I"), []focusEvent{focusIn}},
		// An ESC abandoned by ordinary text, then a clean report later.
		{"abandoned ESC then focus", []byte("\x1bX\x1b[O"), []focusEvent{focusOut}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var s focusScanner
			got := feedFocusScanner(&s, tc.in)
			if !eventsEqual(got, tc.want) {
				t.Fatalf("events = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestFocusScannerSplitAcrossReads proves the scanner is resumable: an ESC [ I delivered one
// byte per Read still decodes a single focusIn, because state persists across step calls.
func TestFocusScannerSplitAcrossReads(t *testing.T) {
	var s focusScanner
	if ev := s.step(0x1b); ev != focusNone {
		t.Fatalf("after ESC: %v, want none", ev)
	}
	if ev := s.step('['); ev != focusNone {
		t.Fatalf("after [: %v, want none", ev)
	}
	if ev := s.step('I'); ev != focusIn {
		t.Fatalf("after I: %v, want focusIn", ev)
	}
}

// TestApplyFocus pins the reducer: focus-in latches focused + needsRepaint; focus-out clears
// focused but MUST NOT clear a still-pending needsRepaint (else a focus-out between a focus-in
// and the loop consuming the latch would drop the repaint).
func TestApplyFocus(t *testing.T) {
	in := applyFocus(focusState{}, focusIn)
	if !in.focused || !in.needsRepaint {
		t.Fatalf("focusIn => %+v, want focused+needsRepaint", in)
	}
	out := applyFocus(in, focusOut)
	if out.focused {
		t.Fatalf("focusOut must clear focused: %+v", out)
	}
	if !out.needsRepaint {
		t.Fatalf("focusOut must NOT clear a pending needsRepaint: %+v", out)
	}
	// quit / none leave focus state untouched.
	if got := applyFocus(focusState{focused: true}, focusQuit); !got.focused {
		t.Fatalf("focusQuit must not change focused: %+v", got)
	}
	if got := applyFocus(focusState{focused: true}, focusNone); !got.focused {
		t.Fatalf("focusNone must not change focused: %+v", got)
	}
}

// TestBackgroundInterval pins the clamp: base*5 bounded to [10s, 30s], never a pause.
func TestBackgroundInterval(t *testing.T) {
	cases := []struct {
		base time.Duration
		want time.Duration
	}{
		{2 * time.Second, 10 * time.Second},  // 2*5=10 -> floor
		{1 * time.Second, 10 * time.Second},  // 1*5=5  -> clamped up to floor
		{4 * time.Second, 20 * time.Second},  // 4*5=20 -> within band
		{10 * time.Second, 30 * time.Second}, // 10*5=50 -> clamped to ceiling
	}
	for _, tc := range cases {
		if got := backgroundInterval(tc.base); got != tc.want {
			t.Fatalf("backgroundInterval(%v) = %v, want %v", tc.base, got, tc.want)
		}
	}
}

// TestEffectiveInterval: focused -> base; unfocused -> the background interval; a base already
// slower than the background floor wins (we never speed a slow poll up because the pane hid).
func TestEffectiveInterval(t *testing.T) {
	base, bg := 2*time.Second, 10*time.Second
	if got := effectiveInterval(true, base, bg); got != base {
		t.Fatalf("focused => %v, want %v", got, base)
	}
	if got := effectiveInterval(false, base, bg); got != bg {
		t.Fatalf("unfocused => %v, want %v", got, bg)
	}
	slow := 60 * time.Second
	if got := effectiveInterval(false, slow, bg); got != slow {
		t.Fatalf("unfocused with slow base => %v, want %v (base wins)", got, slow)
	}
}

// TestWriteFocusBytes pins the exact DECSET 1004 toggle bytes so a regression in the enable/
// disable strings (which would leave a terminal stuck reporting focus) is caught.
func TestWriteFocusBytes(t *testing.T) {
	var on, off bytes.Buffer
	writeFocusEnable(&on)
	writeFocusDisable(&off)
	if on.String() != "\033[?1004h" {
		t.Fatalf("enable = %q, want ESC[?1004h", on.String())
	}
	if off.String() != "\033[?1004l" {
		t.Fatalf("disable = %q, want ESC[?1004l", off.String())
	}
}

// TestStartGuardInfoFocusReader proves the reader goroutine decodes a focus stream off a plain
// io.Reader: it forwards the events in order and fires onQuit when a quit byte arrives.
func TestStartGuardInfoFocusReader(t *testing.T) {
	r := bytes.NewReader([]byte("\x1b[I\x1b[O\x03"))
	quit := make(chan struct{}, 1)
	ch := startGuardInfoFocusReader(r, func() {
		select {
		case quit <- struct{}{}:
		default:
		}
	})
	want := []focusEvent{focusIn, focusOut, focusQuit}
	for i, w := range want {
		select {
		case got := <-ch:
			if got != w {
				t.Fatalf("event %d = %v, want %v", i, got, w)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for event %d (%v)", i, w)
		}
	}
	select {
	case <-quit:
	case <-time.After(2 * time.Second):
		t.Fatal("onQuit was not called for the quit byte")
	}
}

// TestRunInfoOverlayNonTTYEmitsNoFocusBytes is the gate proof: a non-TTY run (a bytes.Buffer
// stdout, the way every overlay test drives the loop) must never enable focus reporting — no
// DECSET 1004 byte may appear — so the piped/redirected/append path is byte-for-byte unchanged.
func TestRunInfoOverlayNonTTYEmitsNoFocusBytes(t *testing.T) {
	c := healthyThenGoneClient(t, 2)
	var stdout, stderr bytes.Buffer
	code := runGuardInfoOverlay(&stdout, &stderr, c, time.Millisecond, false /*once*/, false /*tty*/, 0, 0, "visual")
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	if bytes.Contains(stdout.Bytes(), []byte("\033[?1004")) {
		t.Fatalf("non-TTY run must not emit DECSET 1004 focus-reporting bytes:\n%q", stdout.String())
	}
}
