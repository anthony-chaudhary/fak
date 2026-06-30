package main

import (
	"io"
	"time"
)

// Focus-and-resize awareness for the live `fak info` overlay. An operator typically runs
// several terminal tabs, each with a harness (Claude) AND a fak info pane in it, and switches
// between them. The overlay's in-place redraw (writeGuardInfoFrame) measures the pane size
// ONCE at startup and then redraws with a relative-cursor delta forever, so switching to
// another tab, resizing the window, and switching back leaves the overlay drawing at a stale
// size — sparklines wrap and the pane scrolls. This layer teaches the loop two things:
//
//   - terminal FOCUS (xterm DECSET 1004): when enabled, a focus-reporting terminal sends
//     `ESC [ I` on focus-in and `ESC [ O` on focus-out. On the focus-in edge the loop
//     re-measures the pane and forces ONE clean repaint; while focused-out it THROTTLES (never
//     pauses) its tick cadence so a hidden tab stops churning.
//   - RESIZE (SIGWINCH on POSIX; a per-tick size poll on Windows): re-measure + clean repaint.
//
// Everything in this file is PURE (no I/O except the two tiny escape-byte writers, which take
// an io.Writer so the bytes are test-pinnable). The byte-stream parser, the focus-state
// reducer, and the interval policy are all unit-testable without a TTY — which is the bulk of
// the logic. The thin TTY plumbing (raw mode, the reader goroutine, the select cases) lives in
// runGuardInfoOverlay (info.go) and is gated to a visual-mode TTY with a TTY stdin, so the
// non-TTY / piped / --once / --style line paths are byte-for-byte unchanged.

// focusEvent is the decoded result of feeding one byte to a focusScanner. focusNone means the
// byte advanced (or did not advance) the state machine without completing a recognized event.
type focusEvent int

const (
	focusNone focusEvent = iota
	focusIn              // the terminal reported the pane/tab gained focus (ESC [ I)
	focusOut             // ... lost focus (ESC [ O)
	focusQuit            // a quit control byte (Ctrl-C / Ctrl-D / Ctrl-\) — see the raw-mode note
)

// focus scanner states. A focus report is `ESC [ I` / `ESC [ O`; any OTHER CSI sequence (a
// cursor-position report `ESC [ 12;34 R`, a mode reply, etc.) must be SWALLOWED so its final
// byte is never mistaken for a focus 'I'/'O'.
const (
	focusGround    = iota // no pending escape
	focusSawESC           // saw 0x1b
	focusSawCSI           // saw 0x1b '[' — the next byte decides
	focusCSIParams        // inside a CSI sequence that is NOT a focus report — swallow to its final byte
)

// focusScanner is a resumable byte-at-a-time state machine that recognizes terminal focus
// reports in a raw stdin stream. It is resumable because raw reads can split an escape
// sequence across two Read() calls — feeding `0x1b` then `[` then `I` across three steps must
// still decode one focusIn. Its only state is the current parser state (one int), so a value
// is cheap to copy and trivial to reset.
type focusScanner struct {
	state int
}

// step feeds one byte and returns the event it completes (focusNone for an in-progress or
// inert byte). The control-byte rule comes first: in raw mode term.MakeRaw disables ISIG, so
// Ctrl-C arrives as the byte 0x03 rather than as a SIGINT — the overlay's signal.NotifyContext
// will NOT fire, so the loop must treat 0x03 (and 0x04 Ctrl-D, 0x1c Ctrl-\) as an explicit
// quit. A lone 0x1b always (re)starts an escape, so a truncated/!aborted sequence followed by
// a fresh ESC re-syncs rather than wedging the parser.
func (s *focusScanner) step(b byte) focusEvent {
	switch b {
	case 0x03, 0x04, 0x1c: // Ctrl-C / Ctrl-D / Ctrl-\ — quit (raw mode swallowed the signal)
		s.state = focusGround
		return focusQuit
	case 0x1b: // ESC always (re)starts a sequence, from any state — re-sync point
		s.state = focusSawESC
		return focusNone
	}
	switch s.state {
	case focusSawESC:
		if b == '[' {
			s.state = focusSawCSI
		} else {
			s.state = focusGround // ESC + non-'[' is not a CSI we track
		}
		return focusNone
	case focusSawCSI:
		switch {
		case b == 'I':
			s.state = focusGround
			return focusIn
		case b == 'O':
			s.state = focusGround
			return focusOut
		case b >= 0x30 && b <= 0x3f: // CSI parameter byte (0-9 : ; < = > ?) — some OTHER sequence
			s.state = focusCSIParams
			return focusNone
		case b >= 0x40 && b <= 0x7e: // an immediate final byte that is not I/O — done, not focus
			s.state = focusGround
			return focusNone
		default: // intermediate byte (0x20-0x2f) — keep consuming the sequence
			s.state = focusCSIParams
			return focusNone
		}
	case focusCSIParams:
		// Swallow parameter + intermediate bytes until the final byte (0x40-0x7e) ends the
		// sequence. This is what stops `ESC [ 12;34 R` (a cursor report) from being read as a
		// focus event: its 'R' final byte lands here, not in focusSawCSI.
		if b >= 0x40 && b <= 0x7e {
			s.state = focusGround
		}
		return focusNone
	default: // focusGround and any unknown state: an ordinary byte is inert
		s.state = focusGround
		return focusNone
	}
}

// focusState is the loop's focus bookkeeping. focused tracks whether the pane currently holds
// terminal focus (assumed true at startup — a pane that opens focused, and the universal case
// where focus reporting never arrives at all, must run at full cadence). needsRepaint is a
// latch the loop consumes on its next frame to force ONE clean re-measure + redraw.
type focusState struct {
	focused      bool
	needsRepaint bool
}

// applyFocus folds one event into the state. A focus-in latches needsRepaint (the loop will
// re-measure and repaint cleanly on its next frame). A focus-out clears focused but MUST NOT
// clear a still-pending needsRepaint — otherwise a focus-out arriving between a focus-in and
// the loop consuming the latch would drop the repaint and leave a stale frame. focusQuit and
// focusNone do not change focus state (quit is handled by the loop tearing down).
func applyFocus(s focusState, ev focusEvent) focusState {
	switch ev {
	case focusIn:
		s.focused = true
		s.needsRepaint = true
	case focusOut:
		s.focused = false
	}
	return s
}

// backgroundInterval is the throttled tick cadence for a focused-OUT (hidden) pane: base*5,
// clamped to [10s, 30s]. It is deliberately FINITE — never a pause. The tmux --split info pane
// is left unfocused by design (guard_split.go) and tmux delivers focus per active pane, so a
// visible-but-"unfocused" pane is a real state; a hard pause would freeze it forever. A finite
// throttle keeps it alive, just cheaper.
func backgroundInterval(base time.Duration) time.Duration {
	bg := base * 5
	const lo, hi = 10 * time.Second, 30 * time.Second
	if bg < lo {
		bg = lo
	}
	if bg > hi {
		bg = hi
	}
	return bg
}

// effectiveInterval picks the tick cadence for the current focus state: the base (foreground)
// interval when focused, the throttled background interval when not. A base that already
// exceeds the background floor wins (we never speed a slow poll up just because the pane is
// hidden) — backgroundInterval's clamp handles that, but guarding here keeps the intent local.
func effectiveInterval(focused bool, base, background time.Duration) time.Duration {
	if focused {
		return base
	}
	if base > background {
		return base
	}
	return background
}

// The xterm focus-reporting (DECSET 1004) enable/disable sequences. Enabling makes a
// supporting terminal emit ESC [ I / ESC [ O on focus change; disabling on exit is mandatory
// so the terminal is not left reporting focus to whatever runs next in the pane.
const (
	focusEnable  = "\033[?1004h"
	focusDisable = "\033[?1004l"
)

// writeFocusEnable / writeFocusDisable emit the DECSET 1004 toggles. They take an io.Writer so
// the exact bytes are test-pinnable and so the loop can route them to the same stdout the
// frames use. Errors are ignored: a terminal that does not understand the sequence simply
// never sends focus events, which the loop already degrades to cleanly.
func writeFocusEnable(w io.Writer)  { _, _ = io.WriteString(w, focusEnable) }
func writeFocusDisable(w io.Writer) { _, _ = io.WriteString(w, focusDisable) }

// startGuardInfoFocusReader spawns the raw-stdin reader goroutine for the focus layer. It reads
// r one byte at a time, feeds each through a focusScanner, and forwards every completed event
// (focusIn / focusOut / focusQuit) on a buffered channel the overlay loop selects on. A
// focusQuit (a raw-mode quit byte such as Ctrl-C, which MakeRaw delivered as 0x03 instead of a
// signal) is forwarded AND triggers onQuit so the loop tears down even if the buffered channel
// is momentarily full. The goroutine exits when r returns an error/EOF (the gateway/session
// ended and the process is exiting). It is only started behind the overlay's focusable gate, so
// it never touches a non-TTY or piped stdin.
func startGuardInfoFocusReader(r io.Reader, onQuit func()) <-chan focusEvent {
	ch := make(chan focusEvent, 8)
	go func() {
		var sc focusScanner
		buf := make([]byte, 1)
		for {
			n, err := r.Read(buf)
			if n > 0 {
				if ev := sc.step(buf[0]); ev != focusNone {
					if ev == focusQuit && onQuit != nil {
						onQuit() // ensure the loop wakes even if ch is full
					}
					select {
					case ch <- ev:
					default: // a slow consumer must never block the reader; drop the surplus event
					}
				}
			}
			if err != nil {
				return
			}
		}
	}()
	return ch
}
