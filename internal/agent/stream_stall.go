package agent

// stream_stall.go adds an IDLE-READ deadline to the streaming upstream paths
// (StreamAnthropicRaw, CompleteStream). The planner's *http.Client carries only a
// whole-request Timeout (plannerTimeout, raised to 600s by `fak guard` so a long but
// HEALTHY extended-thinking turn is not cut off mid-stream). That timeout cannot tell a
// long-but-progressing turn apart from one where the upstream API STALLED mid-stream —
// it sent headers and some frames, then went silent (a transient overload / "API
// issue"). With a 600s whole-request ceiling, that stall blocks the SSE scanner on
// resp.Body.Read for the full ten minutes, which the operator experiences as a hang.
//
// The fix is the shape the repo's own boundarylint rule documents for streamed bodies:
// an inter-byte (idle) deadline rather than a whole-request one. stallReader wraps the
// response body and arms a one-shot timer around each Read; any Read that returns bytes
// re-arms it, so a stream emitting steady deltas / `ping` events / SSE keepalives never
// trips. Only a window of true silence fires the timer, which closes the body to unblock
// the read, and the Read is then reported as ErrUpstreamStalled — distinct from a normal
// EOF or a client cancel so the gateway can log the cause and the test can assert it.

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"sync"
	"time"
)

// ErrUpstreamStalled is the sentinel a streaming read returns when the upstream produced
// no bytes for a full idle window — the upstream went silent mid-stream. It is wrapped by
// UpstreamStalledError (which carries the window) so callers can match either form with
// errors.Is / errors.As. It is deliberately distinct from io.EOF (a clean close) and from
// a client context cancel, so a stall is never misreported as a normal end-of-stream.
var ErrUpstreamStalled = errors.New("agent: upstream stream stalled (no bytes within idle window)")

// UpstreamStalledError is returned by the streaming planner paths (CompleteStream,
// StreamAnthropicRaw) when the upstream SSE stream STALLED — it opened (headers + maybe
// some frames) but then emitted nothing for a full idle window. Unlike
// UpstreamUnreachableError (the upstream was never reached) or UpstreamStatusError (it
// answered with a non-200), this fires AFTER a healthy start, so the gateway has usually
// already begun streaming to the client; the gateway maps it to a terminal SSE error
// frame the same way it does any mid-stream upstream error. Idle carries the window that
// elapsed for the OPERATOR LOG; Err is the underlying ErrUpstreamStalled for errors.Is.
type UpstreamStalledError struct {
	Idle time.Duration
	Err  error
}

// Error formats the elapsed idle window as "planner: upstream stalled after <idle> idle".
func (e *UpstreamStalledError) Error() string {
	return fmt.Sprintf("planner: upstream stalled after %s idle", e.Idle)
}

// Unwrap returns the underlying ErrUpstreamStalled sentinel for errors.Is/As.
func (e *UpstreamStalledError) Unwrap() error { return e.Err }

// stallReader wraps a streaming response body with an inter-byte (idle) deadline. A
// single time.AfterFunc timer is re-armed before each Read and stopped after it returns;
// a Read that delivers bytes therefore resets the window, while a Read that blocks longer
// than the window lets the timer fire. The timer callback closes the wrapped body (the
// only way to unblock a Read parked in the transport) and records that the close was a
// stall, so the now-returning Read can be reported as ErrUpstreamStalled rather than the
// raw "use of closed connection" the close produces. A normal io.EOF or a client cancel
// (which closes the body from elsewhere with tripped still false) passes through verbatim.
type stallReader struct {
	rc     io.ReadCloser
	window time.Duration
	timer  *time.Timer

	mu      sync.Mutex
	tripped bool // the timer fired and closed rc — the next Read error is a stall
	closed  bool // rc has been closed (by Close or the timer) — close exactly once
}

// newStallReader wraps rc with an idle deadline of window. A non-positive window disables
// the deadline (the reader is a transparent pass-through), so a caller can opt out without
// a branch at the call site.
func newStallReader(rc io.ReadCloser, window time.Duration) *stallReader {
	s := &stallReader{rc: rc, window: window}
	if window > 0 {
		// Create the timer already stopped; Read arms it per call. AfterFunc has no
		// channel to drain, so Stop/Reset alone manage its lifecycle cleanly.
		s.timer = time.AfterFunc(window, s.trip)
		s.timer.Stop()
	}
	return s
}

// trip is the timer callback: the window elapsed with no completed Read, so the upstream
// is silent. Close the body to unblock the parked Read and mark the close as a stall.
func (s *stallReader) trip() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.tripped = true
	s.closed = true
	_ = s.rc.Close()
}

// Read arms the idle timer for window, performs one underlying Read, then stops the timer.
// The mutex is never held across the blocking rc.Read — only around the small tripped/closed
// bookkeeping — so the timer callback can fire and close the body WHILE this Read is parked,
// which is exactly how the parked Read gets unblocked. A post-trip error is mapped to
// ErrUpstreamStalled; any other error (including io.EOF and a context cancel) is returned
// unchanged.
func (s *stallReader) Read(p []byte) (int, error) {
	if s.timer != nil {
		s.timer.Reset(s.window)
	}
	n, err := s.rc.Read(p)
	if s.timer != nil {
		s.timer.Stop()
	}
	if err != nil {
		s.mu.Lock()
		tripped := s.tripped
		s.mu.Unlock()
		if tripped {
			return n, ErrUpstreamStalled
		}
	}
	return n, err
}

// Close stops the idle timer and closes the wrapped body, idempotently — both an explicit
// defer and the timer callback may reach it, so the underlying body is closed exactly once.
func (s *stallReader) Close() error {
	if s.timer != nil {
		s.timer.Stop()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	return s.rc.Close()
}

// streamStallTimeout is the inter-byte idle deadline applied to a streamed upstream read,
// 60s unless FAK_STREAM_STALL_TIMEOUT_S overrides it (clamped to a sane [5s, 600s] band).
// It mirrors plannerTimeout's shape. 60s sits comfortably above Anthropic's few-second
// ping/keepalive cadence and the prefill-to-first-token gap on a large cached prompt, yet
// an order of magnitude under the 600s whole-request floor `fak guard` sets — so a true
// stall fails in ≤60s instead of hanging for ten minutes, while a healthy stream that is
// still emitting is never tripped. The ceiling is 600s because a window longer than the
// whole-request timeout could never fire.
func streamStallTimeout() time.Duration {
	d := 60 * time.Second
	if v := os.Getenv("FAK_STREAM_STALL_TIMEOUT_S"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 5 && n <= 600 {
			d = time.Duration(n) * time.Second
		}
	}
	return d
}
