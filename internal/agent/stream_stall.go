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
//
// That byte deadline alone answers "is the socket alive?", NOT "is the turn advancing?"
// (#5486). On the Anthropic wire a keepalive `ping` is bytes like any other, so an upstream
// that wedged the generation but still emits keepalives re-arms the idle window forever and
// is indistinguishable from a healthy stream — which is exactly the transient-overload shape
// the deadline was written to catch, and it blocks for the whole 600s ceiling instead. So
// stallReader carries TWO deadlines answering two different questions:
//
//   - LIVENESS (bytes) — the idle timer above. `ping` re-arms it, and should.
//   - PROGRESS (content) — a second, longer timer (the planner's StreamProgressTimeout config
//     field, resolved by streamProgressWindow) armed when the body is wrapped and re-armed
//     ONLY through noteProgress, which the SSE decode loops call for a
//     frame that advances the TURN (content_block_start/delta, message_delta, a finish reason,
//     a tool-call fragment) and deliberately NOT for `ping`/keepalive frames.
//
// Whichever fires first closes the body the same way; stallCause names which one, so the
// UpstreamStalledError the gateway logs says whether the socket died or the turn wedged.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// stallKindIdle, stallKindNoProgress, and stallKindMaxDuration name WHICH deadline tripped
// a stallReader, carried on UpstreamStalledError so the operator log separates "the upstream
// went silent on the wire" from "the upstream kept the socket warm with keepalives but never
// advanced the turn" from "the stream outlived its absolute max-duration budget". All three
// are stalls to the gateway (a 504 upstream_stalled); only the cause differs.
const (
	stallKindIdle          = "idle"
	stallKindNoProgress    = "no-progress"
	stallKindMaxDuration   = "max-duration"
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
//
// Kind names WHICH deadline elapsed — stallKindIdle (no bytes at all) or stallKindNoProgress
// (keepalives kept arriving but no frame advanced the turn, #5486). The zero value reads as
// the idle case, so an existing keyed literal keeps its old meaning.
type UpstreamStalledError struct {
	Idle time.Duration
	Kind string
	Err  error
}

// Error formats the elapsed window, naming the no-progress and max-duration cases distinctly
// so an operator reading the log is not told "silent" about an upstream that was in fact
// still pinging, or "idle" about a stream that was ended by its absolute budget.
func (e *UpstreamStalledError) Error() string {
	switch e.Kind {
	case stallKindNoProgress:
		return fmt.Sprintf("planner: upstream stalled after %s with no content progress (keepalives only)", e.Idle)
	case stallKindMaxDuration:
		return fmt.Sprintf("planner: upstream stream ended by max-duration bound after %s", e.Idle)
	default:
		return fmt.Sprintf("planner: upstream stalled after %s idle", e.Idle)
	}
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
//
// The reader carries up to THREE deadlines answering different questions:
//   - LIVENESS (bytes) — the idle timer above. `ping` re-arms it, and should.
//   - PROGRESS (content) — a second, longer timer (the planner's StreamProgressTimeout config
//     field, resolved by streamProgressWindow) armed when the body is wrapped and re-armed
//     ONLY through noteProgress, which the SSE decode loops call for a
//     frame that advances the TURN (content_block_start/delta, message_delta, a finish reason,
//     a tool-call fragment) and deliberately NOT for `ping`/keepalive frames.
//   - MAX-DURATION (absolute budget) — a third timer armed once when the stream opens, never
//     re-armed by bytes or progress, that ends a stream outliving its configured total budget.
//     Default OFF (zero window). When armed, it fires exactly once at the configured deadline.
type stallReader struct {
	rc     io.ReadCloser
	window time.Duration
	timer  *time.Timer

	// progressWindow/progressTimer are the SECOND deadline (#5486): it runs continuously
	// from the moment the body is wrapped and is reset only by noteProgress, never by a
	// mere Read. A keepalive-only upstream therefore trips it even though every byte it
	// sends keeps re-arming the idle timer above.
	progressWindow time.Duration
	progressTimer  *time.Timer

	// maxDurationWindow/maxDurationTimer are the THIRD deadline (#10672): armed once at
	// stream open with the absolute total-duration budget (FAK_STREAM_MAX_DURATION_S).
	// Never re-armed by bytes or progress — it is the hard ceiling a HEALTHY stream cannot
	// outlive. Zero means OFF (the default, preserving all existing behavior).
	maxDurationWindow time.Duration
	maxDurationTimer  *time.Timer

	mu      sync.Mutex
	tripped bool   // a timer fired and closed rc — the next Read error is a stall
	kind    string // which deadline fired: stallKindIdle, stallKindNoProgress, or stallKindMaxDuration
	closed  bool   // rc has been closed (by Close or a timer) — close exactly once
}

// newStallReader wraps rc with an idle (inter-byte) deadline of window, a content
// PROGRESS deadline of progressWindow, and an absolute MAX-DURATION deadline of
// maxDurationWindow. A non-positive window disables the corresponding deadline (that
// half of the reader is a transparent pass-through), so a caller can opt out without a
// branch at the call site. A progressWindow shorter than window is raised to window: the
// progress deadline is by construction the outer, more forgiving one, and letting it
// undercut the byte deadline would only mislabel a plain dead socket.
func newStallReader(rc io.ReadCloser, window, progressWindow, maxDurationWindow time.Duration) *stallReader {
	if progressWindow > 0 && progressWindow < window {
		progressWindow = window
	}
	s := &stallReader{rc: rc, window: window, progressWindow: progressWindow, maxDurationWindow: maxDurationWindow}
	if window > 0 {
		// Create the timer already stopped; Read arms it per call. AfterFunc has no
		// channel to drain, so Stop/Reset alone manage its lifecycle cleanly.
		s.timer = time.AfterFunc(window, func() { s.trip(stallKindIdle) })
		s.timer.Stop()
	}
	if progressWindow > 0 {
		// Armed immediately and left running: the turn has made no progress yet, and the
		// clock on "no content since the stream opened" starts when the stream opens.
		s.progressTimer = time.AfterFunc(progressWindow, func() { s.trip(stallKindNoProgress) })
	}
	if maxDurationWindow > 0 {
		// Armed once at stream open, never re-armed. This is the absolute budget ceiling.
		s.maxDurationTimer = time.AfterFunc(maxDurationWindow, func() { s.trip(stallKindMaxDuration) })
	}
	return s
}

// trip is the timer callback for all three deadlines: the named window elapsed, so either
// the upstream is silent (idle), it is warm but not advancing the turn (no-progress), or
// the stream outlived its absolute max-duration budget (max-duration). Close the body to
// unblock the parked Read and record which deadline fired. Whichever timer arrives first
// wins; the others find the reader already closed and return.
func (s *stallReader) trip(kind string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.tripped = true
	s.kind = kind
	s.closed = true
	_ = s.rc.Close()
}

// noteProgress records that the stream advanced the TURN and re-arms the progress deadline.
// Callers MUST NOT call it for a `ping`/keepalive frame: re-arming on a frame that carries
// no content is precisely the defect this deadline closes (#5486) — an upstream warm enough
// to ping while producing nothing is the stalled-but-warm case worth catching. Safe to call
// from the decode loop while a Read is parked; a no-op once the reader is closed.
func (s *stallReader) noteProgress() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.progressTimer == nil || s.closed {
		return
	}
	s.progressTimer.Reset(s.progressWindow)
}

// stallCause names the deadline that tripped this reader and the window it enforced, so the
// caller can build an UpstreamStalledError that reports the real cause and a TRUTHFUL
// elapsed window rather than assuming the idle one. A reader that has not tripped reports
// the idle deadline, the one that would fire next.
func (s *stallReader) stallCause() (string, time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch s.kind {
	case stallKindNoProgress:
		return stallKindNoProgress, s.progressWindow
	case stallKindMaxDuration:
		return stallKindMaxDuration, s.maxDurationWindow
	default:
		return stallKindIdle, s.window
	}
}

// Read arms the idle timer for window, performs one underlying Read, then stops the timer.
// The mutex is never held across the blocking rc.Read — only around the small tripped/closed
// bookkeeping — so the timer callback can fire and close the body WHILE this Read is parked,
// which is exactly how the parked Read gets unblocked. A post-trip error is mapped to
// ErrUpstreamStalled; any other error (including io.EOF and a context cancel) is returned
// unchanged.
// Note: the max-duration timer is armed ONCE at stream open (in newStallReader) and is
// never re-armed by Read — it is the absolute budget ceiling.
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

// Close stops all deadlines and closes the wrapped body, idempotently — an explicit defer
// and any timer callback may reach it, so the underlying body is closed exactly once.
func (s *stallReader) Close() error {
	if s.timer != nil {
		s.timer.Stop()
	}
	if s.progressTimer != nil {
		s.progressTimer.Stop()
	}
	if s.maxDurationTimer != nil {
		s.maxDurationTimer.Stop()
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
	return envClampedTimeout("FAK_STREAM_STALL_TIMEOUT_S", 60*time.Second, 5, 600)
}

// streamMaxDuration is the absolute total-duration deadline for a streamed turn.
// Default OFF (0 = no deadline). When set via FAK_STREAM_MAX_DURATION_S, it is clamped
// to a sane [5s, 3600s] band (1 hour ceiling — a window longer than the whole-request
// timeout could never fire, and 600s is the current whole-request floor). A stream that
// outlives this budget is ended with UpstreamStalledError{Kind: "max-duration"} so the
// gateway maps, logs, and receipts it identically to the other stall causes.
func streamMaxDuration() time.Duration {
	return envClampedTimeout("FAK_STREAM_MAX_DURATION_S", 0, 5, 3600)
}

// DefaultStreamProgressTimeout is the CONTENT-progress deadline a planner uses when its
// HTTPPlanner.StreamProgressTimeout config field is left at zero. It is the SINGLE SOURCE OF
// TRUTH for that default — the DefaultCtxViewBudget idiom — so a front door that grows a
// flag for the window defaults it to this constant rather than a bare literal and the two
// can never drift. 300s sits well above the worst prefill-to-first-token gap on a large
// cached prompt and above any extended-thinking pause (thinking streams content_block_deltas,
// which count as progress), yet under the 600s whole-request ceiling `fak guard` sets — a
// window past that ceiling could never fire.
//
// This deliberately is NOT an environment read. A behavioral deadline is configuration, not a
// credential, so it lives on the config surface (internal/envconfiglint's CONFIG_NOT_ENV
// rule); the environment is for declared secrets.
const DefaultStreamProgressTimeout = 300 * time.Second

// streamProgressMinWindow / streamProgressMaxWindow bound a configured content-progress
// deadline. Below the floor the window would undercut the idle deadline and mislabel a plain
// dead socket; above the ceiling it outlasts the 600s whole-request timeout and could never
// fire. A configured value outside the band is not honored — the default is used instead.
const (
	streamProgressMinWindow = 5 * time.Second
	streamProgressMaxWindow = 600 * time.Second
)

// streamProgressWindow resolves the planner's configured content-progress deadline into the
// window newStallReader arms. Zero (unconfigured) is DefaultStreamProgressTimeout; a NEGATIVE
// value is the explicit off switch and disables the deadline; an out-of-band positive value
// falls back to the default rather than being clamped, so a typo'd window never silently
// becomes a different real deadline.
func (p *HTTPPlanner) streamProgressWindow() time.Duration {
	switch {
	case p.StreamProgressTimeout < 0:
		return 0
	case p.StreamProgressTimeout >= streamProgressMinWindow && p.StreamProgressTimeout <= streamProgressMaxWindow:
		return p.StreamProgressTimeout
	default:
		return DefaultStreamProgressTimeout
	}
}

// anthropicFrameAdvancesTurn reports whether an Anthropic SSE frame moved the TURN forward
// (message_start, content_block_start/delta/stop, message_delta/stop, error) rather than
// merely proving the socket is warm. Only `ping` is excluded — it is the wire's keepalive
// and carries nothing about the generation, so counting it as progress would re-arm the
// deadline that exists to catch a pinging-but-wedged upstream (#5486). Frames that omit the
// `event:` line fall back to the payload's own "type", which is where a bare `data:` ping
// still names itself; an undecodable payload counts as progress (it is not a keepalive, and
// the byte deadline still covers the socket).
func anthropicFrameAdvancesTurn(ev AnthropicSSEEvent) bool {
	name := strings.TrimSpace(ev.Event)
	if name == "" {
		var probe struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(ev.Data, &probe) == nil {
			name = strings.TrimSpace(probe.Type)
		}
	}
	return name != "ping"
}

// openAIChunkAdvancesTurn is the OpenAI-wire twin: a decoded stream chunk advances the turn
// when it carries assistant content, reasoning content, a tool-call fragment, or a finish
// reason. A role-only opener, an empty-delta keepalive chunk, or a usage-only trailer does
// not — nor do the SSE comment / non-JSON heartbeat lines the decode loop drops before ever
// reaching here, which is the OpenAI-shaped upstream's equivalent of a `ping`.
func openAIChunkAdvancesTurn(c openAIStreamChunk) bool {
	for _, ch := range c.Choices {
		d := ch.Delta
		if ch.FinishReason != "" || d.Content != "" || d.ReasoningContent != "" || len(d.ToolCalls) > 0 {
			return true
		}
	}
	return false
}

// envClampedTimeout reads a whole-second duration from env key, falling back to def when
// unset/unparseable, and accepting the override only when it lands in [minS, maxS] seconds.
// Shared by the planner whole-request timeout and the stream idle-read deadline.
func envClampedTimeout(key string, def time.Duration, minS, maxS int) time.Duration {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= minS && n <= maxS {
			return time.Duration(n) * time.Second
		}
	}
	return def
}
