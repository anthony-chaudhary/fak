package gateway

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"syscall"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// timeoutNetErr is a net.Error whose Timeout() reports true — a portable stand-in for an I/O
// timeout without depending on a real dial. Used to prove the transient-transport classifier
// catches a timeout the same way it catches a reset.
type timeoutNetErr struct{}

func (timeoutNetErr) Error() string   { return "i/o timeout" }
func (timeoutNetErr) Timeout() bool   { return true }
func (timeoutNetErr) Temporary() bool { return true }

// TestTransientTransportError_ClassifiesOnlyTheTransientFamily is the closing witness for
// #3514's verified HOLE #1: a wire crash reaches the upstream-error fold as a bare net error
// (rs.exhausted wraps it with %w) and MUST be told apart from a deterministic misconfiguration
// and from a plain non-transport error. The classifier is conservative — it matches only the
// transient wire family (reset, broken pipe, truncated read, I/O timeout, a non-dial op error)
// and returns false for everything else — so a systematic failure is never dressed up as a
// recoverable blip.
func TestTransientTransportError_ClassifiesOnlyTheTransientFamily(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		// The transient wire family — each worth one bounded supervisor relaunch.
		{"reset", &net.OpError{Op: "read", Err: syscall.ECONNRESET}, true},
		{"broken_pipe", &net.OpError{Op: "write", Err: syscall.EPIPE}, true},
		{"unexpected_eof", io.ErrUnexpectedEOF, true},
		{"io_timeout", timeoutNetErr{}, true},
		{"non_dial_op_error", &net.OpError{Op: "read", Err: errors.New("connection aborted")}, true},
		{"wrapped_like_exhausted", fmt.Errorf("planner: streaming failed after retries: %w", &net.OpError{Op: "read", Err: syscall.ECONNRESET}), true},
		// NOT transient: a client-aborted turn, a deterministic dial failure, and plain errors.
		{"context_canceled", context.Canceled, false},
		{"wrapped_context_canceled", fmt.Errorf("boom: %w", context.Canceled), false},
		{"deterministic_dial", &net.OpError{Op: "dial", Err: errors.New("connection refused")}, false},
		{"handler_timeout_is_not_a_net_error", http.ErrHandlerTimeout, false},
		{"plain_error", errors.New("nope"), false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := transientTransportError(c.err); got != c.want {
				t.Fatalf("transientTransportError(%q) = %v, want %v", c.name, got, c.want)
			}
		})
	}
}

// TestUpstreamErrorKind_TransportBucket proves a transient wire error surfaces as the DISTINCT
// "transport" kind (not the opaque "other"), including when wrapped exactly as rs.exhausted
// wraps it, while a deterministic dial failure stays "unreachable" and a typed status still wins.
func TestUpstreamErrorKind_TransportBucket(t *testing.T) {
	wireErr := fmt.Errorf("planner: streaming failed after retries: %w", &net.OpError{Op: "read", Err: syscall.ECONNRESET})
	if got := upstreamErrorKind(wireErr); got != "transport" {
		t.Fatalf("upstreamErrorKind(wire) = %q, want transport", got)
	}
	// A deterministic dial failure is surfaced as UpstreamUnreachableError in production — it must
	// stay "unreachable" so the supervisor's transient arm never retries a misconfiguration.
	if got := upstreamErrorKind(&agent.UpstreamUnreachableError{Err: errors.New("dial")}); got != "unreachable" {
		t.Fatalf("a deterministic dial failure = %q, want unreachable", got)
	}
	// A typed status error still wins over the transport check (it is tried first).
	if got := upstreamErrorKind(&agent.UpstreamStatusError{Status: 429}); got != "rate_limited" {
		t.Fatalf("a 429 = %q, want rate_limited (status must outrank the transport check)", got)
	}
}

// TestTransientWireErrorSnapshot_CountsOnlyTransportKind is the guard-facing witness: the
// supervisor's snapshot reflects an observed transient wire error and IGNORES a deterministic
// "unreachable" and a coarse "other" — so a positive delta across a child's run is specific
// evidence of a transient wire crash, the precondition #3514's bounded retry gates on.
func TestTransientWireErrorSnapshot_CountsOnlyTransportKind(t *testing.T) {
	s := &Server{metrics: newGatewayMetrics(time.Now())}
	if got := s.TransientWireErrorSnapshot(); got != 0 {
		t.Fatalf("fresh snapshot = %d, want 0", got)
	}
	// A transient wire error surfaced to the wrapped agent, folded via the SAME observe path the
	// passthrough loop uses on exhaustion.
	s.metrics.observeUpstreamError(fmt.Errorf("planner: streaming failed after retries: %w", &net.OpError{Op: "read", Err: syscall.ECONNRESET}))
	// Noise the snapshot must NOT count: a deterministic dial failure and an unclassifiable error.
	s.metrics.observeUpstreamError(&agent.UpstreamUnreachableError{Err: errors.New("dial")})
	s.metrics.observeUpstreamError(http.ErrHandlerTimeout)

	if got := s.TransientWireErrorSnapshot(); got != 1 {
		t.Fatalf("snapshot after one transient wire error = %d, want 1 (unreachable/other must not count)", got)
	}
	// A nil server/metrics snapshots to 0 rather than panicking, so a caller may snapshot blind.
	if got := (*Server)(nil).TransientWireErrorSnapshot(); got != 0 {
		t.Fatalf("nil-server snapshot = %d, want 0", got)
	}
}
