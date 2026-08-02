package gateway

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"syscall"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestUpstreamErrorKindsSnapshot_CarriesStalledOffTheProcess is the boundary witness for
// #5487. upstreamErrorKind has always minted "stalled" for an idle-deadline failure, but the
// only readers were the in-memory /metrics counter and the stderr FAILED line — both of which
// vanish when a per-invocation `fak guard` gateway exits. This accessor is the one seam that
// carries the FULL classification out of the package so the durable gateway-usage row can
// record it; the two pre-existing accessors deliberately do not (RotationEvidenceSnapshot
// keeps only auth/rate_limited, TransientWireErrorSnapshot only the transport scalar).
func TestUpstreamErrorKindsSnapshot_CarriesStalledOffTheProcess(t *testing.T) {
	s := &Server{metrics: newGatewayMetrics(time.Now())}

	// A fresh server has measured nothing: nil, so the ledger's omitempty field stays ABSENT
	// rather than persisting an empty object that would read as a measured zero.
	if got := s.UpstreamErrorKindsSnapshot(); got != nil {
		t.Fatalf("fresh snapshot = %v, want nil (absent must read NOT INSTRUMENTED, not zero failures)", got)
	}

	// Two stalls plus one of every neighbouring kind, folded through the SAME observe path
	// production uses.
	s.metrics.observeUpstreamError(&agent.UpstreamStalledError{Idle: 90 * time.Second})
	s.metrics.observeUpstreamError(fmt.Errorf("planner: %w", &agent.UpstreamStalledError{Idle: 90 * time.Second}))
	s.metrics.observeUpstreamError(&agent.UpstreamStatusError{Status: http.StatusTooManyRequests})
	s.metrics.observeUpstreamError(&agent.UpstreamUnreachableError{Err: errors.New("dial")})
	s.metrics.observeUpstreamError(fmt.Errorf("planner: streaming failed after retries: %w", &net.OpError{Op: "read", Err: syscall.ECONNRESET}))
	s.metrics.observeUpstreamError(http.ErrHandlerTimeout)

	snap := s.UpstreamErrorKindsSnapshot()
	if snap["stalled"] != 2 {
		t.Fatalf("snapshot[stalled] = %d, want 2 — the stall must leave the process as its OWN kind", snap["stalled"])
	}
	// Every neighbouring kind survives too, and "other" stays at exactly 1 — the single
	// genuinely unclassifiable error. A 3 there would mean the stalls had been swept into
	// the coarse bucket, which is the failure mode this whole kind ladder exists to avoid.
	for kind, want := range map[string]uint64{"rate_limited": 1, "unreachable": 1, "transport": 1, "other": 1} {
		if snap[kind] != want {
			t.Fatalf("snapshot[%s] = %d, want %d — every kind must survive distinctly, not just the stall", kind, snap[kind], want)
		}
	}
	if len(snap) != 5 {
		t.Fatalf("snapshot = %v, want exactly the 5 observed kinds (nothing invented or merged)", snap)
	}

	// The snapshot is a COPY: mutating it must not corrupt the live counter.
	snap["stalled"] = 999
	if again := s.UpstreamErrorKindsSnapshot(); again["stalled"] != 2 {
		t.Fatalf("live counter was aliased by the snapshot: stalled = %d, want 2", again["stalled"])
	}

	// A nil server snapshots to nil rather than panicking, so a caller may snapshot blind —
	// matching TransientWireErrorSnapshot's posture.
	if got := (*Server)(nil).UpstreamErrorKindsSnapshot(); got != nil {
		t.Fatalf("nil-server snapshot = %v, want nil", got)
	}
}
