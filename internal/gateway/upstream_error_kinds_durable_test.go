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

	// Two stalls (one bare, one wrapped) plus one of every neighbouring kind. Each row names
	// the kind upstreamErrorKind (metrics.go:504) must mint for that error, so the expected
	// tally below is DERIVED from this fixture rather than frozen at today's total: adding a
	// row here moves the expectation with it, and a row whose classification silently changes
	// still fails.
	observed := []struct {
		err  error
		kind string
	}{
		{&agent.UpstreamStalledError{Idle: 90 * time.Second}, "stalled"},
		{fmt.Errorf("planner: %w", &agent.UpstreamStalledError{Idle: 90 * time.Second}), "stalled"},
		{&agent.UpstreamStatusError{Status: http.StatusTooManyRequests}, "rate_limited"},
		{&agent.UpstreamUnreachableError{Err: errors.New("dial")}, "unreachable"},
		{fmt.Errorf("planner: streaming failed after retries: %w", &net.OpError{Op: "read", Err: syscall.ECONNRESET}), "transport"},
		{http.ErrHandlerTimeout, "other"},
	}
	// Folded through the SAME observe path production uses.
	want := map[string]uint64{}
	for _, o := range observed {
		s.metrics.observeUpstreamError(o.err)
		want[o.kind]++
	}

	snap := s.UpstreamErrorKindsSnapshot()
	if snap["stalled"] != want["stalled"] {
		t.Fatalf("snapshot[stalled] = %d, want %d — the stall must leave the process as its OWN kind, wrapped or bare", snap["stalled"], want["stalled"])
	}
	// Every neighbouring kind survives distinctly, and the tally is EXACT in both
	// directions: every kind the fixture fed in is present at its own count (nothing
	// merged — "other" reading 3 would mean the stalls had been swept into the coarse
	// bucket, the failure mode this whole kind ladder exists to avoid), and the snapshot
	// carries no key the fixture never observed (nothing invented).
	for kind, n := range want {
		if snap[kind] != n {
			t.Fatalf("snapshot[%s] = %d, want %d — every observed kind must survive distinctly, not just the stall", kind, snap[kind], n)
		}
	}
	for kind, n := range snap {
		if _, ok := want[kind]; !ok {
			t.Fatalf("snapshot invented the kind %q = %d, which nothing in the fixture observed: snapshot = %v", kind, n, snap)
		}
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
