package gateway

import (
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// formatTurnDebugError must render a single flat, payload-free line that LEADS with the failure
// verdict and reason — the mirror of the success line, with the elapsed time rounded to seconds.
func TestFormatTurnDebugError_LeadsWithReason(t *testing.T) {
	line := formatTurnDebugError("trace_x", "anthropic_messages", "stalled", 61*time.Second)
	for _, want := range []string{"fak-turn ", "trace=trace_x", "FAILED", "reason=stalled", "wire=anthropic_messages", "after=61s"} {
		if !strings.Contains(line, want) {
			t.Fatalf("missing %q in: %q", want, line)
		}
	}
	if strings.Contains(line, "\n") {
		t.Fatalf("debug line must be a single row: %q", line)
	}
	// An empty reason must never render blank — it reads "error".
	if l := formatTurnDebugError("t", "w", "", time.Second); !strings.Contains(l, "reason=error") {
		t.Fatalf("empty reason should read 'error': %q", l)
	}
}

// debugErrorReason collapses the shared upstreamErrorKind into the closed glanceable token set,
// folding the two status kinds into one "status".
func TestDebugErrorReason_ClosedTokens(t *testing.T) {
	cases := []struct {
		err  error
		want string
	}{
		{&agent.UpstreamStalledError{Idle: time.Second}, "stalled"},
		{&agent.UpstreamUnreachableError{Err: http.ErrServerClosed}, "unreachable"},
		{&agent.UpstreamStatusError{Status: 404}, "status"},
		{&agent.UpstreamStatusError{Status: 500}, "status"},
		{nil, "error"},
		{http.ErrHandlerTimeout, "error"},
	}
	for _, c := range cases {
		if got := debugErrorReason(c.err); got != c.want {
			t.Errorf("debugErrorReason(%v) = %q, want %q", c.err, got, c.want)
		}
	}
}

// The crux: a FAILED turn must emit a debug line on the default --debug-stats sink even with logf
// nil (--log off) — this is the half that was missing, so a stall is no longer a silent freeze.
func TestRenderTurnDebugError_FiresWithLogfNil(t *testing.T) {
	s := newResetShadowServer() // logf nil — proves the failure line works with --log off
	var sb strings.Builder
	s.debugStatsf = func(format string, args ...any) {
		fmt.Fprintf(&sb, format, args...)
		sb.WriteByte('\n')
	}
	s.renderTurnDebugError("t1", "anthropic_messages", &agent.UpstreamStalledError{Idle: 60 * time.Second}, 61*time.Second)
	out := sb.String()
	if !strings.Contains(out, "fak-turn ") || !strings.Contains(out, "FAILED") {
		t.Fatalf("a failed turn did not emit a debug line: %q", out)
	}
	if !strings.Contains(out, "reason=stalled") {
		t.Fatalf("a stall must render reason=stalled: %q", out)
	}
}

// Gated off: with no sink wired (--debug-stats off / nil), the failure render is a byte-identical
// no-op — no panic, nothing emitted.
func TestRenderTurnDebugError_GatedOffWhenSinkNil(t *testing.T) {
	s := newResetShadowServer() // debugStatsf nil
	s.renderTurnDebugError("t1", "anthropic_messages", &agent.UpstreamStalledError{Idle: time.Second}, time.Second)
	// reaching here without a panic is the assertion (nil sink must short-circuit before format)
}
