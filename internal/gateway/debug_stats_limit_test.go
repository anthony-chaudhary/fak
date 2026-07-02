package gateway

// The gateway half of #2257: a turn that died mid-retry-wait while honoring a CLASSIFIED
// cap 429 must surface the rate-limit truth on every readout — the FAILED debug line
// (reason=rate_limited + the closed cap-kind token + the announced wait + the
// client-disconnect marker) and the upstream-error metric kind — never the catch-all
// "error" the bare context cancel used to collapse into.

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

// capInterruptedErr builds the exact error shape the planner returns when the wrapped
// client hangs up during an honored usage-cap wait (#2257's evidence signature).
func capInterruptedErr() error {
	return fmt.Errorf("planner: %w", &agent.RetryInterruptedError{
		Cause: &agent.UpstreamStatusError{
			Status:      http.StatusTooManyRequests,
			LimitReason: resume.LimitUsage,
		},
		Err:           context.Canceled,
		AnnouncedWait: time.Hour + 10*time.Minute + 36*time.Second,
	})
}

// The FAILED line must read rate_limited + the cap kind + the announced wait + the
// client-gone marker — the truth the operator (and the #2256 supervisor) parks on.
func TestRenderTurnDebugError_ClientCancelDuringCapWait_TellsTheTruth(t *testing.T) {
	s := newResetShadowServer()
	var sb strings.Builder
	s.debugStatsf = func(format string, args ...any) {
		fmt.Fprintf(&sb, format, args...)
		sb.WriteByte('\n')
	}
	s.renderTurnDebugError("t1", "anthropic_messages", capInterruptedErr(), 300*time.Second)
	out := sb.String()
	for _, want := range []string{
		"FAILED",
		"reason=rate_limited", // NOT reason=error: the classification survives the cancel
		"kind=usage_limit",    // the closed vocabulary token `fak resume scan` also speaks
		"after=300s",
		"announced_wait=1h10m36s",
		"client_gone=true",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("FAILED line missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "reason=error") {
		t.Fatalf("classified cap cancel collapsed into reason=error:\n%s", out)
	}
}

// The metric kind must MATCH the line: the upstream-error counter for the interrupted
// cap wait lands in rate_limited, so operators' dashboards and the debug line agree.
func TestObserveUpstreamError_ClientCancelDuringCapWait_CountsRateLimited(t *testing.T) {
	if kind := upstreamErrorKind(capInterruptedErr()); kind != "rate_limited" {
		t.Fatalf("upstreamErrorKind = %q, want rate_limited", kind)
	}
	m := newGatewayMetrics(time.Now())
	m.observeUpstreamError(capInterruptedErr())
	m.upstreamErrMu.Lock()
	got := m.upstreamErrors["rate_limited"]
	m.upstreamErrMu.Unlock()
	if got != 1 {
		t.Fatalf("upstreamErrors[rate_limited] = %d, want 1", got)
	}
}

// A genuinely-unclassified failure keeps today's exact rendering: no kind=, no
// announced_wait=, no client_gone= — and still reason=error.
func TestRenderTurnDebugError_UnclassifiedFailure_Unchanged(t *testing.T) {
	s := newResetShadowServer()
	var sb strings.Builder
	s.debugStatsf = func(format string, args ...any) { fmt.Fprintf(&sb, format, args...) }
	s.renderTurnDebugError("t1", "anthropic_messages", context.Canceled, 5*time.Second)
	out := sb.String()
	if !strings.Contains(out, "reason=error") {
		t.Fatalf("bare cancel must still read reason=error: %q", out)
	}
	for _, banned := range []string{"kind=", "announced_wait=", "client_gone="} {
		if strings.Contains(out, banned) {
			t.Fatalf("unclassified failure grew a %q field: %q", banned, out)
		}
	}
}
