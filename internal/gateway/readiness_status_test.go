package gateway

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// readiness_status_test.go pins the #3051 readiness-vs-liveness contract on the
// SERVED /healthz and /readyz surfaces: an ok:false body (warmup pending, a
// degenerate startup decode, a recent served failure) must answer 503 so an
// OpenAI/opencode client — which treats any 200 as "ready, route work here" —
// never sends the operator's first turn into a cold or broken backend. ok:true
// still answers 200. The identical ok bit drives both endpoints, so their
// verdicts cannot drift.

// serveStatus drives one GET through a handler and returns the recorded status.
func serveStatus(t *testing.T, h func(http.ResponseWriter, *http.Request), path string) int {
	t.Helper()
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodGet, path, nil))
	return rec.Code
}

// TestHealthzNotReadyIs503DuringWarmup is the direct acceptance witness for the
// false-ready bug: before the fix /healthz answered 200 with
// {"ok":false,"warmup_pending":true}; now the status tracks the body.
func TestHealthzNotReadyIs503DuringWarmup(t *testing.T) {
	srv := newTestServer(t)
	srv.ArmWarmupGate()

	rec := httptest.NewRecorder()
	srv.handleHealth(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("warmup-pending /healthz status = %d, want 503", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Fatalf("warmup-pending /healthz Retry-After = %q, want 1", got)
	}

	// The same not-ready body is what /readyz projects; it must agree on BOTH the
	// status and the backoff hint.
	rrec := httptest.NewRecorder()
	srv.handleReady(rrec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rrec.Code != http.StatusServiceUnavailable {
		t.Fatalf("warmup-pending /readyz status = %d, want 503", rrec.Code)
	}
	if got := rrec.Header().Get("Retry-After"); got != "1" {
		t.Fatalf("warmup-pending /readyz Retry-After = %q, want 1", got)
	}

	// Once warmup completes the model can serve: 200 on both surfaces. MarkReady
	// is the gateway's own listener-bound startup gate, which /readyz ANDs in.
	srv.MarkReady()
	srv.MarkWarmupComplete(1234)
	if code := serveStatus(t, srv.handleHealth, "/healthz"); code != http.StatusOK {
		t.Fatalf("warm /healthz status = %d, want 200", code)
	}
	if code := serveStatus(t, srv.handleReady, "/readyz"); code != http.StatusOK {
		t.Fatalf("warm /readyz status = %d, want 200", code)
	}
}

// TestHealthOKTreatsMissingOrNonBoolAsNotReady pins the fail-closed derivation:
// only a literal ok:true is ready, so a malformed body can never masquerade as a
// green light on either surface.
func TestHealthOKTreatsMissingOrNonBoolAsNotReady(t *testing.T) {
	cases := []struct {
		name   string
		health map[string]any
		want   bool
	}{
		{"true", map[string]any{"ok": true}, true},
		{"false", map[string]any{"ok": false}, false},
		{"missing", map[string]any{}, false},
		{"non-bool", map[string]any{"ok": "true"}, false},
		{"nil", map[string]any{"ok": nil}, false},
	}
	for _, c := range cases {
		if got := healthOK(c.health); got != c.want {
			t.Errorf("healthOK(%v) = %v, want %v", c.health, got, c.want)
		}
	}
}
