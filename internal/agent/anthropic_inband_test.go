package agent

import (
	"errors"
	"net/http"
	"testing"
)

// TestAnthropicInBandStatusMapping pins the in-band error `type` -> HTTP status table, and
// with it the retry verdict each type inherits from retryableStatus. The table IS the
// policy: get a status wrong and either a transient overload stops being retried (the bug
// this classification exists to fix) or a permanent request error burns the whole attempt
// budget against the upstream.
func TestAnthropicInBandStatusMapping(t *testing.T) {
	for _, tc := range []struct {
		errType   string
		want      int
		retryable bool
	}{
		{"overloaded_error", 529, true},
		{"api_error", http.StatusInternalServerError, true},
		{"rate_limit_error", http.StatusTooManyRequests, true},
		{"timeout_error", http.StatusRequestTimeout, true},
		{"invalid_request_error", http.StatusBadRequest, false},
		{"authentication_error", http.StatusUnauthorized, false},
		{"billing_error", http.StatusPaymentRequired, false},
		{"permission_error", http.StatusForbidden, false},
		{"not_found_error", http.StatusNotFound, false},
		{"request_too_large", http.StatusRequestEntityTooLarge, false},
		// Case/whitespace robustness: the mapping must not depend on the provider's casing.
		{"  OVERLOADED_ERROR ", 529, true},
		// An UNLABELED frame is deliberately retryable — an in-band refusal rides a request
		// the upstream already accepted, so a capacity flap is likelier than a hard error.
		{"", http.StatusInternalServerError, true},
		{"something_new_from_the_provider", http.StatusInternalServerError, true},
	} {
		got := anthropicInBandStatus(tc.errType)
		if got != tc.want {
			t.Errorf("anthropicInBandStatus(%q) = %d, want %d", tc.errType, got, tc.want)
		}
		e := &UpstreamInBandError{Type: tc.errType, Status: got}
		if e.Retryable() != tc.retryable {
			t.Errorf("%q (HTTP %d): Retryable() = %v, want %v", tc.errType, got, e.Retryable(), tc.retryable)
		}
	}
}

// TestNewAnthropicInBandErrorParsesFrame checks the wire shape is read from the real
// Anthropic frame, and that a frame fak cannot parse still yields a retryable verdict
// rather than being mistaken for a served turn.
func TestNewAnthropicInBandErrorParsesFrame(t *testing.T) {
	e := NewAnthropicInBandError([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`))
	if e.Type != "overloaded_error" || e.Message != "Overloaded" || e.Status != 529 {
		t.Fatalf("parsed = %+v, want {overloaded_error Overloaded 529}", e)
	}
	if !e.Retryable() {
		t.Error("an overloaded_error frame must be retryable")
	}

	// Garbage in, still a usable (and retryable) verdict out.
	for _, raw := range []string{`not json at all`, `{}`, `{"error":{}}`, ``} {
		g := NewAnthropicInBandError([]byte(raw))
		if g == nil {
			t.Fatalf("NewAnthropicInBandError(%q) = nil, want a non-nil unlabeled verdict", raw)
		}
		if g.Status != http.StatusInternalServerError || !g.Retryable() {
			t.Errorf("NewAnthropicInBandError(%q) = %+v, want an unlabeled retryable 500", raw, g)
		}
	}
}

// TestUpstreamInBandErrorStatusErrorSurfaces pins the conversion the gateway relies on: an
// exhausted in-band overload must reach the client as the honest 529 (which
// upstreamErrorStatus renders as overloaded_error), and must be reachable via errors.As
// through the wrapping rs.exhausted applies.
func TestUpstreamInBandErrorStatusErrorSurfaces(t *testing.T) {
	ib := NewAnthropicInBandError([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`))
	se := ib.StatusError()
	if se.Status != 529 {
		t.Fatalf("StatusError().Status = %d, want 529", se.Status)
	}
	if se.RetryAfter != "" {
		t.Errorf("StatusError().RetryAfter = %q, want empty — an in-band frame carries no response header", se.RetryAfter)
	}

	// The retry loop records an in-band retryable through the SAME keepsake as a non-200,
	// so exhaustion surfaces a *UpstreamStatusError the gateway can find.
	var rs retryState
	rs.noteRetryableStatus(ib.Status, []byte(ib.Message), http.Header{}, 400)
	var found *UpstreamStatusError
	if !errors.As(rs.exhausted("planner: streaming failed after retries"), &found) {
		t.Fatal("exhausted() did not carry an *UpstreamStatusError")
	}
	if found.Status != 529 {
		t.Errorf("exhausted status = %d, want 529", found.Status)
	}
}
