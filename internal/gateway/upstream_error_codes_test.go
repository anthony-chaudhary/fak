package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// upstreamErrorStatus must give each operationally-distinct upstream condition its OWN
// OpenAI-style code + client status, instead of collapsing every 4xx into one opaque code:null
// "upstream rejected the request (HTTP n)". A wrapped agent (and an operator) can then tell a
// rate limit from a bad credential from a permission denial — each of which calls for a different
// fix — by branching on the code, not guessing from the bare number.
func TestUpstreamErrorStatus_DistinctCodePerStatus(t *testing.T) {
	cases := []struct {
		status     int
		wantStatus int
		wantCode   string
		// a word that must appear in the client message so the condition is named, not just numbered
		wantMsgSubstr string
	}{
		{400, 400, "upstream_invalid_request", "malformed"},
		{401, 401, "upstream_unauthorized", "credential"},
		{403, 403, "upstream_forbidden", "permission"},
		{404, 404, "upstream_model_not_found", "model"},
		{408, 408, "upstream_request_timeout", "timed out"},
		{413, 413, "upstream_payload_too_large", "too large"},
		{429, 429, "upstream_rate_limited", "rate-limited"},
		{409, 409, "upstream_request_rejected", "rejected"}, // un-enumerated 4xx -> generic
	}
	for _, c := range cases {
		status, code, msg := upstreamErrorStatus(&agent.UpstreamStatusError{Status: c.status})
		if status != c.wantStatus {
			t.Errorf("status %d: client status = %d, want %d", c.status, status, c.wantStatus)
		}
		if code != c.wantCode {
			t.Errorf("status %d: code = %q, want %q", c.status, code, c.wantCode)
		}
		if !strings.Contains(strings.ToLower(msg), c.wantMsgSubstr) {
			t.Errorf("status %d: message %q does not name the condition (want substr %q)", c.status, msg, c.wantMsgSubstr)
		}
	}
}

// errType must give 429 the OpenAI/Anthropic-standard rate_limit_error type so a client that
// branches on `type` backs off on a rate limit but not on a 400. 401/403 stay authentication_error;
// 5xx stays server_error; the rest stay invalid_request_error.
func TestErrType_RateLimitArm(t *testing.T) {
	cases := []struct {
		status int
		want   string
	}{
		{http.StatusTooManyRequests, "rate_limit_error"},
		{http.StatusUnauthorized, "authentication_error"},
		{http.StatusForbidden, "authentication_error"},
		{http.StatusBadRequest, "invalid_request_error"},
		{http.StatusNotFound, "invalid_request_error"},
		{http.StatusBadGateway, "server_error"},
		{http.StatusServiceUnavailable, "server_error"},
		{529, "overloaded_error"}, // Anthropic "Overloaded" gets its own type, not server_error
	}
	for _, c := range cases {
		if got := errType(c.status); got != c.want {
			t.Errorf("errType(%d) = %q, want %q", c.status, got, c.want)
		}
	}
}

// The status ladder (upstreamErrorStatus) and the metric-kind ladder (upstreamErrorKind) are
// hand-mirrored with no shared source of truth, so the live risk is DRIFT: someone re-splits one
// and forgets the other, and the /metrics kind no longer matches the client's code. This test pins
// the pairing for every classified status so that can never silently happen.
func TestUpstreamError_LaddersAgree(t *testing.T) {
	cases := []struct {
		status   int
		wantCode string
		wantKind string
	}{
		{400, "upstream_invalid_request", "status_4xx"},
		{401, "upstream_unauthorized", "auth"},
		{403, "upstream_forbidden", "forbidden"},
		{404, "upstream_model_not_found", "status_4xx"},
		{408, "upstream_request_timeout", "status_4xx"},
		{413, "upstream_payload_too_large", "status_4xx"},
		{429, "upstream_rate_limited", "rate_limited"},
		{409, "upstream_request_rejected", "status_4xx"},
		{500, "", "status_5xx"}, // 5xx keeps the historical code:null shape and the coarse kind
		{503, "", "status_5xx"},
		{529, "upstream_overloaded", "overloaded"}, // 529 is carved out of the coarse 5xx bucket
	}
	for _, c := range cases {
		err := &agent.UpstreamStatusError{Status: c.status}
		_, code, _ := upstreamErrorStatus(err)
		kind := upstreamErrorKind(err)
		if code != c.wantCode {
			t.Errorf("status %d: code = %q, want %q", c.status, code, c.wantCode)
		}
		if kind != c.wantKind {
			t.Errorf("status %d: kind = %q, want %q", c.status, kind, c.wantKind)
		}
	}
}

// The trust-boundary invariant must hold for EVERY new arm: the upstream's raw error body never
// appears in the client-facing message, and no new arm is ever misclassified as an in-kernel OOM.
func TestUpstreamErrorStatus_NoBodyLeakAcrossArms(t *testing.T) {
	const secret = "SECRET_UPSTREAM_BODY_must_never_reach_client"
	for _, status := range []int{400, 401, 403, 404, 408, 413, 422, 429, 500, 503} {
		_, code, msg := upstreamErrorStatus(&agent.UpstreamStatusError{Status: status, Body: secret, RetryAfter: "120"})
		if strings.Contains(msg, secret) {
			t.Fatalf("status %d: upstream body leaked into client message: %q", status, msg)
		}
		if code == "in_kernel_oom" {
			t.Fatalf("status %d: a real upstream error was misclassified as in_kernel_oom", status)
		}
	}
}

// A 5xx that is NOT an overloaded-503-with-Retry-After must keep the byte-identical historical
// envelope (502, code:null, "upstream model error"), so the only 5xx whose surfaced status changes
// is the genuinely-overloaded one.
func TestUpstreamErrorStatus_5xxEnvelopeUnchanged(t *testing.T) {
	status, code, msg := upstreamErrorStatus(&agent.UpstreamStatusError{Status: 500, Body: "provider body"})
	if status != http.StatusBadGateway || code != "" || msg != "upstream model error" {
		t.Fatalf("500 envelope = (%d, %q, %q), want (502, \"\", \"upstream model error\")", status, code, msg)
	}
	// A bare 503 (no Retry-After) also stays the opaque 502 — we only special-case the overloaded
	// 503 the upstream timed for us.
	status, code, _ = upstreamErrorStatus(&agent.UpstreamStatusError{Status: 503})
	if status != http.StatusBadGateway || code != "" {
		t.Fatalf("bare 503 = (%d, %q), want (502, \"\")", status, code)
	}
	// A 503 WITH Retry-After surfaces the real 503 so the client backs off.
	status, _, _ = upstreamErrorStatus(&agent.UpstreamStatusError{Status: 503, RetryAfter: "30"})
	if status != http.StatusServiceUnavailable {
		t.Fatalf("503+Retry-After = %d, want 503", status)
	}
	// A 529 (Anthropic "Overloaded") is carved OUT of the opaque-502 5xx bucket: it surfaces
	// the real 529 with a distinct code, so a client can apply backoff+jitter instead of
	// treating provider-over-capacity like a crash. This holds even with no Retry-After (a 529
	// has none a client can trust).
	status, code, msg = upstreamErrorStatus(&agent.UpstreamStatusError{Status: 529, Body: "provider body"})
	if status != 529 || code != "upstream_overloaded" {
		t.Fatalf("529 envelope = (%d, %q), want (529, \"upstream_overloaded\")", status, code)
	}
	if !strings.Contains(strings.ToLower(msg), "overloaded") {
		t.Fatalf("529 message %q does not name the overloaded condition", msg)
	}
}

// writeUpstreamErr must echo the upstream's Retry-After as a DOWNSTREAM response header (the only
// form an OpenAI/Anthropic SDK or retry middleware reads) on a rate-limited turn — and must NOT
// invent the header when the upstream supplied none, nor leak the body into the JSON envelope.
func TestWriteUpstreamErr_EchoesRetryAfter(t *testing.T) {
	s := &Server{} // nil metrics is tolerated by plannerErrorStatus

	// 429 with Retry-After: the header is echoed, status is 429, the secret body is absent.
	rec := httptest.NewRecorder()
	s.writeUpstreamErr(rec, &agent.UpstreamStatusError{Status: 429, RetryAfter: "30", Body: "SECRET"})
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("429 path: status = %d, want 429", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "30" {
		t.Fatalf("429 path: Retry-After header = %q, want \"30\"", got)
	}
	if strings.Contains(rec.Body.String(), "SECRET") {
		t.Fatalf("429 path: upstream body leaked into envelope: %s", rec.Body.String())
	}
	// The JSON envelope carries the distinct code + rate_limit_error type.
	var env struct {
		Error struct {
			Code string `json:"code"`
			Type string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("429 path: envelope did not decode: %v", err)
	}
	if env.Error.Code != "upstream_rate_limited" || env.Error.Type != "rate_limit_error" {
		t.Fatalf("429 path: envelope = (%q, %q), want (upstream_rate_limited, rate_limit_error)", env.Error.Code, env.Error.Type)
	}

	// No upstream Retry-After: the header must be absent (a clean no-op, never an empty/"0" header).
	rec = httptest.NewRecorder()
	s.writeUpstreamErr(rec, &agent.UpstreamStatusError{Status: 429})
	if _, present := rec.Header()["Retry-After"]; present {
		t.Fatalf("no upstream Retry-After: a Retry-After header must not be written, got %q", rec.Header().Get("Retry-After"))
	}

	// A non-status error (a stall) carries no Retry-After channel at all.
	rec = httptest.NewRecorder()
	s.writeUpstreamErr(rec, &agent.UpstreamStalledError{})
	if _, present := rec.Header()["Retry-After"]; present {
		t.Fatalf("stall: a Retry-After header must not be written")
	}
}
