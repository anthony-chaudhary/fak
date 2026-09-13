package gateway

// unified_memory_admit_test.go — the live-path reproduction witness for issue #12305.
//
// The original issue's witness (`TestMemoryPressureNotification`) proves only the
// in-package governor contract; before this change that governor had zero production
// callers and its 429 CAPACITY_BACKOFF never reached the wire. These tests fail on the
// pre-wiring tree (the governor is never consulted, so the request is admitted and the
// refusal renders as a plain 409) and pass once the served boundary consults it.

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/macobs"
)

// TestUnifiedMemoryGovernorCriticalReturns429CapacityBackoff is the load-bearing repro:
// under a simulated DISPATCH_MEMORYPRESSURE_CRITICAL event, a served request must be
// shed with HTTP 429 and the closed CAPACITY_BACKOFF code — not a 409/503 and not an
// admitted 200.
func TestUnifiedMemoryGovernorCriticalReturns429CapacityBackoff(t *testing.T) {
	srv := &Server{}
	sub := macobs.NewSimulatedSubscriber()
	gov := macobs.NewUnifiedMemoryPressureGovernor(
		sub,
		nil, // no page manager: the refusal contract is independent of reclamation plumbing
		macobs.NewDefaultMTPDepthGovernor(3, 1, 0),
		macobs.DefaultUnifiedMemoryGovernorConfig(),
	)
	if err := gov.Start(context.Background()); err != nil {
		t.Fatalf("gov.Start: %v", err)
	}
	defer func() { _ = gov.Stop() }()
	srv.SetUnifiedMemoryPressureGovernor(gov)

	// NORMAL: the governor admits, so the boundary must not refuse.
	if ref := srv.unifiedMemoryGovernorRefusal("trace-normal"); ref != nil {
		t.Fatalf("NORMAL pressure must not refuse, got %+v", ref)
	}

	// CRITICAL: the governor pauses admissions and the boundary must refuse.
	sub.SimulatePressure(macobs.PressureCritical)
	ref := srv.unifiedMemoryGovernorRefusal("trace-critical")
	if ref == nil {
		t.Fatal("CRITICAL pressure must refuse the served request")
	}
	if ref.Reason != macobs.ReasonCapacityBackoff {
		t.Fatalf("reason = %q, want %q", ref.Reason, macobs.ReasonCapacityBackoff)
	}

	// The refusal must render to the wire as 429 CAPACITY_BACKOFF.
	rec := httptest.NewRecorder()
	writeSessionRefusal(rec, *ref)
	if rec.Code != macobs.StatusCapacityBackoff {
		t.Fatalf("status = %d, want %d", rec.Code, macobs.StatusCapacityBackoff)
	}
	if got := rec.Header().Get("Retry-After"); got == "" {
		t.Error("missing Retry-After header on capacity backoff")
	}
	var body struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal refusal body: %v", err)
	}
	if body.Error.Code != macobs.CodeCapacityBackoff {
		t.Fatalf("error.code = %q, want %q", body.Error.Code, macobs.CodeCapacityBackoff)
	}
	if body.Reason != macobs.ReasonCapacityBackoff {
		t.Fatalf("reason = %q, want %q", body.Reason, macobs.ReasonCapacityBackoff)
	}
}

// TestUnifiedMemoryGovernorWarnStillAdmits pins the WARN semantics: the governor throttles
// MTP and reclaims pages but keeps serving, so the boundary must not shed traffic.
func TestUnifiedMemoryGovernorWarnStillAdmits(t *testing.T) {
	srv := &Server{}
	sub := macobs.NewSimulatedSubscriber()
	gov := macobs.NewUnifiedMemoryPressureGovernor(
		sub, nil,
		macobs.NewDefaultMTPDepthGovernor(3, 1, 0),
		macobs.DefaultUnifiedMemoryGovernorConfig(),
	)
	if err := gov.Start(context.Background()); err != nil {
		t.Fatalf("gov.Start: %v", err)
	}
	defer func() { _ = gov.Stop() }()
	srv.SetUnifiedMemoryPressureGovernor(gov)

	sub.SimulatePressure(macobs.PressureWarn)
	if ref := srv.unifiedMemoryGovernorRefusal("trace-warn"); ref != nil {
		t.Fatalf("WARN pressure must still admit, got refusal %+v", ref)
	}
}

// TestUnifiedMemoryGovernorNilIsInert pins the default: an unarmed host keeps the
// historical served path (no refusal, no 429).
func TestUnifiedMemoryGovernorNilIsInert(t *testing.T) {
	srv := &Server{}
	if ref := srv.unifiedMemoryGovernorRefusal("trace"); ref != nil {
		t.Fatalf("nil governor must be inert, got %+v", ref)
	}
	var nilSrv *Server
	if ref := nilSrv.unifiedMemoryGovernorRefusal("trace"); ref != nil {
		t.Fatalf("nil server must be inert, got %+v", ref)
	}
	rec := httptest.NewRecorder()
	writeSessionRefusal(rec, SessionState{TraceID: "t", Run: "paused", Reason: "operator_reason"})
	if rec.Code != http.StatusConflict {
		t.Fatalf("ordinary refusal status = %d, want 409", rec.Code)
	}
}

// TestUnifiedMemoryGovernorHTTPCritical429 is the full-stack witness: a real POST to the
// served chat-completions handler, with the governor attached and driven to CRITICAL, must
// come back HTTP 429 CAPACITY_BACKOFF on the wire. This is the end-to-end counterpart the
// component-level test above cannot cover (it exercises the seam functions directly).
func TestUnifiedMemoryGovernorHTTPCritical429(t *testing.T) {
	srv := newTestServer(t)

	sub := macobs.NewSimulatedSubscriber()
	gov := macobs.NewUnifiedMemoryPressureGovernor(
		sub, nil,
		macobs.NewDefaultMTPDepthGovernor(3, 1, 0),
		macobs.DefaultUnifiedMemoryGovernorConfig(),
	)
	if err := gov.Start(context.Background()); err != nil {
		t.Fatalf("gov.Start: %v", err)
	}
	defer func() { _ = gov.Stop() }()
	srv.SetUnifiedMemoryPressureGovernor(gov)

	sub.SimulatePressure(macobs.PressureCritical)

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	payload := map[string]any{
		"model":      "test-model",
		"messages":   []map[string]string{{"role": "user", "content": "hello"}},
		"max_tokens": 16,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", "opencode-critical-trace")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST chat: %v", err)
	}
	defer resp.Body.Close()
	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}

	if resp.StatusCode != macobs.StatusCapacityBackoff {
		t.Fatalf("HTTP status = %d, want %d (429 CAPACITY_BACKOFF); body: %s", resp.StatusCode, macobs.StatusCapacityBackoff, string(respBytes))
	}
	if got := resp.Header.Get("Retry-After"); got == "" {
		t.Error("missing Retry-After header on 429 capacity backoff")
	}
	var errResp struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(respBytes, &errResp); err != nil {
		t.Fatalf("decode response: %v; raw: %s", err, string(respBytes))
	}
	if errResp.Error.Code != macobs.CodeCapacityBackoff {
		t.Fatalf("error.code = %q, want %q; body: %s", errResp.Error.Code, macobs.CodeCapacityBackoff, string(respBytes))
	}
	if errResp.Reason != macobs.ReasonCapacityBackoff {
		t.Fatalf("reason = %q, want %q", errResp.Reason, macobs.ReasonCapacityBackoff)
	}
}
