package gateway

// session_routes_test.go — the HTTP contract for the /v1/fak/session control
// surface (#620): observe (GET) and control (POST) over a served session's DRIVE
// state, plus the fail-closed posture (nil injection ⇒ 404, never a silent clean
// reading) and the validation/method rules. Mirrors policy_reload_test.go's trace
// route tests, which are the established pattern for an injected-func route.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSessionObserveRouteReturnsDriveState(t *testing.T) {
	srv := newTestServer(t)
	gotTrace := ""
	srv.observeSession = func(_ context.Context, traceID string) SessionState {
		gotTrace = traceID
		return SessionState{
			TraceID:  traceID,
			Run:      "throttled",
			Budget:   SessionBudget{TurnsLeft: 3, TokensLeft: -1},
			Priority: 5,
			Pace:     SessionPace{MaxTokensPerTurn: 512, MinTurnGapMs: 200},
			Reason:   "operator-throttle",
			Rev:      4,
		}
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Get(ts.URL + "/v1/fak/session/sess-42")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("session observe status = %d, want 200", r.StatusCode)
	}
	var resp SessionState
	if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if gotTrace != "sess-42" || resp.TraceID != "sess-42" || resp.Run != "throttled" ||
		resp.Budget.TurnsLeft != 3 || resp.Budget.TokensLeft != -1 ||
		resp.Priority != 5 || resp.Pace.MaxTokensPerTurn != 512 || resp.Pace.MinTurnGapMs != 200 ||
		resp.Reason != "operator-throttle" || resp.Rev != 4 {
		t.Fatalf("gotTrace=%q response=%+v, want the observed drive state for sess-42", gotTrace, resp)
	}
}

func TestSessionControlRouteAppliesVerb(t *testing.T) {
	srv := newTestServer(t)
	gotTrace, gotVerb := "", ""
	srv.controlSession = func(_ context.Context, traceID, verb string, req SessionControlRequest) (SessionState, bool, error) {
		gotTrace, gotVerb = traceID, verb
		if verb == "priority" && req.Priority == nil {
			t.Fatalf("priority verb lost the body field")
		}
		return SessionState{TraceID: traceID, Run: "running", Priority: 2, Rev: 9}, true, nil
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/fak/session/sess-7/priority", "application/json",
		strings.NewReader(`{"priority":2}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("session control status = %d, want 200", r.StatusCode)
	}
	var resp SessionState
	if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if gotTrace != "sess-7" || gotVerb != "priority" || resp.Priority != 2 || resp.Rev != 9 {
		t.Fatalf("gotTrace=%q verb=%q resp=%+v, want priority control applied to sess-7", gotTrace, gotVerb, resp)
	}
}

func TestSessionRoutesValidationAndDisabled(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Disabled (no observe/control injected) ⇒ 404, not a silent clean reading.
	r, err := http.Get(ts.URL + "/v1/fak/session/sess-9")
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusNotFound {
		t.Fatalf("disabled observe status = %d, want 404", r.StatusCode)
	}
	r, err = http.Post(ts.URL+"/v1/fak/session/sess-9/run", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusNotFound {
		t.Fatalf("disabled control status = %d, want 404", r.StatusCode)
	}

	// Empty trace id ⇒ 400.
	r, err = http.Get(ts.URL + "/v1/fak/session/")
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusBadRequest {
		t.Fatalf("empty-id observe status = %d, want 400", r.StatusCode)
	}

	// POST without a verb ⇒ 400.
	srv.observeSession = func(context.Context, string) SessionState { return SessionState{} }
	srv.controlSession = func(context.Context, string, string, SessionControlRequest) (SessionState, bool, error) {
		t.Fatalf("control func must not be called without a verb")
		return SessionState{}, false, nil
	}
	r, err = http.Post(ts.URL+"/v1/fak/session/sess-9", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusBadRequest {
		t.Fatalf("verb-less POST status = %d, want 400", r.StatusCode)
	}

	// GET with a trailing verb ⇒ 405 (observe is GET on the id path only).
	r, err = http.Get(ts.URL + "/v1/fak/session/sess-9/run")
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET-with-verb status = %d, want 405", r.StatusCode)
	}

	// DELETE (or any other method) ⇒ 405.
	req, err := http.NewRequest(http.MethodDelete, ts.URL+"/v1/fak/session/sess-9", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("DELETE status = %d, want 405", resp.StatusCode)
	}
}

func TestSessionControlRouteErrorAndConflict(t *testing.T) {
	srv := newTestServer(t)
	// A control func that returns an error (malformed verb/body) ⇒ 400.
	srv.controlSession = func(_ context.Context, _, _ string, _ SessionControlRequest) (SessionState, bool, error) {
		return SessionState{}, false, errSentinel
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/fak/session/sess-9/run", "application/json",
		strings.NewReader(`{"run":"paused"}`))
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusBadRequest {
		t.Fatalf("control error status = %d, want 400", r.StatusCode)
	}

	// A control func that refuses (ok=false, err=nil) — terminal or a lost CAS race
	// — ⇒ 409, NOT 400 (the request was well-formed; the state refused it).
	srv.controlSession = func(context.Context, string, string, SessionControlRequest) (SessionState, bool, error) {
		return SessionState{TraceID: "sess-9", Run: "stopped", Rev: 3}, false, nil
	}
	r, err = http.Post(ts.URL+"/v1/fak/session/sess-9/budget", "application/json",
		strings.NewReader(`{"budget":{"turns_left":5}}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusConflict {
		t.Fatalf("control refusal status = %d, want 409", r.StatusCode)
	}
	// The 409 body is an OpenAI-style error envelope, not a SessionState; just
	// assert the status, mirroring how the trace tests assert disabled⇒404 only.
}

// errSentinel is a stable non-nil error for the control-func error path.
var errSentinel = errors.New("malformed control request")
