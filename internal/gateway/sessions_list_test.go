package gateway

// sessions_list_test.go — the HTTP contract for GET /v1/fak/sessions (the
// multi-session DRIVE-state snapshot): the happy path returns every session in the
// injected order with a count, the fail-closed posture (nil injection ⇒ 404), and
// the method rule (GET-only). Mirrors session_routes_test.go's single-session tests.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSessionsListRouteReturnsSnapshot(t *testing.T) {
	srv := newTestServer(t)
	srv.listSessions = func(context.Context) []SessionState {
		return []SessionState{
			{TraceID: "urgent", Run: "running", Priority: 0, Rev: 2},
			{TraceID: "background", Run: "throttled", Priority: 5, Reason: "operator-throttle", Rev: 7},
		}
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Get(ts.URL + "/v1/fak/sessions")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("sessions list status = %d, want 200", r.StatusCode)
	}
	var resp SessionListResponse
	if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Count != 2 || len(resp.Sessions) != 2 {
		t.Fatalf("count=%d sessions=%d, want 2/2", resp.Count, len(resp.Sessions))
	}
	// The wire order is the injected (Snapshot) order: lowest priority first.
	if resp.Sessions[0].TraceID != "urgent" || resp.Sessions[1].TraceID != "background" ||
		resp.Sessions[1].Reason != "operator-throttle" {
		t.Fatalf("sessions = %+v, want urgent then background (with reason)", resp.Sessions)
	}
}

func TestSessionsListRouteDisabledAndMethod(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Disabled (no listSessions injected) ⇒ 404, not a silent empty array.
	r, err := http.Get(ts.URL + "/v1/fak/sessions")
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusNotFound {
		t.Fatalf("disabled list status = %d, want 404", r.StatusCode)
	}

	// POST (or any non-GET) ⇒ 405 even when configured (the list is a read).
	srv.listSessions = func(context.Context) []SessionState { return nil }
	r, err = http.Post(ts.URL+"/v1/fak/sessions", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("POST list status = %d, want 405", r.StatusCode)
	}
}
