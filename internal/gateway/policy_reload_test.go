package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestPolicyReloadRouteInvokesConfiguredReloader(t *testing.T) {
	srv := newTestServer(t)
	calls := 0
	srv.reloadPolicy = func(context.Context) (PolicyReloadResponse, error) {
		calls++
		return PolicyReloadResponse{Reloaded: true, Source: "floor.json", Summary: "posture: fail_closed"}, nil
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/fak/policy/reload", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("reload status = %d, want 200", r.StatusCode)
	}
	var resp PolicyReloadResponse
	if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if calls != 1 || !resp.Reloaded || resp.Source != "floor.json" {
		t.Fatalf("calls=%d response=%+v, want one reload with source", calls, resp)
	}
}

func TestPolicyReloadRouteDisabledWithoutCallback(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/fak/policy/reload", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusNotFound {
		t.Fatalf("disabled reload status = %d, want 404", r.StatusCode)
	}
}

func TestPolicyReloadRouteReportsLoaderFailure(t *testing.T) {
	srv := newTestServer(t)
	srv.reloadPolicy = func(context.Context) (PolicyReloadResponse, error) {
		return PolicyReloadResponse{}, errors.New("bad manifest")
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/fak/policy/reload", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusBadRequest {
		t.Fatalf("failed reload status = %d, want 400", r.StatusCode)
	}
}

func TestTraceResetRouteInvokesConfiguredResetter(t *testing.T) {
	srv := newTestServer(t)
	gotTrace := ""
	srv.resetTrace = func(_ context.Context, traceID string) error {
		gotTrace = traceID
		return nil
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/fak/trace/reset", "application/json", strings.NewReader(`{"trace_id":" trace-1 "}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("trace reset status = %d, want 200", r.StatusCode)
	}
	var resp TraceResetResponse
	if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if gotTrace != "trace-1" || !resp.Reset || resp.TraceID != "trace-1" {
		t.Fatalf("gotTrace=%q response=%+v, want trimmed trace reset", gotTrace, resp)
	}
}

func TestTraceObserveRouteReturnsTaintLevel(t *testing.T) {
	srv := newTestServer(t)
	gotTrace := ""
	srv.observeTrace = func(_ context.Context, traceID string) (string, bool) {
		gotTrace = traceID
		return "tainted", true
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Get(ts.URL + "/v1/fak/trace/sess-9")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("trace observe status = %d, want 200", r.StatusCode)
	}
	var resp TraceObserveResponse
	if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if gotTrace != "sess-9" || resp.TraceID != "sess-9" || resp.Taint != "tainted" || !resp.Dangerous {
		t.Fatalf("gotTrace=%q response=%+v, want observed taint level for sess-9", gotTrace, resp)
	}
}

func TestTraceObserveRouteValidationAndDisabled(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Disabled (no observeTrace injected) => 404, not a silent clean reading.
	r, err := http.Get(ts.URL + "/v1/fak/trace/sess-9")
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusNotFound {
		t.Fatalf("disabled trace observe status = %d, want 404", r.StatusCode)
	}

	srv.observeTrace = func(context.Context, string) (string, bool) { return "trusted", false }

	// Empty trace id on the subtree => 400.
	r, err = http.Get(ts.URL + "/v1/fak/trace/")
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusBadRequest {
		t.Fatalf("empty-id trace observe status = %d, want 400", r.StatusCode)
	}

	// POST to the subtree id-path is not the observe verb => 405 (observe is GET-only).
	r, err = http.Post(ts.URL+"/v1/fak/trace/sess-9", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("POST trace observe status = %d, want 405", r.StatusCode)
	}
}

func TestTraceResetRouteValidationAndDisabled(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/fak/trace/reset", "application/json", strings.NewReader(`{"trace_id":"trace-1"}`))
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusNotFound {
		t.Fatalf("disabled trace reset status = %d, want 404", r.StatusCode)
	}

	srv.resetTrace = func(context.Context, string) error { return nil }
	r, err = http.Post(ts.URL+"/v1/fak/trace/reset", "application/json", strings.NewReader(`{"trace_id":" "}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusBadRequest {
		t.Fatalf("blank trace reset status = %d, want 400", r.StatusCode)
	}
}
