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
