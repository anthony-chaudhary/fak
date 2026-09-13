package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestTurnkeyReadyzGatesWhileWarming proves /readyz cannot claim ready while the
// boot warmup gate is armed-and-incomplete: it must return 503 with Retry-After
// and "warming_up", and only flip to 200 "ready" after markReady.
func TestTurnkeyReadyzGatesWhileWarming(t *testing.T) {
	srv := &turnkeyServer{ready: &readinessGate{}}
	srv.ready.armWarming()

	rec := httptest.NewRecorder()
	srv.handleReadyz(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("warming /readyz code = %d, want 503", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Fatalf("warming /readyz Retry-After = %q, want 1", got)
	}
	var warming struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &warming); err != nil {
		t.Fatal(err)
	}
	if warming.Status != "warming_up" {
		t.Fatalf("warming /readyz status = %q, want warming_up", warming.Status)
	}

	srv.ready.markReady()
	rec = httptest.NewRecorder()
	srv.handleReadyz(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("ready /readyz code = %d, want 200", rec.Code)
	}
	var ready struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &ready); err != nil {
		t.Fatal(err)
	}
	if ready.Status != "ready" {
		t.Fatalf("ready /readyz status = %q, want ready", ready.Status)
	}
}

// TestTurnkeyHealthzReportsReadinessState proves /healthz stays a 200 liveness
// signal but reports the truthful readiness state: warming -> ready:false,
// status:"warming_up"; ready -> ready:true, status:"ok".
func TestTurnkeyHealthzReportsReadinessState(t *testing.T) {
	srv := &turnkeyServer{ready: &readinessGate{}}
	srv.ready.armWarming()

	check := func(wantReady bool, wantStatus string) {
		t.Helper()
		rec := httptest.NewRecorder()
		srv.handleHealthz(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("healthz code = %d, want 200", rec.Code)
		}
		var got struct {
			Ready  bool   `json:"ready"`
			Status string `json:"status"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		if got.Ready != wantReady || got.Status != wantStatus {
			t.Fatalf("healthz = ready:%v status:%q, want ready:%v status:%q", got.Ready, got.Status, wantReady, wantStatus)
		}
	}

	check(false, "warming_up")
	srv.ready.markReady()
	check(true, "ok")
}
