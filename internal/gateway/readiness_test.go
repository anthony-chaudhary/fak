package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestReadyzRequiresStartupAndReusesHealthState(t *testing.T) {
	srv := newAgentSessionsTestServer(t)

	before := httptest.NewRecorder()
	srv.Handler().ServeHTTP(before, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if before.Code != http.StatusServiceUnavailable {
		t.Fatalf("before MarkReady status = %d, want 503; body=%s", before.Code, before.Body.String())
	}
	var notReady map[string]any
	if err := json.Unmarshal(before.Body.Bytes(), &notReady); err != nil {
		t.Fatalf("decode not-ready response: %v", err)
	}
	if notReady["startup_ready"] != false || notReady["ok"] != false {
		t.Fatalf("not-ready response = %#v", notReady)
	}

	srv.MarkReady()
	after := httptest.NewRecorder()
	srv.Handler().ServeHTTP(after, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if after.Code != http.StatusOK {
		t.Fatalf("after MarkReady status = %d, want 200; body=%s", after.Code, after.Body.String())
	}
	var ready map[string]any
	if err := json.Unmarshal(after.Body.Bytes(), &ready); err != nil {
		t.Fatalf("decode ready response: %v", err)
	}
	if ready["startup_ready"] != true || ready["ok"] != true || ready["engine"] != "mock" {
		t.Fatalf("ready response = %#v", ready)
	}
}
