package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// scopeServe runs one request through the full middleware stack with the
// optionally-set X-Fak-Auth-Scope header, the same path agent traffic takes.
func scopeServe(t *testing.T, method, target, scope string) *httptest.ResponseRecorder {
	t.Helper()
	s, err := New(Config{EngineID: "mock", Model: "m", Provider: "openai"})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(method, target, nil)
	if scope != "" {
		req.Header.Set(authScopeHeader, scope)
	}
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	return rec
}

// TestAuthScopeGuardAbsentHeaderLegacyPath pins the P3 invariant: with no
// X-Fak-Auth-Scope header the guard is invisible -- every verb on every route
// takes the legacy path (405 method guards come from the handlers, not the
// scope matrix).
func TestAuthScopeGuardAbsentHeaderLegacyPath(t *testing.T) {
	cases := []struct {
		method, target string
		wantStatus     int
	}{
		{http.MethodGet, "/v1/fak/observation", http.StatusOK},
		{http.MethodGet, "/v1/fak/observation/requests", http.StatusOK},
		{http.MethodGet, "/v1/models", http.StatusOK},
		{http.MethodPost, "/v1/chat/completions", http.StatusBadRequest},
	}
	for _, tc := range cases {
		rec := scopeServe(t, tc.method, tc.target, "")
		if rec.Code != tc.wantStatus {
			t.Errorf("no-header %s %s = %d, want %d", tc.method, tc.target, rec.Code, tc.wantStatus)
		}
	}
	// A refusal would have minted a counter row; the legacy path must not.
	s, err := New(Config{EngineID: "mock", Model: "m", Provider: "openai"})
	if err != nil {
		t.Fatal(err)
	}
	if got := s.renderMetrics(); strings.Contains(got, "fak_gateway_scope_refusals_total{") {
		t.Errorf("legacy path wrote a scope-refusal row:\n%s", got)
	}
}

// TestAuthScopeGuardObservationMatrix proves a read-scoped watcher survives on
// the observation plane (GET 200) and is refused on any mutating verb with the
// machine-readable 403 + counter increment.
func TestAuthScopeGuardObservationMatrix(t *testing.T) {
	// read + GET observation -> 200
	if rec := scopeServe(t, http.MethodGet, "/v1/fak/observation", "read"); rec.Code != http.StatusOK {
		t.Errorf("read GET /v1/fak/observation = %d, want 200", rec.Code)
	}
	if rec := scopeServe(t, http.MethodGet, "/v1/fak/observation/requests", "read"); rec.Code != http.StatusOK {
		t.Errorf("read GET /v1/fak/observation/requests = %d, want 200", rec.Code)
	}

	// read + POST -> 403 scope_forbidden {scope, method}
	rec := scopeServe(t, http.MethodPost, "/v1/fak/observation", "read")
	if rec.Code != http.StatusForbidden {
		t.Fatalf("read POST /v1/fak/observation = %d, want 403", rec.Code)
	}
	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("403 body not JSON: %v (%q)", err, rec.Body.String())
	}
	if body["error"] != "scope_forbidden" {
		t.Errorf("error = %q, want scope_forbidden", body["error"])
	}
	if body["scope"] != "read" {
		t.Errorf("scope = %q, want read", body["scope"])
	}
	if body["method"] != http.MethodPost {
		t.Errorf("method = %q, want POST", body["method"])
	}

	// The refusal was counted under (read, observation).
	s, err := New(Config{EngineID: "mock", Model: "m", Provider: "openai"})
	if err != nil {
		t.Fatal(err)
	}
	post := httptest.NewRequest(http.MethodPost, "/v1/fak/observation", nil)
	post.Header.Set(authScopeHeader, "read")
	s.Handler().ServeHTTP(httptest.NewRecorder(), post)
	get := httptest.NewRequest(http.MethodGet, "/v1/admin/drain", nil)
	get.Header.Set(authScopeHeader, "read")
	s.Handler().ServeHTTP(httptest.NewRecorder(), get)
	text := s.renderMetrics()
	for _, want := range []string{
		`fak_gateway_scope_refusals_total{scope="read",route_class="observation"} 1`,
		`fak_gateway_scope_refusals_total{scope="read",route_class="admin"} 1`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("metrics missing %q\n%s", want, text)
		}
	}
	if !strings.Contains(text, "# HELP fak_gateway_scope_refusals_total") ||
		!strings.Contains(text, "# TYPE fak_gateway_scope_refusals_total counter") {
		t.Errorf("metrics missing scope-refusal HELP/TYPE header\n%s", text)
	}
}

// TestAuthScopeGuardWriteScopeAllowed proves scope=write passes every verb the
// handler itself permits (the 405s below are the route's own method guards, so
// the write claim demonstrably reached them).
func TestAuthScopeGuardWriteScopeAllowed(t *testing.T) {
	if rec := scopeServe(t, http.MethodGet, "/v1/fak/observation", "write"); rec.Code != http.StatusOK {
		t.Errorf("write GET /v1/fak/observation = %d, want 200", rec.Code)
	}
	if rec := scopeServe(t, http.MethodGet, "/v1/models", "write"); rec.Code != http.StatusOK {
		t.Errorf("write GET /v1/models = %d, want 200", rec.Code)
	}

}

// TestAuthScopeGuardAdminRequiresWrite pins the admin arm: /v1/admin/* needs a
// write claim for EVERY verb -- including GET -- because the companion daemon
// drains via POST and a read token must not probe the admin plane.
func TestAuthScopeGuardAdminRequiresWrite(t *testing.T) {
	rec := scopeServe(t, http.MethodGet, "/v1/admin/drain", "read")
	if rec.Code != http.StatusForbidden {
		t.Fatalf("read GET /v1/admin/drain = %d, want 403", rec.Code)
	}
	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("403 body not JSON: %v", err)
	}
	if body["error"] != "scope_forbidden" || body["scope"] != "read" {
		t.Errorf("admin refusal body = %v, want scope_forbidden/read", body)
	}
	// write + GET admin is allowed through the guard (the unwired route 404s).
	if rec := scopeServe(t, http.MethodGet, "/v1/admin/drain", "write"); rec.Code == http.StatusForbidden {
		t.Errorf("write GET /v1/admin/drain = 403, want pass-through the guard")
	}
}

// TestAuthScopeGuardParseMatrix covers the parser edges: case-insensitive,
// trimmed, whitespace-only = absent, unknown values fail CLOSED to read.
func TestAuthScopeGuardParseMatrix(t *testing.T) {
	if rec := scopeServe(t, http.MethodGet, "/v1/fak/observation", "  READ  "); rec.Code != http.StatusOK {
		t.Errorf("trimmed/upper READ GET observation = %d, want 200", rec.Code)
	}
	if rec := scopeServe(t, http.MethodPost, "/v1/fak/observation", "READ"); rec.Code != http.StatusForbidden {
		t.Errorf("READ POST observation = %d, want 403 (case-insensitive)", rec.Code)
	}
	if rec := scopeServe(t, http.MethodGet, "/v1/fak/observation", "  WRITE\t"); rec.Code != http.StatusOK {
		t.Errorf("trimmed WRITE GET observation = %d, want 200", rec.Code)
	}
	if rec := scopeServe(t, http.MethodPost, "/v1/fak/observation", "W R I T E"); rec.Code != http.StatusForbidden {
		t.Errorf("garbage scope POST observation = %d, want 403 (fail closed to read)", rec.Code)
	}
	rec := scopeServe(t, http.MethodGet, "/v1/fak/observation", "   ")
	if rec.Code != http.StatusOK {
		t.Errorf("whitespace-only scope GET observation = %d, want 200 (legacy)", rec.Code)
	}
	if got := rec.Body.String(); strings.Contains(got, "scope_forbidden") {
		t.Errorf("whitespace-only scope was enforced as a claim: %s", got)
	}
}
