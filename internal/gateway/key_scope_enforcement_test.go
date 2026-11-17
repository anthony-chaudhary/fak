package gateway

// key_scope_enforcement_test.go — #14448: the scoped tenant keyset wired through
// gateway.New must ENFORCE the operator-plane gate in withAuth. A --key-principal
// tenant key defaults to the inference class: it reaches the inference wire and
// the tenant self-service surface, gets 403 scope_forbidden on the host-wide
// operator plane (admin, policy/route reload, revoke, control directives), while
// the operator bearer (--require-key-env) stays full-strength everywhere. These
// are the scope-matrix route tests the issue names as the witness.

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTenantScopeServer builds the exact shape `fak serve --require-key-env OP
// --key-principal acme=ACME` produces: one full-strength operator bearer plus
// one inference-class tenant key.
func newTenantScopeServer(t *testing.T) *Server {
	t.Helper()
	srv, err := New(Config{EngineID: "mock", Model: "m", RequireKey: "op-key", KeyPrincipals: map[string]string{"tenant-key": "acme"}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return srv
}

func tenantScopeRequest(t *testing.T, srv *Server, method, path, key string) (int, string) {
	t.Helper()
	r := httptest.NewRequest(method, path, nil)
	if key != "" {
		r.Header.Set("Authorization", "Bearer "+key)
	}
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, r)
	return w.Code, w.Body.String()
}

// TestWithAuthEnforcesTenantKeyScope is the scope-matrix route test: the tenant
// key -> 403 scope_forbidden on every operator-plane route, admitted on the
// inference wire and the tenant self-service surface; operator key -> past the
// gate on the very routes the tenant key drew 403 on; no key -> 401.
func TestWithAuthEnforcesTenantKeyScope(t *testing.T) {
	srv := newTenantScopeServer(t)

	refusals := []struct {
		name   string
		method string
		path   string
	}{
		{"tenant key on the policy reload verb", http.MethodPost, "/v1/fak/policy/reload"},
		{"tenant key on the route reload verb", http.MethodPost, "/v1/fak/route/reload"},
		{"tenant key on the control apply route", http.MethodPost, "/v1/control/apply"},
		{"tenant key on the control directive collection", http.MethodGet, "/v1/fak/control/directives"},
		{"tenant key on the revoke verb", http.MethodPost, "/v1/fak/revoke"},
		{"tenant key on the admin drain route", http.MethodGet, "/v1/admin/drain"},
	}
	for _, c := range refusals {
		code, body := tenantScopeRequest(t, srv, c.method, c.path, "tenant-key")
		if code != http.StatusForbidden {
			t.Errorf("%s: status = %d, want 403 (body: %s)", c.name, code, body)
		}
		if !strings.Contains(body, "scope_forbidden") {
			t.Errorf("%s: refusal body must carry scope_forbidden, got: %s", c.name, body)
		}
	}

	admitted := []struct {
		name   string
		method string
		path   string
	}{
		{"tenant key on the chat completions inference route", http.MethodPost, "/v1/chat/completions"},
		{"tenant key on the anthropic inference route", http.MethodPost, "/v1/messages"},
		{"tenant key on the model listing", http.MethodGet, "/v1/models"},
		{"tenant key on its own session surface", http.MethodGet, "/v1/fak/sessions"},
		{"tenant key on its own task surface", http.MethodGet, "/a2a/v1/tasks"},
	}
	for _, c := range admitted {
		code, body := tenantScopeRequest(t, srv, c.method, c.path, "tenant-key")
		if code == http.StatusUnauthorized || code == http.StatusForbidden {
			t.Errorf("%s: status = %d, want past the auth/scope gate (body: %s)", c.name, code, body)
		}
	}

	// The operator bearer stays full-strength: it passes the gate on the very
	// route the tenant key drew 403 on.
	code, body := tenantScopeRequest(t, srv, http.MethodPost, "/v1/fak/policy/reload", "op-key")
	if code == http.StatusUnauthorized || strings.Contains(body, "scope_forbidden") {
		t.Errorf("operator key on the admin route must pass the scope gate: status = %d, body: %s", code, body)
	}

	// No credential at all is still a plain 401, not a scope refusal.
	code, _ = tenantScopeRequest(t, srv, http.MethodPost, "/v1/fak/policy/reload", "")
	if code != http.StatusUnauthorized {
		t.Errorf("no key on the admin route: status = %d, want 401", code)
	}
}
