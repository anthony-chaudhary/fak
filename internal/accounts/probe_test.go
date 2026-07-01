package accounts

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProbeTokenParsesIdentity(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer sk-ant-oat-test" {
			t.Errorf("missing/wrong bearer: %q", got)
		}
		if got := r.Header.Get("anthropic-beta"); got != oauthBeta {
			t.Errorf("missing beta header: %q", got)
		}
		w.Write([]byte(`{"account":{"uuid":"u-123","email":"who@example.test","full_name":"Who"}}`))
	}))
	defer srv.Close()

	id, err := ProbeToken(srv.Client(), srv.URL, "sk-ant-oat-test")
	if err != nil {
		t.Fatalf("probe: %v", err)
	}
	if id.Email != "who@example.test" || id.AccountUUID != "u-123" || id.FullName != "Who" {
		t.Fatalf("identity = %+v", id)
	}
}

func TestProbeTokenFallbackEmailField(t *testing.T) {
	// Some responses carry email_address instead of email.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"account":{"uuid":"u-9","email_address":"alt@example.test"}}`))
	}))
	defer srv.Close()
	id, err := ProbeToken(srv.Client(), srv.URL, "sk-ant-oat-x")
	if err != nil {
		t.Fatalf("probe: %v", err)
	}
	if id.Email != "alt@example.test" {
		t.Fatalf("email fallback failed: %+v", id)
	}
}

func TestProbeTokenNon2xxIsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()
	if _, err := ProbeToken(srv.Client(), srv.URL, "sk-ant-oat-bad"); err == nil {
		t.Fatalf("a 401 should be an error (token does not work)")
	}
}

func TestProbeTokenEmptyToken(t *testing.T) {
	if _, err := ProbeToken(nil, "", ""); err == nil {
		t.Fatalf("empty token should error")
	}
}

func TestIsScopeError(t *testing.T) {
	// The real body a `claude setup-token` credential draws from the profile endpoint: a valid,
	// serveable token that just lacks the profile scope. This must classify as a scope error so
	// enrollment treats it as "identity pending", not a bad-token failure.
	scopeBody := `{"type":"error","error":{"type":"permission_error","message":"OAuth token does not meet scope requirement any_of(user:profile, user:office)"}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		w.Write([]byte(scopeBody))
	}))
	defer srv.Close()
	_, err := ProbeToken(srv.Client(), srv.URL, "sk-ant-oat-scoped")
	if err == nil {
		t.Fatalf("a 403 should be an error")
	}
	if !IsScopeError(err) {
		t.Errorf("IsScopeError(scope-403) = false, want true; err=%v", err)
	}

	// A plain 401 (genuinely bad token) is NOT a scope error — it must stay in the loud path.
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer bad.Close()
	_, err = ProbeToken(bad.Client(), bad.URL, "sk-ant-oat-bad")
	if IsScopeError(err) {
		t.Errorf("IsScopeError(401) = true, want false")
	}
	if IsScopeError(nil) {
		t.Errorf("IsScopeError(nil) = true, want false")
	}
}
