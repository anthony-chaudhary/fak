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
