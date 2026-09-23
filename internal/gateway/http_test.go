package gateway

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHealthAuthProofRequiresKeyAndValidChallenge(t *testing.T) {
	const key = "configured-local-key"
	nonce := make([]byte, healthAuthNonceBytes)
	for i := range nonce {
		nonce[i] = byte(i + 1)
	}
	challenge := base64.StdEncoding.EncodeToString(nonce)

	keyed, err := New(Config{EngineID: "mock", Model: "test-model", RequireKey: key})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(keyed.Close)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	req.Header.Set(healthAuthChallengeHeader, challenge)
	keyed.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("keyed health status = %d, want 200", rec.Code)
	}
	mac := hmac.New(sha256.New, []byte(key))
	_, _ = mac.Write([]byte(healthAuthProofDomain))
	_, _ = mac.Write(nonce)
	wantProof := base64.StdEncoding.EncodeToString(mac.Sum(nil))
	if got := rec.Header().Get(healthAuthProofHeader); got != wantProof {
		t.Fatalf("health proof = %q, want HMAC for challenge", got)
	}

	for _, malformed := range []string{"", "not-base64", base64.StdEncoding.EncodeToString(nonce[:len(nonce)-1])} {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
		req.Header.Set(healthAuthChallengeHeader, malformed)
		keyed.Handler().ServeHTTP(rec, req)
		if got := rec.Header().Get(healthAuthProofHeader); got != "" {
			t.Fatalf("malformed challenge %q emitted proof %q", malformed, got)
		}
	}

	unkeyed, err := New(Config{EngineID: "mock", Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(unkeyed.Close)
	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/healthz", nil)
	req.Header.Set(healthAuthChallengeHeader, challenge)
	unkeyed.Handler().ServeHTTP(rec, req)
	if got := rec.Header().Get(healthAuthProofHeader); got != "" {
		t.Fatalf("unkeyed health emitted proof %q", got)
	}
}
