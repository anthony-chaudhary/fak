package gateway

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/pkg/gatewayauth"
)

// registerKeyProofTestABI establishes New's prerequisites after any earlier test
// reset the process-wide registry. These tests stay serial; no registry reset is
// needed, and server/lease callback cleanup remains owned by each test.
func registerKeyProofTestABI(t *testing.T) {
	t.Helper()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("mock", engine.MockEngine)
}

func TestHealthAuthProofRequiresKeyAndValidChallenge(t *testing.T) {
	const key = "configured-local-key"
	nonce := make([]byte, healthAuthNonceBytes)
	for i := range nonce {
		nonce[i] = byte(i + 1)
	}
	challenge := base64.StdEncoding.EncodeToString(nonce)

	registerKeyProofTestABI(t)
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

	registerKeyProofTestABI(t)
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

// fak-test:runtime fast est=100ms lane=default
func TestKeyProofHandlerPreservesLeaseAuthorization(t *testing.T) {
	const key = "configured-proof-test-key"
	registerKeyProofTestABI(t)
	keyed, err := New(Config{EngineID: "mock", Model: "test-model", RequireKey: key, AllowLAN: false})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(keyed.Close)
	handler := keyed.Handler()
	nonce := make([]byte, healthAuthNonceBytes)
	for i := range nonce {
		nonce[i] = byte(i + 1)
	}
	challenge := base64.StdEncoding.EncodeToString(nonce)
	mac := hmac.New(sha256.New, []byte(key))
	_, _ = mac.Write([]byte(healthAuthProofDomain))
	_, _ = mac.Write(nonce)
	wantProof := base64.StdEncoding.EncodeToString(mac.Sum(nil))

	for _, tc := range []struct {
		name, method, challenge string
		status                  int
	}{
		{"valid", http.MethodGet, challenge, http.StatusOK},
		{"missing challenge", http.MethodGet, "", http.StatusBadRequest},
		{"invalid base64", http.MethodGet, "not-base64", http.StatusBadRequest},
		{"short challenge", http.MethodGet, base64.StdEncoding.EncodeToString(nonce[:31]), http.StatusBadRequest},
		{"long challenge", http.MethodGet, base64.StdEncoding.EncodeToString(append(append([]byte(nil), nonce...), 0)), http.StatusBadRequest},
		{"POST", http.MethodPost, challenge, http.StatusMethodNotAllowed},
		{"HEAD", http.MethodHead, challenge, http.StatusMethodNotAllowed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, gatewayauth.KeyProofPath, nil)
			req.RemoteAddr = "127.0.0.1:12345"
			req.Header.Set(healthAuthChallengeHeader, tc.challenge)
			if len(req.Header.Values("Authorization")) != 0 {
				t.Fatal("proof request carries Authorization")
			}
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			if rec.Code != tc.status || rec.Header().Get("Cache-Control") != "no-store" {
				t.Fatalf("proof status=%d cache=%q, want %d/no-store", rec.Code, rec.Header().Get("Cache-Control"), tc.status)
			}
			want := ""
			if tc.status == http.StatusOK {
				want = wantProof
				if rec.Body.Len() != 0 {
					t.Fatalf("dedicated proof must not return readiness body: %q", rec.Body.String())
				}
			}
			if got := rec.Header().Get(healthAuthProofHeader); got != want {
				t.Fatalf("proof=%q, want %q", got, want)
			}
			if tc.status == http.StatusMethodNotAllowed && rec.Header().Get("Allow") != http.MethodGet {
				t.Fatalf("Allow=%q, want GET", rec.Header().Get("Allow"))
			}
		})
	}

	for _, path := range []string{gatewayauth.KeyProofPath + "/", gatewayauth.KeyProofPath + "/extra", gatewayauth.KeyProofPath + "-extra"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		req.Header.Set(healthAuthChallengeHeader, challenge)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized || rec.Header().Get(healthAuthProofHeader) != "" {
			t.Fatalf("proof exemption leaked to %s: status=%d", path, rec.Code)
		}
	}

	oldLeaseWrite := leaseWriteFn
	t.Cleanup(func() { SetLeaseWriteFunc(oldLeaseWrite) })
	leaseWrites := 0
	SetLeaseWriteFunc(func(_ context.Context, _ string, req LeaseWriteRequest) (LeaseWriteResult, error) {
		leaseWrites++
		return LeaseWriteResult{OK: false, ID: req.ID, Reason: "LEASE_HELD"}, nil
	})
	for _, op := range []string{"acquire", "renew", "release"} {
		for _, authorization := range []string{"", "Bearer wrong", "Bearer " + key} {
			before := leaseWrites
			req := httptest.NewRequest(http.MethodPost, "/v1/leases/"+op, strings.NewReader(`{"id":"proof-test"}`))
			req.RemoteAddr = "127.0.0.1:12345"
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(healthAuthChallengeHeader, challenge)
			if authorization != "" {
				req.Header.Set("Authorization", authorization)
			}
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			wantStatus, wantWrites := http.StatusUnauthorized, before
			if authorization == "Bearer "+key {
				wantStatus, wantWrites = http.StatusOK, before+1
			}
			if rec.Code != wantStatus || leaseWrites != wantWrites || rec.Header().Get(healthAuthProofHeader) != "" {
				t.Fatalf("lease %s: status=%d writes=%d, want %d/%d", op, rec.Code, leaseWrites, wantStatus, wantWrites)
			}
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestKeyProofHandlerWithoutConfiguredKey(t *testing.T) {
	registerKeyProofTestABI(t)
	unkeyed, err := New(Config{EngineID: "mock", Model: "test-model", AllowLAN: false})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(unkeyed.Close)
	req := httptest.NewRequest(http.MethodGet, gatewayauth.KeyProofPath, nil)
	req.Header.Set(healthAuthChallengeHeader, base64.StdEncoding.EncodeToString(make([]byte, healthAuthNonceBytes)))
	rec := httptest.NewRecorder()
	unkeyed.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable || rec.Header().Get(healthAuthProofHeader) != "" {
		t.Fatalf("unkeyed proof status=%d proof=%q", rec.Code, rec.Header().Get(healthAuthProofHeader))
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestKeyProofHandlerKeysetOnlyPreservesLeaseAuthorization(t *testing.T) {
	const principalKey = "configured-keyset-proof-test-key"
	registerKeyProofTestABI(t)
	keyed, err := New(Config{EngineID: "mock", Model: "test-model", RequireKey: "", KeyPrincipals: map[string]string{principalKey: "proof-test-org"}, AllowLAN: false})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(keyed.Close)
	if keyed.requireKey != "" || keyed.keyset == nil {
		t.Fatal("fixture must be a genuine keyset-only protected gateway")
	}
	handler := keyed.Handler()
	req := httptest.NewRequest(http.MethodGet, gatewayauth.KeyProofPath, nil)
	req.Header.Set(healthAuthChallengeHeader, base64.StdEncoding.EncodeToString(make([]byte, healthAuthNonceBytes)))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable || rec.Header().Get(healthAuthProofHeader) != "" {
		t.Fatalf("keyset-only proof status=%d proof=%q", rec.Code, rec.Header().Get(healthAuthProofHeader))
	}

	oldLeaseWrite := leaseWriteFn
	t.Cleanup(func() { SetLeaseWriteFunc(oldLeaseWrite) })
	leaseWrites := 0
	SetLeaseWriteFunc(func(_ context.Context, _ string, req LeaseWriteRequest) (LeaseWriteResult, error) {
		leaseWrites++
		return LeaseWriteResult{OK: false, ID: req.ID, Reason: "LEASE_HELD"}, nil
	})
	for _, op := range []string{"acquire", "renew", "release"} {
		for _, authorization := range []string{"", "Bearer wrong", "Bearer " + principalKey} {
			before := leaseWrites
			req := httptest.NewRequest(http.MethodPost, "/v1/leases/"+op, strings.NewReader(`{"id":"proof-test"}`))
			req.RemoteAddr = "127.0.0.1:12345"
			req.Header.Set("Content-Type", "application/json")
			if authorization != "" {
				req.Header.Set("Authorization", authorization)
			}
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			wantStatus, wantWrites := http.StatusUnauthorized, before
			if authorization == "Bearer "+principalKey {
				wantStatus, wantWrites = http.StatusOK, before+1
			}
			if rec.Code != wantStatus || leaseWrites != wantWrites || rec.Header().Get(healthAuthProofHeader) != "" {
				t.Fatalf("keyset-only lease %s: status=%d writes=%d, want %d/%d", op, rec.Code, leaseWrites, wantStatus, wantWrites)
			}
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestKeyProofHandlerClearsPreexistingProof(t *testing.T) {
	for _, tc := range []struct {
		name, method, path, challenge, key string
		status                             int
	}{
		{"missing challenge", http.MethodGet, gatewayauth.KeyProofPath, "", "test-key", http.StatusBadRequest},
		{"invalid challenge", http.MethodGet, gatewayauth.KeyProofPath, "not-base64", "test-key", http.StatusBadRequest},
		{"wrong method", http.MethodPost, gatewayauth.KeyProofPath, "", "test-key", http.StatusMethodNotAllowed},
		{"wrong path", http.MethodGet, gatewayauth.KeyProofPath + "/", "", "test-key", http.StatusNotFound},
		{"no key", http.MethodGet, gatewayauth.KeyProofPath, "", "", http.StatusServiceUnavailable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, tc.path, nil)
			req.Header.Set(healthAuthChallengeHeader, tc.challenge)
			rec := httptest.NewRecorder()
			rec.Header().Set(healthAuthProofHeader, "stale-proof")
			gatewayauth.ServeKeyProof(rec, req, tc.key)
			if rec.Code != tc.status || rec.Header().Get(healthAuthProofHeader) != "" {
				t.Fatalf("stale proof survived: status=%d proof=%q, want %d/no proof", rec.Code, rec.Header().Get(healthAuthProofHeader), tc.status)
			}
		})
	}
}
