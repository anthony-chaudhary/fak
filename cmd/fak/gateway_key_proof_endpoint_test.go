package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Independent TEST plane uses only the unchanged existing client wrapper.
// No proposed endpoint constant or new SDK symbol is needed to compile parent.
const gatewayProofEndpointTestKey = "independent-generic-gateway-proof-key"
const gatewayProofEndpointTestBudget = 150 * time.Millisecond
const gatewayProofEndpointTestPath = "/v1/fak/key-proof"

func gatewayProofEndpointTestAnswer(t *testing.T, w http.ResponseWriter, r *http.Request, wrong bool) {
	t.Helper()
	if r.Header.Get("Authorization") != "" {
		t.Error("credential was sent before proof of possession")
	}
	nonce, err := base64.StdEncoding.DecodeString(r.Header.Get("X-Fak-Auth-Challenge"))
	if err != nil || len(nonce) != 32 {
		t.Error("client omitted a valid fresh32 standard-base64 challenge")
		http.Error(w, "bad challenge", http.StatusBadRequest)
		return
	}
	key := gatewayProofEndpointTestKey
	if wrong {
		key = "independent-unrelated-key"
	}
	mac := hmac.New(sha256.New, []byte(key))
	_, _ = mac.Write([]byte("fak-health-v1\x00"))
	_, _ = mac.Write(nonce)
	w.Header().Set("X-Fak-Auth-Proof", base64.StdEncoding.EncodeToString(mac.Sum(nil)))
	w.Header().Set("Content-Type", "application/json")
	if r.URL.Path == gatewayProofEndpointTestPath {
		_, _ = io.WriteString(w, `{"mode":"key_possession","readiness_evaluated":false}`)
	} else {
		_, _ = io.WriteString(w, `{"ok":true,"model":"generic-test-model"}`)
	}
}

// fak-test:runtime fast est=1s
// Suites: TestGatewayKeyProofEndpoint focused and cmd/fak/full.
func TestGatewayKeyProofEndpointPrefersDedicatedOverBlockedHealth(t *testing.T) {
	var dedicated, legacy atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case gatewayProofEndpointTestPath:
			dedicated.Add(1)
			gatewayProofEndpointTestAnswer(t, w, r, false)
		case "/healthz":
			legacy.Add(1)
			<-r.Context().Done()
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	client := server.Client()
	client.Timeout = gatewayProofEndpointTestBudget
	if !probeRouterAuthProof(client, server.URL, gatewayProofEndpointTestKey) {
		t.Error("dedicated possession proof rejected while legacy readiness was blocked")
	}
	if dedicated.Load() != 1 || legacy.Load() != 0 {
		t.Errorf("proof calls dedicated=%d legacy=%d, want1/0", dedicated.Load(), legacy.Load())
	}
}

// fak-test:runtime fast est=1s
// Suites: TestGatewayKeyProofEndpoint focused and cmd/fak/full.
func TestGatewayKeyProofEndpointFallsBackOnlyWhenUnsupported(t *testing.T) {
	for _, status := range []int{http.StatusNotFound, http.StatusMethodNotAllowed, http.StatusNotImplemented} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var mu sync.Mutex
			var paths []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				paths = append(paths, r.URL.Path)
				mu.Unlock()
				if r.Header.Get("Authorization") != "" {
					t.Error("fallback sent bearer")
				}
				switch r.URL.Path {
				case gatewayProofEndpointTestPath:
					w.WriteHeader(status)
				case "/healthz":
					gatewayProofEndpointTestAnswer(t, w, r, false)
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			client := server.Client()
			client.Timeout = gatewayProofEndpointTestBudget
			if !probeRouterAuthProof(client, server.URL, gatewayProofEndpointTestKey) {
				t.Error("legacy producer fallback proof rejected")
			}
			mu.Lock()
			defer mu.Unlock()
			if len(paths) != 2 || paths[0] != gatewayProofEndpointTestPath || paths[1] != "/healthz" {
				t.Errorf("fallback path order=%v, want dedicated then healthz", paths)
			}
		})
	}
}

// fak-test:runtime fast est=1s
// Suites: TestGatewayKeyProofEndpoint focused and cmd/fak/full.
func TestGatewayKeyProofEndpointRejectsWithoutLegacyEscape(t *testing.T) {
	for _, tc := range []struct {
		name         string
		status       int
		wrong, stall bool
	}{
		{name: "invalid-proof", status: http.StatusOK, wrong: true},
		{name: "unauthorized", status: http.StatusUnauthorized},
		{name: "forbidden", status: http.StatusForbidden},
		{name: "server-error", status: http.StatusInternalServerError},
		{name: "timeout", stall: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var dedicated, legacy atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case gatewayProofEndpointTestPath:
					dedicated.Add(1)
					if tc.stall {
						<-r.Context().Done()
						return
					}
					if tc.status == http.StatusOK {
						gatewayProofEndpointTestAnswer(t, w, r, tc.wrong)
						return
					}
					w.WriteHeader(tc.status)
				case "/healthz":
					legacy.Add(1)
					gatewayProofEndpointTestAnswer(t, w, r, false)
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			client := server.Client()
			client.Timeout = gatewayProofEndpointTestBudget
			if probeRouterAuthProof(client, server.URL, gatewayProofEndpointTestKey) {
				t.Error("failed dedicated proof escaped through legacy valid proof")
			}
			if dedicated.Load() != 1 || legacy.Load() != 0 {
				t.Errorf("failed proof paths dedicated=%d legacy=%d, want1/0", dedicated.Load(), legacy.Load())
			}
		})
	}
}

// fak-test:runtime fast est=1s
// Suites: TestGatewayKeyProofEndpoint focused and cmd/fak/full.
func TestGatewayKeyProofEndpointRedirectCannotBorrowProof(t *testing.T) {
	var redirected, legacy atomic.Int64
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		redirected.Add(1)
		gatewayProofEndpointTestAnswer(t, w, r, false)
	}))
	defer target.Close()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == gatewayProofEndpointTestPath {
			http.Redirect(w, r, target.URL+gatewayProofEndpointTestPath, http.StatusTemporaryRedirect)
			return
		}
		if r.URL.Path == "/healthz" {
			legacy.Add(1)
			gatewayProofEndpointTestAnswer(t, w, r, false)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()
	client := server.Client()
	client.Timeout = gatewayProofEndpointTestBudget
	if probeRouterAuthProof(client, server.URL, gatewayProofEndpointTestKey) {
		t.Error("redirected origin borrowed another origin proof")
	}
	if redirected.Load() != 0 || legacy.Load() != 0 {
		t.Errorf("redirected=%d legacy=%d, want0/0", redirected.Load(), legacy.Load())
	}
}

// fak-test:runtime fast est=1s
// Suites: TestGatewayKeyProofEndpoint focused and cmd/fak/full.
func TestGatewayKeyProofEndpointFallbackSharesOneDeadline(t *testing.T) {
	var dedicated, legacy, legacyCompleted atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case gatewayProofEndpointTestPath:
			dedicated.Add(1)
			select {
			case <-time.After(90 * time.Millisecond):
				w.WriteHeader(http.StatusNotFound)
			case <-r.Context().Done():
			}
		case "/healthz":
			legacy.Add(1)
			select {
			case <-time.After(90 * time.Millisecond):
				legacyCompleted.Add(1)
				gatewayProofEndpointTestAnswer(t, w, r, false)
			case <-r.Context().Done():
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	client := server.Client()
	client.Timeout = gatewayProofEndpointTestBudget
	started := time.Now()
	if probeRouterAuthProof(client, server.URL, gatewayProofEndpointTestKey) {
		t.Error("90ms dedicated plus90ms legacy completed despite one150ms total deadline")
	}
	// This wall upper bound absorbs post-deadline goroutine scheduling only.
	// Acceptance still requires rejection of the180ms response and cancellation
	// before the legacy response, so it does not widen the150ms auth budget.
	if elapsed := time.Since(started); elapsed > 200*time.Millisecond {
		t.Errorf("shared-deadline call took %v, want cancellation near150ms", elapsed)
	}
	if dedicated.Load() != 1 || legacy.Load() != 1 || legacyCompleted.Load() != 0 {
		t.Errorf("deadline calls dedicated=%d legacy=%d completed=%d, want1/1/0", dedicated.Load(), legacy.Load(), legacyCompleted.Load())
	}
}
