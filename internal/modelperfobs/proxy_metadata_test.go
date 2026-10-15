package modelperfobs

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fak-test:justify why=contract when=changed:internal/modelperfobs/**
// fak-test:runtime fast est=10ms
func TestProxyPersistsDeclaredCacheStateAndWarmupWithoutInferringAttribution(t *testing.T) {
	for _, tc := range []struct {
		name, declared, wantState, wantSource string
		warmup                                bool
		status                                int
	}{
		{"legacy default", "", "unknown", "unknown", false, 200},
		{"declared warmup", "warm", "warm", "declared", true, 200},
		{"declared steady", "steady", "steady", "declared", false, 200},
		{"failed cold request", "cold", "cold", "declared", false, 503},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend, err := ParseBackend("http://backend.test")
			if err != nil {
				t.Fatal(err)
			}
			ledger := filepath.Join(t.TempDir(), "observations.jsonl")
			var forwardedID string
			now := time.Unix(1700000000, 0)
			proxy := &Proxy{
				Backend: backend, Ledger: ledger, CacheState: tc.declared, Warmup: tc.warmup,
				Now: func() time.Time { now = now.Add(time.Millisecond); return now },
				Client: &http.Client{Transport: metadataRoundTripper(func(r *http.Request) (*http.Response, error) {
					forwardedID = r.Header.Get("X-Fak-Observation-ID")
					return &http.Response{StatusCode: tc.status, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(`{"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`))}, nil
				})},
			}
			response := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test-model"}`))
			proxy.ServeHTTP(response, request)
			if response.Code != tc.status {
				t.Fatalf("response status=%d want %d", response.Code, tc.status)
			}
			data, err := os.ReadFile(ledger)
			if err != nil {
				t.Fatal(err)
			}
			rows, err := ReadObservations(strings.NewReader(string(data)))
			if err != nil {
				t.Fatal(err)
			}
			if len(rows) != 1 {
				t.Fatalf("observations=%d want 1", len(rows))
			}
			got := rows[0]
			if got.CacheState != tc.wantState || got.CacheStateSource != tc.wantSource || got.Warmup != tc.warmup {
				t.Fatalf("cache state=%s source=%s warmup=%t", got.CacheState, got.CacheStateSource, got.Warmup)
			}
			if got.AttributionUnavailable != "backend-does-not-expose-dispatch-attribution" {
				t.Fatalf("API observer attribution unavailable=%q", got.AttributionUnavailable)
			}
			if forwardedID == "" || got.RequestID != forwardedID || response.Header().Get("X-Fak-Observation-ID") != forwardedID {
				t.Fatal("persisted observation lost the request correlation ID")
			}
		})
	}
}

type metadataRoundTripper func(*http.Request) (*http.Response, error)

func (f metadataRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
