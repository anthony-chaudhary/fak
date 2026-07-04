package macbench

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRunDecodeLongUsesBearerAndReportsTokPerSecond(t *testing.T) {
	var sawAuth bool
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			_, _ = w.Write([]byte(`{"ok":true,"engine":"metal","planner":"inkernel","model":"qwen3.6-27b"}`))
		case "/v1/chat/completions":
			if r.Header.Get("Authorization") == "Bearer secret" {
				sawAuth = true
			}
			_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":31,"completion_tokens":64,"total_tokens":95}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	rep, err := Run(context.Background(), Options{
		Gateway:      ts.URL + "/v1",
		Model:        "qwen3.6-27b",
		Key:          "secret",
		Suite:        SuiteDecodeLong,
		DecodeTokens: []int{64},
		HTTPClient:   ts.Client(),
		Now:          func() time.Time { return time.Date(2026, 7, 4, 6, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !sawAuth {
		t.Fatal("chat request did not carry the bearer")
	}
	if rep.Schema != Schema || !strings.HasPrefix(rep.Gateway, "http://127.0.0.1:") || !rep.Health.OK {
		t.Fatalf("unexpected report header: %+v", rep)
	}
	if len(rep.Rows) != 1 {
		t.Fatalf("rows=%d, want 1", len(rep.Rows))
	}
	row := rep.Rows[0]
	if row.CompletionTokens != 64 || row.TokensPerSecond <= 0 || !strings.Contains(row.Headline, "tok/s") {
		t.Fatalf("bad row: %+v", row)
	}
	b, _ := json.Marshal(rep)
	if strings.Contains(string(b), "secret") {
		t.Fatalf("report leaked bearer: %s", b)
	}
}

func TestRunPrefillSweepParsesSSEUsageAndTTFT(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			_, _ = w.Write([]byte(`{"ok":true}`))
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"length\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":128,\"completion_tokens\":16,\"total_tokens\":144}}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	rep, err := Run(context.Background(), Options{
		Gateway:       ts.URL,
		Suite:         SuitePrefillSweep,
		PrefillTokens: []int{128},
		HTTPClient:    ts.Client(),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(rep.Rows) != 1 {
		t.Fatalf("rows=%d, want 1", len(rep.Rows))
	}
	row := rep.Rows[0]
	if row.PromptTokens != 128 || row.CompletionTokens != 16 || row.TTFTSeconds <= 0 || row.PrefillTokensPerSecond <= 0 {
		t.Fatalf("bad prefill row: %+v", row)
	}
}

func TestSanitizeGatewayForReportKeepsLoopbackOnly(t *testing.T) {
	if got := SanitizeGatewayForReport("http://127.0.0.1:8080"); got != "http://127.0.0.1:8080" {
		t.Fatalf("loopback sanitize = %q", got)
	}
	if got := SanitizeGatewayForReport("http://100.64.0.1:8080"); got != "<remote-gateway>" {
		t.Fatalf("remote sanitize = %q", got)
	}
}

func TestRemoteGatewayErrorTextIsSanitized(t *testing.T) {
	rep, err := Run(context.Background(), Options{
		Gateway: "http://example.invalid:8080",
		Suite:   SuiteHealth,
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, errors.New(`Get "http://example.invalid:8080/healthz": context deadline exceeded`)
		})},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	joined := rep.Health.Error + "\n" + strings.Join(rep.Errors, "\n")
	if strings.Contains(joined, "example.invalid") || !strings.Contains(joined, "<remote-gateway>/healthz") {
		t.Fatalf("gateway was not sanitized in errors: %q", joined)
	}
}

func TestElapsedSecondsFloorsNonPositiveDurations(t *testing.T) {
	got := elapsedSeconds(time.Now().Add(time.Second))
	if got != 0.001 {
		t.Fatalf("elapsedSeconds future start = %v, want 0.001", got)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
