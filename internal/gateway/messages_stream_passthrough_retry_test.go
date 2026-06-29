package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestAnthropicMessagesPassthroughStreamRateLimitedSurfacesNoBufferedDoubleHit pins the
// `fak guard -- claude` rate-limit behavior end-to-end: when the real Anthropic API is
// overloaded, the flagship streaming passthrough retries the 429 ITSELF (assertively,
// bounded by FAK_PLANNER_MAX_ATTEMPTS) and, once it has exhausted that budget, surfaces a
// real 429 + the upstream's Retry-After straight to the client — WITHOUT re-issuing the
// same request through the buffered fallback, which would double the load on a (commonly
// shared) upstream account on the very rate limit it just refused. The upstream-hit count
// proves there is no second, buffered burst.
func TestAnthropicMessagesPassthroughStreamRateLimitedSurfacesNoBufferedDoubleHit(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var hits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.Header().Set("Retry-After", "0") // honored as a zero wait => the test does not sleep
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "k", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	inbound := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	// The client sees the REAL rate-limit status (not a generic 502, and not a silently
	// degraded success), with the upstream's Retry-After echoed for its own backoff.
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("client status = %d, want 429", resp.StatusCode)
	}
	if got := resp.Header.Get("Retry-After"); got != "0" {
		t.Fatalf("Retry-After header = %q, want \"0\" (echoed from upstream)", got)
	}
	var env struct {
		Error struct {
			Code string `json:"code"`
			Type string `json:"type"`
		} `json:"error"`
	}
	body, _ := io.ReadAll(resp.Body)
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatalf("error envelope did not decode: %v (body %q)", err, body)
	}
	if env.Error.Code != "upstream_rate_limited" || env.Error.Type != "rate_limit_error" {
		t.Fatalf("envelope = (%q, %q), want (upstream_rate_limited, rate_limit_error)", env.Error.Code, env.Error.Type)
	}

	// The decisive assertion: exactly FAK_PLANNER_MAX_ATTEMPTS upstream hits. The streaming
	// passthrough retried twice and then surfaced the 429 — it did NOT fall through to the
	// buffered path and fire a second 2-attempt burst (which would make this 4).
	if got := atomic.LoadInt32(&hits); got != 2 {
		t.Fatalf("upstream hit %d times, want 2 (streaming retries only — no buffered double-hit)", got)
	}
}
