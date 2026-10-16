package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// The oversized count is one scalar in a small body; no allocation follows it.
// fak-test:runtime fast est=1s
func TestProviderTokenCapOverflowRefusesNextRequest(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	maxInt := int(^uint(0) >> 1)
	// TokenUsage is int64; the provider-facing int can express this witness on 64-bit hosts.
	if int64(maxInt) != int64(^uint64(0)>>1) {
		t.Skip("provider int cannot express int64 boundary on this host")
	}
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/chat/completions" {
			t.Errorf("upstream path=%q", r.URL.Path)
		}
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]string{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}}, "usage": agent.Usage{PromptTokens: maxInt, TotalTokens: maxInt}})
	}))
	defer upstream.Close()
	planner, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), upstream.URL, "test-model", "loopback-only")
	if err != nil {
		t.Fatal(err)
	}
	planner.Client = upstream.Client()
	srv := newTestServer(t)
	srv.planner = planner
	gate := NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxInputTokens: 10}})
	now := time.Unix(1700000000, 0)
	gate.now = func() time.Time { return now }
	gate.windowStart = now
	srv.SetTokenRateGate(gate)
	server := httptest.NewServer(srv.Handler())
	defer server.Close()
	post := func() (int, []byte) {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/v1/chat/completions", strings.NewReader(`{"model":"test-model","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := server.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		return resp.StatusCode, raw
	}
	status, raw := post()
	if status != http.StatusOK {
		t.Fatalf("first status=%d body=%s", status, raw)
	}
	var response ChatResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		t.Fatal(err)
	}
	if response.Usage.PromptTokens != maxInt || response.Usage.TotalTokens != maxInt {
		t.Fatalf("raw usage changed: %+v", response.Usage)
	}
	if snap := gate.Snapshot(); snap.InFlight != 0 || snap.Settled.InputTokens != int64(maxInt) || snap.Settled.OutputTokens != 0 {
		t.Fatalf("settlement not preserved: %+v", snap)
	}
	status, raw = post()
	if status != http.StatusTooManyRequests || calls.Load() != 1 || !strings.Contains(string(raw), "scheduler_overloaded") || !strings.Contains(string(raw), ratelimit.TokenCapInputTokens) {
		t.Fatalf("overflowed token cap reached provider: status=%d calls=%d body=%s", status, calls.Load(), raw)
	}
	if snap := gate.Snapshot(); snap.InFlight != 0 || snap.Settled.InputTokens != int64(maxInt) {
		t.Fatalf("refused request changed settled window: %+v", snap)
	}
}
