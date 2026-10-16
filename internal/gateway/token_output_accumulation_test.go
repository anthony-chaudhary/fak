package gateway

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// Counts are scalars in tiny HTTP bodies. Both initial requests reach the provider
// before either may settle, so a configured cap cannot prevent the overflow setup.
// fak-test:runtime fast est=1s
func TestProviderOutputAccumulationRefusesNextRequest(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	maxInt := int(^uint(0) >> 1)
	if int64(maxInt) != math.MaxInt64 {
		t.Skip("provider int cannot express int64 boundary on this host")
	}
	for _, tc := range []struct {
		name               string
		output             int
		limit, wantSettled int64
		wantStatus         int
	}{
		{"overflow", maxInt, 10, math.MaxInt64, http.StatusTooManyRequests},
		{"ordinary", 2, 10, 4, http.StatusOK},
		{"unlimited", maxInt, 0, math.MaxInt64, http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			initialRequests := []string{"first", "second"}
			const inputTokens, cachedTokens = 2, 1
			entered := make(chan struct{}, len(initialRequests))
			release := make(chan struct{})
			var unblock sync.Once
			var calls atomic.Int32
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				call := calls.Add(1)
				if r.URL.Path != "/chat/completions" {
					t.Errorf("provider path=%q", r.URL.Path)
				}
				_, _ = io.Copy(io.Discard, r.Body)
				if call <= int32(len(initialRequests)) {
					entered <- struct{}{}
					select {
					case <-release:
					case <-r.Context().Done():
						return
					}
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]string{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}}, "usage": agent.Usage{PromptTokens: inputTokens, CompletionTokens: tc.output, PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: cachedTokens}}})
			}))
			defer up.Close()
			planner, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), up.URL, "test-model", "")
			if err != nil {
				t.Fatal(err)
			}
			planner.Client = up.Client()
			srv := newTestServer(t)
			srv.planner = planner
			gate := NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxOutputTokens: tc.limit}})
			now := time.Unix(1700000000, 0)
			gate.now = func() time.Time { return now }
			gate.windowStart = now
			srv.SetTokenRateGate(gate)
			server := httptest.NewServer(srv.Handler())
			defer server.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			type result struct {
				status int
				body   []byte
				err    error
			}
			post := func() result {
				req, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/v1/chat/completions", strings.NewReader(`{"model":"test-model","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}`))
				if err != nil {
					return result{err: err}
				}
				req.Header.Set("Content-Type", "application/json")
				resp, err := server.Client().Do(req)
				if err != nil {
					return result{err: err}
				}
				defer resp.Body.Close()
				raw, err := io.ReadAll(resp.Body)
				return result{status: resp.StatusCode, body: raw, err: err}
			}
			results := make(chan result, len(initialRequests))
			var workers sync.WaitGroup
			defer func() { cancel(); unblock.Do(func() { close(release) }); workers.Wait() }()
			for range initialRequests {
				workers.Add(1)
				go func() { defer workers.Done(); results <- post() }()
			}
			for range initialRequests {
				select {
				case <-entered:
				case <-ctx.Done():
					t.Fatal("both requests did not reach the provider before settlement")
				}
			}
			if snap := gate.Snapshot(); snap.InFlight != len(initialRequests) || snap.Settled != (ratelimit.TokenUsage{}) || snap.Reserved.OutputTokens != int64(len(initialRequests)) {
				t.Fatalf("initial admission was not isolated: %+v", snap)
			}
			unblock.Do(func() { close(release) })
			for range initialRequests {
				var got result
				select {
				case got = <-results:
				case <-ctx.Done():
					t.Fatal("initial request did not finish")
				}
				if got.err != nil || got.status != http.StatusOK {
					t.Fatalf("initial response=%+v", got)
				}
				var response ChatResponse
				if err := json.Unmarshal(got.body, &response); err != nil {
					t.Fatal(err)
				}
				if response.Usage.PromptTokens != inputTokens || response.Usage.CompletionTokens != tc.output || response.Usage.TotalTokens != 0 || response.Usage.PromptTokensDetails == nil || response.Usage.PromptTokensDetails.CachedTokens != cachedTokens {
					t.Fatalf("raw provider usage changed: %+v", response.Usage)
				}
			}
			workers.Wait()
			settled := gate.Snapshot()
			// Exercise the actual next admission before asserting the captured counter,
			// so the causal baseline witnesses the reopened window, not just arithmetic.
			next := post()
			if next.err != nil {
				t.Fatal(next.err)
			}
			if next.status != tc.wantStatus {
				t.Fatalf("output accumulation reopened window: status=%d calls=%d settled=%+v body=%s", next.status, calls.Load(), settled, next.body)
			}
			want := ratelimit.TokenUsage{InputTokens: int64(len(initialRequests) * inputTokens), CachedInputTokens: int64(len(initialRequests) * cachedTokens), UncachedInputTokens: int64(len(initialRequests) * (inputTokens - cachedTokens)), OutputTokens: tc.wantSettled}
			if settled.InFlight != 0 || settled.Settled != want {
				t.Fatalf("output accumulation charge changed: got=%+v want=%+v", settled, want)
			}
			if tc.wantStatus == http.StatusTooManyRequests {
				if calls.Load() != int32(len(initialRequests)) || !strings.Contains(string(next.body), "scheduler_overloaded") || !strings.Contains(string(next.body), ratelimit.TokenCapOutputTokens) {
					t.Fatalf("output refusal identity changed: calls=%d body=%s", calls.Load(), next.body)
				}
				if after := gate.Snapshot(); after.InFlight != 0 || after.Settled != want {
					t.Fatalf("refused request changed settlement: %+v", after)
				}
			} else if calls.Load() != int32(len(initialRequests)+1) {
				t.Fatalf("healthy provider calls=%d", calls.Load())
			}
		})
	}
}
