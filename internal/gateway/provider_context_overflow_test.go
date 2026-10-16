package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// Provider counts are scalars in tiny JSON bodies, not allocation sizes.
// fak-test:runtime fast est=1s
func TestProviderContextOverflowEnforcesSessionBudget(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name        string
		provider    agent.Provider
		usage       agent.Usage
		wantContext int
	}{
		{"anthropic_read_overflow", agent.ProviderAnthropic, agent.Usage{PromptTokens: maxInt, CacheReadInputTokens: 1, TotalTokens: maxInt}, maxInt},
		{"anthropic_create_overflow", agent.ProviderAnthropic, agent.Usage{PromptTokens: maxInt, CacheCreationInputTokens: 1, TotalTokens: maxInt}, maxInt},
		{"anthropic_second_add_overflow", agent.ProviderAnthropic, agent.Usage{PromptTokens: maxInt - 1, CacheReadInputTokens: 1, CacheCreationInputTokens: 1, TotalTokens: maxInt - 1}, maxInt},
		{"deepseek_reconstruction_overflow", agent.ProviderOpenAI, agent.Usage{PromptCacheHitTokens: maxInt, PromptCacheMissTokens: 1}, maxInt},
		{"representable_boundary", agent.ProviderOpenAI, agent.Usage{PromptTokens: maxInt, TotalTokens: maxInt}, maxInt},
		{"ordinary_anthropic", agent.ProviderAnthropic, agent.Usage{PromptTokens: 2, CacheReadInputTokens: 3, CacheCreationInputTokens: 1, TotalTokens: 2}, 6},
		{"ordinary_openai", agent.ProviderOpenAI, agent.Usage{PromptTokens: 6, PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 3}, TotalTokens: 6}, 6},
		{"ordinary_deepseek", agent.ProviderOpenAI, agent.Usage{PromptTokens: 6, PromptCacheHitTokens: 4, PromptCacheMissTokens: 2, TotalTokens: 6}, 6},
		{"omitted_usage", agent.ProviderOpenAI, agent.Usage{}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var requests atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "application/json")
				if tc.provider == agent.ProviderAnthropic {
					if r.URL.Path != "/v1/messages" {
						t.Errorf("Anthropic path=%q", r.URL.Path)
					}
					_ = json.NewEncoder(w).Encode(map[string]any{
						"model": "test-model", "content": []any{map[string]any{"type": "text", "text": "ok"}}, "stop_reason": "end_turn",
						"usage": map[string]int{"input_tokens": tc.usage.PromptTokens, "output_tokens": 0, "cache_read_input_tokens": tc.usage.CacheReadInputTokens, "cache_creation_input_tokens": tc.usage.CacheCreationInputTokens},
					})
				} else {
					if r.URL.Path != "/chat/completions" {
						t.Errorf("OpenAI path=%q", r.URL.Path)
					}
					_ = json.NewEncoder(w).Encode(map[string]any{
						"model": "test-model", "choices": []any{map[string]any{"message": map[string]string{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}}, "usage": tc.usage,
					})
				}
			}))
			defer upstream.Close()
			planner, err := agent.NewProviderHTTPPlanner(string(tc.provider), upstream.URL, "test-model", "loopback-only")
			if err != nil {
				t.Fatal(err)
			}
			planner.Client = upstream.Client()
			srv := newTestServer(t)
			srv.planner = planner
			table := session.NewTable()
			const trace = "provider-context-budget"
			const budget = 10
			table.SetBudget(trace, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: budget})
			wireSessionTableForTest(srv, table)
			gateway := httptest.NewServer(srv.Handler())
			defer gateway.Close()
			post := func() (int, []byte) {
				t.Helper()
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				defer cancel()
				r, err := http.NewRequestWithContext(ctx, http.MethodPost, gateway.URL+"/v1/chat/completions", strings.NewReader(`{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`))
				if err != nil {
					t.Fatal(err)
				}
				r.Header.Set("Content-Type", "application/json")
				r.Header.Set("X-Trace-Id", trace)
				resp, err := gateway.Client().Do(r)
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
			if !reflect.DeepEqual(response.Usage, tc.usage) {
				t.Fatalf("raw provider usage changed: got=%+v want=%+v", response.Usage, tc.usage)
			}
			if got := response.Usage.ContextWindowTokens(); got != tc.wantContext {
				t.Errorf("context usage=%d want=%d", got, tc.wantContext)
			}
			state := table.Get(trace)
			exhausted := tc.wantContext >= budget
			if exhausted {
				if state.Run != session.Draining || state.Reason != session.ReasonBudgetContext || state.Budget.ContextTokensLeft != 0 || state.ContinuationID == "" {
					t.Errorf("positive provider usage escaped context budget: %+v", state)
				}
			} else if state.Budget.ContextTokensLeft != budget-tc.wantContext || state.Run != session.Running {
				t.Errorf("ordinary context accounting changed: %+v", state)
			}
			status, raw = post()
			if exhausted {
				if status != http.StatusConflict || requests.Load() != 1 || !strings.Contains(string(raw), session.ReasonBudgetContext) {
					t.Fatalf("overflow budget admitted another provider call: status=%d requests=%d body=%s", status, requests.Load(), raw)
				}
			} else if status != http.StatusOK || requests.Load() != 2 {
				t.Fatalf("ordinary second turn changed: status=%d requests=%d body=%s", status, requests.Load(), raw)
			}
		})
	}
}
