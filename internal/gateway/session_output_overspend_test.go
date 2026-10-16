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

// The provider reports ordinary scalar counts. The session budget is a
// post-turn accounting boundary, not an estimate of this response's output.
// fak-test:runtime fast est=1s
func TestSessionOutputOverspendRefusesNextRequest(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	for _, tc := range []struct {
		name                     string
		budget, charge, wantLeft int
	}{
		{"overspend", 5, 6, 0},
		{"exact", 5, 5, 0},
		{"ordinary", 5, 2, 3},
		{"unbounded", session.Unbounded, 6, session.Unbounded},
		{"negative_unbounded", -5, 6, -5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var requests atomic.Int32
			usage := agent.Usage{PromptTokens: 2, CompletionTokens: tc.charge, TotalTokens: 2 + tc.charge}
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				_, _ = io.Copy(io.Discard, r.Body)
				if r.URL.Path != "/chat/completions" {
					t.Errorf("provider path=%q", r.URL.Path)
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{
					"model": "test-model", "choices": []any{map[string]any{"message": map[string]string{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}}, "usage": usage,
				})
			}))
			defer upstream.Close()
			planner, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), upstream.URL, "test-model", "loopback-only")
			if err != nil {
				t.Fatal(err)
			}
			planner.Client = upstream.Client()
			srv := newTestServer(t)
			srv.planner = planner
			table := session.NewTable()
			const trace = "session-output-overspend"
			table.SetBudget(trace, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: tc.budget})
			wireSessionTableForTest(srv, table)
			gateway := httptest.NewServer(srv.Handler())
			defer gateway.Close()
			post := func() (int, []byte) {
				t.Helper()
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				defer cancel()
				req, err := http.NewRequestWithContext(ctx, http.MethodPost, gateway.URL+"/v1/chat/completions", strings.NewReader(`{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":6}`))
				if err != nil {
					t.Fatal(err)
				}
				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("X-Trace-Id", trace)
				resp, err := gateway.Client().Do(req)
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
			if !reflect.DeepEqual(response.Usage, usage) {
				t.Errorf("raw provider usage changed: got=%+v want=%+v", response.Usage, usage)
			}
			if len(response.Choices) != 1 || response.Choices[0].Message.Content != "ok" {
				t.Errorf("first answer changed: %+v", response.Choices)
			}
			state := table.Get(trace)
			if state.Budget.TokensLeft != tc.wantLeft || state.Budget.TurnsLeft != session.Unbounded || state.Budget.ContextTokensLeft != 0 || state.Run != session.Running || state.ContinuationID != "" {
				t.Errorf("output debit changed budget or next-boundary timing: %+v", state)
			}
			// The second actual request is sent even when the first balance assertion
			// fails, so the causal baseline exposes the extra provider invocation.
			status, raw = post()
			if tc.wantLeft == 0 {
				if status != http.StatusConflict || requests.Load() != 1 || !strings.Contains(string(raw), session.ReasonBudgetTokens) {
					t.Fatalf("finite output overspend reopened session: budget=%d charge=%d status=%d provider_requests=%d body=%s", tc.budget, tc.charge, status, requests.Load(), raw)
				}
			} else if status != http.StatusOK || requests.Load() != 2 {
				t.Fatalf("ordinary or unbounded session refused: status=%d provider_requests=%d body=%s", status, requests.Load(), raw)
			}
		})
	}
}
