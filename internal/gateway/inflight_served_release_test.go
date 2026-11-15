package gateway

import (
	"context"
	"errors"
	"fmt"
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

// fak-test:runtime integration est=1s lane=default
// Serial: newTestServer and the live inflight gate are global. Estimate is unmeasured.
func TestServedInflightReleasesEveryDimension(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	prior := LiveInflightGate()
	t.Cleanup(func() { SetInflightGate(prior) })
	for _, dimension := range []string{"key", "principal"} {
		for _, tokenChain := range []bool{false, true} {
			for _, providerError := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/token_chain=%v/provider_error=%v", dimension, tokenChain, providerError), func(t *testing.T) {
					var hits atomic.Int32
					upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						hits.Add(1)
						if r.URL.Path != "/chat/completions" || r.Method != http.MethodPost {
							t.Errorf("unexpected upstream route: %s %s", r.Method, r.URL.Path)
							http.NotFound(w, r)
							return
						}
						w.Header().Set("Content-Type", "application/json")
						if providerError {
							w.WriteHeader(http.StatusInternalServerError)
							fmt.Fprint(w, `{"error":{"message":"fixture provider failure"}}`)
							return
						}
						fmt.Fprint(w, `{"model":"inflight-fixture","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
					}))
					defer upstream.Close()
					srv := newTestServer(t)
					srv.keyset = newKeyset(map[string]string{"fixture-key-a": "org-a", "fixture-key-b": "org-b"})
					srv.planner = &agent.HTTPPlanner{Provider: agent.ProviderOpenAI, BaseURL: upstream.URL, ModelID: "inflight-fixture", Client: upstream.Client(), MaxTokens: 8}
					policy := InflightPolicy{MaxInflight: 2, PrincipalSlots: 2, PerKey: map[string]int{"org-a": 1}}
					keys := []string{"fixture-key-a", "fixture-key-a"}
					if dimension == "principal" {
						policy = InflightPolicy{MaxInflight: 2, PrincipalSlots: 1}
						keys = []string{"fixture-key-a", "fixture-key-b", "fixture-key-a"}
					}
					gate := NewInflightGate(policy)
					SetInflightGate(gate)
					t.Cleanup(func() { SetInflightGate(prior) })
					if tokenChain {
						budget := TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 100000, MaxConcurrent: 4, TargetUtilization: 1}}
						srv.SetTokenRateGate(NewTokenRateGate(budget))
						srv.SetPrincipalTokenRates(NewPrincipalTokenRates(budget, 2))
					}
					server := httptest.NewServer(srv.Handler())
					defer server.Close()
					for i, key := range keys {
						ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
						req, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/v1/chat/completions", strings.NewReader(`{"model":"inflight-fixture","max_tokens":8,"messages":[{"role":"user","content":"hello"}]}`))
						if err != nil {
							cancel()
							t.Fatal(err)
						}
						req.Header.Set("Content-Type", "application/json")
						req.Header.Set("Authorization", "Bearer "+key)
						res, err := server.Client().Do(req)
						if err != nil {
							cancel()
							t.Fatal(err)
						}
						body, readErr := io.ReadAll(res.Body)
						res.Body.Close()
						cancel()
						if readErr != nil {
							t.Fatal(readErr)
						}
						want := http.StatusOK
						if providerError {
							want = http.StatusBadGateway
						}
						if res.StatusCode != want || hits.Load() != int32(i+1) {
							t.Fatalf("request %d status=%d upstream=%d, want status=%d upstream=%d; released %s budget must be reusable: %s", i, res.StatusCode, hits.Load(), want, i+1, dimension, body)
						}
						if got := gate.Snapshot().InFlight; got != 0 {
							t.Fatalf("global holds after completed request=%d, want 0", got)
						}
					}
				})
			}
		}
	}
	t.Run("later_gate_refusal_rolls_back", func(t *testing.T) {
		srv := newTestServer(t)
		gate := NewInflightGate(InflightPolicy{MaxInflight: 1, PrincipalSlots: 1, PerKey: map[string]int{"org-a": 1}})
		SetInflightGate(gate)
		t.Cleanup(func() { SetInflightGate(prior) })
		srv.SetTokenRateGate(NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 1}}))
		turn := servedSessionTurn{traceID: "inflight-rollback", maxTokens: 8}
		ctx := WithPrincipal(context.Background(), "org-a")
		lease, err := srv.beginServedAdmission(ctx, turn, nil, nil, 8)
		var refusal *AdmissionError
		if lease != nil || !errors.As(err, &refusal) || refusal.Verdict != VerdictShed {
			t.Fatalf("later budget refusal=(%v,%v), want nil lease and shed", lease, err)
		}
		if got := gate.Snapshot().InFlight; got != 0 {
			t.Fatalf("refused request retained %d global holds", got)
		}
		// The later gate's refusal must leave the same key and principal reusable.
		srv.SetTokenRateGate(nil)
		lease, err = srv.beginServedAdmission(ctx, turn, nil, nil, 8)
		if err != nil || lease == nil {
			t.Fatalf("retry after refusal=(%v,%v), want admission", lease, err)
		}
		lease.SettleUsage(agent.Usage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3})
		lease.Release()
		lease.Release()
		if got := gate.Snapshot().InFlight; got != 0 {
			t.Fatalf("settle then repeated release retained %d global holds", got)
		}
	})
}
