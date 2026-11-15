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
	"github.com/anthony-chaudhary/fak/internal/kvbudget"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// fak-test:runtime integration est=1s lane=default
// Serial: newTestServer and both live admission gates are global. Estimate is unmeasured.
func TestServedRateAdmissionPreservesLeaseLifecycle(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	priorRate, priorInflight := LiveRequestRateGate(), LiveInflightGate()
	restore := func() { SetRequestRateGate(priorRate); SetInflightGate(priorInflight) }
	t.Cleanup(restore)
	for _, tokenChain := range []bool{false, true} {
		for _, providerError := range []bool{false, true} {
			t.Run(fmt.Sprintf("served/token_chain=%v/provider_error=%v", tokenChain, providerError), func(t *testing.T) {
				t.Cleanup(restore)
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
					fmt.Fprint(w, `{"model":"rate-lifecycle-fixture","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
				}))
				defer upstream.Close()
				srv := newTestServer(t)
				srv.keyset = newKeyset(map[string]string{"fixture-key-a": "org-a"})
				srv.planner = &agent.HTTPPlanner{Provider: agent.ProviderOpenAI, BaseURL: upstream.URL, ModelID: "rate-lifecycle-fixture", Client: upstream.Client(), MaxTokens: 8}
				inflight := NewInflightGate(InflightPolicy{MaxInflight: 1, PrincipalSlots: 2})
				SetInflightGate(inflight)
				rate := NewRequestRateGate(RateBucketPolicy{RequestsPerSecond: 16, RequestBurst: 16, PrincipalSlots: 1})
				var now atomic.Int64
				now.Store(time.Unix(1, 0).UnixNano())
				rate.SetClock(func() time.Time { return time.Unix(0, now.Load()).UTC() })
				SetRequestRateGate(rate)
				if tokenChain {
					budget := TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 1000000, MaxConcurrent: 4, TargetUtilization: 1}}
					srv.SetTokenRateGate(NewTokenRateGate(budget))
					srv.SetPrincipalTokenRates(NewPrincipalTokenRates(budget, 2))
				}
				server := httptest.NewServer(srv.Handler())
				defer server.Close()
				for i := 0; i < 2; i++ {
					ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
					req, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/v1/chat/completions", strings.NewReader(`{"model":"rate-lifecycle-fixture","max_tokens":8,"messages":[{"role":"user","content":"hello"}]}`))
					if err != nil {
						cancel()
						t.Fatal(err)
					}
					req.Header.Set("Content-Type", "application/json")
					req.Header.Set("Authorization", "Bearer fixture-key-a")
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
						t.Fatalf("request %d status=%d upstream=%d, want status=%d upstream=%d: %s", i, res.StatusCode, hits.Load(), want, i+1, body)
					}
					if got := inflight.Snapshot().InFlight; got != 0 {
						t.Fatalf("completed request retained %d global inflight holds", got)
					}
					// An admitted request consumes one rate token even when its provider fails.
					if got, want := rate.BucketFor("org-a").Credit, int64(15-i)*rateTokenNs; got != want {
						t.Fatalf("rate credit=%d want=%d after request %d", got, want, i)
					}
				}
				// Once the bucket refills, a released carrier permits distinct-principal reuse.
				now.Add(int64(time.Second))
				next, err := rate.Take("org-b", 1)
				if err != nil || next == nil {
					t.Fatalf("released rate seat was not reusable: (%v,%v)", next, err)
				}
				next.Release()
			})
		}
	}
	for _, cancelled := range []bool{false, true} {
		t.Run(fmt.Sprintf("fixed_capacity_refusal/cancelled=%v", cancelled), func(t *testing.T) {
			t.Cleanup(restore)
			srv := newTestServer(t)
			SetInflightGate(nil)
			rate := NewRequestRateGate(RateBucketPolicy{RequestsPerSecond: 1, RequestBurst: 1, PrincipalSlots: 1})
			rate.SetClock(func() time.Time { return time.Unix(1, 0).UTC() })
			SetRequestRateGate(rate)
			policy := DefaultAdmissionPolicy()
			policy.MaxNumSeqs = 1
			srv.SetAdmissionController(NewAdmissionController(policy))
			srv.SetWarmupCapacity(kvbudget.WarmupCapacity{UsableBytes: 1, BytesPerToken: 1, FixedBytesPerStream: 2})
			ctx, cancel := context.WithCancel(WithPrincipal(context.Background(), "org-a"))
			defer cancel()
			if cancelled {
				cancel()
			}
			lease, err := srv.beginServedAdmission(ctx, servedSessionTurn{}, nil, nil, 1)
			var refusal *fixedWarmupCapacityError
			if lease != nil || (cancelled && !errors.Is(err, context.Canceled)) || (!cancelled && !errors.As(err, &refusal)) {
				t.Fatalf("fixed capacity refusal=(%v,%v), cancelled=%v", lease, err, cancelled)
			}
			if got := rate.BucketFor("org-a").Credit; got != rateTokenNs {
				t.Fatalf("refused request retained rate debit: credit=%d want=%d", got, int64(rateTokenNs))
			}
			next, err := rate.Take("org-b", 1)
			if err != nil || next == nil {
				t.Fatalf("refused request retained rate seat: (%v,%v)", next, err)
			}
			next.Release()
		})
	}
}
