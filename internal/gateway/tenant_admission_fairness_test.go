package gateway

// fak-test:runtime integration est=2s lane=gateway
// Two-tenant fairness witness for #14453: with the per-tenant request-rate, in-flight,
// and token-window gates wired PER SERVER onto a real served path, the over-budget
// tenant receives a 429 rate_limit_error while the other tenant's success rate is
// unchanged, and the per-tenant shed counters are exported on /metrics. The gates are
// Server state, not process-global: a second Server on the same process stays inert
// even while the first is saturated.

import (
	"context"
	"encoding/json"
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

func TestTenantAdmissionTwoTenantFairness(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")

	t.Run("rate_gate_http_two_tenant", func(t *testing.T) {
		var hits atomic.Int32
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			hits.Add(1)
			if r.URL.Path != "/chat/completions" || r.Method != http.MethodPost {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"model":"tenant-fairness-fixture","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
		}))
		defer upstream.Close()

		srv := newTestServer(t)
		srv.keyset = newKeyset(map[string]string{"fixture-key-a": "org-a", "fixture-key-b": "org-b"})
		srv.planner = &agent.HTTPPlanner{Provider: agent.ProviderOpenAI, BaseURL: upstream.URL, ModelID: "tenant-fairness-fixture", Client: upstream.Client(), MaxTokens: 8}
		// Burst 2 at a frozen clock: exactly two requests fit before the bucket empties.
		rg := NewRequestRateGate(RateBucketPolicy{RequestsPerSecond: 2, RequestBurst: 2})
		rg.SetClock(func() time.Time { return time.Unix(0, 0).UTC() })
		srv.SetRequestRateGate(rg)

		server := httptest.NewServer(srv.Handler())
		defer server.Close()
		post := func(key string) (int, string) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/v1/chat/completions", strings.NewReader(`{"model":"tenant-fairness-fixture","max_tokens":8,"messages":[{"role":"user","content":"hello"}]}`))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer "+key)
			res, err := server.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			body, _ := io.ReadAll(res.Body)
			return res.StatusCode, string(body)
		}

		// org-a spends its burst: two admits, then the 429 OpenAI-shaped rate_limit_error.
		for i := 0; i < 2; i++ {
			status, body := post("fixture-key-a")
			if status != http.StatusOK {
				t.Fatalf("org-a request %d status=%d, want 200: %s", i+1, status, body)
			}
		}
		status, body := post("fixture-key-a")
		if status != http.StatusTooManyRequests {
			t.Fatalf("org-a third request status=%d, want 429: %s", status, body)
		}
		var envelope struct {
			Error struct {
				Message string `json:"message"`
				Type    string `json:"type"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(body), &envelope); err != nil {
			t.Fatalf("429 body is not the OpenAI error envelope: %v\n%s", err, body)
		}
		if envelope.Error.Type != "rate_limit_error" {
			t.Fatalf("429 error.type=%q, want rate_limit_error", envelope.Error.Type)
		}
		if !strings.Contains(envelope.Error.Message, "request rate limit") {
			t.Fatalf("429 message=%q does not name the fired rate class", envelope.Error.Message)
		}

		// org-b is untouched by org-a's saturation: full success rate.
		for i := 0; i < 2; i++ {
			status, body := post("fixture-key-b")
			if status != http.StatusOK {
				t.Fatalf("org-b request %d status=%d, want 200 — a neighbour's saturation must not shed this tenant: %s", i+1, status, body)
			}
		}
		if got := hits.Load(); got != 4 {
			t.Fatalf("upstream hits=%d, want 4 — the shed request never reached the provider", got)
		}

		// Counters exported per tenant: only org-a was shed, once.
		sheds := rg.ShedCounts()
		if sheds["org-a"] != 1 {
			t.Fatalf("org-a shed counter=%d, want 1", sheds["org-a"])
		}
		if _, ok := sheds["org-b"]; ok {
			t.Fatalf("org-b holds a shed entry, want none: %v", sheds)
		}
		out := srv.renderMetrics()
		if !strings.Contains(out, "fak_tenant_admission_shed_total{tenant=\"org-a\",gate=\"rate\"} 1\n") {
			t.Fatalf("live /metrics missing the per-tenant rate shed row:\n%s", out)
		}

		// Per-SERVER isolation: a second Server on this process is inert even though
		// srv's gate is saturated (the pre-#14453 process-global would have leaked).
		plain := newTestServer(t)
		lease, err := plain.beginServedAdmission(context.Background(), servedSessionTurn{traceID: "tenant-isolation"}, nil, nil, 1)
		if err != nil || lease != nil {
			t.Fatalf("unwired-gate beginServedAdmission = (%v, %v), want the historical inert (nil, nil)", lease, err)
		}
	})

	t.Run("inflight_per_principal", func(t *testing.T) {
		srv := newTestServer(t)
		ig := NewInflightGate(InflightPolicy{PerPrincipal: 1})
		srv.SetInflightGate(ig)
		turn := servedSessionTurn{traceID: "tenant-inflight", maxTokens: 1}

		ctxA := WithPrincipal(context.Background(), "org-a")
		leaseA, err := srv.beginServedAdmission(ctxA, turn, nil, nil, 1)
		if err != nil || leaseA == nil {
			t.Fatalf("org-a beginServedAdmission = (%v, %v), want an admitted lease", leaseA, err)
		}
		_, err = srv.beginServedAdmission(ctxA, turn, nil, nil, 1)
		var ae *AdmissionError
		if !errors.As(err, &ae) || ae.Verdict != VerdictShed || !strings.HasPrefix(ae.Reason, "inflight per-key") {
			t.Fatalf("org-a saturated err = %v, want the per-principal (key) class named in the shed", err)
		}
		// org-b is unaffected: its own principal carrier still has room.
		ctxB := WithPrincipal(context.Background(), "org-b")
		leaseB, err := srv.beginServedAdmission(ctxB, turn, nil, nil, 1)
		if err != nil || leaseB == nil {
			t.Fatalf("org-b beginServedAdmission = (%v, %v), want admit — org-a's cap must not touch org-b", leaseB, err)
		}
		leaseA.Release()
		leaseB.Release()
		if got := ig.Snapshot().InFlight; got != 0 {
			t.Fatalf("inflight after both releases = %d, want 0", got)
		}
		if ig.ShedCounts()["org-a"] != 1 {
			t.Fatalf("org-a inflight shed counter=%d, want 1", ig.ShedCounts()["org-a"])
		}
		if !strings.Contains(srv.renderMetrics(), "fak_tenant_admission_shed_total{tenant=\"org-a\",gate=\"inflight\"} 1\n") {
			t.Fatalf("live /metrics missing the per-tenant inflight shed row:\n%s", srv.renderMetrics())
		}
	})

	t.Run("token_window_per_tenant", func(t *testing.T) {
		srv := newTestServer(t)
		book := NewPrincipalTokenRates(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxTotalTokens: 200}}, 0)
		srv.SetPrincipalTokenRates(book)
		turn := servedSessionTurn{traceID: "tenant-tokens", maxTokens: 60}
		msgs := []agent.Message{{Role: "user", Content: strings.Repeat("x", 196)}} // ~49-token estimate + 60 max_tokens = 109

		ctxA := WithPrincipal(context.Background(), "org-a")
		leaseA, err := srv.beginServedAdmission(ctxA, turn, msgs, nil, 60)
		if err != nil || leaseA == nil {
			t.Fatalf("org-a first beginServedAdmission = (%v, %v), want an admitted lease", leaseA, err)
		}
		_, err = srv.beginServedAdmission(ctxA, turn, msgs, nil, 60)
		var ae *AdmissionError
		if !errors.As(err, &ae) || ae.Verdict != VerdictShed || !strings.Contains(ae.Reason, `principal "org-a"`) {
			t.Fatalf("org-a over-cap err = %v, want the shed naming its own allotment", err)
		}
		// org-b reserves against its OWN allotment: org-a's spent window does not touch it.
		ctxB := WithPrincipal(context.Background(), "org-b")
		leaseB, err := srv.beginServedAdmission(ctxB, turn, msgs, nil, 60)
		if err != nil || leaseB == nil {
			t.Fatalf("org-b beginServedAdmission = (%v, %v), want admit — org-a's window must not starve org-b", leaseB, err)
		}
		leaseA.Release()
		leaseB.Release()
		if book.ShedCounts()[`principal "org-a"`] != 1 {
			t.Fatalf("org-a token shed counter=%d, want 1", book.ShedCounts()[`principal "org-a"`])
		}
		out := srv.renderMetrics()
		if !strings.Contains(out, "fak_tenant_admission_shed_total") || !strings.Contains(out, `,gate="token"} 1`) {
			t.Fatalf("live /metrics missing the per-tenant token shed row:\n%s", out)
		}
	})
}
