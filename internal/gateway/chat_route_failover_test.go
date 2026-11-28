package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func newOpenAIUpstreamMock(t *testing.T, status int, replyText string, seenModels *[]string, hits *atomic.Int32) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits != nil {
			hits.Add(1)
		}
		var body struct {
			Model string `json:"model"`
		}
		if raw, err := io.ReadAll(r.Body); err == nil {
			_ = json.Unmarshal(raw, &body)
		}
		if seenModels != nil {
			*seenModels = append(*seenModels, body.Model)
		}
		if status != http.StatusOK {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			_, _ = w.Write([]byte(`{"error":{"message":"upstream failed"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"model":   body.Model,
			"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": replyText}, "finish_reason": "stop"}},
			"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
		})
	}))
	t.Cleanup(ts.Close)
	return ts
}

func newTestFailoverGateway(t *testing.T, roster *modelroute.Roster, tracker *routeCircuitTracker) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("mock", echoEngine{})
	srv, err := New(Config{
		EngineID:      "mock",
		Model:         "boot-model",
		RouteAccounts: roster,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if tracker != nil {
		srv.routeCircuit = tracker
	}
	t.Cleanup(srv.Close)
	return srv
}

func postChatCompletion(t *testing.T, handler http.Handler, model string) *httptest.ResponseRecorder {
	t.Helper()
	payload, _ := json.Marshal(map[string]any{
		"model":    model,
		"messages": []map[string]any{{"role": "user", "content": "hello"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	return rec
}

// fak-test:runtime fast est=50ms
func TestFailoverOn503And429UsesFallbackUpstreamModel(t *testing.T) {
	for _, status := range []int{http.StatusServiceUnavailable, http.StatusTooManyRequests} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var primaryModels, fallbackModels []string
			var primaryHits, fallbackHits atomic.Int32

			primary := newOpenAIUpstreamMock(t, status, "", &primaryModels, &primaryHits)
			fallback := newOpenAIUpstreamMock(t, http.StatusOK, "from-fallback", &fallbackModels, &fallbackHits)

			t.Setenv("TEST_FB_KEY", "sk-fb")
			roster := &modelroute.Roster{
				Version: modelroute.RosterVersion,
				Accounts: []modelroute.Account{
					{ID: "local-gpu", Kind: modelroute.KindLocal, BaseURL: primary.URL + "/v1"},
					{ID: "cloud-fb", Kind: modelroute.KindOpenAI, BaseURL: fallback.URL + "/v1", CredEnv: "TEST_FB_KEY"},
				},
				Bindings: []modelroute.Binding{
					{
						Model:         "default",
						Account:       "local-gpu",
						UpstreamModel: "local-qwen",
						Fallbacks: []modelroute.FallbackTarget{
							{Account: "cloud-fb", UpstreamModel: "gpt-4o-mini"},
						},
					},
				},
			}

			srv := newTestFailoverGateway(t, roster, nil)
			rec := postChatCompletion(t, srv.Handler(), "default")
			if rec.Code != http.StatusOK {
				t.Fatalf("status=%d, want 200; body=%s", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), "from-fallback") {
				t.Fatalf("body=%s, want response from fallback", rec.Body.String())
			}
			if primaryHits.Load() == 0 {
				t.Fatalf("expected primary to be attempted first")
			}
			if fallbackHits.Load() != 1 {
				t.Fatalf("fallback hits=%d, want 1", fallbackHits.Load())
			}
			if len(fallbackModels) != 1 || fallbackModels[0] != "gpt-4o-mini" {
				t.Fatalf("fallback wire models=%v, want [gpt-4o-mini]", fallbackModels)
			}
		})
	}
}

// fak-test:runtime fast est=50ms
func TestFailoverCooldownSkipsPrimaryAndRecoversAfterExpiry(t *testing.T) {
	now := time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC)
	tracker := newRouteCircuitTracker(30*time.Second, func() time.Time { return now })

	var primaryStatus atomic.Int32
	primaryStatus.Store(http.StatusServiceUnavailable)
	var primaryHits, fallbackHits atomic.Int32

	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHits.Add(1)
		st := int(primaryStatus.Load())
		if st != http.StatusOK {
			w.WriteHeader(st)
			_, _ = w.Write([]byte(`{"error":{"message":"down"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-p",
			"object":  "chat.completion",
			"model":   "local-qwen",
			"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": "from-primary"}, "finish_reason": "stop"}},
		})
	}))
	t.Cleanup(primary.Close)

	fallback := newOpenAIUpstreamMock(t, http.StatusOK, "from-fallback", nil, &fallbackHits)
	t.Setenv("TEST_FB_KEY", "sk-fb")

	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "local-gpu", Kind: modelroute.KindLocal, BaseURL: primary.URL + "/v1"},
			{ID: "cloud-fb", Kind: modelroute.KindOpenAI, BaseURL: fallback.URL + "/v1", CredEnv: "TEST_FB_KEY"},
		},
		Bindings: []modelroute.Binding{
			{
				Model:         "default",
				Account:       "local-gpu",
				UpstreamModel: "local-qwen",
				Fallbacks:     []modelroute.FallbackTarget{{Account: "cloud-fb", UpstreamModel: "gpt-4o-mini"}},
			},
		},
	}

	srv := newTestFailoverGateway(t, roster, tracker)
	handler := srv.Handler()

	// Turn 1: primary returns 503 -> fails over to fallback and puts primary in cooldown.
	rec1 := postChatCompletion(t, handler, "default")
	if rec1.Code != http.StatusOK || !strings.Contains(rec1.Body.String(), "from-fallback") {
		t.Fatalf("turn 1 status=%d body=%s", rec1.Code, rec1.Body.String())
	}
	hitsAfterFirst := primaryHits.Load()

	// Turn 2 (10s later, still within 30s cooldown): primary is skipped completely!
	now = now.Add(10 * time.Second)
	rec2 := postChatCompletion(t, handler, "default")
	if rec2.Code != http.StatusOK || !strings.Contains(rec2.Body.String(), "from-fallback") {
		t.Fatalf("turn 2 status=%d body=%s", rec2.Code, rec2.Body.String())
	}
	if primaryHits.Load() != hitsAfterFirst {
		t.Fatalf("expected primary to be skipped during cooldown; hits went %d -> %d", hitsAfterFirst, primaryHits.Load())
	}

	// Turn 3 (25s later -> 35s total, past 30s cooldown): primary is healthy again and tried first!
	primaryStatus.Store(http.StatusOK)
	now = now.Add(25 * time.Second)
	rec3 := postChatCompletion(t, handler, "default")
	if rec3.Code != http.StatusOK || !strings.Contains(rec3.Body.String(), "from-primary") {
		t.Fatalf("turn 3 status=%d body=%s", rec3.Code, rec3.Body.String())
	}
	if cooling, _ := tracker.isCoolingDown("local-gpu"); cooling {
		t.Fatalf("expected primary cooldown to clear after recovery")
	}
}

// fak-test:runtime fast est=50ms
func TestFailoverDoesNotTriggerOn400Or401(t *testing.T) {
	for _, status := range []int{http.StatusBadRequest, http.StatusUnauthorized} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var fallbackHits atomic.Int32
			primary := newOpenAIUpstreamMock(t, status, "", nil, nil)
			fallback := newOpenAIUpstreamMock(t, http.StatusOK, "should-not-reach", nil, &fallbackHits)
			t.Setenv("TEST_FB_KEY", "sk-fb")

			roster := &modelroute.Roster{
				Version: modelroute.RosterVersion,
				Accounts: []modelroute.Account{
					{ID: "local-gpu", Kind: modelroute.KindLocal, BaseURL: primary.URL + "/v1"},
					{ID: "cloud-fb", Kind: modelroute.KindOpenAI, BaseURL: fallback.URL + "/v1", CredEnv: "TEST_FB_KEY"},
				},
				Bindings: []modelroute.Binding{
					{
						Model:         "default",
						Account:       "local-gpu",
						UpstreamModel: "local-qwen",
						Fallbacks:     []modelroute.FallbackTarget{{Account: "cloud-fb", UpstreamModel: "gpt-4o-mini"}},
					},
				},
			}

			tracker := newRouteCircuitTracker(30*time.Second, nil)
			srv := newTestFailoverGateway(t, roster, tracker)
			rec := postChatCompletion(t, srv.Handler(), "default")
			if rec.Code == http.StatusOK {
				t.Fatalf("expected non-200 error for %d, got 200: %s", status, rec.Body.String())
			}
			if fallbackHits.Load() != 0 {
				t.Fatalf("fallback must NOT be called on %d, got %d hits", status, fallbackHits.Load())
			}
			if cooling, _ := tracker.isCoolingDown("local-gpu"); cooling {
				t.Fatalf("400/401 client error must not trip circuit cooldown")
			}
		})
	}
}

// fak-test:runtime fast est=50ms
func TestFailoverLocalOnlyRefusesRemoteFailClosed(t *testing.T) {
	var remoteHits atomic.Int32
	localSrv := newOpenAIUpstreamMock(t, http.StatusServiceUnavailable, "", nil, nil)
	remoteSrv := newOpenAIUpstreamMock(t, http.StatusOK, "remote-leak", nil, &remoteHits)
	t.Setenv("TEST_REMOTE_KEY", "sk-remote")

	localPlanner, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), localSrv.URL+"/v1", "local-m", "")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner local: %v", err)
	}
	remotePlanner, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), remoteSrv.URL+"/v1", "remote-m", "sk-remote")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner remote: %v", err)
	}

	fp := &failoverPlanner{
		requestedModel: "private",
		residencyFloor: "local_only",
		candidates: []failoverCandidate{
			{
				target:   modelroute.Target{Account: "local-gpu", Kind: modelroute.KindLocal, UpstreamModel: "local-m", ResidencyFloor: "local_only"},
				planner:  localPlanner,
				locality: localitySelfHosted,
			},
			{
				target:   modelroute.Target{Account: "cloud-fb", Kind: modelroute.KindOpenAI, UpstreamModel: "remote-m", ResidencyFloor: "local_only"},
				planner:  remotePlanner,
				locality: localityVendor,
			},
		},
		circuit: newRouteCircuitTracker(30*time.Second, nil),
	}

	_, err = fp.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "secret"}}, nil)
	if err == nil || !strings.Contains(err.Error(), "RESIDENCY_FAILOVER_DENIED") {
		t.Fatalf("err = %v, want RESIDENCY_FAILOVER_DENIED", err)
	}
	if remoteHits.Load() != 0 {
		t.Fatalf("remote account was contacted (%d hits) despite local_only residency floor", remoteHits.Load())
	}
}

// fak-test:runtime fast est=50ms
func TestFailoverPrincipalIsolationExcludesUnauthorizedFallback(t *testing.T) {
	var primaryHits, tenantBHits atomic.Int32
	primary := newOpenAIUpstreamMock(t, http.StatusServiceUnavailable, "", nil, &primaryHits)
	tenantBFB := newOpenAIUpstreamMock(t, http.StatusOK, "tenant-b-secret", nil, &tenantBHits)
	t.Setenv("TEST_TB_KEY", "sk-tb")

	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "acct-a", Kind: modelroute.KindLocal, BaseURL: primary.URL + "/v1", Principals: []string{"tenant-a"}},
			{ID: "acct-b", Kind: modelroute.KindOpenAI, BaseURL: tenantBFB.URL + "/v1", CredEnv: "TEST_TB_KEY", Principals: []string{"tenant-b"}},
		},
		Bindings: []modelroute.Binding{
			{
				Model:         "shared-model",
				Account:       "acct-a",
				UpstreamModel: "m-a",
				Fallbacks:     []modelroute.FallbackTarget{{Account: "acct-b", UpstreamModel: "m-b"}},
			},
		},
	}

	srv := newTestFailoverGateway(t, roster, nil)

	// tenant-a: only acct-a is admitted; acct-b fallback is excluded, so 503 never spills to tenant-b.
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	ctxA := WithPrincipal(context.Background(), "tenant-a")
	bA, err := srv.bindChatRoute(ctxA, "shared-model")
	if err != nil {
		t.Fatalf("bindChatRoute(tenant-a): %v", err)
	}
	if _, err := bA.Planner.Complete(ctxA, []agent.Message{{Role: agent.RoleUser, Content: "hi"}}, nil); err == nil {
		t.Fatalf("expected primary 503 error for tenant-a without failing over to tenant-b")
	}
	if tenantBHits.Load() != 0 {
		t.Fatalf("tenant-b fallback was hit (%d times) for tenant-a request", tenantBHits.Load())
	}

	// tenant-c: neither account is admitted -> bindChatRoute fails (403 in prepareChatRoute).
	ctxC := WithPrincipal(context.Background(), "tenant-c")
	if _, err := srv.bindChatRoute(ctxC, "shared-model"); err == nil {
		t.Fatalf("expected bindChatRoute to deny tenant-c")
	}
}

// fak-test:runtime fast est=10ms
func TestIsFailoverEligibleError(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"canceled", context.Canceled, false},
		{"deadline", context.DeadlineExceeded, true},
		{"eof", io.EOF, true},
		{"unreachable", &agent.UpstreamUnreachableError{Err: errors.New("connection refused")}, true},
		{"400", &agent.UpstreamStatusError{Status: 400}, false},
		{"401", &agent.UpstreamStatusError{Status: 401}, false},
		{"403", &agent.UpstreamStatusError{Status: 403}, false},
		{"404", &agent.UpstreamStatusError{Status: 404}, false},
		{"408", &agent.UpstreamStatusError{Status: 408}, true},
		{"422", &agent.UpstreamStatusError{Status: 422}, false},
		{"429", &agent.UpstreamStatusError{Status: 429}, true},
		{"500", &agent.UpstreamStatusError{Status: 500}, true},
		{"502", &agent.UpstreamStatusError{Status: 502}, true},
		{"503", &agent.UpstreamStatusError{Status: 503}, true},
		{"504", &agent.UpstreamStatusError{Status: 504}, true},
	}
	for _, tc := range cases {
		if got := isFailoverEligibleError(tc.err); got != tc.want {
			t.Errorf("%s: isFailoverEligibleError(%v) = %v, want %v", tc.name, tc.err, got, tc.want)
		}
	}
}

type stubStreamFailoverPlanner struct {
	model string
	fn    func(ctx context.Context, sink agent.StreamSink) (*agent.Completion, error)
}

func (s *stubStreamFailoverPlanner) Model() string { return s.model }
func (s *stubStreamFailoverPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, errors.New("unexpected non-streaming call")
}
func (s *stubStreamFailoverPlanner) StreamingSupported() bool { return true }
func (s *stubStreamFailoverPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return s.fn(ctx, sink)
}

// fak-test:runtime fast est=50ms
func TestFailoverCompleteStreamRetriesPreChunkOnly(t *testing.T) {
	t.Run("pre_chunk_503_fails_over_via_http", func(t *testing.T) {
		var primaryHits, fallbackHits atomic.Int32
		primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			primaryHits.Add(1)
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		t.Cleanup(primary.Close)

		fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fallbackHits.Add(1)
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"model\":\"fb\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"from-stream-fallback\"},\"finish_reason\":null}]}\n\n")
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
		}))
		t.Cleanup(fallback.Close)

		p1, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), primary.URL+"/v1", "p-model", "")
		if err != nil {
			t.Fatalf("NewProviderHTTPPlanner primary: %v", err)
		}
		p2, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), fallback.URL+"/v1", "fb-model", "")
		if err != nil {
			t.Fatalf("NewProviderHTTPPlanner fallback: %v", err)
		}
		fp := &failoverPlanner{
			requestedModel: "default",
			candidates: []failoverCandidate{
				{target: modelroute.Target{Account: "local-gpu", Kind: modelroute.KindLocal, UpstreamModel: "p-model"}, planner: p1, locality: localitySelfHosted},
				{target: modelroute.Target{Account: "cloud-fb", Kind: modelroute.KindOpenAI, UpstreamModel: "fb-model"}, planner: p2, locality: localityVendor},
			},
			circuit: newRouteCircuitTracker(30*time.Second, nil),
		}

		var got strings.Builder
		comp, err := fp.CompleteStream(context.Background(), func(d string) error { got.WriteString(d); return nil },
			[]agent.Message{{Role: agent.RoleUser, Content: "hi"}}, nil)
		if err != nil {
			t.Fatalf("CompleteStream unexpected err: %v", err)
		}
		if got.String() != "from-stream-fallback" || comp.Message.Content != "from-stream-fallback" {
			t.Fatalf("streamed=%q content=%q, want from-stream-fallback", got.String(), comp.Message.Content)
		}
		if primaryHits.Load() != 1 || fallbackHits.Load() != 1 {
			t.Fatalf("hits primary=%d fallback=%d, want 1/1", primaryHits.Load(), fallbackHits.Load())
		}
	})

	t.Run("mid_stream_after_emit_does_not_fail_over", func(t *testing.T) {
		var fallbackHits atomic.Int32
		fp := &failoverPlanner{
			requestedModel: "default",
			candidates: []failoverCandidate{
				{
					target: modelroute.Target{Account: "local-gpu", Kind: modelroute.KindLocal, UpstreamModel: "p-model"},
					planner: &stubStreamFailoverPlanner{
						model: "p-model",
						fn: func(_ context.Context, sink agent.StreamSink) (*agent.Completion, error) {
							_ = sink("partial-")
							return nil, &agent.UpstreamStatusError{Status: http.StatusServiceUnavailable}
						},
					},
					locality: localitySelfHosted,
				},
				{
					target: modelroute.Target{Account: "cloud-fb", Kind: modelroute.KindOpenAI, UpstreamModel: "fb-model"},
					planner: &stubStreamFailoverPlanner{
						model: "fb-model",
						fn: func(context.Context, agent.StreamSink) (*agent.Completion, error) {
							fallbackHits.Add(1)
							return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "dup"}}, nil
						},
					},
					locality: localityVendor,
				},
			},
		}

		var got strings.Builder
		_, err := fp.CompleteStream(context.Background(), func(d string) error { got.WriteString(d); return nil },
			[]agent.Message{{Role: agent.RoleUser, Content: "hi"}}, nil)
		if err == nil {
			t.Fatal("expected error after mid-stream failure, got nil")
		}
		if got.String() != "partial-" {
			t.Fatalf("got streamed prefix %q, want partial-", got.String())
		}
		if fallbackHits.Load() != 0 {
			t.Fatalf("fallback hits=%d, want 0 after partial stream output", fallbackHits.Load())
		}
	})
}

// fak-test:runtime fast est=50ms
func TestChatRouteFailoverContextOverflowProactive(t *testing.T) {
	var primaryHits, smallFBHits, largeFBHits atomic.Int32
	primary := newOpenAIUpstreamMock(t, http.StatusOK, "from-primary", nil, &primaryHits)
	smallFB := newOpenAIUpstreamMock(t, http.StatusOK, "from-small-fb", nil, &smallFBHits)
	largeFB := newOpenAIUpstreamMock(t, http.StatusOK, "from-large-fb", nil, &largeFBHits)
	t.Setenv("TEST_LARGE_KEY", "sk-large")

	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "local-gpu", Kind: modelroute.KindLocal, BaseURL: primary.URL + "/v1", ContextTokens: 100},
			{ID: "local-small", Kind: modelroute.KindLocal, BaseURL: smallFB.URL + "/v1", ContextTokens: 100},
			{ID: "cloud-large", Kind: modelroute.KindOpenAI, BaseURL: largeFB.URL + "/v1", CredEnv: "TEST_LARGE_KEY", ContextTokens: 4096},
		},
		Bindings: []modelroute.Binding{
			{
				Model:         "default",
				Account:       "local-gpu",
				UpstreamModel: "local-qwen",
				Fallbacks: []modelroute.FallbackTarget{
					{Account: "local-small", UpstreamModel: "local-qwen-small"},
					{Account: "cloud-large", UpstreamModel: "gpt-4o"},
				},
			},
		},
	}

	tracker := newRouteCircuitTracker(30*time.Second, nil)
	srv := newTestFailoverGateway(t, roster, tracker)

	longPrompt := strings.Repeat("word ", 200) // ~1000 chars -> ~250 estimated tokens > 100
	payload, _ := json.Marshal(map[string]any{
		"model":    "default",
		"messages": []map[string]any{{"role": "user", "content": longPrompt}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "from-large-fb") {
		t.Fatalf("status=%d body=%s, want 200 from-large-fb", rec.Code, rec.Body.String())
	}
	if primaryHits.Load() != 0 || smallFBHits.Load() != 0 || largeFBHits.Load() != 1 {
		t.Fatalf("hits primary=%d smallFB=%d largeFB=%d, want 0/0/1",
			primaryHits.Load(), smallFBHits.Load(), largeFBHits.Load())
	}
	if cooling, _ := tracker.isCoolingDown("local-gpu"); cooling {
		t.Fatalf("proactive context overflow must not put primary into circuit cooldown")
	}
	snap := tracker.snapshot()
	if snap.accounts["local-gpu"].totalFailovers != 1 {
		t.Fatalf("local-gpu totalFailovers=%d, want 1", snap.accounts["local-gpu"].totalFailovers)
	}
}

// fak-test:runtime fast est=50ms
func TestChatRouteFailoverContextOverflowReactive(t *testing.T) {
	t.Run("context_length_exceeded_escalates_past_insufficient_fallback", func(t *testing.T) {
		var primaryHits, sameFBHits, largeFBHits atomic.Int32
		primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			primaryHits.Add(1)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"This model's maximum context length is 8192 tokens.","type":"invalid_request_error","code":"context_length_exceeded"}}`))
		}))
		t.Cleanup(primary.Close)

		sameFB := newOpenAIUpstreamMock(t, http.StatusOK, "from-same-fb", nil, &sameFBHits)
		largeFB := newOpenAIUpstreamMock(t, http.StatusOK, "from-large-fb", nil, &largeFBHits)
		t.Setenv("TEST_LARGE_KEY", "sk-large")

		roster := &modelroute.Roster{
			Version: modelroute.RosterVersion,
			Accounts: []modelroute.Account{
				{ID: "local-gpu", Kind: modelroute.KindLocal, BaseURL: primary.URL + "/v1", ContextTokens: 8192},
				{ID: "local-same", Kind: modelroute.KindLocal, BaseURL: sameFB.URL + "/v1", ContextTokens: 8192},
				{ID: "cloud-large", Kind: modelroute.KindOpenAI, BaseURL: largeFB.URL + "/v1", CredEnv: "TEST_LARGE_KEY", ContextTokens: 128000},
			},
			Bindings: []modelroute.Binding{
				{
					Model:         "default",
					Account:       "local-gpu",
					UpstreamModel: "local-qwen",
					Fallbacks: []modelroute.FallbackTarget{
						{Account: "local-same", UpstreamModel: "local-qwen-2"},
						{Account: "cloud-large", UpstreamModel: "gpt-4o"},
					},
				},
			},
		}

		tracker := newRouteCircuitTracker(30*time.Second, nil)
		srv := newTestFailoverGateway(t, roster, tracker)
		rec := postChatCompletion(t, srv.Handler(), "default")
		if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "from-large-fb") {
			t.Fatalf("status=%d body=%s, want 200 from-large-fb", rec.Code, rec.Body.String())
		}
		if primaryHits.Load() == 0 || sameFBHits.Load() != 0 || largeFBHits.Load() != 1 {
			t.Fatalf("hits primary=%d sameFB=%d largeFB=%d, want >=1/0/1",
				primaryHits.Load(), sameFBHits.Load(), largeFBHits.Load())
		}
		if cooling, _ := tracker.isCoolingDown("local-gpu"); cooling {
			t.Fatalf("context_length_exceeded must not put primary into circuit cooldown")
		}
		snap := tracker.snapshot()
		if snap.accounts["local-gpu"].totalFailovers != 1 {
			t.Fatalf("local-gpu totalFailovers=%d, want 1", snap.accounts["local-gpu"].totalFailovers)
		}
	})

	t.Run("non_overflow_400_does_not_escalate", func(t *testing.T) {
		var fallbackHits atomic.Int32
		primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"invalid parameter","type":"invalid_request_error","code":"invalid_request"}}`))
		}))
		t.Cleanup(primary.Close)
		fallback := newOpenAIUpstreamMock(t, http.StatusOK, "should-not-reach", nil, &fallbackHits)
		t.Setenv("TEST_LARGE_KEY", "sk-large")

		roster := &modelroute.Roster{
			Version: modelroute.RosterVersion,
			Accounts: []modelroute.Account{
				{ID: "local-gpu", Kind: modelroute.KindLocal, BaseURL: primary.URL + "/v1", ContextTokens: 8192},
				{ID: "cloud-large", Kind: modelroute.KindOpenAI, BaseURL: fallback.URL + "/v1", CredEnv: "TEST_LARGE_KEY", ContextTokens: 128000},
			},
			Bindings: []modelroute.Binding{
				{
					Model:         "default",
					Account:       "local-gpu",
					UpstreamModel: "local-qwen",
					Fallbacks:     []modelroute.FallbackTarget{{Account: "cloud-large", UpstreamModel: "gpt-4o"}},
				},
			},
		}

		srv := newTestFailoverGateway(t, roster, nil)
		rec := postChatCompletion(t, srv.Handler(), "default")
		if rec.Code == http.StatusOK {
			t.Fatalf("expected non-200 for non-overflow 400, got 200: %s", rec.Body.String())
		}
		if fallbackHits.Load() != 0 {
			t.Fatalf("fallback hits=%d, want 0 for non-overflow 400", fallbackHits.Load())
		}
	})
}

// fak-test:runtime fast est=50ms
func TestChatRouteFailoverContextOverflowLocalOnlyFailsClosed(t *testing.T) {
	t.Run("proactive_local_only_refuses_remote_fallback", func(t *testing.T) {
		var localHits, remoteHits atomic.Int32
		localSrv := newOpenAIUpstreamMock(t, http.StatusOK, "local-ok", nil, &localHits)
		remoteSrv := newOpenAIUpstreamMock(t, http.StatusOK, "remote-leak", nil, &remoteHits)

		localPlanner, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), localSrv.URL+"/v1", "local-m", "")
		if err != nil {
			t.Fatalf("NewProviderHTTPPlanner local: %v", err)
		}
		remotePlanner, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), remoteSrv.URL+"/v1", "remote-m", "sk-remote")
		if err != nil {
			t.Fatalf("NewProviderHTTPPlanner remote: %v", err)
		}

		fp := &failoverPlanner{
			requestedModel: "private",
			residencyFloor: "local_only",
			candidates: []failoverCandidate{
				{
					target:   modelroute.Target{Account: "local-gpu", Kind: modelroute.KindLocal, UpstreamModel: "local-m", ContextTokens: 100, ResidencyFloor: "local_only"},
					planner:  localPlanner,
					locality: localitySelfHosted,
				},
				{
					target:   modelroute.Target{Account: "cloud-fb", Kind: modelroute.KindOpenAI, UpstreamModel: "remote-m", ContextTokens: 128000, ResidencyFloor: "local_only"},
					planner:  remotePlanner,
					locality: localityVendor,
				},
			},
			circuit: newRouteCircuitTracker(30*time.Second, nil),
		}

		longPrompt := strings.Repeat("secret ", 150) // > 100 tokens
		_, err = fp.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: longPrompt}}, nil)
		if err == nil || !strings.Contains(err.Error(), "RESIDENCY_FAILOVER_DENIED") {
			t.Fatalf("err = %v, want RESIDENCY_FAILOVER_DENIED", err)
		}
		if localHits.Load() != 0 || remoteHits.Load() != 0 {
			t.Fatalf("hits local=%d remote=%d, want 0/0", localHits.Load(), remoteHits.Load())
		}
	})

	t.Run("reactive_local_only_refuses_remote_fallback", func(t *testing.T) {
		var remoteHits atomic.Int32
		localSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"code":"context_length_exceeded","message":"maximum context length is 4096"}}`))
		}))
		t.Cleanup(localSrv.Close)
		remoteSrv := newOpenAIUpstreamMock(t, http.StatusOK, "remote-leak", nil, &remoteHits)

		localPlanner, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), localSrv.URL+"/v1", "local-m", "")
		if err != nil {
			t.Fatalf("NewProviderHTTPPlanner local: %v", err)
		}
		remotePlanner, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), remoteSrv.URL+"/v1", "remote-m", "sk-remote")
		if err != nil {
			t.Fatalf("NewProviderHTTPPlanner remote: %v", err)
		}

		fp := &failoverPlanner{
			requestedModel: "private",
			residencyFloor: "local_only",
			candidates: []failoverCandidate{
				{
					target:   modelroute.Target{Account: "local-gpu", Kind: modelroute.KindLocal, UpstreamModel: "local-m", ContextTokens: 4096, ResidencyFloor: "local_only"},
					planner:  localPlanner,
					locality: localitySelfHosted,
				},
				{
					target:   modelroute.Target{Account: "cloud-fb", Kind: modelroute.KindOpenAI, UpstreamModel: "remote-m", ContextTokens: 128000, ResidencyFloor: "local_only"},
					planner:  remotePlanner,
					locality: localityVendor,
				},
			},
			circuit: newRouteCircuitTracker(30*time.Second, nil),
		}

		_, err = fp.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "short"}}, nil)
		if err == nil || !strings.Contains(err.Error(), "RESIDENCY_FAILOVER_DENIED") {
			t.Fatalf("err = %v, want RESIDENCY_FAILOVER_DENIED", err)
		}
		if remoteHits.Load() != 0 {
			t.Fatalf("remote account was contacted (%d hits) despite local_only residency floor", remoteHits.Load())
		}
	})

	t.Run("all_local_candidates_insufficient_returns_context_overflow", func(t *testing.T) {
		var local1Hits, local2Hits atomic.Int32
		local1 := newOpenAIUpstreamMock(t, http.StatusOK, "l1", nil, &local1Hits)
		local2 := newOpenAIUpstreamMock(t, http.StatusOK, "l2", nil, &local2Hits)

		p1, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), local1.URL+"/v1", "m1", "")
		if err != nil {
			t.Fatalf("NewProviderHTTPPlanner 1: %v", err)
		}
		p2, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), local2.URL+"/v1", "m2", "")
		if err != nil {
			t.Fatalf("NewProviderHTTPPlanner 2: %v", err)
		}

		fp := &failoverPlanner{
			requestedModel: "private",
			residencyFloor: "local_only",
			candidates: []failoverCandidate{
				{target: modelroute.Target{Account: "local-1", Kind: modelroute.KindLocal, UpstreamModel: "m1", ContextTokens: 100, ResidencyFloor: "local_only"}, planner: p1, locality: localitySelfHosted},
				{target: modelroute.Target{Account: "local-2", Kind: modelroute.KindLocal, UpstreamModel: "m2", ContextTokens: 150, ResidencyFloor: "local_only"}, planner: p2, locality: localitySelfHosted},
			},
		}

		longPrompt := strings.Repeat("secret ", 150) // > 150 tokens
		_, err = fp.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: longPrompt}}, nil)
		if !errors.Is(err, agent.ErrContextOverflow) {
			t.Fatalf("err = %v, want ErrContextOverflow", err)
		}
		if local1Hits.Load() != 0 || local2Hits.Load() != 0 {
			t.Fatalf("hits local1=%d local2=%d, want 0/0", local1Hits.Load(), local2Hits.Load())
		}
	})
}

// fak-test:runtime fast est=50ms
func TestChatRouteFailoverContextOverflowStream(t *testing.T) {
	var primaryHits, largeHits atomic.Int32
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"code":"context_length_exceeded","message":"maximum context length is 4096"}}`))
	}))
	t.Cleanup(primary.Close)

	large := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		largeHits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"model\":\"fb\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"from-large-stream\"},\"finish_reason\":null}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(large.Close)

	p1, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), primary.URL+"/v1", "local-m", "")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner primary: %v", err)
	}
	p2, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAI), large.URL+"/v1", "large-m", "")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner large: %v", err)
	}

	fp := &failoverPlanner{
		requestedModel: "default",
		candidates: []failoverCandidate{
			{target: modelroute.Target{Account: "local-gpu", Kind: modelroute.KindLocal, UpstreamModel: "local-m", ContextTokens: 4096}, planner: p1, locality: localitySelfHosted},
			{target: modelroute.Target{Account: "cloud-large", Kind: modelroute.KindOpenAI, UpstreamModel: "large-m", ContextTokens: 128000}, planner: p2, locality: localityVendor},
		},
		circuit: newRouteCircuitTracker(30*time.Second, nil),
	}

	var got strings.Builder
	comp, err := fp.CompleteStream(context.Background(), func(d string) error { got.WriteString(d); return nil },
		[]agent.Message{{Role: agent.RoleUser, Content: "short"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream err: %v", err)
	}
	if got.String() != "from-large-stream" || comp.Message.Content != "from-large-stream" {
		t.Fatalf("streamed=%q content=%q, want from-large-stream", got.String(), comp.Message.Content)
	}
	if primaryHits.Load() == 0 || largeHits.Load() != 1 {
		t.Fatalf("hits primary=%d large=%d, want >=1/1", primaryHits.Load(), largeHits.Load())
	}
}
