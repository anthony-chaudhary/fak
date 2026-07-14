package gateway

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestUpstreamResponseObserverSeesProviderHeaders proves the Config seam end to end:
// a planner built by newProxyPlanner with an UpstreamResponseObserver reports every
// upstream response's status + headers (here the provider's account rate-limit
// headers) to the host, and a nil observer leaves the transport untouched.
func TestUpstreamResponseObserverSeesProviderHeaders(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Anthropic-Ratelimit-Unified-5h-Utilization", "0.34")
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	var mu sync.Mutex
	var gotStatus int
	var gotUtil string
	cfg := Config{
		Provider: "anthropic",
		UpstreamResponseObserver: func(status int, h http.Header) {
			mu.Lock()
			defer mu.Unlock()
			gotStatus = status
			gotUtil = h.Get("Anthropic-Ratelimit-Unified-5h-Utilization")
		},
	}
	planner, err := newProxyPlanner(cfg, "claude-test", []string{upstream.URL})
	if err != nil {
		t.Fatal(err)
	}
	hp, ok := planner.(*agent.HTTPPlanner)
	if !ok {
		t.Fatalf("planner = %T, want *agent.HTTPPlanner", planner)
	}
	resp, err := hp.Client.Get(upstream.URL)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	mu.Lock()
	defer mu.Unlock()
	if gotStatus != http.StatusOK {
		t.Fatalf("observed status = %d, want 200", gotStatus)
	}
	if gotUtil != "0.34" {
		t.Fatalf("observed utilization header = %q, want 0.34", gotUtil)
	}
}

func TestNilUpstreamObserverLeavesTransportUnchanged(t *testing.T) {
	planner, err := newProxyPlanner(Config{Provider: "anthropic"}, "claude-test", []string{"https://api.anthropic.com"})
	if err != nil {
		t.Fatal(err)
	}
	hp := planner.(*agent.HTTPPlanner)
	if hp.Client.Transport != nil {
		t.Fatalf("nil observer must not install a transport, got %T", hp.Client.Transport)
	}
}

func TestProxyPlannerForwardsExtraHeaders(t *testing.T) {
	var gotAuth, gotAccount string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotAccount = r.Header.Get("ChatGPT-Account-Id")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	planner, err := newProxyPlanner(Config{
		Provider:     "openai-responses",
		APIKey:       "subscription-token",
		ExtraHeaders: map[string]string{"ChatGPT-Account-Id": "acct-gateway"},
	}, "gpt-test", []string{upstream.URL})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := planner.Complete(t.Context(), []agent.Message{{Role: agent.RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("complete: %v", err)
	}
	if gotAuth != "Bearer subscription-token" {
		t.Fatalf("Authorization = %q, want bearer subscription-token", gotAuth)
	}
	if gotAccount != "acct-gateway" {
		t.Fatalf("ChatGPT-Account-Id = %q, want acct-gateway", gotAccount)
	}
}

func TestUpstreamObserverReportsOnlyTransientTransportErrors(t *testing.T) {
	transient := errors.New("read: connection reset by peer")
	seen := 0
	tr := &upstreamObserveTransport{base: roundTripFunc(func(*http.Request) (*http.Response, error) { return nil, transient }), observeError: func(error) { seen++ }}
	req, _ := http.NewRequest(http.MethodGet, "http://provider.invalid", nil)
	_, _ = tr.RoundTrip(req)
	if seen != 1 {
		t.Fatalf("transient transport observations=%d, want 1", seen)
	}

	seen = 0
	tr.base = roundTripFunc(func(*http.Request) (*http.Response, error) { return nil, errors.New("permanent malformed request") })
	_, _ = tr.RoundTrip(req)
	if seen != 0 {
		t.Fatalf("permanent transport error observed as transient")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
