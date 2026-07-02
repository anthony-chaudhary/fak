package gateway

import (
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
