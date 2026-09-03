package gateway

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestReplicaRouterRetryTelemetry(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")

	var attempts int32
	upstreamHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&attempts, 1) == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"model":"fleet-model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	})

	upstream1 := httptest.NewServer(upstreamHandler)
	defer upstream1.Close()
	upstream2 := httptest.NewServer(upstreamHandler)
	defer upstream2.Close()

	var debugMu sync.Mutex
	var debugLines []string
	debugStatsf := func(format string, args ...any) {
		debugMu.Lock()
		defer debugMu.Unlock()
		debugLines = append(debugLines, fmt.Sprintf(format, args...))
	}

	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	srv, err := New(Config{
		EngineID:        "test",
		Model:           "fleet-model",
		BaseURL:         upstream1.URL,
		ReplicaBaseURLs: []string{upstream2.URL},
		Provider:        "openai",
		VDSO:            true,
		DebugStatsf:     debugStatsf,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp ChatResponse
	code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{
		Model:    "fleet-model",
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "hi"}},
	}, &resp)
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Message.Content != "ok" {
		t.Fatalf("response = %+v, want choice with content 'ok'", resp)
	}

	vars := srv.debugVars(time.Now())
	if vars.Upstream.Retries != 1 {
		t.Fatalf("Upstream.Retries = %d, want 1", vars.Upstream.Retries)
	}

	metrics := srv.renderMetrics()
	if !strings.Contains(metrics, "fak_gateway_upstream_retries_total 1") {
		t.Fatalf("metrics missing fak_gateway_upstream_retries_total 1:\n%s", metrics)
	}

	debugMu.Lock()
	captured := strings.Join(debugLines, "\n")
	debugMu.Unlock()
	if !strings.Contains(captured, "fak-turn retry") {
		t.Fatalf("captured debug stats missing 'fak-turn retry':\n%s", captured)
	}
}

func TestReplicaRouterWalkPlanners(t *testing.T) {
	var nilRouter *ReplicaRouter
	nilRouter.WalkPlanners(func(agent.Planner) {
		t.Fatal("nil router should not walk planners")
	})

	p1 := agent.NewHTTPPlanner("http://localhost:1", "m1", "")
	p2 := agent.NewHTTPPlanner("http://localhost:2", "m2", "")
	router, err := NewReplicaRouter("fleet", []PlannerReplica{
		{Name: "r1", Planner: p1},
		{Name: "r2", Planner: p2},
	})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}

	var visited []agent.Planner
	router.WalkPlanners(func(p agent.Planner) {
		visited = append(visited, p)
	})
	if len(visited) != 2 || visited[0] != p1 || visited[1] != p2 {
		t.Fatalf("WalkPlanners visited %+v, want [%p, %p]", visited, p1, p2)
	}

	var walkedHTTP []*agent.HTTPPlanner
	walkHTTPPlanners(router, func(hp *agent.HTTPPlanner) {
		walkedHTTP = append(walkedHTTP, hp)
	})
	if len(walkedHTTP) != 2 || walkedHTTP[0] != p1 || walkedHTTP[1] != p2 {
		t.Fatalf("walkHTTPPlanners visited %+v, want [%p, %p]", walkedHTTP, p1, p2)
	}
}

func TestDualPlannerWalkPlanners(t *testing.T) {
	var nilDual *DualPlanner
	nilDual.WalkPlanners(func(agent.Planner) {
		t.Fatal("nil dual planner should not walk planners")
	})

	proxy := agent.NewHTTPPlanner("http://localhost:1", "proxy-m", "")
	local := &replicaRouterTestPlanner{name: "local-m"}
	dual, err := NewDualPlanner(proxy, local, "local")
	if err != nil {
		t.Fatalf("NewDualPlanner: %v", err)
	}

	var visited []agent.Planner
	dual.WalkPlanners(func(p agent.Planner) {
		visited = append(visited, p)
	})
	if len(visited) != 2 || visited[0] != proxy || visited[1] != local {
		t.Fatalf("WalkPlanners visited %+v, want [%p, %p]", visited, proxy, local)
	}

	var walkedHTTP []*agent.HTTPPlanner
	walkHTTPPlanners(dual, func(hp *agent.HTTPPlanner) {
		walkedHTTP = append(walkedHTTP, hp)
	})
	if len(walkedHTTP) != 1 || walkedHTTP[0] != proxy {
		t.Fatalf("walkHTTPPlanners visited %+v, want [%p]", walkedHTTP, proxy)
	}
}
