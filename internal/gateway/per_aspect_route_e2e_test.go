package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// simulatedLocalGPUBackend represents one local GPU serving process (1-8 GPUs on a node)
// behind an OpenAI-compatible httptest.Server, tracking real prefix-KV LRU residency hits.
type simulatedLocalGPUBackend struct {
	name   string
	server *httptest.Server

	mu         sync.Mutex
	cache      *residentSet
	requests   int
	hits       int
	underpower atomic.Bool
	refuse403  atomic.Bool
	transport  atomic.Bool
}

func newSimulatedLocalGPUBackend(t *testing.T, name string, cacheCap int) *simulatedLocalGPUBackend {
	t.Helper()
	b := &simulatedLocalGPUBackend{
		name:  name,
		cache: newResidentSet(cacheCap),
	}
	b.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if b.transport.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		if b.refuse403.Load() {
			w.WriteHeader(http.StatusForbidden)
			_, _ = io.WriteString(w, `{"error":{"message":"policy denied"}}`)
			return
		}

		raw, _ := io.ReadAll(r.Body)
		var req struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(raw, &req)

		prefixKey := ""
		if len(req.Messages) > 0 {
			prefixKey = req.Messages[0].Content
		}

		b.mu.Lock()
		b.requests++
		hit := prefixKey != "" && b.cache.holds(prefixKey)
		if hit {
			b.hits++
		}
		if prefixKey != "" {
			b.cache.admit(prefixKey)
		}
		b.mu.Unlock()

		if b.underpower.Load() {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnprocessableEntity)
			_, _ = io.WriteString(w, `{"error":{"code":"underpowered","message":"local model could not solve multi-turn step"}}`)
			return
		}

		content := fmt.Sprintf("gpu=%s hit=%v", b.name, hit)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"model":%q,"choices":[{"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":4,"total_tokens":20}}`,
			req.Model, content)
	}))
	t.Cleanup(b.server.Close)
	return b
}

func (b *simulatedLocalGPUBackend) stats() (requests, hits int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.requests, b.hits
}

// fak-test:justify why=integration when=changed:internal/gateway/**
// fak-test:runtime fast est=120ms lane=default
func TestKVAwareFleetRoutingLiveReplicaDispatchBeatsBlind(t *testing.T) {
	const (
		numGPUs     = 4
		capPerGPU   = 4
		numFamilies = 12 // fits fleet (16), exceeds any single GPU (4)
	)
	backends := make([]*simulatedLocalGPUBackend, numGPUs)
	replicas := make([]PlannerReplica, numGPUs)
	mem := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 2,
		Probe:          func(context.Context, WorkerSpec) bool { return true },
	})
	for i := 0; i < numGPUs; i++ {
		name := fmt.Sprintf("gpu-%d", i)
		backends[i] = newSimulatedLocalGPUBackend(t, name, capPerGPU)
		hp, err := agent.NewProviderHTTPPlanner("openai", backends[i].server.URL, "qwen3-8b", "")
		if err != nil {
			t.Fatalf("NewProviderHTTPPlanner(%s): %v", name, err)
		}
		replicas[i] = PlannerReplica{Name: name, Planner: hp, Endpoint: backends[i].server.URL}
		if err := mem.Add(WorkerSpec{ID: name, Engine: EngineNative, Endpoint: backends[i].server.URL, Models: []string{"qwen3-8b"}}); err != nil {
			t.Fatalf("Add(%s): %v", name, err)
		}
	}
	ctx := context.Background()
	mem.ProbeOnce(ctx)

	kvRouter, err := NewReplicaDispatch("qwen3-8b", replicas)
	if err != nil {
		t.Fatalf("NewReplicaDispatch(kv): %v", err)
	}
	kvPolicy := NewFleetCacheRouter(numGPUs, capPerGPU).AsPickPolicy()
	kvRouter.WithMembership(mem).WithPickPolicy(kvPolicy)

	// Build a shared-prefix stream where distinct user turns share system-prompt prefixes
	// across 12 families, exceeding any single GPU's 4-slot cache.
	type turnReq struct {
		msgs []agent.Message
	}
	var stream []turnReq
	for round := 0; round < 4; round++ {
		for fam := 0; fam < numFamilies; fam++ {
			stream = append(stream, turnReq{
				msgs: []agent.Message{
					{Role: agent.RoleSystem, Content: fmt.Sprintf("shared-system-scaffold-%d", fam)},
					{Role: agent.RoleUser, Content: fmt.Sprintf("turn-%d-fam-%d", round, fam)},
				},
			})
		}
	}

	// Measure KV-aware live ReplicaDispatch over the 4 local GPU httptest.Servers.
	for i, tr := range stream {
		if _, err := kvRouter.Complete(ctx, tr.msgs, nil); err != nil {
			t.Fatalf("kv Complete(%d): %v", i, err)
		}
	}
	var kvReqs, kvHits int
	for _, b := range backends {
		r, h := b.stats()
		kvReqs += r
		kvHits += h
	}
	kvHitRate := float64(kvHits) / float64(kvReqs)

	// Compare against cache-blind ReplicaDispatch (nil PickPolicy -> HRW keyed over full
	// message slice including varying user turn suffix, which scatters shared prefixes).
	blindBackends := make([]*simulatedLocalGPUBackend, numGPUs)
	blindReplicas := make([]PlannerReplica, numGPUs)
	for i := 0; i < numGPUs; i++ {
		name := fmt.Sprintf("blind-gpu-%d", i)
		blindBackends[i] = newSimulatedLocalGPUBackend(t, name, capPerGPU)
		hp, err := agent.NewProviderHTTPPlanner("openai", blindBackends[i].server.URL, "qwen3-8b", "")
		if err != nil {
			t.Fatalf("NewProviderHTTPPlanner(blind %s): %v", name, err)
		}
		blindReplicas[i] = PlannerReplica{Name: name, Planner: hp, Endpoint: blindBackends[i].server.URL}
	}
	blindRouter, err := NewReplicaDispatch("qwen3-8b", blindReplicas)
	if err != nil {
		t.Fatalf("NewReplicaDispatch(blind): %v", err)
	}
	for i, tr := range stream {
		if _, err := blindRouter.Complete(ctx, tr.msgs, nil); err != nil {
			t.Fatalf("blind Complete(%d): %v", i, err)
		}
	}
	var blindReqs, blindHits int
	for _, b := range blindBackends {
		r, h := b.stats()
		blindReqs += r
		blindHits += h
	}
	blindHitRate := float64(blindHits) / float64(blindReqs)

	if kvReqs != len(stream) || blindReqs != len(stream) {
		t.Fatalf("request counts kv=%d blind=%d, want %d", kvReqs, blindReqs, len(stream))
	}
	if kvHitRate <= blindHitRate {
		t.Fatalf("live KV-aware ReplicaDispatch hit rate %.4f (%d/%d) did not beat cache-blind %.4f (%d/%d)",
			kvHitRate, kvHits, kvReqs, blindHitRate, blindHits, blindReqs)
	}
	// 12 cold misses + 36 shared-prefix hits = 75% backend cache hit rate.
	if kvHits != len(stream)-numFamilies {
		t.Fatalf("live KV-aware backend hits = %d, want %d (every shared-prefix continuation hits home GPU)", kvHits, len(stream)-numFamilies)
	}
	t.Logf("live 4-GPU ReplicaDispatch prefix hit rate: kv-aware=%.4f (%d/%d) vs blind=%.4f (%d/%d)",
		kvHitRate, kvHits, kvReqs, blindHitRate, blindHits, blindReqs)
}

// fak-test:justify why=integration when=changed:internal/gateway/**
// fak-test:runtime fast est=80ms lane=default
func TestKVAwareFleetRoutingSlackSheddingAndAffinity(t *testing.T) {
	const numGPUs = 8 // single-node 8-GPU local ceiling
	backends := make([]*simulatedLocalGPUBackend, numGPUs)
	replicas := make([]PlannerReplica, numGPUs)
	mem := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 2,
		Probe:          func(context.Context, WorkerSpec) bool { return true },
	})
	for i := 0; i < numGPUs; i++ {
		name := fmt.Sprintf("gpu-%d", i)
		backends[i] = newSimulatedLocalGPUBackend(t, name, 8)
		hp, err := agent.NewProviderHTTPPlanner("openai", backends[i].server.URL, "qwen3-32b", "")
		if err != nil {
			t.Fatalf("NewProviderHTTPPlanner(%s): %v", name, err)
		}
		replicas[i] = PlannerReplica{Name: name, Planner: hp, Endpoint: backends[i].server.URL}
		if err := mem.Add(WorkerSpec{ID: name, Engine: EngineNative, Endpoint: backends[i].server.URL, Models: []string{"qwen3-32b"}}); err != nil {
			t.Fatalf("Add(%s): %v", name, err)
		}
	}
	ctx := context.Background()
	mem.ProbeOnce(ctx)

	policy := NewSlackTolerantFleetPickPolicy(numGPUs, 8, 1.5)
	router, err := NewReplicaDispatch("qwen3-32b", replicas)
	if err != nil {
		t.Fatalf("NewReplicaDispatch: %v", err)
	}
	router.WithMembership(mem).WithPickPolicy(policy)

	sharedMsgs := []agent.Message{
		{Role: agent.RoleSystem, Content: "agent-scaffold-v1"},
		{Role: agent.RoleUser, Content: "step-1"},
	}

	// Turn 1 + Turn 2 home onto gpu-0 and hit gpu-0 while unsaturated.
	comp1, err := router.Complete(ctx, sharedMsgs, nil)
	if err != nil {
		t.Fatalf("Complete 1: %v", err)
	}
	if !strings.Contains(comp1.Message.Content, "gpu=gpu-0") {
		t.Fatalf("first placement = %q, want gpu-0", comp1.Message.Content)
	}
	comp2, err := router.Complete(ctx, []agent.Message{
		{Role: agent.RoleSystem, Content: "agent-scaffold-v1"},
		{Role: agent.RoleUser, Content: "step-2"},
	}, nil)
	if err != nil {
		t.Fatalf("Complete 2: %v", err)
	}
	if !strings.Contains(comp2.Message.Content, "gpu=gpu-0 hit=true") {
		t.Fatalf("warm continuation = %q, want gpu-0 hit=true", comp2.Message.Content)
	}

	// Within slack (load 1 on gpu-0 vs 0 on peers <= 1.5*max(0,1)): sticks to gpu-0.
	if err := mem.Acquire("gpu-0"); err != nil {
		t.Fatalf("Acquire(gpu-0): %v", err)
	}
	compWithin, err := router.Complete(ctx, sharedMsgs, nil)
	if err != nil {
		t.Fatalf("Complete within slack: %v", err)
	}
	if !strings.Contains(compWithin.Message.Content, "gpu=gpu-0 hit=true") {
		t.Fatalf("within slack should stay on gpu-0, got %q", compWithin.Message.Content)
	}

	// Saturate gpu-0 past slack (load 3 > 1.5*1): sheds cleanly to least-loaded gpu-1.
	for i := 0; i < 2; i++ {
		if err := mem.Acquire("gpu-0"); err != nil {
			t.Fatalf("Acquire(gpu-0): %v", err)
		}
	}
	compShed, err := router.Complete(ctx, sharedMsgs, nil)
	if err != nil {
		t.Fatalf("Complete shed: %v", err)
	}
	if strings.Contains(compShed.Message.Content, "gpu=gpu-0") {
		t.Fatalf("saturated gpu-0 should shed past slack bound, got %q", compShed.Message.Content)
	}
	if !strings.Contains(compShed.Message.Content, "gpu=gpu-1") {
		t.Fatalf("shed target = %q, want least-loaded gpu-1", compShed.Message.Content)
	}
	for i := 0; i < 3; i++ {
		mem.Release("gpu-0")
	}

	// Verify AffinityPicker contract explicitly, including skew shedding and drain handling.
	affRepl, ok := policy.PickWithAffinity(replicas, prefixSegments(sharedMsgs), "sess-alpha", func(name string) int {
		return 0
	})
	if !ok || affRepl.Name == "" {
		t.Fatalf("PickWithAffinity failed")
	}
	affRepl2, ok := policy.PickWithAffinity(replicas, []string{"unrelated"}, "sess-alpha", func(name string) int {
		return 0
	})
	if !ok || affRepl2.Name != affRepl.Name {
		t.Fatalf("sticky affinity drifted from %q to %q", affRepl.Name, affRepl2.Name)
	}
	affShed, ok := policy.PickWithAffinity(replicas, []string{"unrelated"}, "sess-alpha", func(name string) int {
		if name == affRepl.Name {
			return 5
		}
		return 0
	})
	if !ok || affShed.Name == affRepl.Name {
		t.Fatalf("overloaded affinity target %q did not shed: got %q", affRepl.Name, affShed.Name)
	}
	if st := policy.Stats(); st.Requests < 4 || st.Sheds == 0 {
		t.Fatalf("unexpected FleetCachePickStats: %+v", st)
	}
}

type perAspectFixture struct {
	localGPUs  []*simulatedLocalGPUBackend
	local32B   *simulatedLocalGPUBackend
	frontier   *simulatedLocalGPUBackend
	frontierN  *atomic.Int32
	srv        *Server
	router     *PerAspectRouter
	httpServer *httptest.Server
}

func newPerAspectFixture(t *testing.T, numLocalGPUs int, bounds modelroute.EscalationBounds, minFailures int, autoRetry bool) *perAspectFixture {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	engine.RegisterResidencyGate()

	localGPUs := make([]*simulatedLocalGPUBackend, numLocalGPUs)
	replicaURLs := make([]string, numLocalGPUs)
	for i := 0; i < numLocalGPUs; i++ {
		b := newSimulatedLocalGPUBackend(t, fmt.Sprintf("local-gpu-%d", i), 16)
		localGPUs[i] = b
		replicaURLs[i] = b.server.URL
	}
	local32B := newSimulatedLocalGPUBackend(t, "local-32b", 16)
	frontier := newSimulatedLocalGPUBackend(t, "frontier", 16)

	t.Setenv("FAK_TEST_FRONTIER_KEY", "frontier-secret-key")
	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "local-8b-node", Kind: modelroute.KindLocal, BaseURL: localGPUs[0].server.URL},
			{ID: "local-32b-node", Kind: modelroute.KindLocal, BaseURL: local32B.server.URL},
			{ID: "frontier-api", Kind: modelroute.KindOpenAI, BaseURL: frontier.server.URL, CredEnv: "FAK_TEST_FRONTIER_KEY"},
		},
		Bindings: []modelroute.Binding{
			{Model: "local-8b", Account: "local-8b-node", UpstreamModel: "qwen3-8b"},
			{Model: "local-32b", Account: "local-32b-node", UpstreamModel: "qwen3-32b"},
			{Model: "frontier", Account: "frontier-api", UpstreamModel: "gpt-frontier"},
		},
		Default: "local-8b-node",
	}

	manifest := &modelroute.Manifest{
		Version: modelroute.Version,
		Default: modelroute.Plan{Members: []modelroute.Member{{Model: "local-8b", Role: "primary"}}},
		Rules: []modelroute.Rule{
			{
				Name:  "single-turn-tool-local",
				Match: modelroute.Match{Aspect: modelroute.AspectToolCall},
				Plan:  modelroute.Plan{Members: []modelroute.Member{{Model: "local-8b", Role: "primary"}}},
			},
			{
				Name:  "single-turn-query-local",
				Match: modelroute.Match{Aspect: modelroute.AspectQuery},
				Plan:  modelroute.Plan{Members: []modelroute.Member{{Model: "local-8b", Role: "primary"}}},
			},
			{
				Name:  "step-32b-local",
				Match: modelroute.Match{Aspect: modelroute.AspectStep},
				Plan:  modelroute.Plan{Members: []modelroute.Member{{Model: "local-32b", Role: "primary"}}},
			},
			{
				Name:  "request-local",
				Match: modelroute.Match{Aspect: modelroute.AspectRequest},
				Plan:  modelroute.Plan{Members: []modelroute.Member{{Model: "local-8b", Role: "primary"}}},
			},
		},
	}

	// Register kernel engines for /v1/fak/syscall route-before-adjudicate witnesses.
	abi.RegisterEngine("local:local-8b-node/qwen3-8b", markerEngine{"local-8b"})
	abi.RegisterEngine("local:local-32b-node/qwen3-32b", markerEngine{"local-32b"})
	abi.RegisterEngine("openai:frontier-api/gpt-frontier", markerEngine{"frontier"})

	srv, err := New(Config{
		EngineID:        "test",
		Model:           "qwen3-8b-fleet",
		Provider:        "openai",
		ReplicaBaseURLs: replicaURLs,
		RouteManifest:   manifest,
		RouteAccounts:   roster,
		VDSO:            true,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	t.Cleanup(srv.Close)

	if numLocalGPUs > 1 {
		fleetPolicy := NewSlackTolerantFleetPickPolicy(numLocalGPUs, 16, 1.5)
		if !srv.UseFleetPickPolicy(context.Background(), fleetPolicy) {
			t.Fatal("UseFleetPickPolicy returned false")
		}
	}

	candidates := []modelroute.Candidate{
		{Model: "local-8b", Capability: modelroute.TierT2, Measured: true},
		{Model: "local-32b", Capability: modelroute.TierT1, Measured: true},
		{Model: "frontier", Capability: modelroute.TierT0, Measured: true},
	}
	router, err := NewPerAspectRouter(PerAspectRouterConfig{
		Roster:                    roster,
		Manifest:                  manifest,
		Candidates:                candidates,
		Bounds:                    bounds,
		MinFailuresBeforeEscalate: minFailures,
		MaxSameRungRetries:        2,
		LocalFleetModels:          []string{"local-8b"},
		AutoRetryInRequest:        autoRetry,
		Planners: map[string]agent.Planner{
			"local-8b": srv.FleetRouter(),
		},
	})
	if err != nil {
		t.Fatalf("NewPerAspectRouter: %v", err)
	}

	hs := httptest.NewServer(router.WrapHandler(srv))
	t.Cleanup(hs.Close)
	return &perAspectFixture{
		localGPUs:  localGPUs,
		local32B:   local32B,
		frontier:   frontier,
		srv:        srv,
		router:     router,
		httpServer: hs,
	}
}

func postAspectChat(t *testing.T, url, aspect, workItem, workClass, complexity, sensitivity string, msgs []agent.Message, tools []agent.ToolDef) (int, http.Header, []byte) {
	t.Helper()
	body, _ := json.Marshal(ChatRequest{
		Messages: msgs,
		Tools:    tools,
	})
	req, err := http.NewRequest(http.MethodPost, url+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if aspect != "" {
		req.Header.Set(HeaderFakAspect, aspect)
	}
	if workItem != "" {
		req.Header.Set(HeaderFakWorkItem, workItem)
	}
	if workClass != "" {
		req.Header.Set(HeaderFakWorkClass, workClass)
	}
	if complexity != "" {
		req.Header.Set(HeaderFakComplexity, complexity)
	}
	if sensitivity != "" {
		req.Header.Set(HeaderFakSensitivity, sensitivity)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, resp.Header, raw
}

func postAspectSyscall(t *testing.T, url, aspect, workItem, sensitivity string, sc SyscallRequest) (int, http.Header, SyscallResponse) {
	t.Helper()
	body, _ := json.Marshal(sc)
	req, err := http.NewRequest(http.MethodPost, url+"/v1/fak/syscall", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if aspect != "" {
		req.Header.Set(HeaderFakAspect, aspect)
	}
	if workItem != "" {
		req.Header.Set(HeaderFakWorkItem, workItem)
	}
	if sensitivity != "" {
		req.Header.Set(HeaderFakSensitivity, sensitivity)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/fak/syscall: %v", err)
	}
	defer resp.Body.Close()
	var out SyscallResponse
	raw, _ := io.ReadAll(resp.Body)
	_ = json.Unmarshal(raw, &out)
	return resp.StatusCode, resp.Header, out
}

// fak-test:justify why=integration when=changed:internal/gateway/**
// fak-test:runtime fast est=100ms lane=default
func TestPerAspectRouteSingleTurnLocal3BTo32BAcrossAspects(t *testing.T) {
	fx := newPerAspectFixture(t, 4, modelroute.EscalationBounds{
		Ceiling:     modelroute.ZoneVendor,
		MaxAttempts: 2,
	}, 1, false)

	// Exercise all four aspects (request, tool_call, query, step) via live HTTP.
	cases := []struct {
		aspect    string
		wantModel string
	}{
		{"request", "local-8b"},
		{"tool_call", "local-8b"},
		{"query", "local-8b"},
		{"step", "local-32b"},
	}
	for _, tc := range cases {
		code, hdr, raw := postAspectChat(t, fx.httpServer.URL, tc.aspect, "single-"+tc.aspect, "", "", "",
			[]agent.Message{{Role: agent.RoleUser, Content: "hello " + tc.aspect}}, nil)
		if code != http.StatusOK {
			t.Fatalf("aspect=%s HTTP %d: %s", tc.aspect, code, raw)
		}
		if gotZone := hdr.Get(HeaderFakRouteZone); gotZone != string(modelroute.ZoneDevice) {
			t.Fatalf("aspect=%s zone = %q, want device (local 3B-32B)", tc.aspect, gotZone)
		}
		if gotModel := hdr.Get(HeaderFakRouteModel); gotModel != tc.wantModel {
			t.Fatalf("aspect=%s model = %q, want %q", tc.aspect, gotModel, tc.wantModel)
		}
		if hdr.Get(HeaderFakRouteEscalated) != "false" {
			t.Fatalf("aspect=%s unexpectedly escalated: %v", tc.aspect, hdr)
		}
		if hdr.Get(HeaderFakEscalationAct) != string(modelroute.ActionAccept) {
			t.Fatalf("aspect=%s action = %q, want accept", tc.aspect, hdr.Get(HeaderFakEscalationAct))
		}
		if tc.wantModel == "local-32b" && !strings.Contains(string(raw), "gpu=local-32b") {
			t.Fatalf("aspect=%s did not reach local-32b backend: %s", tc.aspect, raw)
		}
	}
	if l32Reqs, _ := fx.local32B.stats(); l32Reqs != 1 {
		t.Fatalf("local-32b backend requests = %d, want 1", l32Reqs)
	}

	// Single-turn tool call via /v1/fak/syscall also routes locally (local-8b) through abi.Kernel.
	scCode, scHdr, scResp := postAspectSyscall(t, fx.httpServer.URL, "tool_call", "sc-single-1", "", SyscallRequest{
		Tool:      "allow_search_kb",
		Arguments: json.RawMessage(`{"q":"status"}`),
		ReadOnly:  true,
		TraceID:   "trace-sc-1",
	})
	if scCode != http.StatusOK || scResp.Verdict.Kind != "ALLOW" {
		t.Fatalf("syscall status=%d verdict=%+v", scCode, scResp.Verdict)
	}
	if scResp.Result == nil || scResp.Result.Meta["engine"] != "local-8b" {
		t.Fatalf("syscall routed to %+v, want local-8b", scResp.Result)
	}
	if scHdr.Get(HeaderFakRouteZone) != string(modelroute.ZoneDevice) {
		t.Fatalf("syscall header zone = %q, want device", scHdr.Get(HeaderFakRouteZone))
	}

	// Frontier backend must receive ZERO calls when single-turn local aspects succeed.
	if fReqs, _ := fx.frontier.stats(); fReqs != 0 {
		t.Fatalf("frontier received %d call(s) on single-turn local workload, want 0", fReqs)
	}

	// Verify Prometheus routing observability recorded all four aspects.
	mResp, err := http.Get(fx.httpServer.URL + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer mResp.Body.Close()
	metricsRaw, _ := io.ReadAll(mResp.Body)
	metricsStr := string(metricsRaw)
	for _, asp := range []string{"request", "tool_call", "query", "step"} {
		wantMetric := fmt.Sprintf(`fak_gateway_routing_decisions_by_aspect_total{aspect=%q}`, asp)
		if !strings.Contains(metricsStr, wantMetric) {
			t.Fatalf("/metrics missing %s", wantMetric)
		}
	}
}

// fak-test:justify why=integration when=changed:internal/gateway/**
// fak-test:runtime fast est=100ms lane=default
func TestPerAspectRouteMultiTurnFailureEscalatesViaAfterAttempt(t *testing.T) {
	// Require 2 consecutive underpowered failures on local 8B replicas before escalating
	// via modelroute.AfterAttempt to frontier.
	fx := newPerAspectFixture(t, 4, modelroute.EscalationBounds{
		Ceiling:     modelroute.ZoneVendor,
		MaxAttempts: 2,
	}, 2, false)

	for _, b := range fx.localGPUs {
		b.underpower.Store(true)
	}

	workItem := "multi-turn-bugfix-42"
	msgs := []agent.Message{
		{Role: agent.RoleSystem, Content: "fix complex invariant"},
		{Role: agent.RoleUser, Content: "attempt tool edit"},
	}

	// Turn 1: lands on local GPU replica, fails underpowered -> stays local (retry same rung).
	code1, hdr1, _ := postAspectChat(t, fx.httpServer.URL, "tool_call", workItem, "", "", "", msgs, nil)
	if code1 != http.StatusUnprocessableEntity {
		t.Fatalf("turn 1 status = %d, want 422", code1)
	}
	if hdr1.Get(HeaderFakRouteZone) != string(modelroute.ZoneDevice) {
		t.Fatalf("turn 1 zone = %q, want device", hdr1.Get(HeaderFakRouteZone))
	}
	if hdr1.Get(HeaderFakEscalationAct) != string(modelroute.ActionRetrySameRung) {
		t.Fatalf("turn 1 action = %q, want retry-same-rung", hdr1.Get(HeaderFakEscalationAct))
	}

	// Turn 2: still local, fails underpowered again -> triggers modelroute.AfterAttempt escalation!
	code2, hdr2, _ := postAspectChat(t, fx.httpServer.URL, "tool_call", workItem, "", "", "", msgs, nil)
	if code2 != http.StatusUnprocessableEntity {
		t.Fatalf("turn 2 status = %d, want 422", code2)
	}
	if hdr2.Get(HeaderFakRouteZone) != string(modelroute.ZoneDevice) {
		t.Fatalf("turn 2 zone = %q, want device", hdr2.Get(HeaderFakRouteZone))
	}
	if hdr2.Get(HeaderFakEscalationAct) != string(modelroute.ActionEscalate) ||
		hdr2.Get(HeaderFakEscalationWhy) != modelroute.ReasonEarnedByUnderpower {
		t.Fatalf("turn 2 escalation = %q/%q, want %q/%q",
			hdr2.Get(HeaderFakEscalationAct), hdr2.Get(HeaderFakEscalationWhy),
			modelroute.ActionEscalate, modelroute.ReasonEarnedByUnderpower)
	}

	// Turn 3: work item is now escalated to frontier (ZoneVendor) and succeeds!
	code3, hdr3, raw3 := postAspectChat(t, fx.httpServer.URL, "tool_call", workItem, "", "", "", msgs, nil)
	if code3 != http.StatusOK {
		t.Fatalf("turn 3 escalated status = %d: %s", code3, raw3)
	}
	if hdr3.Get(HeaderFakRouteZone) != string(modelroute.ZoneVendor) ||
		hdr3.Get(HeaderFakRouteModel) != "frontier" ||
		hdr3.Get(HeaderFakRouteEscalated) != "true" {
		t.Fatalf("turn 3 headers = zone:%q model:%q escalated:%q, want vendor/frontier/true",
			hdr3.Get(HeaderFakRouteZone), hdr3.Get(HeaderFakRouteModel), hdr3.Get(HeaderFakRouteEscalated))
	}
	if !strings.Contains(string(raw3), "gpu=frontier") {
		t.Fatalf("turn 3 did not reach frontier backend: %s", raw3)
	}
	if fReqs, _ := fx.frontier.stats(); fReqs != 1 {
		t.Fatalf("frontier requests = %d, want 1", fReqs)
	}
	st, ok := fx.router.State(workItem)
	if !ok || st.PriorEscalations != 1 || st.Attempts != 3 {
		t.Fatalf("tracked work item state = %+v, want PriorEscalations=1 Attempts=3", st)
	}

	// Also exercise AutoRetryInRequest=true across an 8-GPU local fleet + frontier.
	fxAuto := newPerAspectFixture(t, 8, modelroute.EscalationBounds{
		Ceiling:     modelroute.ZoneVendor,
		MaxAttempts: 1,
	}, 1, true)
	for _, b := range fxAuto.localGPUs {
		b.underpower.Store(true)
	}
	codeAuto, hdrAuto, rawAuto := postAspectChat(t, fxAuto.httpServer.URL, "query", "auto-item-1", "", "", "", msgs, nil)
	if codeAuto != http.StatusOK || !strings.Contains(string(rawAuto), "gpu=frontier") {
		t.Fatalf("auto-escalate status=%d body=%s", codeAuto, rawAuto)
	}
	if hdrAuto.Get(HeaderFakRouteZone) != string(modelroute.ZoneVendor) || hdrAuto.Get(HeaderFakRouteEscalated) != "true" {
		t.Fatalf("auto-escalate headers = zone:%q escalated:%q", hdrAuto.Get(HeaderFakRouteZone), hdrAuto.Get(HeaderFakRouteEscalated))
	}
	var localHits int
	for _, b := range fxAuto.localGPUs {
		r, _ := b.stats()
		localHits += r
	}
	if localHits != len(fxAuto.localGPUs) {
		t.Fatalf("local 8-GPU fleet hits before escalation = %d, want %d (one per local GPU replica)", localHits, len(fxAuto.localGPUs))
	}
	if fReqs, _ := fxAuto.frontier.stats(); fReqs != 1 {
		t.Fatalf("auto-escalate frontier requests = %d, want 1", fReqs)
	}
}

// fak-test:justify why=integration when=changed:internal/gateway/**
// fak-test:runtime fast est=120ms lane=default
func TestPerAspectRouteEscalationBoundsAndFailClosed(t *testing.T) {
	t.Run("device_ceiling_blocks_frontier_and_fails_closed", func(t *testing.T) {
		fx := newPerAspectFixture(t, 4, modelroute.EscalationBounds{
			Ceiling:     modelroute.ZoneDevice,
			MaxAttempts: 3,
		}, 1, true)
		for _, b := range fx.localGPUs {
			b.underpower.Store(true)
		}

		code, hdr, _ := postAspectChat(t, fx.httpServer.URL, "tool_call", "onbox-work-1", "", "", "",
			[]agent.Message{{Role: agent.RoleUser, Content: "sensitive local task"}}, nil)
		if code != http.StatusServiceUnavailable {
			t.Fatalf("status = %d, want 503 fail-closed when ceiling=device", code)
		}
		if hdr.Get(HeaderFakEscalationAct) != string(modelroute.ActionStop) ||
			hdr.Get(HeaderFakEscalationWhy) != modelroute.ReasonAtCeiling {
			t.Fatalf("stop headers = %q/%q, want stop/%s",
				hdr.Get(HeaderFakEscalationAct), hdr.Get(HeaderFakEscalationWhy), modelroute.ReasonAtCeiling)
		}
		if fReqs, _ := fx.frontier.stats(); fReqs != 0 {
			t.Fatalf("frontier received %d calls under ZoneDevice ceiling, want 0", fReqs)
		}
	})

	t.Run("zero_bounds_grant_no_escalation", func(t *testing.T) {
		fx := newPerAspectFixture(t, 2, modelroute.EscalationBounds{}, 1, true)
		for _, b := range fx.localGPUs {
			b.underpower.Store(true)
		}

		code, hdr, _ := postAspectChat(t, fx.httpServer.URL, "query", "ungranted-1", "", "", "",
			[]agent.Message{{Role: agent.RoleUser, Content: "query"}}, nil)
		if code != http.StatusServiceUnavailable {
			t.Fatalf("status = %d, want 503 fail-closed with zero bounds", code)
		}
		if hdr.Get(HeaderFakEscalationAct) != string(modelroute.ActionStop) ||
			hdr.Get(HeaderFakEscalationWhy) != modelroute.ReasonNoCeiling {
			t.Fatalf("stop headers = %q/%q, want stop/%s",
				hdr.Get(HeaderFakEscalationAct), hdr.Get(HeaderFakEscalationWhy), modelroute.ReasonNoCeiling)
		}
		if fReqs, _ := fx.frontier.stats(); fReqs != 0 {
			t.Fatalf("frontier received %d calls under zero bounds, want 0", fReqs)
		}
	})

	t.Run("exhausted_budget_and_top_rung_fail_closed", func(t *testing.T) {
		fx := newPerAspectFixture(t, 2, modelroute.EscalationBounds{
			Ceiling:     modelroute.ZoneVendor,
			MaxAttempts: 1,
		}, 1, false)
		for _, b := range fx.localGPUs {
			b.underpower.Store(true)
		}
		fx.local32B.underpower.Store(true)
		fx.frontier.underpower.Store(true)

		// Attempt 1 on local -> escalates (prior 0 -> 1).
		_, hdr1, _ := postAspectChat(t, fx.httpServer.URL, "step", "budget-1", "", "", "",
			[]agent.Message{{Role: agent.RoleUser, Content: "hard step"}}, nil)
		if hdr1.Get(HeaderFakEscalationAct) != string(modelroute.ActionEscalate) {
			t.Fatalf("attempt 1 action = %q, want escalate", hdr1.Get(HeaderFakEscalationAct))
		}
		// Attempt 2 on frontier -> fails underpowered at top rung -> ActionStop and exhausted.
		_, hdr2, _ := postAspectChat(t, fx.httpServer.URL, "step", "budget-1", "", "", "",
			[]agent.Message{{Role: agent.RoleUser, Content: "hard step"}}, nil)
		if hdr2.Get(HeaderFakEscalationAct) != string(modelroute.ActionStop) ||
			hdr2.Get(HeaderFakEscalationWhy) != modelroute.ReasonAtTopRung {
			t.Fatalf("attempt 2 headers = %q/%q, want stop/%s",
				hdr2.Get(HeaderFakEscalationAct), hdr2.Get(HeaderFakEscalationWhy), modelroute.ReasonAtTopRung)
		}
		// Attempt 3 on exhausted work item -> refused fail-closed before dialing any backend.
		beforeF, _ := fx.frontier.stats()
		code3, hdr3, _ := postAspectChat(t, fx.httpServer.URL, "step", "budget-1", "", "", "",
			[]agent.Message{{Role: agent.RoleUser, Content: "hard step"}}, nil)
		afterF, _ := fx.frontier.stats()
		if code3 != http.StatusServiceUnavailable || hdr3.Get(HeaderFakEscalationAct) != string(modelroute.ActionStop) {
			t.Fatalf("exhausted attempt status=%d headers=%v, want 503/stop", code3, hdr3)
		}
		if afterF != beforeF {
			t.Fatalf("exhausted work item still dialed frontier: %d -> %d", beforeF, afterF)
		}
	})

	t.Run("refusal_and_sensitive_payload_never_escalate_to_frontier", func(t *testing.T) {
		fx := newPerAspectFixture(t, 2, modelroute.EscalationBounds{
			Ceiling:     modelroute.ZoneVendor,
			MaxAttempts: 5,
		}, 1, true)

		// Policy refusal (kernel DENY on /v1/fak/syscall) -> FailRefused -> ActionStop.
		scCode, scHdr, scResp := postAspectSyscall(t, fx.httpServer.URL, "tool_call", "refused-sc-1", "", SyscallRequest{
			Tool:      "deny_rm_rf",
			Arguments: json.RawMessage(`{}`),
			TraceID:   "trace-deny-1",
		})
		if scCode != http.StatusOK || scResp.Verdict.Kind != "DENY" {
			t.Fatalf("refused syscall status=%d verdict=%+v", scCode, scResp.Verdict)
		}
		if scHdr.Get(HeaderFakEscalationAct) != string(modelroute.ActionStop) ||
			scHdr.Get(HeaderFakEscalationWhy) != modelroute.ReasonRefusalNotRetried {
			t.Fatalf("refusal headers = %q/%q, want stop/%s",
				scHdr.Get(HeaderFakEscalationAct), scHdr.Get(HeaderFakEscalationWhy), modelroute.ReasonRefusalNotRetried)
		}

		// Sensitive tenant payload clamps Ceiling to ZoneDevice: underpowered local failure stops on-box.
		for _, b := range fx.localGPUs {
			b.underpower.Store(true)
		}
		codeSens, hdrSens, _ := postAspectChat(t, fx.httpServer.URL, "request", "tenant-sens-1", "", "", "tenant",
			[]agent.Message{{Role: agent.RoleUser, Content: "tenant secret"}}, nil)
		if codeSens != http.StatusServiceUnavailable || hdrSens.Get(HeaderFakEscalationWhy) != modelroute.ReasonAtCeiling {
			t.Fatalf("sensitive payload status=%d reason=%q, want 503/%s",
				codeSens, hdrSens.Get(HeaderFakEscalationWhy), modelroute.ReasonAtCeiling)
		}
		if fReqs, _ := fx.frontier.stats(); fReqs != 0 {
			t.Fatalf("frontier received %d calls on refusal/sensitive workload, want 0", fReqs)
		}
	})

	t.Run("transport_failure_retries_local_never_escalates", func(t *testing.T) {
		// Verify direct PerAspectRouter.Execute handling of FailTransport: retries same rung,
		// never escalates to frontier, and fails closed when same-rung retries exhaust.
		fx := newPerAspectFixture(t, 2, modelroute.EscalationBounds{
			Ceiling:     modelroute.ZoneVendor,
			MaxAttempts: 3,
		}, 1, false)
		attempts := 0
		res, err := fx.router.Execute(context.Background(), AspectRouteRequest{
			WorkItemID: "transport-item-1",
			Subject:    modelroute.Subject{Aspect: modelroute.AspectQuery},
		}, func(_ context.Context, dec AspectRouteDecision) (modelroute.AttemptResult, error) {
			attempts++
			if dec.Placement.Zone != modelroute.ZoneDevice {
				t.Fatalf("transport retry escalated off device to %q", dec.Placement.Zone)
			}
			if attempts == 1 {
				return modelroute.AttemptResult{Fail: modelroute.FailTransport}, errors.New("connection reset")
			}
			return modelroute.AttemptResult{Succeeded: true, Verify: modelroute.VerifyWitness}, nil
		})
		if err != nil {
			t.Fatalf("Execute transport recovery failed: %v", err)
		}
		if attempts != 2 || res.FinalVerdict.Action != modelroute.ActionAccept || res.FinalDecision.PriorEscalations != 0 {
			t.Fatalf("unexpected transport recovery result: attempts=%d res=%+v", attempts, res)
		}
		if fReqs, _ := fx.frontier.stats(); fReqs != 0 {
			t.Fatalf("frontier received %d calls during local transport retry, want 0", fReqs)
		}
	})
}
