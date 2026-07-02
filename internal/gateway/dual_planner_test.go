package gateway

// dual_planner_test.go — the small-local-model-ALONGSIDE-API seam (dual_planner.go).
// The contract under test: in dual mode the proxy side is the DEFAULT (an omitted or
// API model id is byte-for-byte the proxy-only path — including the Anthropic
// byte-preserving passthrough), while a request naming the local model id (or the
// literal "local") decodes in-kernel and must NEVER ride raw bytes upstream.

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// dualSide is a minimal recording planner standing in for one side of the split.
type dualSide struct {
	id    string
	calls int
}

func (p *dualSide) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "from:" + p.id}}, nil
}

func (p *dualSide) Model() string { return p.id }

// dualStreamSide additionally implements the StreamingPlanner seam.
type dualStreamSide struct {
	dualSide
	streamCalls int
}

func (p *dualStreamSide) StreamingSupported() bool { return true }

func (p *dualStreamSide) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.streamCalls++
	if err := sink("stream:" + p.id); err != nil {
		return nil, err
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "stream:" + p.id}}, nil
}

func TestDualPlannerRoutesByRequestedModel(t *testing.T) {
	proxy := &dualSide{id: "claude-sonnet-5"}
	local := &dualSide{id: "qwen2.5-coder:3b"}
	d, err := NewDualPlanner(proxy, local, "qwen2.5-coder:3b")
	if err != nil {
		t.Fatal(err)
	}
	if got := d.Model(); got != "claude-sonnet-5" {
		t.Errorf("Model() = %q, want the PROXY id (the default side)", got)
	}
	if got := d.LocalModelID(); got != "qwen2.5-coder:3b" {
		t.Errorf("LocalModelID() = %q, want the configured local id", got)
	}

	cases := []struct {
		name      string
		reqModel  string
		wantLocal bool
	}{
		{"omitted model routes proxy", "", false},
		{"proxy model id routes proxy", "claude-sonnet-5", false},
		{"unknown model id routes proxy (the upstream surfaces its own 404)", "gpt-77", false},
		{"local model id routes local", "qwen2.5-coder:3b", true},
		{"local id is case-insensitive", "QWEN2.5-Coder:3B", true},
		{"local id is whitespace-tolerant", "  qwen2.5-coder:3b ", true},
		{"the literal alias routes local", "local", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := d.RoutesLocal(tc.reqModel); got != tc.wantLocal {
				t.Fatalf("RoutesLocal(%q) = %v, want %v", tc.reqModel, got, tc.wantLocal)
			}
			proxy.calls, local.calls = 0, 0
			comp, err := d.Complete(context.Background(), nil, nil, agent.WithModel(tc.reqModel))
			if err != nil {
				t.Fatal(err)
			}
			wantFrom, wantCalls := "from:claude-sonnet-5", &proxy.calls
			if tc.wantLocal {
				wantFrom, wantCalls = "from:qwen2.5-coder:3b", &local.calls
			}
			if comp.Message.Content != wantFrom {
				t.Errorf("Complete(model=%q) answered by %q, want %q", tc.reqModel, comp.Message.Content, wantFrom)
			}
			if *wantCalls != 1 || proxy.calls+local.calls != 1 {
				t.Errorf("Complete(model=%q) call counts proxy=%d local=%d — exactly one side must serve", tc.reqModel, proxy.calls, local.calls)
			}
		})
	}
}

func TestNewDualPlannerValidates(t *testing.T) {
	if _, err := NewDualPlanner(nil, &dualSide{id: "l"}, "l"); err == nil {
		t.Error("nil proxy must be refused")
	}
	if _, err := NewDualPlanner(&dualSide{id: "p"}, nil, "l"); err == nil {
		t.Error("nil local must be refused")
	}
	// An empty local id defaults to "local" rather than capturing omitted-model requests.
	d, err := NewDualPlanner(&dualSide{id: "p"}, &dualSide{id: ""}, "")
	if err != nil {
		t.Fatal(err)
	}
	if got := d.LocalModelID(); got != "local" {
		t.Errorf("empty local id defaulted to %q, want \"local\"", got)
	}
	if d.RoutesLocal("") {
		t.Error("an omitted model must route to the proxy (default side), never local")
	}
}

// The dual planner advertises the planner-seam stream when either side can stream, and
// EMULATES for a picked side that cannot (one final fragment) — the gateway never has
// to unwind a streaming path it committed to before knowing the requested model.
func TestDualPlannerStreamEmulation(t *testing.T) {
	proxy := &dualStreamSide{dualSide: dualSide{id: "api-model"}}
	local := &dualSide{id: "tiny-local"} // NOT a StreamingPlanner (like InKernelPlanner)
	d, err := NewDualPlanner(proxy, local, "tiny-local")
	if err != nil {
		t.Fatal(err)
	}
	if !d.StreamingSupported() {
		t.Fatal("StreamingSupported() = false, want true when the proxy side streams")
	}

	// API-bound request: delegated to the proxy's live stream.
	var got []string
	sink := func(delta string) error { got = append(got, delta); return nil }
	if _, err := d.CompleteStream(context.Background(), sink, nil, nil, agent.WithModel("api-model")); err != nil {
		t.Fatal(err)
	}
	if proxy.streamCalls != 1 || strings.Join(got, "|") != "stream:api-model" {
		t.Errorf("API-bound stream: streamCalls=%d deltas=%v, want the proxy's live stream", proxy.streamCalls, got)
	}

	// Local-bound request: the non-streaming side is emulated as one final fragment.
	got = nil
	comp, err := d.CompleteStream(context.Background(), sink, nil, nil, agent.WithModel("tiny-local"))
	if err != nil {
		t.Fatal(err)
	}
	if local.calls != 1 {
		t.Errorf("local side Complete calls = %d, want 1 (buffered emulation)", local.calls)
	}
	if strings.Join(got, "|") != "from:tiny-local" || comp.Message.Content != "from:tiny-local" {
		t.Errorf("local-bound stream: deltas=%v comp=%q, want the buffered content as one fragment", got, comp.Message.Content)
	}

	// Neither side streams → the planner seam reports no stream (buffered path).
	nostream, err := NewDualPlanner(&dualSide{id: "p"}, &dualSide{id: "l"}, "l")
	if err != nil {
		t.Fatal(err)
	}
	if nostream.StreamingSupported() {
		t.Error("StreamingSupported() = true with no streaming side, want false")
	}
}

// Per-request passthrough: in dual mode an API-bound request keeps the Anthropic
// byte-preserving passthrough (prompt-cache preservation), while a request addressed
// to the LOCAL model must not be a passthrough — its bytes decode in-kernel and never
// reach the remote API.
func TestAnthropicPassthroughForDualMode(t *testing.T) {
	proxy := &agent.HTTPPlanner{Provider: agent.ProviderAnthropic}
	local := &dualSide{id: "tiny-local"}
	d, err := NewDualPlanner(proxy, local, "tiny-local")
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{planner: d, logf: func(string, ...any) {}}

	if !s.anthropicPassthrough() {
		t.Error("planner-level anthropicPassthrough() must unwrap the dual proxy side (anthropic wire)")
	}
	if !s.anthropicPassthroughFor("") {
		t.Error("omitted model must stay a passthrough (the proxy is the default side)")
	}
	if !s.anthropicPassthroughFor("claude-sonnet-5") {
		t.Error("an API model id must stay a passthrough")
	}
	if s.anthropicPassthroughFor("tiny-local") {
		t.Error("a LOCAL-model request must NOT be a passthrough — its bytes must never ride upstream")
	}
	if s.anthropicPassthroughFor("local") {
		t.Error("the literal \"local\" alias must NOT be a passthrough")
	}

	// The unwrap helper feeds the passthrough relay + retry observability seams.
	if unwrapHTTPPlanner(d) != proxy {
		t.Error("unwrapHTTPPlanner(dual) must expose the proxy side")
	}
	if unwrapHTTPPlanner(local) != nil {
		t.Error("unwrapHTTPPlanner(non-HTTP planner) must be nil")
	}
}

// New()-level wiring: a live proxy AND a loaded in-kernel model+tokenizer in one
// Config selects the dual planner (historically the proxy silently won and the loaded
// weights were dead), /healthz reports planner:"dual", and /v1/models advertises BOTH
// ids so an OpenAI-wire client can discover the local side.
func TestNewSelectsDualPlannerAndAdvertisesBothModels(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	srv, err := New(Config{
		EngineID:      "test",
		Model:         "api-default",
		BaseURL:       "http://127.0.0.1:1/v1",
		Provider:      "openai",
		InKernelModel: &model.Model{},
		Tokenizer:     &tokenizer.Tokenizer{},
		LocalModelID:  "qwen2.5-coder:3b",
		VDSO:          true,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)

	if kind := plannerKind(srv.planner); kind != "dual" {
		t.Fatalf("plannerKind = %q, want \"dual\" when BaseURL and InKernelModel+Tokenizer are both configured", kind)
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var health map[string]any
	getJSON(t, ts.URL+"/healthz", &health)
	if health["planner"] != "dual" {
		t.Errorf(`/healthz planner = %v, want "dual"`, health["planner"])
	}

	var models struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	resp, err := ts.Client().Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(&models); err != nil {
		t.Fatal(err)
	}
	ids := make([]string, 0, len(models.Data))
	for _, m := range models.Data {
		ids = append(ids, m.ID)
	}
	joined := strings.Join(ids, ",")
	if !strings.Contains(joined, "api-default") || !strings.Contains(joined, "qwen2.5-coder:3b") {
		t.Errorf("/v1/models = %v, want BOTH the API default and the local model id", ids)
	}
}
