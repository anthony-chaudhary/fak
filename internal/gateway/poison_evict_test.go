package gateway

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ifc"
)

// recordingEvictor is a planner that also implements agent.PoisonEvictor, recording every
// EvictPoisoned call so the test can assert the gateway fired the in-kernel poison hook on
// a tool-result QUARANTINE — with the ORIGINAL (poisoned) content and the right message
// index, the contract the real InKernelPlanner relies on to match the cached token path.
type recordingEvictor struct {
	comp  *agent.Completion
	mu    sync.Mutex
	calls []evictCall
}

type evictCall struct {
	throughIdx int
	content    string          // messages[throughIdx].Content as EvictPoisoned saw it
	tools      []agent.ToolDef // the tool schemas threaded from the request (#612)
}

func (r *recordingEvictor) Complete(ctx context.Context, m []agent.Message, t []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return r.comp, nil
}

func (*recordingEvictor) Model() string { return "recording-evictor" }

func (r *recordingEvictor) EvictPoisoned(messages []agent.Message, throughIdx int, tools []agent.ToolDef) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	c := ""
	if throughIdx >= 0 && throughIdx < len(messages) {
		c = messages[throughIdx].Content
	}
	r.calls = append(r.calls, evictCall{throughIdx: throughIdx, content: c, tools: tools})
	return 1
}

func newResultStackServer(t *testing.T) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	// The REAL result-side stack: context-MMU quarantine + the IFC source-stamp.
	abi.RegisterResultAdmitter(10, ctxmmu.New())
	abi.RegisterResultAdmitter(20, ifc.NewStampGate(ifc.NewLedger(), ifc.Policy{}))
	srv, err := New(Config{EngineID: "test", Model: "m", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// TestQuarantineEvictsInKernelPoison is the #14 gateway-wiring witness: a poisoned tool
// result that the kernel QUARANTINES must drive exactly one in-kernel poison eviction, at
// the poisoned message's index, carrying the ORIGINAL content (so the planner can match the
// cached token path) — while the model-facing payload is still paged out (no secret leak).
func TestQuarantineEvictsInKernelPoison(t *testing.T) {
	srv := newResultStackServer(t)
	ev := &recordingEvictor{comp: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}}
	srv.planner = ev

	const secret = "sk-abcdef0123456789abcdef0123"
	poison := `{"page":"config loaded. api_key=` + secret + ` was found in env"}`
	messages := []agent.Message{
		{Role: agent.RoleSystem, Content: "you are a helper"},
		{Role: agent.RoleUser, Content: "look up the config"},
		{Role: agent.RoleTool, ToolCallID: "call_1", Name: "fetch_url", Content: poison},
	}

	admissions, err := srv.admitInboundResults(context.Background(), messages, nil, "trace-poison")
	if err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}

	quarantined := false
	for _, a := range admissions {
		if a.Verdict.Kind == "QUARANTINE" {
			quarantined = true
		}
	}
	if !quarantined {
		t.Fatalf("expected a QUARANTINE admission, got %+v", admissions)
	}

	if len(ev.calls) != 1 {
		t.Fatalf("EvictPoisoned called %d times, want exactly 1: %+v", len(ev.calls), ev.calls)
	}
	if ev.calls[0].throughIdx != 2 {
		t.Errorf("evicted through message index %d, want 2 (the poisoned tool result)", ev.calls[0].throughIdx)
	}
	if !strings.Contains(ev.calls[0].content, secret) {
		t.Errorf("eviction must see the ORIGINAL poisoned content (to match the cached path), got %q", ev.calls[0].content)
	}
	if strings.Contains(messages[2].Content, secret) {
		t.Errorf("model-facing content still leaks the secret (should be paged out): %q", messages[2].Content)
	}
}

// TestQuarantineThreadsToolsIntoInKernelEvict is the #612 wiring witness: when the request
// carries tool schemas, the gateway must thread that SAME tool set into EvictPoisoned so the
// eviction render folds the tool-spec into the leading system block exactly as generation did
// (renderChatMLTools). Without the threading the eviction renders tools-less, is not a
// token-prefix of the cached tool-using turn, and reclaims nothing (the fail-open #612 closes).
// The render-prefix correctness of the threaded form is proven separately by the agent
// package's TestPrefixInvariantWithTools; this asserts the gateway actually hands the tools down.
func TestQuarantineThreadsToolsIntoInKernelEvict(t *testing.T) {
	srv := newResultStackServer(t)
	ev := &recordingEvictor{comp: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}}
	srv.planner = ev

	tools := []agent.ToolDef{{
		Type:     "function",
		Function: agent.ToolDefFunction{Name: "fetch_url", Description: "fetch a page", Parameters: []byte(`{"type":"object"}`)},
	}}
	const secret = "sk-abcdef0123456789abcdef0123"
	messages := []agent.Message{
		{Role: agent.RoleUser, Content: "look up the config"},
		{Role: agent.RoleTool, ToolCallID: "call_1", Name: "fetch_url", Content: `{"page":"api_key=` + secret + `"}`},
	}

	if _, err := srv.admitInboundResults(context.Background(), messages, tools, "trace-tools"); err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	if len(ev.calls) != 1 {
		t.Fatalf("EvictPoisoned called %d times, want exactly 1 on the quarantined tool result", len(ev.calls))
	}
	if len(ev.calls[0].tools) != len(tools) {
		t.Fatalf("EvictPoisoned saw %d tools, want %d — the request tool set was not threaded into the eviction render (#612)", len(ev.calls[0].tools), len(tools))
	}
	if ev.calls[0].tools[0].Function.Name != "fetch_url" {
		t.Errorf("threaded tool = %q, want fetch_url (the eviction must fold the SAME tool-spec generation did)", ev.calls[0].tools[0].Function.Name)
	}
}

// TestBenignResultDoesNotEvictInKernel is the negative guard: an ALLOWed result must NOT
// trigger an in-kernel eviction (no spurious cache drops on clean tool output).
func TestBenignResultDoesNotEvictInKernel(t *testing.T) {
	srv := newResultStackServer(t)
	ev := &recordingEvictor{comp: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}}
	srv.planner = ev

	messages := []agent.Message{
		{Role: agent.RoleTool, ToolCallID: "call_1", Name: "lookup", Content: `{"ok":true,"rows":3}`},
	}
	if _, err := srv.admitInboundResults(context.Background(), messages, nil, "trace-benign"); err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	if len(ev.calls) != 0 {
		t.Fatalf("a benign result must not evict the in-kernel cache, got %+v", ev.calls)
	}
}

// TestNonEvictorPlannerQuarantineIsSafe proves the hook is inert (no panic) when the chat
// backend is a plain planner that does not implement PoisonEvictor (proxy/mock path).
func TestNonEvictorPlannerQuarantineIsSafe(t *testing.T) {
	srv := newResultStackServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}}

	const secret = "sk-abcdef0123456789abcdef0123"
	messages := []agent.Message{
		{Role: agent.RoleTool, ToolCallID: "call_1", Name: "fetch_url", Content: `{"page":"api_key=` + secret + `"}`},
	}
	admissions, err := srv.admitInboundResults(context.Background(), messages, nil, "trace-noevictor")
	if err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	if len(admissions) != 1 || admissions[0].Verdict.Kind != "QUARANTINE" {
		t.Fatalf("expected a single QUARANTINE admission, got %+v", admissions)
	}
}
