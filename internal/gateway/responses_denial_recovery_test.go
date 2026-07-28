package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// sequencePlanner answers successive Complete calls from a script and records the
// messages each call was given. It is what makes the #5212 regression end-to-end: the
// defect is precisely that the SECOND sample never happened, so a planner that can
// distinguish "called once" from "called twice — and here is what it was told" is the
// instrument the fix has to move.
type sequencePlanner struct {
	mu    sync.Mutex
	comps []*agent.Completion
	seen  [][]agent.Message
}

func (p *sequencePlanner) Complete(ctx context.Context, m []agent.Message, t []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.seen = append(p.seen, append([]agent.Message(nil), m...))
	if len(p.comps) == 0 {
		return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant}, FinishReason: "stop"}, nil
	}
	comp := p.comps[0]
	if len(p.comps) > 1 {
		p.comps = p.comps[1:]
	}
	return comp, nil
}

func (*sequencePlanner) Model() string { return "sequence" }

func (p *sequencePlanner) calls() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.seen)
}

func (p *sequencePlanner) request(i int) []agent.Message {
	p.mu.Lock()
	defer p.mu.Unlock()
	if i >= len(p.seen) {
		return nil
	}
	return p.seen[i]
}

func toolCallTurn(id, name, args string) *agent.Completion {
	return &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{
			{ID: id, Type: "function", Function: agent.Func{Name: name, Arguments: args}},
		}},
		FinishReason: "tool_calls",
	}
}

// TestResponsesDenialReturnsControlToTheModel is the #5212 regression.
//
// A guarded Codex turn proposes one shell call; the floor refuses it. Before this fix
// the gateway dropped the call, put its own remediation prose in an assistant message,
// and returned `completed` — a shape Codex reads as the model's final answer and answers
// with `task_complete`, leaving the requested work untouched.
//
// The turn must instead hand the refusal back to the model as a structured tool result
// and give it another actuation opportunity in the SAME turn. The proof is threefold:
// the second sample happened, it was told which call was refused and why, and what
// reaches the client is a real function_call rather than a denial-only message.
func TestResponsesDenialReturnsControlToTheModel(t *testing.T) {
	srv := newTestServer(t)
	planner := &sequencePlanner{comps: []*agent.Completion{
		toolCallTurn("call_denied", "deny_shell", `{"command":"bash install-openssh-linux.sh"}`),
		toolCallTurn("call_safe", "allow_read", `{"path":"AGENTS.md"}`),
	}}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, resp := postResponses(t, ts.URL, map[string]any{
		"model": "m",
		"input": "fix the install script, then copy the setup concept over",
		"tools": []map[string]any{
			{"type": "function", "name": "deny_shell"},
			{"type": "function", "name": "allow_read"},
		},
	})
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}

	// 1. The turn re-sampled instead of ending on the denial.
	if got := planner.calls(); got != 2 {
		t.Fatalf("planner calls = %d, want 2 (the denial must return control to the model, not end the turn)", got)
	}

	// 2. The model was told, as a tool result keyed by its own call id, that the call
	//    did not run and that the task is still open.
	recovery := planner.request(1)
	var denialResult *agent.Message
	for i := range recovery {
		if recovery[i].Role == agent.RoleTool && recovery[i].ToolCallID == "call_denied" {
			denialResult = &recovery[i]
			break
		}
	}
	if denialResult == nil {
		t.Fatalf("recovery request carried no tool result for the refused call: %+v", recovery)
	}
	for _, want := range []string{"FAK_DENIED", "POLICY_BLOCK", "call_id=call_denied", "remains OPEN"} {
		if !strings.Contains(denialResult.Content, want) {
			t.Errorf("denial tool result missing %q:\n%s", want, denialResult.Content)
		}
	}

	// 3. What the client receives is an actuation, not a terminal denial message.
	calls := functionCallItems(resp.Output)
	if _, ok := calls["allow_read"]; !ok {
		t.Fatalf("recovered turn did not reach the client as a function_call: %+v", resp.Output)
	}
	if _, leaked := calls["deny_shell"]; leaked {
		t.Errorf("refused call leaked to the client: %+v", resp.Output)
	}
	if resp.Status != "completed" || resp.IncompleteDetails != nil {
		t.Errorf("recovered turn = %q/%+v, want a plain completed turn", resp.Status, resp.IncompleteDetails)
	}
	if strings.Contains(messageText(resp.Output), "BLOCKED_BY_GUARD") {
		t.Errorf("a recovered turn must not report itself blocked: %q", messageText(resp.Output))
	}
	// The refusal still rides the wire extension — it happened, and a turn that hides it
	// would leave the client unable to explain why the model changed tools.
	if resp.Fak == nil || len(resp.Fak.Adjudications) != 2 {
		t.Fatalf("fak.adjudications = %+v, want both the refusal and the recovered call", resp.Fak)
	}
}

// TestResponsesDenialUnrecoverableIsTypedBlockedState covers the other half of #5212:
// when the model answers a refusal with another refused call, the turn genuinely cannot
// continue — and that must surface as an explicit blocked state carrying the denied call
// ids, NOT as a normal completion a status consumer would score as `done`.
func TestResponsesDenialUnrecoverableIsTypedBlockedState(t *testing.T) {
	srv := newTestServer(t)
	planner := &sequencePlanner{comps: []*agent.Completion{
		toolCallTurn("call_one", "deny_shell", `{"command":"bash install-openssh-linux.sh"}`),
		toolCallTurn("call_two", "selfmod_edit", `{"path":"internal/abi/abi.go"}`),
	}}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, resp := postResponses(t, ts.URL, map[string]any{
		"model": "m",
		"input": "fix the install script",
		"tools": []map[string]any{
			{"type": "function", "name": "deny_shell"},
			{"type": "function", "name": "selfmod_edit"},
		},
	})
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if got := planner.calls(); got != 2 {
		t.Fatalf("planner calls = %d, want 2 (recovery is attempted exactly once)", got)
	}
	if resp.Status != "incomplete" {
		t.Errorf("status = %q, want incomplete — a guard-blocked turn is not a completed one", resp.Status)
	}
	if resp.IncompleteDetails == nil || resp.IncompleteDetails.Reason != deniedGuardIncompleteReason {
		t.Errorf("incomplete_details = %+v, want reason %q", resp.IncompleteDetails, deniedGuardIncompleteReason)
	}
	text := messageText(resp.Output)
	for _, want := range []string{"BLOCKED_BY_GUARD", "needs_operator=true", "call_one", "call_two", "POLICY_BLOCK", "SELF_MODIFY"} {
		if !strings.Contains(text, want) {
			t.Errorf("blocked note missing %q:\n%s", want, text)
		}
	}
	if len(functionCallItems(resp.Output)) != 0 {
		t.Errorf("a blocked turn must actuate nothing: %+v", resp.Output)
	}
	// The denial→terminal-turn transition is counted, and counted ONCE per HTTP turn.
	// This is the input the "repeated keep going after the same denial class" alert reads;
	// before #5212 the Responses wire recorded nothing here, so a Codex session stopping on
	// the same refusal turn after turn was indistinguishable from a run of clean completions.
	// Counting the recovery sample's own refusal again would double-count one operator-visible
	// stop, so exactly-once is the assertion, not merely non-zero.
	if stops, _ := srv.metrics.denyAllSnapshot(); stops != 1 {
		t.Errorf("deny-all stops = %d, want exactly 1 for one denial→terminal turn", stops)
	}
}

// TestResponsesDenialThenModelAuthoredAnswerStaysCompleted guards the boundary the fix
// must NOT cross. A denial is not a completion — but a model that reads the refusal and
// then writes a real answer HAS authored a terminal response, and that turn stays a
// normal completion. Marking it blocked would trade a false `done` for a false `stuck`.
func TestResponsesDenialThenModelAuthoredAnswerStaysCompleted(t *testing.T) {
	srv := newTestServer(t)
	planner := &sequencePlanner{comps: []*agent.Completion{
		toolCallTurn("call_denied", "deny_shell", `{"command":"bash install-openssh-linux.sh"}`),
		{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "I cannot run that command; here is the manual step instead."},
			FinishReason: "stop",
		},
	}}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, resp := postResponses(t, ts.URL, map[string]any{
		"model": "m",
		"input": "fix the install script",
		"tools": []map[string]any{{"type": "function", "name": "deny_shell"}},
	})
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if resp.Status != "completed" || resp.IncompleteDetails != nil {
		t.Fatalf("status = %q/%+v, want completed — the model authored this answer", resp.Status, resp.IncompleteDetails)
	}
	text := messageText(resp.Output)
	if !strings.Contains(text, "here is the manual step instead") {
		t.Errorf("model-authored answer missing from output: %q", text)
	}
	if strings.Contains(text, "BLOCKED_BY_GUARD") {
		t.Errorf("model-authored answer wrongly reported blocked: %q", text)
	}
}

// TestResponsesDenialRecoveryStandsDownOnOptOut proves Config.DenialRecoveryOff is a real
// kill switch: no second sample is spent, and the turn falls through to the same typed
// blocked state rather than silently reverting to the false-completion shape.
func TestResponsesDenialRecoveryStandsDownOnOptOut(t *testing.T) {
	srv := newTestServer(t)
	// The stand-down reaches the server THROUGH Config, so witness that copy first —
	// otherwise setting the unexported field by hand below would pass even if New dropped
	// the declaration on the floor and no host could ever actually stand recovery down.
	wired, err := New(Config{EngineID: "test", Model: "test-model", DenialRecoveryOff: true})
	if err != nil {
		t.Fatalf("New with the stand-down declared: %v", err)
	}
	if wired.denialRecoveryEnabled() {
		t.Fatal("Config.DenialRecoveryOff never reached the server: recovery is still armed")
	}
	if !srv.denialRecoveryEnabled() {
		t.Fatal("an undeclared Config must leave recovery ARMED — that is the shipped default")
	}
	srv.denialRecoveryOff = true
	planner := &sequencePlanner{comps: []*agent.Completion{
		toolCallTurn("call_denied", "deny_shell", `{"command":"bash install-openssh-linux.sh"}`),
	}}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, resp := postResponses(t, ts.URL, map[string]any{
		"model": "m",
		"input": "fix the install script",
		"tools": []map[string]any{{"type": "function", "name": "deny_shell"}},
	})
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if got := planner.calls(); got != 1 {
		t.Fatalf("planner calls = %d, want 1 (the opt-out must spend no recovery sample)", got)
	}
	if resp.Status != "completed" {
		t.Errorf("status = %q: with recovery stood down the turn keeps its prior shape", resp.Status)
	}
}

// TestResponsesDenialOnlyPredicate pins the predicate that decides whether a turn would
// reach the client as guard prose and nothing else — the single condition the whole fix
// hangs on.
func TestResponsesDenialOnlyPredicate(t *testing.T) {
	call := []agent.ToolCall{{ID: "c1"}}
	cases := []struct {
		name        string
		kept        []agent.ToolCall
		dropped     int
		content     string
		bodyRefused bool
		servedText  string
		want        bool
	}{
		{name: "all refused, no prose", dropped: 1, want: true},
		{name: "all refused, whitespace prose", dropped: 1, content: "  \n ", want: true},
		{name: "all refused, body blanked by the guard", dropped: 1, content: "", bodyRefused: true, want: true},
		{name: "a call survived", kept: call, dropped: 1},
		{name: "nothing was refused", dropped: 0, content: "done"},
		{name: "model authored prose", dropped: 1, content: "here is the answer"},
		{name: "vdso served the read inline", dropped: 1, servedText: "cached bytes"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := turnIsDenialOnly(tc.kept, tc.dropped, tc.content, tc.bodyRefused, tc.servedText); got != tc.want {
				t.Fatalf("turnIsDenialOnly = %v, want %v", got, tc.want)
			}
		})
	}
}
