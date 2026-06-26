// Package agenttest is the public test harness for fak agent workflows (#238, D-008):
// deterministic fixtures, a tool-call assertion library, mock tool responses, and
// reproduce-from-transcript replay.
//
// It is the agent-loop analogue of net/http/httptest — import it from a _test.go (or any
// host harness) to drive an agent.Planner over mock tools with no network, no model, and
// no GPU, then assert the tool-call pattern the workflow produced. The four pieces map
// one-to-one onto the issue's acceptance:
//
//   - Test agent workflows deterministically — ScriptedPlanner + RunSession is a pure
//     function of (planner, tools, task); AssertReproducible witnesses it.
//   - Assert tool call patterns — the Match*/Assert* library over a Run's tool events.
//   - Mock responses — MockTools, a deterministic tool-name → response registry.
//   - Reproduce from transcript — ReplayPlanner re-emits a recorded transcript's
//     assistant turns so a captured run becomes a regression fixture.
//
// Nothing here touches the kernel or the wire: the harness reuses the real agent message
// vocabulary (agent.Message / agent.ToolCall / agent.Planner) so it tests REAL agent
// workflows, not a parallel toy model.
package agenttest

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// DefaultMaxTurns bounds a session when the caller passes a non-positive maxTurns, so a
// runaway script (one that never emits a final answer) terminates instead of looping.
const DefaultMaxTurns = 16

// SystemPrompt is the neutral system message the harness seeds a session with. It is
// irrelevant to a ScriptedPlanner/ReplayPlanner (which ignore context), but a planner
// that inspects messages still sees a well-formed transcript.
const SystemPrompt = "You are a test agent. Use the provided tools to complete the task, then reply with a short final answer."

// ToolEvent is one tool call the harness observed during a run, paired with the mock
// response it was answered with. These are the assertable surface: which tool, the raw
// JSON arguments the planner emitted (verbatim, never re-marshaled), and the response
// the mock returned.
type ToolEvent struct {
	Turn     int    `json:"turn"`     // 1-based model turn that emitted this call
	Tool     string `json:"tool"`     // tool name
	Args     string `json:"args"`     // raw JSON arguments as emitted
	Response string `json:"response"` // the mock tool's response
	ID       string `json:"id"`       // tool_call id (for transcript round-trip)
	Mocked   bool   `json:"mocked"`   // false when no mock was registered for Tool
}

// Run is the full deterministic outcome of driving a planner over mock tools: the final
// message transcript, every tool call observed in order, and the planner's final answer.
type Run struct {
	Task        string          `json:"task"`
	Messages    []agent.Message `json:"messages"`
	Tools       []ToolEvent     `json:"tools"`
	FinalAnswer string          `json:"final_answer"`
	Turns       int             `json:"turns"`
	HitTurnCap  bool            `json:"hit_turn_cap"`
}

// ToolNames returns the ordered list of tool names called across the whole run — the
// primary surface the pattern assertions match against.
func (r Run) ToolNames() []string {
	out := make([]string, len(r.Tools))
	for i, e := range r.Tools {
		out[i] = e.Tool
	}
	return out
}

// UnmockedTools returns the distinct tool names the run called that had no registered
// mock, in first-seen order — a loud signal that a fixture under-specified its mocks.
func (r Run) UnmockedTools() []string {
	seen := map[string]bool{}
	var out []string
	for _, e := range r.Tools {
		if !e.Mocked && !seen[e.Tool] {
			seen[e.Tool] = true
			out = append(out, e.Tool)
		}
	}
	return out
}

// Responder computes a mock tool response from the raw JSON arguments a planner emitted.
// It MUST be deterministic — the same args must always yield the same response — so a
// run stays reproducible.
type Responder func(args string) string

// MockTools is a deterministic registry of mock tool responses keyed by tool name. A
// tool with no registered response returns a deterministic error payload and is recorded
// with Mocked=false, so a missing mock is a loud, assertable failure rather than a silent
// empty result.
type MockTools struct {
	static map[string]string
	funcs  map[string]Responder
	order  []string // registration order, for a stable Catalog()
}

// NewMockTools returns an empty mock-tool registry.
func NewMockTools() *MockTools {
	return &MockTools{static: map[string]string{}, funcs: map[string]Responder{}}
}

// Respond registers a fixed response for tool. A later Respond/RespondFunc for the same
// tool replaces the earlier one. Returns the registry for chaining.
func (m *MockTools) Respond(tool, response string) *MockTools {
	m.track(tool)
	m.static[tool] = response
	return m
}

// RespondFunc registers a deterministic response function for tool; it takes precedence
// over a static response for the same tool. Returns the registry for chaining.
func (m *MockTools) RespondFunc(tool string, fn Responder) *MockTools {
	m.track(tool)
	m.funcs[tool] = fn
	return m
}

func (m *MockTools) track(tool string) {
	if _, ok := m.static[tool]; ok {
		return
	}
	if _, ok := m.funcs[tool]; ok {
		return
	}
	m.order = append(m.order, tool)
}

// Catalog returns one agent.ToolDef per registered tool, in registration order, so a
// planner that inspects the advertised tool catalog sees exactly the mocked tools.
func (m *MockTools) Catalog() []agent.ToolDef {
	out := make([]agent.ToolDef, 0, len(m.order))
	for _, name := range m.order {
		out = append(out, agent.ToolDef{
			Type: "function",
			Function: agent.ToolDefFunction{
				Name:       name,
				Parameters: json.RawMessage(`{"type":"object"}`),
			},
		})
	}
	return out
}

// answer resolves a tool call to its mock response. The bool reports whether a mock was
// registered; when false the response is a deterministic error payload so the loop can
// proceed and a test can assert on the miss.
func (m *MockTools) answer(call agent.ToolCall) (string, bool) {
	name := call.Function.Name
	if fn, ok := m.funcs[name]; ok {
		return fn(call.Function.Arguments), true
	}
	if r, ok := m.static[name]; ok {
		return r, true
	}
	return fmt.Sprintf(`{"error":"agenttest: no mock registered for tool %q"}`, name), false
}

// RunSession drives planner over the mock tools to completion. It seeds a system+user
// transcript, then each turn asks the planner for the next assistant message, dispatches
// every tool call to the mock registry, and appends the tool responses — exactly the
// shape of the real agent loop (internal/agent.RunArm) but with no kernel, no network,
// and no model, so the whole run is a pure function of (planner, tools, task). It stops
// at the planner's final answer (a turn with no tool calls) or at maxTurns.
func RunSession(planner agent.Planner, tools *MockTools, task string, maxTurns int) (Run, error) {
	if planner == nil {
		return Run{}, fmt.Errorf("agenttest: nil planner")
	}
	if tools == nil {
		tools = NewMockTools()
	}
	if maxTurns <= 0 {
		maxTurns = DefaultMaxTurns
	}
	run := Run{Task: task}
	msgs := []agent.Message{
		{Role: agent.RoleSystem, Content: SystemPrompt},
		{Role: agent.RoleUser, Content: task},
	}
	catalog := tools.Catalog()
	ctx := context.Background()
	for turn := 1; turn <= maxTurns; turn++ {
		comp, err := planner.Complete(ctx, msgs, catalog)
		if err != nil {
			return run, fmt.Errorf("agenttest: turn %d: %w", turn, err)
		}
		run.Turns = turn
		asst := comp.Message
		asst.Role = agent.RoleAssistant
		msgs = append(msgs, asst)
		if len(asst.ToolCalls) == 0 {
			run.FinalAnswer = asst.Content
			run.Messages = msgs
			return run, nil
		}
		for _, tc := range asst.ToolCalls {
			resp, mocked := tools.answer(tc)
			run.Tools = append(run.Tools, ToolEvent{
				Turn:     turn,
				Tool:     tc.Function.Name,
				Args:     tc.Function.Arguments,
				Response: resp,
				ID:       tc.ID,
				Mocked:   mocked,
			})
			msgs = append(msgs, agent.Message{
				Role:       agent.RoleTool,
				ToolCallID: tc.ID,
				Name:       tc.Function.Name,
				Content:    resp,
			})
		}
	}
	run.HitTurnCap = true
	run.Messages = msgs
	return run, nil
}
