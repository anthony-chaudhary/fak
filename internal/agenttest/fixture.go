package agenttest

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Fixture bundles a deterministic agent-workflow test case into one serializable value:
// a task, the mock tool responses, the scripted planner turns, and the expected tool-call
// sequence / final answer. It is the "agent test fixture" of #238 — a self-contained
// scenario that runs with no model in the loop and asserts the workflow's tool-call
// pattern. Because every field is plain data, a fixture can be authored in Go or loaded
// from JSON, and a run is a pure function of the fixture (the reproducibility guarantee).
type Fixture struct {
	Name         string            `json:"name"`
	Task         string            `json:"task"`
	MaxTurns     int               `json:"max_turns,omitempty"`
	Mocks        map[string]string `json:"mocks,omitempty"`         // tool -> static response
	Script       []FixtureTurn     `json:"script"`                  // the scripted planner turns
	ExpectTools  []string          `json:"expect_tools,omitempty"`  // exact tool sequence
	ExpectAnswer string            `json:"expect_answer,omitempty"` // substring of the final answer
}

// FixtureTurn is one declared assistant turn: tool calls, or a final answer when Calls is
// empty.
type FixtureTurn struct {
	Calls  []FixtureCall `json:"calls,omitempty"`
	Answer string        `json:"answer,omitempty"`
}

// FixtureCall is one declared tool call.
type FixtureCall struct {
	Tool string `json:"tool"`
	Args string `json:"args,omitempty"`
}

// LoadFixture decodes a fixture from JSON, rejecting unknown keys so a typo'd field is a
// loud error rather than a silent no-op.
func LoadFixture(data []byte) (Fixture, error) {
	var f Fixture
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&f); err != nil {
		return Fixture{}, fmt.Errorf("agenttest: load fixture: %w", err)
	}
	return f, nil
}

// JSON serializes the fixture to indented JSON.
func (f Fixture) JSON() ([]byte, error) { return json.MarshalIndent(f, "", "  ") }

// tools builds a MockTools registry from the fixture's static mocks.
func (f Fixture) tools() *MockTools {
	m := NewMockTools()
	for tool, resp := range f.Mocks {
		m.Respond(tool, resp)
	}
	return m
}

// planner builds a FRESH ScriptedPlanner from the fixture's script. A new planner per
// call is what makes Run reproducible by construction.
func (f Fixture) planner() *ScriptedPlanner {
	turns := make([]Turn, 0, len(f.Script))
	for _, ft := range f.Script {
		if len(ft.Calls) == 0 {
			turns = append(turns, AnswerTurn(ft.Answer))
			continue
		}
		calls := make([]Call, 0, len(ft.Calls))
		for _, c := range ft.Calls {
			calls = append(calls, Call{Tool: c.Tool, Args: c.Args})
		}
		turns = append(turns, CallTurn(calls...))
	}
	return NewScriptedPlanner(turns...)
}

// Run executes the fixture once over a fresh planner and fresh mock tools. Two calls to
// Run on the same fixture produce identical Runs (AssertReproducible witnesses this).
func (f Fixture) Run() (Run, error) {
	return RunSession(f.planner(), f.tools(), f.Task, f.MaxTurns)
}

// Verify runs the fixture and asserts its declared expectations (ExpectTools sequence,
// ExpectAnswer substring) through t. With no expectations declared it only fails on a
// run error — a useful smoke check.
func (f Fixture) Verify(t T) {
	t.Helper()
	r, err := f.Run()
	if err != nil {
		t.Errorf("fixture %q: run failed: %v", f.Name, err)
		return
	}
	if f.ExpectTools != nil {
		AssertToolSequence(t, r, f.ExpectTools...)
	}
	if f.ExpectAnswer != "" {
		AssertFinalAnswer(t, r, f.ExpectAnswer)
	}
}

// AssertReproducible runs the fixture twice from scratch and fails t if the two runs
// differ in their tool-call sequence, arguments, responses, or final answer — the
// executable form of the "reproducibility guarantee" (#238).
func AssertReproducible(t T, f Fixture) {
	t.Helper()
	a, err := f.Run()
	if err != nil {
		t.Errorf("fixture %q: first run failed: %v", f.Name, err)
		return
	}
	b, err := f.Run()
	if err != nil {
		t.Errorf("fixture %q: second run failed: %v", f.Name, err)
		return
	}
	if diff := runDiff(a, b); diff != "" {
		t.Errorf("fixture %q is not reproducible: %s", f.Name, diff)
	}
}

// runDiff returns a human-readable description of the first difference between two runs'
// observable outcomes, or "" when they are identical.
func runDiff(a, b Run) string {
	if a.FinalAnswer != b.FinalAnswer {
		return fmt.Sprintf("final answer differs: %q vs %q", a.FinalAnswer, b.FinalAnswer)
	}
	if len(a.Tools) != len(b.Tools) {
		return fmt.Sprintf("tool-call count differs: %d vs %d", len(a.Tools), len(b.Tools))
	}
	for i := range a.Tools {
		x, y := a.Tools[i], b.Tools[i]
		switch {
		case x.Tool != y.Tool:
			return fmt.Sprintf("call #%d tool differs: %q vs %q", i, x.Tool, y.Tool)
		case x.Args != y.Args:
			return fmt.Sprintf("call #%d args differ: %q vs %q", i, x.Args, y.Args)
		case x.Response != y.Response:
			return fmt.Sprintf("call #%d response differs: %q vs %q", i, x.Response, y.Response)
		}
	}
	return ""
}
