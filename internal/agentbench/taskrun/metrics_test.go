package taskrun

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func metricsCatalog() []agent.ToolDef {
	return []agent.ToolDef{
		{Type: "function", Function: agent.ToolDefFunction{Name: "Read", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"limit":{"type":"integer"}},"required":["file_path"]}`)}},
		{Type: "function", Function: agent.ToolDefFunction{Name: "Bash", Parameters: json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false}`)}},
	}
}

func metricsCall(name, args string) agent.ToolCall {
	return agent.ToolCall{ID: name + args, Type: "function", Function: agent.Func{Name: name, Arguments: args}}
}

func metricsTurns(calls ...[]agent.ToolCall) []PlannerTurn {
	turns := make([]PlannerTurn, len(calls))
	for i, c := range calls {
		turns[i] = PlannerTurn{ToolCalls: c}
	}
	return turns
}

func floatOrNil(p *float64) any {
	if p == nil {
		return nil
	}
	return *p
}

// fak-test:runtime fast est=10ms lane=default
func TestReduceToolCallsClassifiesInvalidCalls(t *testing.T) {
	turns := metricsTurns(
		[]agent.ToolCall{metricsCall("Read", `{"file_path":"a.go"}`), metricsCall("Read", `{"file_path":`)},
		[]agent.ToolCall{metricsCall("Delete", `{"file_path":"a.go"}`), metricsCall("Read", `{"limit":3}`)},
		[]agent.ToolCall{metricsCall("Read", `{"file_path":7}`), metricsCall("Bash", `{"command":"go test","cwd":"/"}`)},
		[]agent.ToolCall{metricsCall("Read", `{"file_path":"b.go","limit":2.5}`), metricsCall("Bash", `{"command":"ls"}`)},
	)
	got := reduceToolCalls(turns, metricsCatalog(), 12)
	want := ToolCallMetrics{Measured: true, Total: 8, Invalid: 6, InvalidJSON: 1, UnknownTool: 1, SchemaInvalid: 4, MaxIdenticalRun: 1}
	if got.Validity == nil || *got.Validity != 0.25 {
		t.Fatalf("validity = %v, want 0.25", floatOrNil(got.Validity))
	}
	got.Validity = nil
	if got != want {
		t.Fatalf("metrics = %+v\nwant      %+v", got, want)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestReduceToolCallsLoopDetection(t *testing.T) {
	read := func(args string) agent.ToolCall { return metricsCall("Read", args) }
	cases := []struct {
		name       string
		turns      []PlannerTurn
		cap        int
		repeat     bool
		capHit     bool
		maxRun     int
		validity   any
		totalCalls int
	}{
		{"three identical across turns with reordered keys", metricsTurns(
			[]agent.ToolCall{read(`{"file_path":"a.go","limit":1}`)},
			[]agent.ToolCall{read(`{"limit":1, "file_path":"a.go"}`)},
			[]agent.ToolCall{read(`{"file_path":"a.go","limit":1}`)},
		), 12, true, false, 3, 1.0, 3},
		{"two identical then different", metricsTurns(
			[]agent.ToolCall{read(`{"file_path":"a.go"}`), read(`{"file_path":"a.go"}`), read(`{"file_path":"b.go"}`), read(`{"file_path":"a.go"}`)},
		), 12, false, false, 2, 1.0, 4},
		{"turn cap hit while still calling tools", metricsTurns(
			[]agent.ToolCall{read(`{"file_path":"a.go"}`)},
			[]agent.ToolCall{read(`{"file_path":"b.go"}`)},
		), 2, false, true, 1, 1.0, 2},
		{"final answer at the cap is not a cap hit", append(metricsTurns(
			[]agent.ToolCall{read(`{"file_path":"a.go"}`)},
		), PlannerTurn{Content: "done"}), 2, false, false, 1, 1.0, 1},
		{"no tool calls has no validity", []PlannerTurn{{Content: "done"}}, 12, false, false, 0, nil, 0},
	}
	for _, tc := range cases {
		got := reduceToolCalls(tc.turns, metricsCatalog(), tc.cap)
		if got.Repeating != tc.repeat || got.TurnCapHit != tc.capHit || got.Loop != (tc.repeat || tc.capHit) || got.MaxIdenticalRun != tc.maxRun || floatOrNil(got.Validity) != tc.validity || got.Total != tc.totalCalls {
			t.Fatalf("%s: got repeat=%t cap=%t loop=%t run=%d validity=%v total=%d", tc.name, got.Repeating, got.TurnCapHit, got.Loop, got.MaxIdenticalRun, floatOrNil(got.Validity), got.Total)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestAggregateTrials(t *testing.T) {
	v := func(f float64) *float64 { return &f }
	tasks := []TaskReceipt{
		{Accepted: true, ToolCalls: ToolCallMetrics{Measured: true, Total: 10, Invalid: 1, Validity: v(0.9)}},
		{ToolCalls: ToolCallMetrics{Measured: true, Total: 6, Invalid: 3, Loop: true}},
		{Accepted: true, ToolCalls: ToolCallMetrics{Measured: true, Total: 4}},
		{Error: "fixture controls did not prove red/green"},
	}
	got := SummarizeTrials(tasks)
	if got.Trials != 4 || got.Accepted != 2 || got.MeasuredTrials != 3 || got.ToolCallsTotal != 20 || got.ToolCallsInvalid != 4 || got.Loops != 1 {
		t.Fatalf("counts = %+v", got)
	}
	if floatOrNil(got.AcceptRate) != 0.5 || floatOrNil(got.ToolCallValidity) != 0.8 || floatOrNil(got.StuckRate) != 0.333333 {
		t.Fatalf("rates accept=%v validity=%v loop=%v", floatOrNil(got.AcceptRate), floatOrNil(got.ToolCallValidity), floatOrNil(got.StuckRate))
	}
	empty := SummarizeTrials(nil)
	if empty.AcceptRate != nil || empty.ToolCallValidity != nil || empty.StuckRate != nil {
		t.Fatalf("empty aggregate rates must be nil: %+v", empty)
	}
}
