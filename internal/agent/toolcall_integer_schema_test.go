package agent

import (
	"encoding/json"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/agent/**
// fak-test:runtime medium est=2s lane=default
func TestCompleteIntegerSchemaArguments(t *testing.T) {
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{
		Name:       "record_count",
		Parameters: json.RawMessage(`{"type":"object","properties":{"count":{"type":"integer"},"label":{"type":"string"}}}`),
	}}}
	for _, tc := range []struct {
		name, value, want string
	}{
		{"whole float string", `"42.0"`, `42`},
		{"exponent string", `"4.2e1"`, `42`},
		{"exact large string", `"9007199254740993.0"`, `9007199254740993`},
		{"fraction string", `"42.5"`, `"42.5"`},
		{"nonfinite string", `"Infinity"`, `"Infinity"`},
		{"unbounded exponent", `"1e1000000000"`, `"1e1000000000"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := `<tool_call>{"name":"record_count","arguments":{"count":` + tc.value + `,"label":"42.0","undeclared":"42.0"}}</tool_call>`
			comp := inKernelDecodeToCompletion(t, raw, "stop", tools)
			if len(comp.Message.ToolCalls) != 1 || comp.ToolCallsDropped {
				t.Fatal("ordinary completion failed to preserve the offered tool call")
			}
			var args map[string]json.RawMessage
			if err := json.Unmarshal([]byte(comp.Message.ToolCalls[0].Function.Arguments), &args); err != nil {
				t.Fatal(err)
			}
			if string(args["count"]) != tc.want || string(args["label"]) != `"42.0"` || string(args["undeclared"]) != `"42.0"` {
				t.Fatalf("arguments=%s, want count=%s with string and undeclared fields preserved", comp.Message.ToolCalls[0].Function.Arguments, tc.want)
			}
		})
	}
}
