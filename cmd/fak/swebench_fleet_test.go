package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/swebench"
)

func TestGatewayBaseURL(t *testing.T) {
	cases := map[string]string{
		"localhost:8080":        "http://localhost:8080/v1",
		"http://h:9/v1":         "http://h:9/v1",
		"https://api.x.com/v1/": "https://api.x.com/v1",
		"":                      "",
		"host:1/v1":             "http://host:1/v1",
	}
	for in, want := range cases {
		if got := gatewayBaseURL(in); got != want {
			t.Errorf("gatewayBaseURL(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestFleetPlannerMessageRoundTrip witnesses the swebench<->agent type mapping the
// adapter relies on (tool calls, tool results, and tool defs all carry through).
func TestFleetPlannerMessageRoundTrip(t *testing.T) {
	msgs := []swebench.ChatMessage{
		{Role: "system", Content: "sys"},
		{Role: "assistant", ToolCalls: []swebench.ChatToolCall{{ID: "c1", Name: "read_file", Args: `{"path":"x"}`}}},
		{Role: "tool", ToolCallID: "c1", Name: "read_file", Content: "file body"},
	}
	am := toAgentMessages(msgs)
	if len(am) != 3 {
		t.Fatalf("toAgentMessages len = %d, want 3", len(am))
	}
	tc := am[1].ToolCalls[0]
	if tc.ID != "c1" || tc.Type != "function" || tc.Function.Name != "read_file" || tc.Function.Arguments != `{"path":"x"}` {
		t.Errorf("tool call not mapped: %+v", tc)
	}
	if am[2].ToolCallID != "c1" || am[2].Name != "read_file" || am[2].Content != "file body" {
		t.Errorf("tool result not mapped: %+v", am[2])
	}

	at := toAgentTools([]swebench.ChatTool{{Name: "finish", Description: "d", Parameters: `{"type":"object"}`}})
	if len(at) != 1 || at[0].Type != "function" || at[0].Function.Name != "finish" || string(at[0].Function.Parameters) != `{"type":"object"}` {
		t.Errorf("tool def not mapped: %+v", at)
	}

	out := fromAgentMessage(agent.Message{
		Role:      "assistant",
		ToolCalls: []agent.ToolCall{{ID: "c2", Type: "function", Function: agent.Func{Name: "write_file", Arguments: `{"path":"y"}`}}},
	})
	if len(out.ToolCalls) != 1 || out.ToolCalls[0].ID != "c2" || out.ToolCalls[0].Name != "write_file" || out.ToolCalls[0].Args != `{"path":"y"}` {
		t.Errorf("fromAgentMessage tool call not mapped: %+v", out)
	}
}
