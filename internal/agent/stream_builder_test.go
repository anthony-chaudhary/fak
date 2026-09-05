package agent

import (
	"context"
	"strings"
	"testing"
)

// TestStreamToolCallBuilderAccumulationInterleaved verifies that multiple tool calls
// arriving with interleaved chunks, out-of-order indices, and multiple argument fragments
// are correctly accumulated via strings.Builder without loss or corruption, and returned
// in sorted order by index.
func TestStreamToolCallBuilderAccumulationInterleaved(t *testing.T) {
	const body = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":1,\"id\":\"call_b\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"key\\\":\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"calculate\",\"arguments\":\"{\\\"expr\\\":\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"\\\"user_123\\\"\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":2,\"id\":\"call_c\",\"type\":\"function\",\"function\":{\"name\":\"notify\",\"arguments\":\"{\\\"msg\\\":\\\"hi\\\"}\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"2 + 2\\\"\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\",\\\"timeout\\\":30}\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"}\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"

	srv, _ := sseServer(t, body)
	p := NewHTTPPlanner(srv.URL, "m", "")

	sinkCalls := 0
	sink := func(string) error {
		sinkCalls++
		return nil
	}

	comp, err := p.CompleteStream(context.Background(), sink, []Message{{Role: RoleUser, Content: "run tasks"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}

	if sinkCalls != 0 {
		t.Fatalf("sink called %d times on a tool call turn; tool calls must stay buffered", sinkCalls)
	}

	if len(comp.Message.ToolCalls) != 3 {
		t.Fatalf("tool calls count = %d, want 3: %+v", len(comp.Message.ToolCalls), comp.Message.ToolCalls)
	}

	// Tool calls must be sorted by index: 0, 1, 2
	tc0 := comp.Message.ToolCalls[0]
	if tc0.ID != "call_a" || tc0.Function.Name != "calculate" || tc0.Function.Arguments != `{"expr":"2 + 2"}` {
		t.Errorf("tc[0] = %+v, want ID=call_a, Name=calculate, Arguments={\"expr\":\"2 + 2\"}", tc0)
	}

	tc1 := comp.Message.ToolCalls[1]
	if tc1.ID != "call_b" || tc1.Function.Name != "lookup" || tc1.Function.Arguments != `{"key":"user_123","timeout":30}` {
		t.Errorf("tc[1] = %+v, want ID=call_b, Name=lookup, Arguments={\"key\":\"user_123\",\"timeout\":30}", tc1)
	}

	tc2 := comp.Message.ToolCalls[2]
	if tc2.ID != "call_c" || tc2.Function.Name != "notify" || tc2.Function.Arguments != `{"msg":"hi"}` {
		t.Errorf("tc[2] = %+v, want ID=call_c, Name=notify, Arguments={\"msg\":\"hi\"}", tc2)
	}

	if comp.FinishReason != "tool_calls" {
		t.Errorf("finish reason = %q, want tool_calls", comp.FinishReason)
	}

	// Raw transcript must retain all wire frames including data and DONE lines
	if !strings.Contains(string(comp.Raw), "call_b") || !strings.Contains(string(comp.Raw), "[DONE]") {
		t.Errorf("comp.Raw missing expected frames: %s", string(comp.Raw))
	}
}

// TestStreamToolCallBuilderLastNonEmptyFields verifies that subsequent chunks preserve
// non-empty IDs, types, and names from earlier chunks, and that later non-empty updates overwrite them.
func TestStreamToolCallBuilderLastNonEmptyFields(t *testing.T) {
	const body = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_initial\",\"type\":\"custom_type\",\"function\":{\"name\":\"do_action\",\"arguments\":\"{\\\"step\\\":1\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\",\\\"step\\\":2\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_updated\",\"function\":{\"arguments\":\"}\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"

	srv, _ := sseServer(t, body)
	p := NewHTTPPlanner(srv.URL, "m", "")

	comp, err := p.CompleteStream(context.Background(), nil, []Message{{Role: RoleUser, Content: "act"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}

	if len(comp.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(comp.Message.ToolCalls))
	}

	tc := comp.Message.ToolCalls[0]
	if tc.ID != "call_updated" {
		t.Errorf("tc.ID = %q, want call_updated (last non-empty)", tc.ID)
	}
	if tc.Type != "custom_type" {
		t.Errorf("tc.Type = %q, want custom_type (preserved from first chunk)", tc.Type)
	}
	if tc.Function.Name != "do_action" {
		t.Errorf("tc.Function.Name = %q, want do_action (preserved from first chunk)", tc.Function.Name)
	}
	if tc.Function.Arguments != `{"step":1,"step":2}` {
		t.Errorf("tc.Function.Arguments = %q, want {\"step\":1,\"step\":2}", tc.Function.Arguments)
	}
}

// TestStreamToolCallAccDirectUnit verifies the accumulator struct directly:
// default callType, strings.Builder accumulation, and toolCall materialization.
func TestStreamToolCallAccDirectUnit(t *testing.T) {
	acc := &streamToolCallAcc{callType: "function"}
	acc.id = "call_xyz"
	acc.name = "search"
	acc.arguments.WriteString("{\"q\":")
	acc.arguments.WriteString("\"golang\"")
	acc.arguments.WriteString("}")

	tc := acc.toolCall()
	if tc.ID != "call_xyz" {
		t.Errorf("ID = %q, want call_xyz", tc.ID)
	}
	if tc.Type != "function" {
		t.Errorf("Type = %q, want function", tc.Type)
	}
	if tc.Function.Name != "search" {
		t.Errorf("Name = %q, want search", tc.Function.Name)
	}
	if tc.Function.Arguments != `{"q":"golang"}` {
		t.Errorf("Arguments = %q, want {\"q\":\"golang\"}", tc.Function.Arguments)
	}

	// Verify sortedIndices with non-sequential unordered map keys
	m := map[int]*streamToolCallAcc{
		9:  acc,
		1:  acc,
		4:  acc,
		-2: acc,
	}
	indices := sortedIndices(m)
	want := []int{-2, 1, 4, 9}
	if len(indices) != len(want) {
		t.Fatalf("indices length = %d, want %d", len(indices), len(want))
	}
	for i, idx := range indices {
		if idx != want[i] {
			t.Errorf("indices[%d] = %d, want %d", i, idx, want[i])
		}
	}
}
