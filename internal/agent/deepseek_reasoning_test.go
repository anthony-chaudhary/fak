package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestOpenAIAdapterPreservesDeepSeekReasoningContent(t *testing.T) {
	adapter, err := NewTranscriptAdapter(ProviderOpenAI)
	if err != nil {
		t.Fatal(err)
	}
	comp, err := adapter.ParseResponse([]byte(`{
		"model":"deepseek-v4-pro",
		"choices":[{
			"message":{
				"role":"assistant",
				"reasoning_content":"I should answer directly.",
				"content":"final answer"
			},
			"finish_reason":"stop"
		}],
		"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}
	}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if comp.Message.Content != "final answer" {
		t.Fatalf("content = %q, want final answer", comp.Message.Content)
	}
	if comp.Message.ReasoningContent != "I should answer directly." {
		t.Fatalf("reasoning_content = %q", comp.Message.ReasoningContent)
	}
	if len(comp.Message.ToolCalls) != 0 {
		t.Fatalf("reasoning-only response produced tool calls: %+v", comp.Message.ToolCalls)
	}
}

func TestOpenAIAdapterDeepSeekReasoningDoesNotLiftToolCalls(t *testing.T) {
	adapter, err := NewTranscriptAdapter(ProviderOpenAI)
	if err != nil {
		t.Fatal(err)
	}
	comp, err := adapter.ParseResponse([]byte(`{
		"choices":[{
			"message":{
				"role":"assistant",
				"reasoning_content":"<tool_call>{\"name\":\"Bash\",\"arguments\":{\"command\":\"rm -rf /tmp/x\"}}</tool_call>",
				"content":"I will not call a tool."
			},
			"finish_reason":"stop"
		}]
	}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if comp.Message.Content != "I will not call a tool." {
		t.Fatalf("content = %q", comp.Message.Content)
	}
	if !strings.Contains(comp.Message.ReasoningContent, "Bash") {
		t.Fatalf("reasoning_content not preserved: %q", comp.Message.ReasoningContent)
	}
	if len(comp.Message.ToolCalls) != 0 || comp.FinishReason == "tool_calls" {
		t.Fatalf("reasoning_content must not be lifted into executable calls: finish=%q calls=%+v", comp.FinishReason, comp.Message.ToolCalls)
	}
}

func TestOpenAIAdapterDeepSeekReasoningWithStructuredToolCalls(t *testing.T) {
	adapter, err := NewTranscriptAdapter(ProviderOpenAI)
	if err != nil {
		t.Fatal(err)
	}
	comp, err := adapter.ParseResponse([]byte(`{
		"choices":[{
			"message":{
				"role":"assistant",
				"reasoning_content":"The user asked for weather; call the weather tool.",
				"content":"",
				"tool_calls":[{
					"id":"call_1",
					"type":"function",
					"function":{"name":"weather","arguments":"{\"city\":\"SFO\"}"}
				}]
			},
			"finish_reason":"tool_calls"
		}]
	}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if comp.Message.ReasoningContent == "" {
		t.Fatal("reasoning_content was dropped")
	}
	if comp.Message.Content != "" {
		t.Fatalf("reasoning leaked into content: %q", comp.Message.Content)
	}
	if len(comp.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1: %+v", len(comp.Message.ToolCalls), comp.Message.ToolCalls)
	}
	tc := comp.Message.ToolCalls[0]
	if tc.Function.Name != "weather" || !strings.Contains(tc.Function.Arguments, "SFO") {
		t.Fatalf("bad structured tool call: %+v", tc)
	}
}

func TestOpenAIAdapterRoundTripsDeepSeekReasoningOnToolResultTurn(t *testing.T) {
	adapter, err := NewTranscriptAdapter(ProviderOpenAI)
	if err != nil {
		t.Fatal(err)
	}
	extra, err := ParseExtraBodyJSON(`{"thinking":{"type":"enabled"},"reasoning_effort":"max"}`)
	if err != nil {
		t.Fatalf("parse DeepSeek extra body: %v", err)
	}
	body, err := adapter.MarshalRequest(adapterRequest{
		Model: "deepseek-v4-pro",
		Messages: []Message{
			{
				Role:             RoleAssistant,
				ReasoningContent: "I need the lookup result before answering.",
				ToolCalls: []ToolCall{{
					ID:       "call_1",
					Type:     "function",
					Function: Func{Name: "lookup", Arguments: `{"id":"42"}`},
				}},
			},
			{Role: RoleTool, ToolCallID: "call_1", Content: `{"name":"Ada"}`},
		},
		Tools: []ToolDef{{
			Type: "function",
			Function: ToolDefFunction{
				Name:       "lookup",
				Parameters: json.RawMessage(`{"type":"object","properties":{"id":{"type":"string"}}}`),
			},
		}},
		MaxTokens: 128,
		ExtraBody: extra,
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	var req struct {
		ReasoningEffort string `json:"reasoning_effort"`
		Thinking        struct {
			Type string `json:"type"`
		} `json:"thinking"`
		Messages []Message `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("decode marshaled body: %v\n%s", err, body)
	}
	if req.ReasoningEffort != "max" || req.Thinking.Type != "enabled" {
		t.Fatalf("DeepSeek thinking controls missing: %s", body)
	}
	if len(req.Messages) != 2 {
		t.Fatalf("messages = %d, want 2: %s", len(req.Messages), body)
	}
	if req.Messages[0].ReasoningContent != "I need the lookup result before answering." {
		t.Fatalf("assistant reasoning_content was not round-tripped: %+v", req.Messages[0])
	}
	if len(req.Messages[0].ToolCalls) != 1 || req.Messages[1].Role != RoleTool || req.Messages[1].ToolCallID != "call_1" {
		t.Fatalf("tool-call continuation was not preserved: %+v", req.Messages)
	}
}
