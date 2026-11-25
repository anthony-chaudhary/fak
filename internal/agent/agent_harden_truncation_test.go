package agent

import (
	"errors"
	"testing"
)

func TestAgentHardenTruncatedInkling2ToolCall(t *testing.T) {
	cases := []struct {
		name, content, tool string
	}{
		{"mid-args", `<|message_model|>Write<|content_invoke_tool_json|>{"name":"Write","arguments":{"content":"abc`, "Write"},
		{"before-name", `<|message_model|>Write<|content_invoke_tool_json|>{"na`, "Write"},
		{"bare-marker", `<|content_invoke_tool_json|>`, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			comp := &Completion{FinishReason: "length", Message: Message{Role: RoleAssistant, Content: tc.content}}
			sig, err := TruncatedToolCall(comp)
			if !errors.Is(err, ErrTruncatedToolCall) {
				t.Fatalf("err = %v, want ErrTruncatedToolCall", err)
			}
			if sig.ToolName != tc.tool {
				t.Fatalf("tool = %q, want %q", sig.ToolName, tc.tool)
			}
		})
	}
	closed := &Completion{FinishReason: "length", Message: Message{Role: RoleAssistant,
		Content: `<|message_model|>Write<|content_invoke_tool_json|>{"name":"Write","args":{}}<|end_message|>`}}
	if _, err := TruncatedToolCall(closed); err != nil {
		t.Fatalf("closed block reported truncated: %v", err)
	}
}
