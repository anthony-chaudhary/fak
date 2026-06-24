package agent

import (
	"strings"
	"testing"
)

// These tests exercise the in-kernel planner's tool-call SEAM — the normalization that
// InKernelPlanner.Complete runs on its raw text completion — without booting a weighted
// model (which OOMs under WSL). They pin the exact transformation Complete applies after
// decode: lift the model's text-form <tool_call> into structured Message.ToolCalls, set
// the finish reason, and fail closed on a truncated call.

// inKernelNormalize mirrors the post-decode steps of InKernelPlanner.Complete (the two
// lines after the raw Completion is built), so the seam is testable on a synthesized
// completion. Keep it in lockstep with Complete.
func inKernelNormalize(content, finishReason string) *Completion {
	comp := &Completion{
		Message:      Message{Role: "assistant", Content: content},
		FinishReason: finishReason,
	}
	comp = normalizeCompletionToolCalls(comp)
	if len(comp.Message.ToolCalls) == 0 && strings.Contains(comp.Message.Content, "<tool_call>") {
		comp.ToolCallsDropped = true
	}
	return comp
}

// TestCompleteLiftsTextToolCall: a well-formed Hermes <tool_call> (Qwen2.5's native
// dialect) decoded by the in-kernel forward is lifted to a structured ToolCall and the
// finish reason becomes tool_calls — the signal the gateway adjudicator + Anthropic wire
// read to emit a tool_use block.
func TestCompleteLiftsTextToolCall(t *testing.T) {
	comp := inKernelNormalize(`<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>`, "stop")
	if len(comp.Message.ToolCalls) != 1 {
		t.Fatalf("want 1 lifted tool call, got %d (content=%q)", len(comp.Message.ToolCalls), comp.Message.Content)
	}
	if comp.Message.ToolCalls[0].Function.Name != "Bash" {
		t.Fatalf("lifted call name = %q, want Bash", comp.Message.ToolCalls[0].Function.Name)
	}
	if comp.FinishReason != "tool_calls" {
		t.Fatalf("finish reason = %q, want tool_calls", comp.FinishReason)
	}
	if comp.ToolCallsDropped {
		t.Fatalf("a well-formed call must not be flagged dropped")
	}
}

// TestCompleteLiftsMultipleToolCalls: two <tool_call> blocks in one turn both lift.
func TestCompleteLiftsMultipleToolCalls(t *testing.T) {
	comp := inKernelNormalize(
		`<tool_call>{"name": "Read", "arguments": {"path": "a"}}</tool_call>`+
			"\n"+`<tool_call>{"name": "Read", "arguments": {"path": "b"}}</tool_call>`, "stop")
	if len(comp.Message.ToolCalls) != 2 {
		t.Fatalf("want 2 lifted tool calls, got %d", len(comp.Message.ToolCalls))
	}
}

// TestCompleteMalformedToolCallFailsClosed: a TRUNCATED/unclosed <tool_call> the lift
// cannot recover sets ToolCallsDropped so the conformance gate refuses the turn rather
// than leaking a half-formed call into Claude Code's context. The content is preserved.
func TestCompleteMalformedToolCallFailsClosed(t *testing.T) {
	comp := inKernelNormalize(`sure, calling it: <tool_call>{"name": "Bash", "argum`, "length")
	if len(comp.Message.ToolCalls) != 0 {
		t.Fatalf("a truncated call must lift 0 structured calls, got %d", len(comp.Message.ToolCalls))
	}
	if !comp.ToolCallsDropped {
		t.Fatalf("a truncated <tool_call> must set ToolCallsDropped (fail closed)")
	}
	if !strings.Contains(comp.Message.Content, "<tool_call>") {
		t.Fatalf("content must be preserved for the operator to see the truncation")
	}
}

// TestCompletePlainChatUnaffected: a turn with no tool call is unchanged — plain chat
// still works on the in-kernel path.
func TestCompletePlainChatUnaffected(t *testing.T) {
	comp := inKernelNormalize("2 + 2 is 4.", "stop")
	if len(comp.Message.ToolCalls) != 0 {
		t.Fatalf("plain chat must lift 0 tool calls, got %d", len(comp.Message.ToolCalls))
	}
	if comp.ToolCallsDropped {
		t.Fatalf("plain chat must not be flagged dropped")
	}
	if comp.FinishReason != "stop" {
		t.Fatalf("plain chat finish reason changed to %q", comp.FinishReason)
	}
	if comp.Message.Content != "2 + 2 is 4." {
		t.Fatalf("plain chat content changed: %q", comp.Message.Content)
	}
}
