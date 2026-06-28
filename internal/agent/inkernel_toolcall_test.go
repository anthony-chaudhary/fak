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

// inKernelDecodeToCompletion mirrors the FULL post-decode pipeline of
// InKernelPlanner.Complete in the exact order Complete runs it: splitReasoning FIRST
// (Qwen3.5/Ornith <think> block → ReasoningContent, post-</think> answer → Content),
// THEN the tool-call lift over Content, THEN the truncated-call fail-closed flag. Unlike
// inKernelNormalize (which only covers the tool-call steps), this exercises the
// reasoning+tool-call COMPOSITION — the ordering issue #1059 names — so a regression that
// lifts the tool call before stripping reasoning (or feeds <think> text into the lift)
// would fail here. Keep it in lockstep with Complete (inkernel_planner.go lines ~667-687).
func inKernelDecodeToCompletion(raw, finishReason string) *Completion {
	reasoning, content := splitReasoning(raw)
	comp := &Completion{
		Message:      Message{Role: "assistant", Content: content, ReasoningContent: reasoning},
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

// TestReasoningThenToolCallOrdering: a reasoning model (Ornith / Qwen3.5) that opens the
// turn with a <think> block and THEN emits a tool call must have the reasoning stripped to
// ReasoningContent and the tool call lifted from the post-</think> content — the
// reasoning-before-call ordering issue #1059's acceptance names. The think text must NOT
// leak into Content (and thus into Claude Code's context), and the lift must still see the
// tool call that follows it. This is the composition splitReasoning → lift, in Complete's
// order.
func TestReasoningThenToolCallOrdering(t *testing.T) {
	raw := "<think>The user wants the directory listing. I'll run ls.</think>" +
		`<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>`
	comp := inKernelDecodeToCompletion(raw, "stop")

	// The reasoning is split off into ReasoningContent, not left in Content.
	if !strings.Contains(comp.Message.ReasoningContent, "directory listing") {
		t.Fatalf("reasoning not captured in ReasoningContent: %q", comp.Message.ReasoningContent)
	}
	if strings.Contains(comp.Message.Content, "<think>") || strings.Contains(comp.Message.Content, "directory listing") {
		t.Fatalf("reasoning leaked into Content: %q", comp.Message.Content)
	}
	// The tool call that FOLLOWS the reasoning still lifts.
	if len(comp.Message.ToolCalls) != 1 {
		t.Fatalf("want 1 tool call lifted after the reasoning block, got %d (content=%q)",
			len(comp.Message.ToolCalls), comp.Message.Content)
	}
	if comp.Message.ToolCalls[0].Function.Name != "Bash" {
		t.Fatalf("lifted call name = %q, want Bash", comp.Message.ToolCalls[0].Function.Name)
	}
	if comp.FinishReason != "tool_calls" {
		t.Fatalf("finish reason = %q, want tool_calls", comp.FinishReason)
	}
	if comp.ToolCallsDropped {
		t.Fatalf("a well-formed reasoning+call turn must not be flagged dropped")
	}
}

// TestReasoningThenMultipleToolCalls: reasoning followed by MORE THAN ONE tool call — the
// multi-call + reasoning-before-call combination — lifts every call and still strips the
// reasoning. Guards the composition against a regression that only handles a single call
// after a think block.
func TestReasoningThenMultipleToolCalls(t *testing.T) {
	raw := "<think>Read both files to compare them.</think>" +
		`<tool_call>{"name": "Read", "arguments": {"path": "a"}}</tool_call>` + "\n" +
		`<tool_call>{"name": "Read", "arguments": {"path": "b"}}</tool_call>`
	comp := inKernelDecodeToCompletion(raw, "stop")
	if !strings.Contains(comp.Message.ReasoningContent, "compare them") {
		t.Fatalf("reasoning not captured: %q", comp.Message.ReasoningContent)
	}
	if len(comp.Message.ToolCalls) != 2 {
		t.Fatalf("want 2 tool calls after the reasoning block, got %d", len(comp.Message.ToolCalls))
	}
}

// TestReasoningWithTruncatedToolCallFailsClosed: a <think> block followed by a TRUNCATED
// tool call must still fail closed (ToolCallsDropped) — the reasoning split must not mask
// the unclosed-call refuse path. The reasoning is captured; the dropped flag fires on the
// post-reasoning content.
func TestReasoningWithTruncatedToolCallFailsClosed(t *testing.T) {
	raw := "<think>I'll call Bash now.</think>" +
		`calling it: <tool_call>{"name": "Bash", "argum`
	comp := inKernelDecodeToCompletion(raw, "length")
	if !strings.Contains(comp.Message.ReasoningContent, "call Bash") {
		t.Fatalf("reasoning not captured ahead of the truncated call: %q", comp.Message.ReasoningContent)
	}
	if len(comp.Message.ToolCalls) != 0 {
		t.Fatalf("a truncated call after reasoning must lift 0 calls, got %d", len(comp.Message.ToolCalls))
	}
	if !comp.ToolCallsDropped {
		t.Fatalf("a truncated <tool_call> after a reasoning block must still set ToolCallsDropped")
	}
}
