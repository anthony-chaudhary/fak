package agent

import "testing"

// TestRenderChatML pins the ChatML shape the in-kernel planner feeds the tokenizer:
// one folded system block, role-tagged turns, an open assistant turn at the end, and
// tool results rendered as user context. A drift here changes every served turn.
func TestRenderChatML(t *testing.T) {
	got := renderChatML([]Message{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
		{Role: "user", Content: "bye"},
	})
	want := "<|im_start|>system\nbe brief<|im_end|>\n" +
		"<|im_start|>user\nhi<|im_end|>\n" +
		"<|im_start|>assistant\nhello<|im_end|>\n" +
		"<|im_start|>user\nbye<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("renderChatML drift:\nwant: %q\n got: %q", want, got)
	}
}

// TestRenderChatMLToolResult ensures a tool-role message reads as user context to
// the model (no structured tool-call emission from this planner yet).
func TestRenderChatMLToolResult(t *testing.T) {
	got := renderChatML([]Message{
		{Role: "user", Content: "run it"},
		{Role: "tool", Name: "read_file", Content: "notes"},
	})
	want := "<|im_start|>user\nrun it<|im_end|>\n" +
		"<|im_start|>user\nread_file: notes<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("tool-result render drift:\nwant: %q\n got: %q", want, got)
	}
}

// TestRenderChatMLNoSystem keeps generation working when there is no system message.
func TestRenderChatMLNoSystem(t *testing.T) {
	got := renderChatML([]Message{{Role: "user", Content: "ping"}})
	want := "<|im_start|>user\nping<|im_end|>\n<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("no-system render drift:\nwant: %q\n got: %q", want, got)
	}
}
