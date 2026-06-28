package agent

import (
	"strings"
	"testing"
)

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

// TestRenderChatMLToolResult ensures a tool-role message reads as user context to the
// model in Qwen2.5's canonical <tool_response> grammar (the multi-turn tool flow a
// tool-trained model recognizes).
func TestRenderChatMLToolResult(t *testing.T) {
	got := renderChatML([]Message{
		{Role: "user", Content: "run it"},
		{Role: "tool", Name: "read_file", Content: "notes"},
	})
	want := "<|im_start|>user\nrun it<|im_end|>\n" +
		"<|im_start|>user\n<tool_response>\nread_file: notes\n</tool_response><|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("tool-result render drift:\nwant: %q\n got: %q", want, got)
	}
}

// TestRenderChatMLToolsNilByteIdentical pins the load-bearing invariant: with nil tools
// and no structured tool call/result, renderChatMLTools is byte-for-byte identical to the
// historical renderChatML — so radix KV reuse and poison eviction (which render with nil
// tools) keep the exact pre-tool token path.
func TestRenderChatMLToolsNilByteIdentical(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}
	if a, b := renderChatML(msgs), renderChatMLTools(msgs, nil); a != b {
		t.Fatalf("nil-tools render diverged from historical:\n hist: %q\n tools: %q", a, b)
	}
}

// TestRenderChatMLInjectsToolSchemas asserts the tool JSON-schemas land INSIDE the single
// leading folded system block (the constraint that keeps the tool spec part of every
// token-prefix), with the <tools> signatures and the <tool_call> instruction.
func TestRenderChatMLInjectsToolSchemas(t *testing.T) {
	tools := []ToolDef{{
		Type: "function",
		Function: ToolDefFunction{
			Name:        "Bash",
			Description: "run a shell command",
			Parameters:  []byte(`{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}`),
		},
	}}
	got := renderChatMLTools([]Message{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "list files"},
	}, tools)
	// The schema must be inside the ONE leading system block: everything from the first
	// <|im_start|>system to its <|im_end|> is the system block; the tool spec lives there.
	sysStart := strings.Index(got, "<|im_start|>system\n")
	sysEnd := strings.Index(got, "<|im_end|>")
	if sysStart != 0 || sysEnd < 0 {
		t.Fatalf("expected a leading system block, got: %q", got)
	}
	sysBlock := got[len("<|im_start|>system\n"):sysEnd]
	for _, want := range []string{"be brief", "<tools>", `"name":"Bash"`, `"command"`, "</tools>", "<tool_call>"} {
		if !strings.Contains(sysBlock, want) {
			t.Fatalf("system block missing %q\nblock: %q", want, sysBlock)
		}
	}
	// Exactly one system block (the spec folded in, not a second block).
	if n := strings.Count(got, "<|im_start|>system"); n != 1 {
		t.Fatalf("want exactly 1 system block, got %d:\n%q", n, got)
	}
}

// TestRenderChatMLToolsNoSystemMessage: tools but no system message still produces one
// leading system block carrying only the tool spec.
func TestRenderChatMLToolsNoSystemMessage(t *testing.T) {
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "ls", Parameters: []byte(`{}`)}}}
	got := renderChatMLTools([]Message{{Role: "user", Content: "go"}}, tools)
	if !strings.HasPrefix(got, "<|im_start|>system\n") {
		t.Fatalf("tools with no system message should still emit a leading system block: %q", got)
	}
	if !strings.Contains(got, `"name":"ls"`) {
		t.Fatalf("tool spec missing: %q", got)
	}
}

// TestRenderChatMLToolCallHistoryRoundTrips renders an assistant turn carrying a
// structured ToolCall, then feeds the rendered <tool_call> text back through
// LiftTextToolCalls and asserts the same name+arguments are recovered — proving
// render/lift symmetry, which is what closes the multi-turn agent loop.
func TestRenderChatMLToolCallHistoryRoundTrips(t *testing.T) {
	asst := Message{Role: "assistant", ToolCalls: []ToolCall{{
		ID:       "call_0",
		Function: Func{Name: "Bash", Arguments: `{"command":"ls -la"}`},
	}}}
	got := renderTranscriptTools([]Message{asst}, nil)
	if !strings.Contains(got, "<tool_call>") || !strings.Contains(got, `"name": "Bash"`) {
		t.Fatalf("assistant tool_call not rendered canonically: %q", got)
	}
	// Extract just the assistant turn body and round-trip it through the lift.
	lifted := LiftTextToolCalls(Message{Role: "assistant", Content: got})
	if len(lifted.ToolCalls) != 1 {
		t.Fatalf("round-trip lift recovered %d calls, want 1\nrendered: %q", len(lifted.ToolCalls), got)
	}
	if lifted.ToolCalls[0].Function.Name != "Bash" {
		t.Fatalf("round-trip name = %q, want Bash", lifted.ToolCalls[0].Function.Name)
	}
	if !strings.Contains(lifted.ToolCalls[0].Function.Arguments, "ls -la") {
		t.Fatalf("round-trip args lost the command: %q", lifted.ToolCalls[0].Function.Arguments)
	}
}

// TestPrefixInvariantWithTools extends the radix prefix property to a transcript WITH a
// tools system block: renderTranscriptTools(full[:k+1], tools) must be a string-prefix of
// renderChatMLTools(full, tools) for every k (the tokenizer-free witness; the token-level
// gate lives in zz_prefix_probe_test.go for the nil-tools case).
func TestPrefixInvariantWithTools(t *testing.T) {
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Bash", Description: "shell", Parameters: []byte(`{"type":"object"}`)}}}
	full := []Message{
		{Role: "system", Content: "be careful"},
		{Role: "user", Content: "list the files"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c0", Function: Func{Name: "Bash", Arguments: `{"command":"ls"}`}}}},
		{Role: "tool", Name: "Bash", Content: "a.go b.go"},
		{Role: "assistant", Content: "two files"},
	}
	cached := renderChatMLTools(full, tools)
	for k := range full {
		ev := renderTranscriptTools(full[:k+1], tools)
		if !strings.HasPrefix(cached, ev) {
			t.Errorf("throughIdx=%d: renderTranscriptTools is NOT a string-prefix of renderChatMLTools\n ev=%q\nful=%q", k, ev, cached)
		}
	}
}

// TestToolsLessEvictRenderMissesCachedToolTurn measures the cost #612 closes: when a turn is
// generated WITH tools (renderChatMLTools folds the tool-spec into the leading system block), the
// historical tools-LESS eviction render (renderTranscript) is NOT a string-prefix of the cached
// turn — it diverges right at the folded tool-spec, so EvictPrefix walks off at the system block
// and reclaims NOTHING (the fail-open this issue documents). Threading the SAME tools makes the
// eviction render a genuine prefix again (the positive direction is TestPrefixInvariantWithTools).
// This is the deterministic, tokenizer-free witness that the un-reclaimed span is real on EVERY
// tool-using turn (not negligible), and a regression guard against reverting EvictPoisoned to the
// tools-less render.
func TestToolsLessEvictRenderMissesCachedToolTurn(t *testing.T) {
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "fetch_url", Description: "fetch", Parameters: []byte(`{"type":"object"}`)}}}
	full := []Message{
		{Role: "system", Content: "be careful"},
		{Role: "user", Content: "look it up"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c0", Function: Func{Name: "fetch_url", Arguments: `{"u":"x"}`}}}},
		{Role: "tool", Name: "fetch_url", Content: "secret leaked"},
	}
	poisonIdx := len(full) - 1 // the tool result is the poisoned message
	cached := renderChatMLTools(full, tools)

	// OLD behavior: EvictPoisoned rendered renderTranscript (tools-less). It diverges from the
	// cached turn at the folded tool-spec, so it is NOT a prefix -> EvictPrefix reclaims nothing.
	toolsLess := renderTranscript(full[:poisonIdx+1])
	if strings.HasPrefix(cached, toolsLess) {
		t.Fatalf("tools-less eviction render is unexpectedly a prefix of the tool-bearing cached turn — the #612 reuse loss would not exist\n toolsLess=%q\n   cached=%q", toolsLess, cached)
	}

	// NEW behavior (#612): rendering the SAME tools makes the eviction path a genuine prefix, so
	// the radix walk lands on the cached node and the poison span is reclaimed.
	withTools := renderTranscriptTools(full[:poisonIdx+1], tools)
	if !strings.HasPrefix(cached, withTools) {
		t.Fatalf("tools-aware eviction render must be a prefix of the cached turn (the #612 fix)\n withTools=%q\n    cached=%q", withTools, cached)
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
