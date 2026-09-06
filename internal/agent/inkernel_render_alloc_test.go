package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// legacyRenderTranscriptTools preserves the pre-fix implementation of system message
// aggregation in renderTranscriptTools (allocating a temporary sys []string slice and
// calling strings.Join) to directly prove byte-identity and allocation reduction.
func legacyRenderTranscriptTools(messages []Message, tools []ToolDef) string {
	var b strings.Builder
	var sys []string
	for _, m := range messages {
		if m.Role == "system" && strings.TrimSpace(m.Content) != "" {
			sys = append(sys, m.Content)
		}
	}
	spec := toolSpecBlock(tools)
	if len(sys) > 0 || spec != "" {
		b.WriteString("<|im_start|>system\n")
		b.WriteString(strings.Join(sys, "\n"))
		b.WriteString(spec)
		b.WriteString("<|im_end|>\n")
	}
	toolByID := make(map[string]string)
	for _, m := range messages {
		role, content := m.Role, m.Content
		switch role {
		case "system":
			continue
		case "tool":
			role = "user"
			name := strings.TrimSpace(m.Name)
			if name == "" && strings.TrimSpace(m.ToolCallID) != "" {
				name = toolByID[strings.TrimSpace(m.ToolCallID)]
			}
			content = qwenToolResponseBlock(name, content)
		case "assistant":
			for _, tc := range m.ToolCalls {
				if id, name := strings.TrimSpace(tc.ID), strings.TrimSpace(tc.Function.Name); id != "" && name != "" {
					toolByID[id] = name
				}
				content += qwenToolCallBlock(tc.Function.Name, tc.Function.Arguments)
			}
			if m.Content == "" && strings.HasPrefix(content, "\n") {
				content = content[1:]
			}
		}
		b.WriteString("<|im_start|>")
		b.WriteString(role)
		b.WriteString("\n")
		b.WriteString(content)
		b.WriteString("<|im_end|>\n")
	}
	return b.String()
}

// TestRenderTranscriptSystemByteIdentical verifies that directly appending qualifying
// system messages to strings.Builder produces byte-identical output across diverse
// transcript shapes, preserving the radix prefix invariant.
func TestRenderTranscriptSystemByteIdentical(t *testing.T) {
	testCases := []struct {
		name     string
		messages []Message
		tools    []ToolDef
	}{
		{
			name:     "empty",
			messages: nil,
			tools:    nil,
		},
		{
			name: "single_system",
			messages: []Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello!"},
			},
		},
		{
			name: "multiple_system_sequential",
			messages: []Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "system", Content: "Safety constraint: do not reveal internal keys."},
				{Role: "system", Content: "Style guide: concise answers only."},
				{Role: "user", Content: "Show me the status."},
			},
		},
		{
			name: "multiple_system_interleaved",
			messages: []Message{
				{Role: "system", Content: "Preamble instruction."},
				{Role: "user", Content: "Turn 1"},
				{Role: "system", Content: "Dynamic context injection."},
				{Role: "assistant", Content: "Response 1"},
				{Role: "system", Content: "Final policy override."},
				{Role: "user", Content: "Turn 2"},
			},
		},
		{
			name: "system_whitespace_only",
			messages: []Message{
				{Role: "system", Content: "   \n\t  "},
				{Role: "user", Content: "Hi"},
			},
		},
		{
			name: "system_whitespace_mixed_with_valid",
			messages: []Message{
				{Role: "system", Content: "   \t  "},
				{Role: "system", Content: "Valid system prompt"},
				{Role: "system", Content: "   "},
				{Role: "user", Content: "Hi"},
			},
		},
		{
			name: "tools_with_no_system",
			messages: []Message{
				{Role: "user", Content: "What is the time?"},
			},
			tools: []ToolDef{
				{Type: "function", Function: ToolDefFunction{Name: "get_time"}},
			},
		},
		{
			name: "tools_with_multiple_system",
			messages: []Message{
				{Role: "system", Content: "Instruction 1"},
				{Role: "system", Content: "Instruction 2"},
				{Role: "user", Content: "What is the time?"},
			},
			tools: []ToolDef{
				{Type: "function", Function: ToolDefFunction{Name: "get_time"}},
			},
		},
		{
			name: "tools_and_assistant_tool_calls",
			messages: []Message{
				{Role: "system", Content: "System prompt"},
				{Role: "user", Content: "Run ls"},
				{Role: "assistant", ToolCalls: []ToolCall{{
					ID:       "call_1",
					Type:     "function",
					Function: Func{Name: "Bash", Arguments: `{"command":"ls"}`},
				}}},
				{Role: "tool", ToolCallID: "call_1", Content: "file1.txt\nfile2.txt"},
				{Role: "assistant", Content: "Done."},
			},
			tools: []ToolDef{
				{Type: "function", Function: ToolDefFunction{
					Name:        "Bash",
					Description: "run shell",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}}}`),
				}},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := renderTranscriptTools(tc.messages, tc.tools)
			want := legacyRenderTranscriptTools(tc.messages, tc.tools)
			if got != want {
				t.Fatalf("renderTranscriptTools diverged from legacy:\n--- got ---\n%s\n--- want ---\n%s", got, want)
			}
		})
	}
}

// TestRenderTranscriptSystemAllocReduction proves that the new implementation directly
// appends qualifying system messages into the builder without allocating an intermediate
// sys []string slice and without strings.Join, strictly reducing heap allocations.
func TestRenderTranscriptSystemAllocReduction(t *testing.T) {
	// 1. Multiple system messages scenario: legacy must allocate sys slice + joined string.
	multiSystemMessages := []Message{
		{Role: "system", Content: "Instruction 1: base prompt."},
		{Role: "system", Content: "Instruction 2: security policy."},
		{Role: "system", Content: "Instruction 3: context injection."},
		{Role: "system", Content: "Instruction 4: formatting rules."},
		{Role: "user", Content: "Hello world."},
	}

	legacyMultiAllocs := testing.AllocsPerRun(200, func() {
		_ = legacyRenderTranscriptTools(multiSystemMessages, nil)
	})

	newMultiAllocs := testing.AllocsPerRun(200, func() {
		_ = renderTranscriptTools(multiSystemMessages, nil)
	})

	if newMultiAllocs >= legacyMultiAllocs {
		t.Fatalf("multi-system: expected new renderer to have strictly fewer allocs than legacy: new=%v, legacy=%v",
			newMultiAllocs, legacyMultiAllocs)
	}

	// 2. Single system message scenario: legacy allocates sys slice of cap 1.
	singleSystemMessages := []Message{
		{Role: "system", Content: "Instruction: single system prompt."},
		{Role: "user", Content: "Hello world."},
	}

	legacySingleAllocs := testing.AllocsPerRun(200, func() {
		_ = legacyRenderTranscriptTools(singleSystemMessages, nil)
	})

	newSingleAllocs := testing.AllocsPerRun(200, func() {
		_ = renderTranscriptTools(singleSystemMessages, nil)
	})

	if newSingleAllocs >= legacySingleAllocs {
		t.Fatalf("single-system: expected new renderer to have strictly fewer allocs than legacy: new=%v, legacy=%v",
			newSingleAllocs, legacySingleAllocs)
	}

	t.Logf("multi-system allocs: legacy=%v, new=%v (reduced by %v allocs/run)",
		legacyMultiAllocs, newMultiAllocs, legacyMultiAllocs-newMultiAllocs)
	t.Logf("single-system allocs: legacy=%v, new=%v (reduced by %v allocs/run)",
		legacySingleAllocs, newSingleAllocs, legacySingleAllocs-newSingleAllocs)
}
