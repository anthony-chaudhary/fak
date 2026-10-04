package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// batchedToolResultInbound is ONE user turn carrying two parallel tool_result
// blocks (the shape Claude Code emits for an assistant turn with two tool calls),
// preceded by the assistant turn that issued both calls.
const batchedToolResultInbound = `{
  "model": "claude-sonnet-4",
  "max_tokens": 256,
  "system": "rules",
  "messages": [
    {"role": "user", "content": "do two things"},
    {"role": "assistant", "content": [{"type": "text", "text": "calling both"},
      {"type": "tool_use", "id": "toolu_1", "name": "alpha", "input": {"x": 1}},
      {"type": "tool_use", "id": "toolu_2", "name": "beta", "input": {"y": 2}}]},
    {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "toolu_1", "content": "A-ok"},
      {"type": "tool_result", "tool_use_id": "toolu_2", "content": "B-ok"}]}
  ]
}`

func decodeBatched(t *testing.T) []Message {
	t.Helper()
	req, err := DecodeAnthropicMessagesRequest([]byte(batchedToolResultInbound))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	return req.Messages
}

func toolResultRunLen(msgs []Message) int {
	n := 0
	for _, m := range msgs {
		if m.Role == RoleTool {
			n++
		}
	}
	return n
}

// TestParallelToolResultsRecollapseAnthropic is the #5797 witness: one inbound
// user turn holding two tool_result blocks must re-encode to exactly ONE Anthropic
// user turn carrying both tool_result blocks, not two stacked user turns.
func TestParallelToolResultsRecollapseAnthropic(t *testing.T) {
	msgs := decodeBatched(t)
	if got := toolResultRunLen(msgs); got != 2 {
		t.Fatalf("canonical fan-out: want 2 RoleTool messages, got %d", got)
	}
	// Canonical array must not stack user turns (the fan-out shape the adjudicator
	// exempts); assert the invariant directly so a marshal-side fix cannot mask it.
	for i := 1; i < len(msgs); i++ {
		if msgs[i].Role == RoleUser && msgs[i-1].Role == RoleUser {
			t.Fatalf("canonical array stacks user turns at %d", i)
		}
	}

	adapter, err := NewTranscriptAdapter(ProviderAnthropic)
	if err != nil {
		t.Fatal(err)
	}
	body, err := adapter.MarshalRequest(adapterRequest{Model: "claude-sonnet-4", Messages: msgs, MaxTokens: 256})
	if err != nil {
		t.Fatal(err)
	}
	var req struct {
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type      string `json:"type"`
				ToolUseID string `json:"tool_use_id"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatal(err)
	}
	// Count only the user turns that CARRY tool_result blocks: the inbound "do two
	// things" user turn is a legitimate separate turn, not part of the run.
	userTurns, toolResults := 0, 0
	var ids []string
	for _, m := range req.Messages {
		if m.Role != "user" {
			continue
		}
		carries := false
		for _, b := range m.Content {
			if b.Type == "tool_result" {
				carries = true
				toolResults++
				ids = append(ids, b.ToolUseID)
			}
		}
		if carries {
			userTurns++
		}
	}
	if userTurns != 1 {
		t.Fatalf("anthropic: want 1 tool-result user turn, got %d (wire stacks parallel tool results)", userTurns)
	}
	if toolResults != 2 {
		t.Fatalf("anthropic: want 2 tool_result blocks, got %d", toolResults)
	}
	if len(ids) != 2 || ids[0] != "toolu_1" || ids[1] != "toolu_2" {
		t.Fatalf("anthropic: tool_result order not preserved: %v", ids)
	}
}

// TestParallelToolResultsRecollapseGemini proves the same re-collapse on the
// Gemini wire: two functionResponse parts in ONE user content, not two contents.
func TestParallelToolResultsRecollapseGemini(t *testing.T) {
	msgs := decodeBatched(t)
	adapter, err := NewTranscriptAdapter(ProviderGemini)
	if err != nil {
		t.Fatal(err)
	}
	body, err := adapter.MarshalRequest(adapterRequest{Model: "gemini-2.5-pro", Messages: msgs, MaxTokens: 256})
	if err != nil {
		t.Fatal(err)
	}
	var req struct {
		Contents []struct {
			Role  string `json:"role"`
			Parts []struct {
				FunctionResponse *struct {
					Name string `json:"name"`
					ID   string `json:"id"`
				} `json:"functionResponse"`
			} `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatal(err)
	}
	userContents, responses := 0, 0
	for _, c := range req.Contents {
		if c.Role != "user" {
			continue
		}
		before := responses
		for _, p := range c.Parts {
			if p.FunctionResponse != nil {
				responses++
			}
		}
		if responses > before {
			userContents++
		}
	}
	if userContents != 1 {
		t.Fatalf("gemini: want 1 tool-result user content, got %d", userContents)
	}
	if responses != 2 {
		t.Fatalf("gemini: want 2 functionResponse parts, got %d", responses)
	}
}

func openAIToolTurnCounts(t *testing.T, asText bool) (userTurnsWithToolResponse, nativeToolMessages int) {
	t.Helper()
	msgs := decodeBatched(t)
	adapter, err := NewTranscriptAdapter(ProviderOpenAI)
	if err != nil {
		t.Fatal(err)
	}
	body, err := adapter.MarshalRequest(adapterRequest{
		Model: "qwen3-coder", Messages: msgs, MaxTokens: 256, OpenAIToolMessagesAsText: asText,
	})
	if err != nil {
		t.Fatal(err)
	}
	var req struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatal(err)
	}
	for _, m := range req.Messages {
		if m.Role == "tool" {
			nativeToolMessages++
		}
		if strings.Contains(m.Content, "<tool_response>") {
			userTurnsWithToolResponse++
		}
	}
	return userTurnsWithToolResponse, nativeToolMessages
}

// TestParallelToolResultsRecollapseOpenAIText proves the opt-in text lowering
// collapses the run into one <tool_response> turn carrying both results.
func TestParallelToolResultsRecollapseOpenAIText(t *testing.T) {
	userTurns, native := openAIToolTurnCounts(t, true)
	if native != 0 {
		t.Fatalf("as-text: want 0 native role=tool messages, got %d", native)
	}
	if userTurns != 1 {
		t.Fatalf("as-text: want 1 user turn carrying <tool_response>, got %d (stacked)", userTurns)
	}
}

// TestParallelToolResultsOpenAINativeUnchanged is the no-change arm: the default
// OpenAI wire keeps one native role=tool message per result (no re-collapse).
func TestParallelToolResultsOpenAINativeUnchanged(t *testing.T) {
	userTurns, native := openAIToolTurnCounts(t, false)
	if userTurns != 0 {
		t.Fatalf("native: want no <tool_response> text turns, got %d", userTurns)
	}
	if native != 2 {
		t.Fatalf("native: want 2 native role=tool messages, got %d", native)
	}
}

// TestSingleToolResultUnchanged pins no behaviour change for the single-result case.
func TestSingleToolResultUnchanged(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "one"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Type: "function", Function: Func{Name: "only", Arguments: "{}"}}}},
		{Role: RoleTool, ToolCallID: "c1", Name: "only", Content: "one result"},
	}
	adapter, err := NewTranscriptAdapter(ProviderAnthropic)
	if err != nil {
		t.Fatal(err)
	}
	body, err := adapter.MarshalRequest(adapterRequest{Model: "m", Messages: msgs, MaxTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	var req struct {
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type string `json:"type"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatal(err)
	}
	toolResultTurns := 0
	for _, m := range req.Messages {
		if m.Role != "user" {
			continue
		}
		for _, b := range m.Content {
			if b.Type == "tool_result" {
				toolResultTurns++
				break
			}
		}
	}
	if toolResultTurns != 1 { // exactly one tool-result turn, unchanged from pre-fix
		t.Fatalf("single tool result: want 1 tool-result user turn, got %d", toolResultTurns)
	}
}
