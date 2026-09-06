package gateway

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestResponsesReasoningDecodeDropsItems proves that historical items of type "reasoning"
// and "thought" are explicitly dropped during decodeResponsesInput.
func TestResponsesReasoningDecodeDropsItems(t *testing.T) {
	inputJSON := []byte(`[
		{"type": "reasoning", "summary": [{"type": "summary_text", "text": "historical reasoning item"}]},
		{"type": "thought", "summary": [{"type": "summary_text", "text": "historical thought item"}]},
		{"type": "message", "role": "user", "content": "user question"},
		{"type": "message", "role": "assistant", "content": "assistant response"}
	]`)

	msgs, err := decodeResponsesInput(inputJSON, "")
	if err != nil {
		t.Fatalf("decodeResponsesInput error: %v", err)
	}

	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages (reasoning items dropped), got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Role != agent.RoleUser || msgs[0].Content != "user question" {
		t.Fatalf("msg 0 unexpected: %+v", msgs[0])
	}
	if msgs[1].Role != agent.RoleAssistant || msgs[1].Content != "assistant response" {
		t.Fatalf("msg 1 unexpected: %+v", msgs[1])
	}
}

// TestResponsesReasoningContentTextStripsThinkingBlocks proves that <think>...</think> and <thought>...</thought>
// blocks are stripped from assistant messages.
func TestResponsesReasoningContentTextStripsThinkingBlocks(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		role     string
		expected string
	}{
		{
			name:     "assistant with think tag",
			input:    `"<think>inner monologue</think>Visible answer."`,
			role:     "assistant",
			expected: "Visible answer.",
		},
		{
			name:     "assistant with thought tag",
			input:    `"<thought>pondering steps</thought>Visible answer."`,
			role:     "assistant",
			expected: "Visible answer.",
		},
		{
			name:     "assistant case insensitive THINK",
			input:    `"<THINK>shouting internally</THINK>Normal text."`,
			role:     "assistant",
			expected: "Normal text.",
		},
		{
			name:     "assistant unclosed think tag",
			input:    `"<think>streaming cut off mid thought"`,
			role:     "assistant",
			expected: "",
		},
		{
			name:     "assistant structured parts with think",
			input:    `[{"type":"output_text","text":"<think>part 1</think>Part 1"},{"type":"output_text","text":"<thought>part 2</thought>Part 2"}]`,
			role:     "assistant",
			expected: "Part 1\nPart 2",
		},
		{
			name:     "user message with think tag remains unstripped",
			input:    `"User says <think>do not strip me</think>"`,
			role:     "user",
			expected: "User says <think>do not strip me</think>",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := responsesContentText(json.RawMessage(tc.input), tc.role)
			if got != tc.expected {
				t.Fatalf("responsesContentText(%s, %s) = %q, want %q", tc.input, tc.role, got, tc.expected)
			}
		})
	}
}

// TestResponsesReasoningHandleClearsAssistantReasoningContent proves historical assistant messages
// have ReasoningContent cleared before calling completeServed.
func TestResponsesReasoningHandleClearsAssistantReasoningContent(t *testing.T) {
	srv := newTestServer(t)
	planner := &capturingResponsesPlanner{
		comp: &agent.Completion{
			Message: agent.Message{Role: agent.RoleAssistant, Content: "reply"},
		},
	}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := map[string]any{
		"model": "test-model",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "hello"},
			map[string]any{"type": "message", "role": "assistant", "content": "<think>old thought</think>old answer"},
			map[string]any{"type": "message", "role": "user", "content": "next question"},
		},
	}

	status, resp := postResponses(t, ts.URL, body)
	if status != 200 || resp.Status != "completed" {
		t.Fatalf("expected 200 completed, got status=%d resp=%+v", status, resp)
	}

	for _, msg := range planner.messages {
		if msg.Role == agent.RoleAssistant {
			if msg.ReasoningContent != "" {
				t.Fatalf("assistant message still has ReasoningContent: %q", msg.ReasoningContent)
			}
			if strings.Contains(msg.Content, "<think>") || strings.Contains(msg.Content, "old thought") {
				t.Fatalf("assistant message content still has thinking tags: %q", msg.Content)
			}
		}
	}
}

// TestResponsesReasoningUsagePreservesReasoningTokens proves responsesUsageFrom correctly preserves
// ReasoningTokens from current turn completion.
func TestResponsesReasoningUsagePreservesReasoningTokens(t *testing.T) {
	// Unit test on responsesUsageFrom.
	u := agent.Usage{
		PromptTokens:     200,
		CompletionTokens: 100,
		TotalTokens:      300,
		CompletionTokensDetails: &agent.UsageCompletionTokenDetails{
			ReasoningTokens: 64,
		},
	}
	ru := responsesUsageFrom(u)
	if ru.OutputTokensDetails == nil {
		t.Fatal("expected non-nil OutputTokensDetails")
	}
	if ru.OutputTokensDetails.ReasoningTokens != 64 {
		t.Fatalf("ReasoningTokens = %d, want 64", ru.OutputTokensDetails.ReasoningTokens)
	}

	// End-to-end test through POST /v1/responses.
	srv := newTestServer(t)
	srv.planner = &reasoningTokensPlanner{
		reasoningTokens: 88,
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := map[string]any{
		"model": "test-model",
		"input": "test prompt",
	}

	status, resp := postResponses(t, ts.URL, body)
	if status != 200 {
		t.Fatalf("expected 200, got %d", status)
	}
	if resp.Usage.OutputTokensDetails == nil {
		t.Fatal("expected resp.Usage.OutputTokensDetails to be non-nil")
	}
	if resp.Usage.OutputTokensDetails.ReasoningTokens != 88 {
		t.Fatalf("resp.Usage.OutputTokensDetails.ReasoningTokens = %d, want 88", resp.Usage.OutputTokensDetails.ReasoningTokens)
	}
}

type reasoningTokensPlanner struct {
	reasoningTokens int
}

func (p *reasoningTokensPlanner) Complete(ctx context.Context, m []agent.Message, t []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, Content: "result"},
		Usage: agent.Usage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
			CompletionTokensDetails: &agent.UsageCompletionTokenDetails{
				ReasoningTokens: p.reasoningTokens,
			},
		},
	}, nil
}

func (*reasoningTokensPlanner) Model() string { return "reasoning-test" }
