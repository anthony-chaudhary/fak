package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestDecodeInboundNoEmptyCollapse is the parent-#3118 repro: the inbound Anthropic decoder must
// never fold a message made ENTIRELY of text-less / unrecognized content blocks to empty content.
// Each case covers a distinct block class (image, a synthetic unknown type, thinking/redacted) and
// asserts the decoded canonical message is non-empty and well-formed.
func TestDecodeInboundNoEmptyCollapse(t *testing.T) {
	cases := []struct {
		name         string
		role         string
		content      string // the JSON array for message.content
		wantNonEmpty bool   // the decoded turn must carry non-empty content or a first-class field
		wantSubstr   string // a marker/text the decoded content must contain (empty = skip)
	}{
		{
			name:         "image_only_user_turn",
			role:         "user",
			content:      `[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}]`,
			wantNonEmpty: true,
			wantSubstr:   "[image]",
		},
		{
			name:         "unknown_type_user_turn",
			role:         "user",
			content:      `[{"type":"server_tool_use","id":"x","name":"web_search"}]`,
			wantNonEmpty: true,
			wantSubstr:   "[server_tool_use block]",
		},
		{
			name:         "image_only_tool_result",
			role:         "user",
			content:      `[{"type":"tool_result","tool_use_id":"t","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]}]`,
			wantNonEmpty: true,
			wantSubstr:   "[image]",
		},
		{
			name:         "unknown_only_tool_result",
			role:         "user",
			content:      `[{"type":"tool_result","tool_use_id":"t","content":[{"type":"tool_reference","tool_name":"Read"}]}]`,
			wantNonEmpty: true,
			wantSubstr:   "[tool_reference block]",
		},
		{
			name:         "image_on_assistant_turn",
			role:         "assistant",
			content:      `[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]`,
			wantNonEmpty: true,
			wantSubstr:   "[image]",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			msgs := decodeAnthropicMessage(anthropicInboundMessage{
				Role:    tc.role,
				Content: json.RawMessage(tc.content),
			})
			if len(msgs) == 0 {
				t.Fatalf("decoded to ZERO messages — the turn collapsed entirely")
			}
			// Every emitted message must carry a non-empty content or a first-class field; none may
			// be an empty tool/user/assistant turn (the shape a strict downstream 400s).
			var joined strings.Builder
			for _, m := range msgs {
				joined.WriteString(m.Content)
				if m.Role == RoleTool && m.Content == "" {
					t.Fatalf("emitted an EMPTY tool message (tool_call_id=%q) — empty-content collapse", m.ToolCallID)
				}
			}
			if tc.wantNonEmpty && strings.TrimSpace(joined.String()) == "" {
				t.Fatalf("all decoded content is empty:\n%+v", msgs)
			}
			if tc.wantSubstr != "" && !strings.Contains(joined.String(), tc.wantSubstr) {
				t.Fatalf("decoded content %q missing marker %q", joined.String(), tc.wantSubstr)
			}
		})
	}
}

// TestDecodeInboundThinkingRoundTrip is the #3120 repro (the inbound mirror of the adapter-side
// anthropic_thinking_test.go): a replayed assistant turn carrying thinking + redacted_thinking must
// survive inbound decode with its reasoning text, signature, and redacted payloads intact, and then
// re-encode thinking-FIRST with the signature so the Anthropic round-trip is wire-valid (no 400).
func TestDecodeInboundThinkingRoundTrip(t *testing.T) {
	content := `[` +
		`{"type":"thinking","thinking":"let me reason step by step","signature":"sig-xyz"},` +
		`{"type":"redacted_thinking","data":"ENCRYPTED=="},` +
		`{"type":"text","text":"the answer is 42"},` +
		`{"type":"tool_use","id":"toolu_1","name":"calc","input":{"x":1}}` +
		`]`
	msgs := decodeAnthropicMessage(anthropicInboundMessage{Role: "assistant", Content: json.RawMessage(content)})
	if len(msgs) != 1 {
		t.Fatalf("want 1 assistant message, got %d: %+v", len(msgs), msgs)
	}
	m := msgs[0]
	if m.Thinking != "let me reason step by step" {
		t.Errorf("Thinking = %q, want preserved reasoning", m.Thinking)
	}
	if m.ThinkingSignature != "sig-xyz" {
		t.Errorf("ThinkingSignature = %q, want sig-xyz", m.ThinkingSignature)
	}
	if len(m.RedactedThinking) != 1 || m.RedactedThinking[0] != "ENCRYPTED==" {
		t.Errorf("RedactedThinking = %v, want [ENCRYPTED==]", m.RedactedThinking)
	}
	if m.Content != "the answer is 42" {
		t.Errorf("Content = %q, want the text block", m.Content)
	}
	if len(m.ToolCalls) != 1 || m.ToolCalls[0].Function.Name != "calc" {
		t.Fatalf("ToolCalls = %+v, want the calc tool_use", m.ToolCalls)
	}

	// Re-encode via the outbound adapter and confirm the thinking block is FIRST and signed.
	adapter, err := NewTranscriptAdapter(ProviderAnthropic)
	if err != nil {
		t.Fatal(err)
	}
	body, err := adapter.MarshalRequest(adapterRequest{Model: "m", MaxTokens: 64, Messages: []Message{m}})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var req struct {
		Messages []struct {
			Content []struct {
				Type      string `json:"type"`
				Signature string `json:"signature"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("re-decode marshaled body: %v", err)
	}
	if len(req.Messages) != 1 || len(req.Messages[0].Content) == 0 {
		t.Fatalf("re-encoded body has no assistant content: %s", body)
	}
	first := req.Messages[0].Content[0]
	if first.Type != "thinking" {
		t.Fatalf("first re-encoded block is %q, want thinking-first (Anthropic ordering): %s", first.Type, body)
	}
	if first.Signature != "sig-xyz" {
		t.Fatalf("re-encoded thinking block lost its signature: %s", body)
	}
}

// TestDecodeInboundThinkingOnlyTurnSurvives guards the alternation invariant: a thinking-ONLY
// assistant turn (no text, no tool_use) must NOT decode to zero messages — dropping it desyncs the
// assistant/user alternation and loses the signature a later turn needs (#3120).
func TestDecodeInboundThinkingOnlyTurnSurvives(t *testing.T) {
	content := `[{"type":"thinking","thinking":"quiet reasoning","signature":"sig-only"}]`
	msgs := decodeAnthropicMessage(anthropicInboundMessage{Role: "assistant", Content: json.RawMessage(content)})
	if len(msgs) != 1 {
		t.Fatalf("thinking-only turn decoded to %d messages, want 1 (must not vanish)", len(msgs))
	}
	if msgs[0].Thinking != "quiet reasoning" || msgs[0].ThinkingSignature != "sig-only" {
		t.Fatalf("thinking-only turn lost its reasoning/signature: %+v", msgs[0])
	}
}
