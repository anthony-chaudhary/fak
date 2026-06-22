package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestAnthropicPreservesThinkingBlocks proves the #466 fix: the Anthropic response
// parser no longer drops extended-thinking / redacted_thinking content blocks, and
// the request marshaler re-emits them (signature included) so a thinking turn
// round-trips upstream instead of being silently lost through the proxy.
func TestAnthropicPreservesThinkingBlocks(t *testing.T) {
	adapter, err := NewTranscriptAdapter(ProviderAnthropic)
	if err != nil {
		t.Fatal(err)
	}

	raw := `{"model":"claude-test","content":[` +
		`{"type":"thinking","thinking":"let me reason about this","signature":"sig-abc"},` +
		`{"type":"redacted_thinking","data":"ENCRYPTED=="},` +
		`{"type":"text","text":"the answer is 42"},` +
		`{"type":"tool_use","id":"toolu_1","name":"calc","input":{"x":1}}` +
		`],"stop_reason":"tool_use","usage":{"input_tokens":3,"output_tokens":5}}`

	comp, err := adapter.ParseResponse([]byte(raw))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if comp.Message.Thinking != "let me reason about this" {
		t.Errorf("Thinking = %q, want preserved reasoning text", comp.Message.Thinking)
	}
	if comp.Message.ThinkingSignature != "sig-abc" {
		t.Errorf("ThinkingSignature = %q, want sig-abc", comp.Message.ThinkingSignature)
	}
	if len(comp.Message.RedactedThinking) != 1 || comp.Message.RedactedThinking[0] != "ENCRYPTED==" {
		t.Errorf("RedactedThinking = %v, want [ENCRYPTED==]", comp.Message.RedactedThinking)
	}
	if comp.Message.Content != "the answer is 42" {
		t.Errorf("Content = %q, want the text block", comp.Message.Content)
	}
	if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].Function.Name != "calc" {
		t.Fatalf("ToolCalls = %+v, want the calc tool_use", comp.Message.ToolCalls)
	}

	// The preserved thinking must round-trip back upstream: re-marshal the assistant
	// turn and confirm the thinking + redacted_thinking blocks are emitted, with the
	// thinking block FIRST (the Anthropic API requires that ordering).
	body, err := adapter.MarshalRequest(adapterRequest{
		Model:     "m",
		MaxTokens: 64,
		Messages:  []Message{comp.Message},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var req struct {
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type      string `json:"type"`
				Thinking  string `json:"thinking"`
				Signature string `json:"signature"`
				Data      string `json:"data"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("decode marshaled request: %v (body=%s)", err, body)
	}
	if len(req.Messages) != 1 {
		t.Fatalf("want 1 assistant message, got %d", len(req.Messages))
	}
	blocks := req.Messages[0].Content
	if len(blocks) < 3 {
		t.Fatalf("want thinking+redacted+text(+tool_use) blocks, got %d: %s", len(blocks), body)
	}
	if blocks[0].Type != "thinking" || blocks[0].Thinking != "let me reason about this" || blocks[0].Signature != "sig-abc" {
		t.Errorf("first block must be the signed thinking block, got %+v", blocks[0])
	}
	if blocks[1].Type != "redacted_thinking" || blocks[1].Data != "ENCRYPTED==" {
		t.Errorf("second block must be the redacted_thinking block, got %+v", blocks[1])
	}
	if !strings.Contains(string(body), `"the answer is 42"`) {
		t.Errorf("text block lost in round-trip: %s", body)
	}
}

// TestAnthropicNoThinkingIsClean confirms the additive fields stay absent when the
// model returns no thinking (no empty thinking blocks leak into the request).
func TestAnthropicNoThinkingIsClean(t *testing.T) {
	adapter, err := NewTranscriptAdapter(ProviderAnthropic)
	if err != nil {
		t.Fatal(err)
	}
	comp, err := adapter.ParseResponse([]byte(`{"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if comp.Message.Thinking != "" || comp.Message.ThinkingSignature != "" || comp.Message.RedactedThinking != nil {
		t.Errorf("expected no thinking fields, got %+v", comp.Message)
	}
	body, err := adapter.MarshalRequest(adapterRequest{Model: "m", MaxTokens: 8, Messages: []Message{comp.Message}})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(body), "thinking") {
		t.Errorf("no thinking block should be emitted, got %s", body)
	}
}
