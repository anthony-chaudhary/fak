// fak-test:runtime fast est=1s lane=default
package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type stubReasoningInk2Gen struct {
	ids []int
}

func (g *stubReasoningInk2Gen) Generate(_ context.Context, _ []int, _ int, _ map[int]bool, emit func(int) bool) (Inkling2LoopGeneration, error) {
	for _, id := range g.ids {
		if !emit(id) {
			return Inkling2LoopGeneration{Generated: len(g.ids), Stopped: true}, nil
		}
	}
	return Inkling2LoopGeneration{Generated: len(g.ids), Stopped: true}, nil
}

func newReasoningUsageTestTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := map[string]int{"Ġ": 0, "Ċ": 1}
	for ch := byte(32); ch < 127; ch++ {
		vocab[string(ch)] = int(ch)
	}
	added := []map[string]any{
		{"id": 200, "content": "<|message_model|>", "special": true},
		{"id": 201, "content": "<|content_thinking|>", "special": true},
		{"id": 202, "content": "<|content_text|>", "special": true},
		{"id": 203, "content": "<|end_message|>", "special": true},
		{"id": 204, "content": Inkling2EndOfTurnMarker, "special": true},
		{"id": 205, "content": "<|message_user|>", "special": true},
		{"id": 206, "content": "<|message_system|>", "special": true},
	}
	raw, err := json.Marshal(map[string]any{
		"model":        map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}},
		"decoder":      map[string]any{"type": "ByteLevel"},
		"added_tokens": added,
	})
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.ParseJSON(raw)
	if err != nil {
		t.Fatalf("parse tokenizer: %v", err)
	}
	return tok
}

func TestPopulateReasoningTokensUsage(t *testing.T) {
	tok := newReasoningUsageTestTokenizer(t)

	comp := &Completion{
		Message: Message{Role: "assistant", Content: "b", ReasoningContent: "aaa"},
		Usage:   Usage{CompletionTokens: 4},
	}
	populateReasoningTokensUsage(comp, tok)
	if comp.Usage.CompletionTokensDetails == nil || comp.Usage.CompletionTokensDetails.ReasoningTokens != 3 {
		t.Fatalf("ReasoningTokens = %v, want 3", comp.Usage.CompletionTokensDetails)
	}

	// Empty ReasoningContent leaves CompletionTokensDetails nil.
	plain := &Completion{
		Message: Message{Role: "assistant", Content: "b"},
		Usage:   Usage{CompletionTokens: 2},
	}
	populateReasoningTokensUsage(plain, tok)
	if plain.Usage.CompletionTokensDetails != nil {
		t.Fatalf("expected nil CompletionTokensDetails, got %+v", plain.Usage.CompletionTokensDetails)
	}

	// Clamps to CompletionTokens when encoded reasoning exceeds CompletionTokens.
	clamped := &Completion{
		Message: Message{Role: "assistant", Content: "b", ReasoningContent: "aaaa"},
		Usage:   Usage{CompletionTokens: 2},
	}
	populateReasoningTokensUsage(clamped, tok)
	if clamped.Usage.CompletionTokensDetails == nil || clamped.Usage.CompletionTokensDetails.ReasoningTokens != 2 {
		t.Fatalf("clamped ReasoningTokens = %v, want 2", clamped.Usage.CompletionTokensDetails)
	}
}

func TestInkling2LoopPlannerPopulatesReasoningTokens(t *testing.T) {
	tok := newReasoningUsageTestTokenizer(t)
	ids, err := tok.Encode("<|message_model|><|content_thinking|>aaa<|end_message|><|message_model|><|content_text|>b<|end_message|>")
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	planner, err := NewInkling2LoopPlanner(&stubReasoningInk2Gen{ids: ids}, tok, "inkling2-mini", model.Config{ModelType: "inkling2_loop"}, 1024)
	if err != nil {
		t.Fatalf("NewInkling2LoopPlanner: %v", err)
	}
	comp, err := planner.Complete(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if comp.Message.ReasoningContent != "aaa" {
		t.Fatalf("ReasoningContent = %q, want %q", comp.Message.ReasoningContent, "aaa")
	}
	if comp.Usage.ReasoningTokens() == 0 {
		t.Fatalf("expected non-zero ReasoningTokens, got 0 (details=%+v)", comp.Usage.CompletionTokensDetails)
	}
}
