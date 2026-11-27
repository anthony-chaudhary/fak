// fak-test:runtime fast est=1s lane=default
package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

type inkling2WireReplayGen struct {
	greedyCalls int
	logitCalls  int
	scripted    [][]int
	turn        int
}

func (g *inkling2WireReplayGen) Generate(_ context.Context, _ []int, maxNew int, stops map[int]bool, emit func(int) bool) (Inkling2LoopGeneration, error) {
	g.greedyCalls++
	return g.emitScripted(maxNew, stops, emit)
}

func (g *inkling2WireReplayGen) GenerateLogits(_ context.Context, _ []int, maxNew int, stops map[int]bool, sample func([]float32) (int, error), emit func(int) bool) (Inkling2LoopGeneration, error) {
	g.logitCalls++
	// Exercise the sampler closure so non-greedy sampling is verified end-to-end.
	if _, err := sample(ink2LogitRow(map[int]float32{'A': 2.0, 'B': 1.9, 'C': 1.8})); err != nil {
		return Inkling2LoopGeneration{}, err
	}
	return g.emitScripted(maxNew, stops, emit)
}

func (g *inkling2WireReplayGen) emitScripted(maxNew int, stops map[int]bool, emit func(int) bool) (Inkling2LoopGeneration, error) {
	ids := g.scripted[g.turn%len(g.scripted)]
	g.turn++
	emitted := 0
	for _, id := range ids {
		if emitted >= maxNew {
			break
		}
		if stops[id] {
			return Inkling2LoopGeneration{Generated: emitted + 1, Stopped: true}, nil
		}
		emitted++
		if !emit(id) {
			return Inkling2LoopGeneration{Generated: emitted, Stopped: true}, nil
		}
	}
	return Inkling2LoopGeneration{Generated: emitted, Stopped: true}, nil
}

func TestInkling2DefaultWireReplayNonGreedyUniqueIDsAndReasoningTokens(t *testing.T) {
	t.Parallel()
	// Sanitized OpenCode /v1/chat/completions payload: stream=true,
	// max_completion_tokens=16384, reasoning_effort="high", tools[], and
	// explicitly omitted temperature, top_p, top_k, presence_penalty.
	const openCodePayload = `{
		"model": "aegis/inkling2-mini",
		"stream": true,
		"max_completion_tokens": 16384,
		"reasoning_effort": "high",
		"messages": [
			{"role": "system", "content": "You are a coding assistant."},
			{"role": "user", "content": "Check git status and go.mod."}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "Bash",
					"description": "Run a shell command",
					"parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
				}
			}
		]
	}`

	var wire struct {
		Model               string    `json:"model"`
		Stream              bool      `json:"stream"`
		MaxCompletionTokens int       `json:"max_completion_tokens"`
		ReasoningEffort     string    `json:"reasoning_effort"`
		Temperature         *float64  `json:"temperature"`
		TopP                *float64  `json:"top_p"`
		TopK                *int      `json:"top_k"`
		PresencePenalty     *float64  `json:"presence_penalty"`
		Messages            []Message `json:"messages"`
		Tools               []ToolDef `json:"tools"`
	}
	if err := json.Unmarshal([]byte(openCodePayload), &wire); err != nil {
		t.Fatalf("unmarshal sanitized payload: %v", err)
	}
	if wire.Temperature != nil || wire.TopP != nil || wire.TopK != nil || wire.PresencePenalty != nil {
		t.Fatalf("expected omitted sampling params in wire payload")
	}

	tok := inkling2StreamTokenizer(t)
	encodeText := func(s string) []int {
		ids, err := tok.Encode(s)
		if err != nil {
			t.Fatalf("encode %q: %v", s, err)
		}
		return ids
	}
	// Build two turns of raw Inkling-2 output (reasoning + tool call + EOT).
	turn1IDs := encodeText(`<|message_model|><|content_thinking|>check status<|end_message|><|message_model|>Bash<|content_invoke_tool_json|>{"name":"Bash","args":{"command":"git status"}}<|end_message|>` + Inkling2EndOfTurnMarker)
	turn2IDs := encodeText(`<|message_model|><|content_thinking|>read go.mod<|end_message|><|message_model|>Bash<|content_invoke_tool_json|>{"name":"Bash","args":{"command":"cat go.mod"}}<|end_message|>` + Inkling2EndOfTurnMarker)

	gen := &inkling2WireReplayGen{scripted: [][]int{turn1IDs, turn2IDs}}
	p := inkling2StreamPlanner(t, gen)

	opts := []SampleOpt{
		WithModel(wire.Model),
		WithMaxTokens(wire.MaxCompletionTokens),
		WithReasoningEffort(wire.ReasoningEffort),
		WithClientTemperature(wire.Temperature),
		WithTopP(wire.TopP),
		WithChatWireTopK(wire.TopK),
		WithPresencePenalty(wire.PresencePenalty),
	}

	msgs := append([]Message(nil), wire.Messages...)
	seenIDs := map[string]struct{}{}
	for turn := 1; turn <= 2; turn++ {
		var streamedReasoning strings.Builder
		ctx := WithStreamReasoningSink(t.Context(), func(delta string) error {
			streamedReasoning.WriteString(delta)
			return nil
		})
		comp, err := p.CompleteStream(ctx, func(string) error { return nil }, msgs, wire.Tools, opts...)
		if err != nil {
			t.Fatalf("turn %d CompleteStream: %v", turn, err)
		}
		if comp.Message.ReasoningContent == "" {
			t.Fatalf("turn %d: expected non-empty reasoning_content", turn)
		}
		if comp.Usage.CompletionTokensDetails == nil || comp.Usage.CompletionTokensDetails.ReasoningTokens <= 0 {
			t.Fatalf("turn %d: reasoning_tokens = %#v, want > 0", turn, comp.Usage.CompletionTokensDetails)
		}
		if len(comp.Message.ToolCalls) != 1 {
			t.Fatalf("turn %d: tool_calls = %d, want 1", turn, len(comp.Message.ToolCalls))
		}
		id := comp.Message.ToolCalls[0].ID
		if _, dup := seenIDs[id]; dup {
			t.Fatalf("turn %d: duplicate tool_calls[].id %q", turn, id)
		}
		seenIDs[id] = struct{}{}
		msgs = append(msgs, comp.Message, Message{Role: RoleTool, ToolCallID: id, Content: "ok"})
	}

	if gen.logitCalls != 2 || gen.greedyCalls != 0 {
		t.Fatalf("omitted temperature on wire: logitCalls=%d greedyCalls=%d, want 2/0", gen.logitCalls, gen.greedyCalls)
	}

	// Explicit temperature=0 with presence_penalty != 0 must still route to GenerateLogits.
	zero := 0.0
	pres := 0.5
	_, err := p.Complete(t.Context(), wire.Messages, wire.Tools, WithClientTemperature(&zero), WithPresencePenalty(&pres))
	if err != nil {
		t.Fatalf("explicit temp=0 + presence_penalty: %v", err)
	}
	if gen.logitCalls != 3 || gen.greedyCalls != 0 {
		t.Fatalf("temp=0 + presence_penalty: logitCalls=%d greedyCalls=%d, want 3/0", gen.logitCalls, gen.greedyCalls)
	}
}
