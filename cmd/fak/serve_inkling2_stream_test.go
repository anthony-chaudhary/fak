package main

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// inkling2StreamGateSession holds the decode of token at until gate fires, so a
// test can observe what reached the client while the turn is still decoding.
type inkling2StreamGateSession struct {
	*inkling2FakeSession
	at    int
	gate  <-chan struct{}
	early bool
}

func (g *inkling2StreamGateSession) Step(token int) (int, error) {
	if g.i+1 == g.at {
		select {
		case <-g.gate:
			g.early = true
		case <-time.After(5 * time.Second):
		}
	}
	return g.inkling2FakeSession.Step(token)
}

func TestInkling2AdapterPlannerStreamsContentBeforeTurnEnds(t *testing.T) {
	tok := inkling2FakeTokenizer(t)
	text := `<|message_model|><|content_text|>Hello streaming world<|end_message|>`
	turn := text + `<|message_model|>bash<|content_invoke_tool_json|>{"name":"bash","args":{"command":"pwd"}}<|end_message|><|content_model_end_sampling|>`
	canned, err := tok.Encode(turn)
	if err != nil {
		t.Fatal(err)
	}
	textIDs, err := tok.Encode(text)
	if err != nil {
		t.Fatal(err)
	}
	gate := make(chan struct{}, 1)
	var session *inkling2StreamGateSession
	eng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		session = &inkling2StreamGateSession{inkling2FakeSession: &inkling2FakeSession{canned: canned}, at: len(textIDs), gate: gate}
		return session, nil
	})
	p, err := newServeInkling2LoopPlanner(eng, tok, "inkling2-mini", 4096)
	if err != nil {
		t.Fatal(err)
	}
	tools := []agent.ToolDef{{Type: "function", Function: agent.ToolDefFunction{Name: "bash", Description: "run a command",
		Parameters: json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}`)}}}
	var mu sync.Mutex
	var deltas []string
	sink := func(delta string) error {
		mu.Lock()
		deltas = append(deltas, delta)
		mu.Unlock()
		select {
		case gate <- struct{}{}:
		default:
		}
		return nil
	}
	comp, err := p.CompleteStream(context.Background(), sink, []agent.Message{{Role: "user", Content: "where am I"}}, tools)
	if err != nil {
		t.Fatal(err)
	}
	if !session.early {
		t.Fatal("no content delta reached the client while the tool call was still decoding")
	}
	if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].Function.Name != "bash" || comp.FinishReason != "tool_calls" {
		t.Fatalf("completion = %+v, want one bash tool call", comp)
	}
	mu.Lock()
	got := strings.Join(deltas, "")
	mu.Unlock()
	if got != comp.Message.Content || got != "Hello streaming world" {
		t.Fatalf("streamed %q, content %q", got, comp.Message.Content)
	}
}
