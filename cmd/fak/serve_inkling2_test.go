package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// inkling2FakeTokenizer has one id per printable ASCII byte plus the Inkling-2
// specials, so any canned turn encodes and decodes exactly.
func inkling2FakeTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := map[string]int{"Ġ": 0, "Ċ": 1}
	for c := 33; c < 127; c++ {
		vocab[string(rune(c))] = c
	}
	var added []map[string]any
	for i, s := range []string{"<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
		"<|content_text|>", "<|content_thinking|>", "<|content_invoke_tool_json|>", "<|content_xml|>",
		"<|end_message|>", "<|content_model_end_sampling|>"} {
		added = append(added, map[string]any{"id": 200 + i, "content": s, "special": true})
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
		t.Fatalf("parse fake tokenizer: %v", err)
	}
	return tok
}

// inkling2FakeSession replays a canned token stream: Prefill returns the first
// token, each Step the next, and it records what it was fed.
type inkling2FakeSession struct {
	canned  []int
	i       int
	prompt  []int
	fed     []int
	closed  bool
	stepErr error
}

func (s *inkling2FakeSession) Prefill(prompt []int) (int, error) {
	s.prompt = append([]int(nil), prompt...)
	return s.canned[0], nil
}

func (s *inkling2FakeSession) Step(token int) (int, error) {
	if s.stepErr != nil {
		return 0, s.stepErr
	}
	s.fed = append(s.fed, token)
	s.i++
	return s.canned[s.i], nil
}

func (s *inkling2FakeSession) Close() error { s.closed = true; return nil }

func inkling2FakeEngine(canned []int) (*inkling2LoopEngine, *[]*inkling2FakeSession) {
	var opened []*inkling2FakeSession
	return &inkling2LoopEngine{open: func() (ggufload.Inkling2LoopSession, error) {
		s := &inkling2FakeSession{canned: canned}
		opened = append(opened, s)
		return s, nil
	}}, &opened
}

func TestInkling2AdapterStopsOnEndOfTurnNotEndMessage(t *testing.T) {
	const endMessage, endTurn = 208, 209
	eng, opened := inkling2FakeEngine([]int{201, 204, 'h', 'i', endMessage, 201, 204, 'x', endMessage, endTurn, 'z', 'z'})
	var got []int
	res, err := eng.Generate(context.Background(), []int{200, 'q'}, 100, map[int]bool{endTurn: true}, func(id int) bool {
		got = append(got, id)
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []int{201, 204, 'h', 'i', endMessage, 201, 204, 'x', endMessage}
	if !res.Stopped || res.Generated != len(want) || !inkling2EqualInts(got, want) {
		t.Fatalf("emitted %v (stopped=%v n=%d), want %v stopped at the end-of-turn marker", got, res.Stopped, res.Generated, want)
	}
	if s := (*opened)[0]; !s.closed || !inkling2EqualInts(s.prompt, []int{200, 'q'}) {
		t.Fatalf("session prompt %v closed=%v", s.prompt, s.closed)
	}
}

func TestInkling2AdapterMaxTokensAndFreshSessionPerRequest(t *testing.T) {
	eng, opened := inkling2FakeEngine([]int{'a', 'b', 'c', 'd', 'e'})
	for req := 0; req < 2; req++ {
		var got []int
		res, err := eng.Generate(context.Background(), []int{'p'}, 3, nil, func(id int) bool {
			got = append(got, id)
			return true
		})
		if err != nil {
			t.Fatal(err)
		}
		if res.Stopped || res.Generated != 3 || !inkling2EqualInts(got, []int{'a', 'b', 'c'}) {
			t.Fatalf("request %d: emitted %v stopped=%v, want a b c on the length limit", req, got, res.Stopped)
		}
	}
	if len(*opened) != 2 || len((*opened)[1].fed) != 2 {
		t.Fatalf("opened %d sessions, want one fresh session per request", len(*opened))
	}
}

func TestInkling2AdapterEmitFalseAndStepErrorEndTheTurn(t *testing.T) {
	eng, _ := inkling2FakeEngine([]int{'a', 'b', 'c'})
	res, err := eng.Generate(context.Background(), []int{'p'}, 10, nil, func(int) bool { return false })
	if err != nil || !res.Stopped || res.Generated != 1 {
		t.Fatalf("emit=false: res=%+v err=%v", res, err)
	}
	boom := errors.New("device fault")
	failing := &inkling2LoopEngine{open: func() (ggufload.Inkling2LoopSession, error) {
		return &inkling2FakeSession{canned: []int{'a', 'b'}, stepErr: boom}, nil
	}}
	if _, err := failing.Generate(context.Background(), []int{'p'}, 10, nil, func(int) bool { return true }); !errors.Is(err, boom) {
		t.Fatalf("step error = %v, want %v", err, boom)
	}
}

func TestInkling2AdapterPlannerRendersInklingAndLiftsToolCall(t *testing.T) {
	tok := inkling2FakeTokenizer(t)
	turn := `<|message_model|><|content_text|>ok<|end_message|><|message_model|>bash<|content_invoke_tool_json|>{"name":"bash","args":{"command":"pwd"}}<|end_message|><|content_model_end_sampling|>trailing`
	canned, err := tok.Encode(turn)
	if err != nil {
		t.Fatal(err)
	}
	eng, opened := inkling2FakeEngine(canned)
	p, err := newServeInkling2LoopPlanner(eng, tok, "inkling2-mini", 4096)
	if err != nil {
		t.Fatal(err)
	}
	tools := []agent.ToolDef{{Type: "function", Function: agent.ToolDefFunction{Name: "bash", Description: "run a command",
		Parameters: json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}`)}}}
	comp, err := p.Complete(context.Background(), []agent.Message{{Role: "user", Content: "where am I"}}, tools)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := tok.Decode((*opened)[0].prompt)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt, "<|message_user|>") || strings.Contains(prompt, "<|im_start|>") || !strings.Contains(prompt, "tool_declare") {
		t.Fatalf("prompt did not render through the Inkling-2 template:\n%s", prompt)
	}
	if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].Function.Name != "bash" ||
		comp.Message.ToolCalls[0].Function.Arguments != `{"command":"pwd"}` || comp.FinishReason != "tool_calls" {
		t.Fatalf("completion = %+v, want one bash tool call", comp)
	}
	if strings.Contains(comp.Message.Content, "trailing") || strings.Contains(comp.Message.Content, "<|") {
		t.Fatalf("content leaked markers or post-stop text: %q", comp.Message.Content)
	}
	if comp.Model != "inkling2-mini" || comp.Usage.PromptTokens != len((*opened)[0].prompt) {
		t.Fatalf("model %q usage %+v", comp.Model, comp.Usage)
	}
}

func TestInkling2AdapterPlannerRefusesPromptPastContext(t *testing.T) {
	tok := inkling2FakeTokenizer(t)
	eng, opened := inkling2FakeEngine([]int{'a'})
	p, err := newServeInkling2LoopPlanner(eng, tok, "inkling2", 8)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Complete(context.Background(), []agent.Message{{Role: "user", Content: "a long enough prompt"}}, nil); err == nil {
		t.Fatal("prompt past the context window was accepted")
	}
	if len(*opened) != 0 {
		t.Fatal("refused request still opened a device session")
	}
}

func TestInkling2PlannerNeedsEndOfTurnToken(t *testing.T) {
	tok, err := tokenizer.ParseJSON([]byte(`{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"decoder":{"type":"ByteLevel"}}`))
	if err != nil {
		t.Fatal(err)
	}
	eng, _ := inkling2FakeEngine([]int{0})
	if _, err := newServeInkling2LoopPlanner(eng, tok, "inkling2", 64); err == nil {
		t.Fatal("tokenizer without the end-of-turn marker accepted")
	}
}

// writeInkling2HeaderGGUF writes a tensor-less GGUF v3 header carrying only
// general.architecture.
func writeInkling2HeaderGGUF(t *testing.T, arch string) string {
	t.Helper()
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(0))
	w(uint64(1))
	str("general.architecture")
	w(uint32(8))
	str(arch)
	path := filepath.Join(t.TempDir(), arch+".gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestInkling2GGUFSniffAndGenericLoaderRefusal(t *testing.T) {
	path := writeInkling2HeaderGGUF(t, serveInkling2LoopArch)
	if !serveIsInkling2LoopGGUF(path) {
		t.Fatal("inkling2_loop header not recognised")
	}
	if serveIsInkling2LoopGGUF(writeInkling2HeaderGGUF(t, "llama")) || serveIsInkling2LoopGGUF(t.TempDir()) || serveIsInkling2LoopGGUF("") {
		t.Fatal("non-looped input recognised as inkling2_loop")
	}
	_, _, _, _, _, err := loadServeInKernelModelPlacedErr(path, nil, false, false, 0, nil, 1, nil)
	var refusal *fakmodel.UnsupportedInkling2LoopExecutionError
	if !errors.As(err, &refusal) || refusal.Backend != "cpu" || !strings.Contains(err.Error(), "--backend cuda") {
		t.Fatalf("CPU load of an inkling2_loop GGUF = %v, want the typed cpu refusal naming the cuda route", err)
	}
	if err := serveRefuseInkling2LoopGeneric(nil); err == nil || strings.Contains(err.Error(), "not implemented") {
		t.Fatalf("generic refusal = %v", err)
	}
}

func inkling2EqualInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
