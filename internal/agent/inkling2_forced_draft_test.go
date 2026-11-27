package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func inkling2ForcedDraftTestTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := map[string]int{
		"<unk>": 0,
		"\n":    1,
		"\t":    2,
	}
	nextID := len(vocab)
	for b := 0x20; b <= 0x7e; b++ {
		ch := string(byte(b))
		if b == ' ' {
			ch = "Ġ"
		}
		vocab[ch] = nextID
		nextID++
	}
	for _, piece := range []string{
		"name",
		"arguments",
		"read_file",
		"path",
		`{"name":"`,
		`","arguments":{"path":"`,
		`"}}`,
		"go",
		"mod",
	} {
		if _, exists := vocab[piece]; !exists {
			vocab[piece] = nextID
			nextID++
		}
	}
	specials := []string{
		"<|end_message|>",
		Inkling2EndOfTurnMarker,
		"<|message_user|>",
		"<|message_system|>",
		"<|message_model|>",
		"<|content_text|>",
		"<|content_thinking|>",
		"<|content_invoke_tool_json|>",
		"<|content_xml|>",
	}
	added := make([]map[string]any, 0, len(specials))
	for _, s := range specials {
		vocab[s] = nextID
		added = append(added, map[string]any{"id": nextID, "content": s, "special": true})
		nextID++
	}
	doc := map[string]any{
		"model": map[string]any{
			"type":   "BPE",
			"vocab":  vocab,
			"merges": []string{},
		},
		"decoder":      map[string]any{"type": "ByteLevel"},
		"added_tokens": added,
	}
	raw, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.ParseJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

type inkling2StubDraftGen struct {
	draftCalls int
	logitCalls int
	genCalls   int
	stepPreds  func(seq []int) int
	stopID     int
}

func (g *inkling2StubDraftGen) Generate(ctx context.Context, prompt []int, maxNew int, stops map[int]bool, emit func(int) bool) (Inkling2LoopGeneration, error) {
	g.genCalls++
	return g.run(ctx, prompt, maxNew, stops, nil, emit)
}

func (g *inkling2StubDraftGen) GenerateLogits(ctx context.Context, prompt []int, maxNew int, stops map[int]bool, sample func([]float32) (int, error), emit func(int) bool) (Inkling2LoopGeneration, error) {
	g.logitCalls++
	return g.run(ctx, prompt, maxNew, stops, nil, emit)
}

func (g *inkling2StubDraftGen) GenerateDraft(ctx context.Context, prompt []int, maxNew int, stops map[int]bool, sample func([]float32) (int, error), draft Inkling2LoopDrafter, emit func(int) bool) (Inkling2LoopGeneration, error) {
	g.draftCalls++
	return g.run(ctx, prompt, maxNew, stops, draft, emit)
}

func (g *inkling2StubDraftGen) run(_ context.Context, _ []int, maxNew int, stops map[int]bool, draft Inkling2LoopDrafter, emit func(int) bool) (Inkling2LoopGeneration, error) {
	var out Inkling2LoopGeneration
	var emitted []int
	if draft != nil {
		for _, fid := range draft.NextForced(0, false) {
			if out.Generated >= maxNew || stops[fid] {
				break
			}
			out.Generated++
			emitted = append(emitted, fid)
			if !emit(fid) {
				out.Stopped = true
				return out, nil
			}
		}
	}
	for out.Generated < maxNew {
		next := g.stepPreds(emitted)
		if stops[next] {
			out.Stopped = true
			break
		}
		out.Generated++
		emitted = append(emitted, next)
		if !emit(next) {
			out.Stopped = true
			break
		}
		if draft != nil && out.Generated < maxNew {
			for _, fid := range draft.NextForced(next, true) {
				if out.Generated >= maxNew || stops[fid] {
					break
				}
				out.Generated++
				emitted = append(emitted, fid)
				if !emit(fid) {
					out.Stopped = true
					return out, nil
				}
			}
		}
	}
	return out, nil
}

// fak-test:runtime fast est=50ms lane=default
func TestInkling2ForcedDraftVocabAndStateProgression(t *testing.T) {
	tok := inkling2ForcedDraftTestTokenizer(t)
	gen := &inkling2StubDraftGen{}
	cfg := model.Config{ModelType: "inkling2_loop", VocabSize: tok.Vocab()}
	p, err := NewInkling2LoopPlanner(gen, tok, "inkling2-mini", cfg, 4096)
	if err != nil {
		t.Fatal(err)
	}

	tools := []ToolDef{{
		Type: "function",
		Function: ToolDefFunction{
			Name:       "read_file",
			Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}`),
		},
	}}
	sp := applySampleOpts(WithToolChoice(json.RawMessage(`{"type":"function","function":{"name":"read_file"}}`)), WithReasoningEffort(EffortTierNone))
	d := p.newForcedDrafter(sp, tools)
	if d == nil {
		t.Fatal("newForcedDrafter returned nil for forced tool call")
	}

	// Prefix vocab must be cached once per planner.
	if p.forcedDraftVocab() != p.forcedDraftVocab() {
		t.Fatal("forcedDraftVocab rebuilt PrefixVocab instead of reusing cached instance")
	}

	initIDs := d.NextForced(0, false)
	if len(initIDs) == 0 {
		t.Fatal("expected deterministic initial forced tokens for forced tool schema")
	}
	initText, err := tok.Decode(initIDs)
	if err != nil {
		t.Fatal(err)
	}
	wantPrefix := `{"name":"read_file","arguments":{"path":"`
	if initText != wantPrefix {
		t.Fatalf("initial forced text = %q, want %q", initText, wantPrefix)
	}

	// While inside the free string value, NextForced must emit zero forced tokens.
	valIDs, err := tok.Encode("go.mod")
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range valIDs {
		if mid := d.NextForced(id, true); len(mid) != 0 {
			t.Fatalf("emitted false forced tokens %v inside branching string value", mid)
		}
	}

	// Closing quote triggers the deterministic suffix through completion.
	quoteIDs, err := tok.Encode(`"`)
	if err != nil || len(quoteIDs) != 1 {
		t.Fatalf("encode quote: ids=%v err=%v", quoteIDs, err)
	}
	tailIDs := d.NextForced(quoteIDs[0], true)
	if len(tailIDs) == 0 {
		t.Fatal("expected deterministic closing forced tokens after string quote")
	}
	tailText, err := tok.Decode(tailIDs)
	if err != nil {
		t.Fatal(err)
	}
	if tailText != `}}` {
		t.Fatalf("closing forced text = %q, want %q", tailText, `}}`)
	}
	full := initText + "go.mod" + `"` + tailText
	var parsed struct {
		Name      string            `json:"name"`
		Arguments map[string]string `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(full), &parsed); err != nil {
		t.Fatalf("drafted payload %q is not valid JSON: %v", full, err)
	}
	if parsed.Name != "read_file" || parsed.Arguments["path"] != "go.mod" {
		t.Fatalf("parsed drafted call = %+v, want read_file(go.mod)", parsed)
	}
	if !d.state.IsComplete() {
		t.Fatalf("expected completed grammar state after %q", full)
	}

	// Mismatch tolerance: unexpected non-JSON character disables drafting cleanly.
	d2 := p.newForcedDrafter(sp, tools)
	_ = d2.NextForced(0, false)
	for _, id := range valIDs {
		_ = d2.NextForced(id, true)
	}
	_ = d2.NextForced(quoteIDs[0], true)
	if bad := d2.NextForced(quoteIDs[0], true); len(bad) != 0 {
		t.Fatalf("expected nil after completed/mismatched state, got %v", bad)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestInkling2ForcedDraftCompleteUsesGenerateDraftAndPreservesParity(t *testing.T) {
	tok := inkling2ForcedDraftTestTokenizer(t)
	stopID, _ := tok.SpecialID(Inkling2EndOfTurnMarker)
	tools := []ToolDef{{
		Type: "function",
		Function: ToolDefFunction{
			Name:       "read_file",
			Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}`),
		},
	}}
	valIDs, err := tok.Encode(`go.mod"`)
	if err != nil {
		t.Fatal(err)
	}
	idx := 0
	gen := &inkling2StubDraftGen{
		stopID: stopID,
		stepPreds: func(seq []int) int {
			if idx < len(valIDs) {
				tokID := valIDs[idx]
				idx++
				return tokID
			}
			return stopID
		},
	}
	cfg := model.Config{ModelType: "inkling2_loop", VocabSize: tok.Vocab()}
	p, err := NewInkling2LoopPlanner(gen, tok, "inkling2-mini", cfg, 4096)
	if err != nil {
		t.Fatal(err)
	}

	choice := json.RawMessage(`{"type":"function","function":{"name":"read_file"}}`)
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "read go.mod"}}, tools, WithToolChoice(choice), WithReasoningEffort(EffortTierNone), WithMaxTokens(64))
	if err != nil {
		t.Fatal(err)
	}
	if gen.draftCalls != 1 {
		t.Fatalf("draftCalls = %d, want 1", gen.draftCalls)
	}
	if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].Function.Name != "read_file" || comp.Message.ToolCalls[0].Function.Arguments != `{"path":"go.mod"}` {
		t.Fatalf("unexpected forced tool completion: %+v", comp.Message.ToolCalls)
	}

	// Guards: omitted reasoning effort (no <|message_model|><|content_text|> prefill), effort > 0, penalties, Write tool, plain response_format, and auto tool_choice stay off.
	if p.newForcedDrafter(applySampleOpts(WithToolChoice(choice)), tools) != nil {
		t.Fatal("expected ForcedDraft inactive when reasoning effort is omitted")
	}
	if p.newForcedDrafter(applySampleOpts(WithToolChoice(choice), WithReasoningEffort("high")), tools) != nil {
		t.Fatal("expected ForcedDraft inactive when reasoning effort > 0")
	}
	pen := 0.5
	if p.newForcedDrafter(applySampleOpts(WithToolChoice(choice), WithReasoningEffort(EffortTierNone), WithFrequencyPenalty(&pen)), tools) != nil {
		t.Fatal("expected ForcedDraft inactive when frequency penalty set")
	}
	writeTools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Write", Parameters: json.RawMessage(`{"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}`)}}}
	if p.newForcedDrafter(applySampleOpts(WithToolChoice(json.RawMessage(`{"type":"function","function":{"name":"Write"}}`)), WithReasoningEffort(EffortTierNone)), writeTools) != nil {
		t.Fatal("expected ForcedDraft inactive for Write artifact tool")
	}
	if p.newForcedDrafter(applySampleOpts(WithResponseFormat(json.RawMessage(`{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"status":{"type":"string"}},"required":["status"]}}}`)), WithReasoningEffort(EffortTierNone)), nil) != nil {
		t.Fatal("expected ForcedDraft inactive for non-tool response_format")
	}
	if p.newForcedDrafter(applySampleOpts(WithReasoningEffort(EffortTierNone)), tools) != nil {
		t.Fatal("expected ForcedDraft inactive for unconstrained auto tool_choice")
	}
	if got := strings.TrimSpace(comp.Message.Content); got != "" {
		t.Fatalf("forced tool completion leaked text content: %q", got)
	}
}
