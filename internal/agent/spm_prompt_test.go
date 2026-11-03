package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// fak-test:runtime fast est=100ms lane=default
func TestSPMOrdinaryRawPromptOwnsSpecialPolicy(t *testing.T) {
	pieces := []string{"<unk>", "<s>", "</s>", "a", "b", "ab"}
	scores := []float32{0, 0, 0, 0, 0, 2}
	types := []int32{2, 3, 3, 1, 1, 1}
	for b := 0; b < 256; b++ {
		pieces = append(pieces, fmt.Sprintf("<0x%02X>", b))
		scores = append(scores, -100)
		types = append(types, 6)
	}
	tok, err := tokenizer.FromGGMLSPM(pieces, scores, types, tokenizer.SPMOptions{BOS: 1, EOS: 2, AddBOS: true, AddEOS: true})
	if err != nil {
		t.Fatal(err)
	}
	bos := 1
	m := model.NewSynthetic(model.Config{ModelType: "llama", HiddenSize: 8, NumLayers: 1, NumHeads: 2, NumKVHeads: 1, HeadDim: 4,
		IntermediateSize: 16, VocabSize: len(pieces), RMSNormEps: 1e-5, RopeTheta: 10000, MaxPositionEmbeddings: 128, BOSTokenID: &bos, EOSTokenID: 2})
	p := NewInKernelPlanner(m, tok, "synthetic-spm", false, nil, false)
	messages := []Message{{Role: RoleUser, Content: "ab"}}
	got, err := p.EncodePrompt(context.Background(), messages, nil, WithRawPrompt(), WithMaxTokens(1))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.TokenIDs, []int{1, 5, 2}) || got.PromptTokens != 3 || got.RendererID != "llama-spm-raw-v1" || got.ReservedOutputTokens != 1 {
		t.Fatalf("ordinary policy: %+v", got)
	}
	if _, err := p.EncodePrompt(context.Background(), messages, nil, WithMaxTokens(1)); err == nil {
		t.Fatal("SPM chat silently rendered")
	}
	if _, err := p.EncodePrompt(context.Background(), append(messages, messages...), nil, WithRawPrompt(), WithMaxTokens(1)); err == nil {
		t.Fatal("multi-message raw accepted")
	}
	structured := []Message{{Role: RoleUser, Content: "ab", ContentWasStructured: true}}
	if _, err := p.EncodePrompt(context.Background(), structured, nil, WithRawPrompt(), WithMaxTokens(1)); err == nil {
		t.Fatal("structured carrier accepted")
	}
	for _, raw := range []string{
		`{"role":"user","content":"ab","name":"tool"}`,
		`{"role":"user","content":"ab","tool_call_id":"call"}`,
		`{"role":"user","content":"ab","tool_calls":[{"id":"call","type":"function","function":{"name":"f","arguments":"{}"}}]}`,
		`{"role":"user","content":"ab","function_call":{"name":"f","arguments":"{}"}}`,
	} {
		var msg Message
		if err := json.Unmarshal([]byte(raw), &msg); err != nil {
			t.Fatal(err)
		}
		if _, err := p.EncodePrompt(context.Background(), []Message{msg}, nil, WithRawPrompt(), WithMaxTokens(1)); err == nil {
			t.Fatalf("tool-bearing user silently stripped: %s", raw)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.EncodePrompt(ctx, messages, nil, WithRawPrompt()); err != context.Canceled {
		t.Fatalf("cancel: %v", err)
	}
	m.Cfg.EOSTokenID = 0
	if _, err := p.EncodePrompt(context.Background(), messages, nil, WithRawPrompt(), WithMaxTokens(1)); err == nil {
		t.Fatal("mismatched EOS stop admitted")
	}
	m.Cfg.EOSTokenID = 2
	badBOS := 0
	m.Cfg.BOSTokenID = &badBOS
	if _, err := p.EncodePrompt(context.Background(), messages, nil, WithRawPrompt(), WithMaxTokens(1)); err == nil {
		t.Fatal("mismatched BOS admitted")
	}
	m.Cfg.BOSTokenID = &bos
	for _, raw := range []string{"ordinary raw text", "<think>\n"} {
		got, selected := p.spmRawStreamPrompt([]Message{{Role: RoleUser, Content: raw}}, applySampleOpts(WithRawPrompt()))
		if !selected || got != raw {
			t.Fatalf("stream seed differs from exact raw input: %q", raw)
		}
	}
	seed, selected := p.spmRawStreamPrompt([]Message{{Role: RoleUser, Content: "<think>\n"}}, applySampleOpts(WithRawPrompt()))
	if !selected || seed != "<think>\n" {
		t.Fatal("stream projection reconstructed chat instead of raw seed")
	}
	var visible strings.Builder
	projector := newInKernelStreamProjector(func(piece string) error { visible.WriteString(piece); return nil }, nil, strings.HasSuffix(seed, qwenThinkAssistantSeed))
	if err := projector.feed("hidden"); err != nil {
		t.Fatal(err)
	}
	if visible.Len() != 0 {
		t.Fatal("raw seeded reasoning leaked")
	}
	if err := projector.feed("</think>visible"); err != nil {
		t.Fatal(err)
	}
	if err := projector.finish("visible", false); err != nil {
		t.Fatal(err)
	}
	if visible.String() != "visible" {
		t.Fatalf("stream seed projection %q", visible.String())
	}
	generic, err := tok.Encode("ab")
	if err != nil || !reflect.DeepEqual(generic, []int{5}) {
		t.Fatal("generic encoding changed")
	}
}
