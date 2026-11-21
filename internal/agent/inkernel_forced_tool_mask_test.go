package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

var inKernelForcedProbeTool = []ToolDef{{Type: "function", Function: ToolDefFunction{
	Name:       "record_probe",
	Parameters: json.RawMessage(`{"type":"object","properties":{"probe":{"type":"string","enum":["alpha","beta"]}},"required":["probe"],"additionalProperties":false}`),
}}}

func inKernelCheckForcedProbeCall(t *testing.T, comp *Completion) {
	t.Helper()
	if comp.ToolCallsDropped || comp.FinishReason != "tool_calls" || len(comp.Message.ToolCalls) != 1 || comp.Message.Content != "" {
		t.Fatalf("forced tool turn: dropped=%v reason=%v finish=%q calls=%d content=%q", comp.ToolCallsDropped, comp.ToolCallsDroppedReason, comp.FinishReason, len(comp.Message.ToolCalls), comp.Message.Content)
	}
	call := comp.Message.ToolCalls[0]
	if call.Type != "function" || call.Function.Name != "record_probe" {
		t.Fatalf("forced call = %#v", call)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		t.Fatalf("arguments are not a string-valued object: %q: %v", call.Function.Arguments, err)
	}
	if len(args) != 1 || (args["probe"] != "alpha" && args["probe"] != "beta") {
		t.Fatalf("arguments violate the tool schema: %q", call.Function.Arguments)
	}
}

// fak-test:runtime fast est=600ms lane=default
func TestInKernelForcedToolChoiceMasksEnvelope(t *testing.T) {
	// The user text never names the probe value, so only the masked decode can
	// produce schema-valid arguments; the text-scraping fallback finds nothing.
	messages := []Message{{Role: RoleUser, Content: "go"}}
	for _, choice := range []json.RawMessage{
		json.RawMessage(`{"type":"function","function":{"name":"record_probe"}}`),
		json.RawMessage(`"required"`),
	} {
		for _, temperature := range []float64{0, 0.8} {
			t.Run(string(choice)+"/"+string(rune('0'+int(temperature*10))), func(t *testing.T) {
				p := nativeStructuredTestPlanner(t)
				opts := []SampleOpt{WithToolChoice(choice), WithTemperature(&temperature), WithMaxTokens(128)}
				comp, err := p.Complete(context.Background(), messages, inKernelForcedProbeTool, opts...)
				if err != nil {
					t.Fatal(err)
				}
				inKernelCheckForcedProbeCall(t, comp)

				var streamed strings.Builder
				comp, err = p.CompleteStream(context.Background(), func(delta string) error {
					streamed.WriteString(delta)
					return nil
				}, messages, inKernelForcedProbeTool, append(opts, WithPerTokenStream(true))...)
				if err != nil {
					t.Fatal(err)
				}
				inKernelCheckForcedProbeCall(t, comp)
				if streamed.String() != "" {
					t.Fatalf("masked envelope leaked into the content stream: %q", streamed.String())
				}
			})
		}
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestInKernelForcedToolChoiceMaskAfterReasoning(t *testing.T) {
	p := nativeStructuredReasoningPlanner(t, false)
	open, close, body, eos := nativeStructuredReasoningIDs(t, p)
	zero := 0.0
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "go"}}, inKernelForcedProbeTool,
		WithToolChoice(json.RawMessage(`{"type":"function","function":{"name":"record_probe"}}`)), WithThinkingBudget(2), WithTemperature(&zero),
		WithLogitBias(map[int]float64{body: 100, open: 60, close: -100, eos: 200}), WithMaxTokens(128))
	if err != nil {
		t.Fatal(err)
	}
	inKernelCheckForcedProbeCall(t, comp)
	if comp.Message.ReasoningContent != "xx" {
		t.Fatalf("reasoning = %q, want the two budgeted reasoning tokens", comp.Message.ReasoningContent)
	}
}

// fak-test:runtime fast est=1.5s lane=default
func TestInKernelForcedToolChoiceUnsupportedSchemaFallsBack(t *testing.T) {
	choice := json.RawMessage(`{"type":"function","function":{"name":"record_probe"}}`)
	messages := []Message{{Role: RoleUser, Content: "Set probe to alpha."}}
	for name, parameters := range map[string]string{
		"keyword":           `{"type":"object","properties":{"probe":{"type":"string","pattern":"^a"}},"required":["probe"]}`,
		"open-object":       `{"type":"object","properties":{"probe":{"type":"string"}},"required":["probe"],"additionalProperties":true}`,
		"optional-overflow": `{"type":"object","properties":{"probe":{"type":"string"},"a":{"type":"string"},"b":{"type":"string"},"c":{"type":"string"},"d":{"type":"string"},"e":{"type":"string"}},"required":["probe"]}`,
		"anyOf":             `{"type":"object","anyOf":[{"required":["probe"]}],"properties":{"probe":{"type":"string"}},"required":["probe"]}`,
		"top-level-$defs":   `{"type":"object","$defs":{"p":{"type":"string"}},"properties":{"probe":{"$ref":"#/$defs/p"}},"required":["probe"]}`,
	} {
		t.Run(name, func(t *testing.T) {
			tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "record_probe", Parameters: json.RawMessage(parameters)}}}
			p := nativeStructuredTestPlanner(t)
			// An immediate stop keeps the unmasked decode on the text fallback;
			// a mask would hold the stop until a complete envelope.
			comp, err := p.Complete(context.Background(), messages, tools, WithToolChoice(choice), WithMaxTokens(16), WithLogitBias(map[int]float64{0: 200}))
			if err != nil {
				t.Fatal(err)
			}
			want := ToolCall{ID: "call_forced_0", Type: "function", Function: Func{Name: "record_probe", Arguments: `{"probe":"alpha"}`}}
			if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0] != want || comp.FinishReason != "tool_calls" {
				t.Fatalf("unsupported schema left today's forced-tool path: %+v", comp)
			}
		})
	}
	t.Run("Write keeps the artifact path", func(t *testing.T) {
		write := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Write", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string"}},"required":["file_path","content","mode"],"additionalProperties":false}`)}}}
		p := nativeStructuredTestPlanner(t)
		comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "Create index.html."}}, write,
			WithToolChoice(json.RawMessage(`{"type":"function","function":{"name":"Write"}}`)), WithMaxTokens(16), WithLogitBias(map[int]float64{0: 200}))
		if err != nil {
			t.Fatal(err)
		}
		if len(comp.Message.ToolCalls) != 0 || !comp.ToolCallsDropped {
			t.Fatalf("forced Write left the artifact path: %+v", comp)
		}
	})
	t.Run("reasoning open is not one token", func(t *testing.T) {
		var doc map[string]any
		if err := json.Unmarshal([]byte(buildByteVocab()), &doc); err != nil {
			t.Fatal(err)
		}
		vocab := doc["model"].(map[string]any)["vocab"].(map[string]any)
		vocab[thinkClose] = len(vocab)
		doc["added_tokens"] = append(doc["added_tokens"].([]any), map[string]any{"id": vocab[thinkClose], "content": thinkClose, "special": true})
		data, err := json.Marshal(doc)
		if err != nil {
			t.Fatal(err)
		}
		tok, err := tokenizer.ParseJSON(data)
		if err != nil {
			t.Fatal(err)
		}
		cfg := tinyConcurrencyConfig()
		cfg.VocabSize, cfg.EOSTokenID = tok.Vocab(), 0
		m := model.NewSynthetic(cfg)
		m.Quantize()
		p := NewInKernelPlanner(m, tok, "synthetic-forced-tool-reasoning", false, nil, false)
		p.batchDecode = false
		comp, err := p.Complete(context.Background(), messages, inKernelForcedProbeTool, WithToolChoice(choice), WithThinkingBudget(2), WithMaxTokens(16), WithLogitBias(map[int]float64{0: 200}))
		if err != nil {
			t.Fatal(err)
		}
		if want := `{"probe":"alpha"}`; len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].ID != "call_forced_0" || comp.Message.ToolCalls[0].Function.Arguments != want {
			t.Fatalf("unarmable reasoning left today's forced-tool path: %+v", comp)
		}
	})
	t.Run("native receipt", func(t *testing.T) {
		p := nativeStructuredTestPlanner(t)
		_, err := p.Complete(context.Background(), messages, inKernelForcedProbeTool, WithToolChoice(choice), WithMaxTokens(16), WithNativeInferenceReceipt(true))
		if errors.Is(err, ErrNativeStructuredUnsupported) {
			t.Fatalf("forced tool with a receipt was newly refused: %v", err)
		}
	})
	t.Run("response_format still refused", func(t *testing.T) {
		p := nativeStructuredTestPlanner(t)
		_, err := p.Complete(context.Background(), messages, inKernelForcedProbeTool, WithToolChoice(choice), WithResponseFormat(nativeStructuredTestFormat))
		if !errors.Is(err, ErrNativeStructuredUnsupported) {
			t.Fatalf("response_format with a forced tool: %v", err)
		}
	})
}
