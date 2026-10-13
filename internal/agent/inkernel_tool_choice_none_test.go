package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:justify why=regression when=changed:internal/agent/**
// fak-test:runtime fast est=100ms lane=default
func TestInKernelRenderPromptToolChoiceNone(t *testing.T) {
	messages := []Message{
		{Role: RoleSystem, Content: "Answer briefly."},
		{Role: RoleUser, Content: "Read the previous result."},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "read-old", Type: "function", Function: Func{Name: "Read", Arguments: `{"path":"historical.txt"}`}}}},
		{Role: RoleTool, Name: "Read", ToolCallID: "read-old", Content: "historical-result-marker"},
		{Role: RoleUser, Content: "Explain that result."},
	}
	tools := []ToolDef{
		{Type: "function", Function: ToolDefFunction{Name: "Read", Description: "read-schema-only-marker", Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`)}},
		{Type: "function", Function: ToolDefFunction{Name: "Search", Description: "search-schema-only-marker", Parameters: json.RawMessage(`{"type":"object"}`)}},
	}
	responseFormat := json.RawMessage(`{"type":"json_object"}`)
	snapshot := func(choice json.RawMessage) []byte {
		b, err := json.Marshal([]any{messages, tools, responseFormat, choice})
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	for _, family := range []struct {
		name string
		cfg  model.Config
	}{
		{name: "chatml"},
		{name: "ornith", cfg: model.Config{ModelType: "qwen3_5", LayerTypes: []string{"linear_attention", "full_attention"}}},
		{name: "deepseek41", cfg: model.Config{ModelType: "deepseek41"}},
	} {
		t.Run(family.name, func(t *testing.T) {
			planner := NewInKernelPlannerWithConfig(&model.Model{Cfg: family.cfg}, loadProbeTok(t), "tool-choice-prompt", false, nil, false, InKernelPlannerConfig{ContextTokens: 8192})
			for _, choice := range []struct {
				name string
				raw  json.RawMessage
			}{
				{name: "absent"},
				{name: "auto", raw: json.RawMessage(`"auto"`)},
				{name: "required", raw: json.RawMessage(`"required"`)},
				{name: "named", raw: json.RawMessage(`{"type":"function","function":{"name":"Read"}}`)},
				{name: "none", raw: json.RawMessage(`"none"`)},
				{name: "none-whitespace", raw: json.RawMessage(" \n\"none\"\t")},
				{name: "none-escaped", raw: json.RawMessage(`"n\u006fne"`)},
			} {
				t.Run(choice.name, func(t *testing.T) {
					before := snapshot(choice.raw)
					sp := SampleParams{ResponseFormat: responseFormat, ToolChoice: choice.raw}
					got := inKernelRenderPrompt(messages, tools, family.cfg, sp)
					if !strings.Contains(got, "historical.txt") || !strings.Contains(got, "historical-result-marker") {
						t.Fatal("historical tool call or result disappeared")
					}
					if strings.HasPrefix(choice.name, "none") {
						want := inKernelRenderPrompt(messages, nil, family.cfg, sp)
						if got != want {
							t.Fatal("tool_choice none advertised schemas: rendered prompt differs from tool-free request")
						}
						withTools, err := planner.EncodePrompt(context.Background(), messages, tools, WithMaxTokens(1), WithResponseFormat(responseFormat), WithToolChoice(choice.raw))
						if err != nil {
							t.Fatal(err)
						}
						withoutTools, err := planner.EncodePrompt(context.Background(), messages, nil, WithMaxTokens(1), WithResponseFormat(responseFormat), WithToolChoice(choice.raw))
						if err != nil {
							t.Fatal(err)
						}
						if withTools.RenderedSHA256 != withoutTools.RenderedSHA256 || !reflect.DeepEqual(withTools.TokenIDs, withoutTools.TokenIDs) {
							t.Fatal("native prompt encoding differs from tool-free request")
						}
					} else if choice.name == "named" {
						if !strings.Contains(got, `"const":"Read"`) || strings.Contains(got, "search-schema-only-marker") {
							t.Fatal("named choice lost its forced tool constraint or advertised unrelated schemas")
						}
					} else if !strings.Contains(got, "read-schema-only-marker") || !strings.Contains(got, "search-schema-only-marker") {
						t.Fatal("ordinary or required choice lost its advertised schemas")
					}
					if !bytes.Equal(before, snapshot(choice.raw)) {
						t.Fatal("prompt preparation mutated caller-owned request data")
					}
				})
			}
		})
	}
}

// fak-test:justify why=regression when=changed:internal/agent/**
// fak-test:runtime medium est=2s lane=default
func TestInKernelRenderPromptToolChoiceNoneComplete(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	planner := NewInKernelPlanner(m, loadProbeTok(t), "tool-choice-complete", false, nil, false)
	messages := []Message{{Role: RoleUser, Content: "Say hi."}}
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Read", Description: "read-schema-only-marker", Parameters: json.RawMessage(`{"type":"object"}`)}}}
	opts := []SampleOpt{WithMaxTokens(1), WithToolChoice(json.RawMessage(`"none"`)), WithNativeInferenceReceipt(true)}
	var baseline *Completion
	for _, advertised := range [][]ToolDef{nil, tools} {
		got, err := planner.Complete(context.Background(), messages, advertised, opts...)
		if err != nil {
			t.Fatal(err)
		}
		if got.NativeInference == nil {
			t.Fatal("ordinary native completion omitted requested prompt receipt")
		}
		if baseline != nil && (got.NativeInference.RenderedSHA256 != baseline.NativeInference.RenderedSHA256 || !reflect.DeepEqual(got.NativeInference.PromptTokenIDs, baseline.NativeInference.PromptTokenIDs) || got.Usage.PromptTokens != baseline.Usage.PromptTokens) {
			t.Fatal("ordinary native completion still prefills advertised tools with tool_choice none")
		}
		baseline = got
	}
	content := `<tool_call>{"name":"Read","arguments":{}}</tool_call>`
	parsed := inKernelDecodeToCompletion(t, content, "stop", tools, WithToolChoice(json.RawMessage(`"none"`)))
	if len(parsed.Message.ToolCalls) != 1 || parsed.Message.ToolCalls[0].Function.Name != "Read" {
		t.Fatal("prompt schema suppression changed the independent offered-tool parsing contract")
	}
}
