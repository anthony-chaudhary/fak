package agent

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func inkling2TestTools() []ToolDef {
	return []ToolDef{
		{Type: "function", Function: ToolDefFunction{
			Name:        "Bash",
			Description: "Run a shell command",
			Parameters:  json.RawMessage(`{"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}`),
		}},
		{Function: ToolDefFunction{Name: "Read"}},
	}
}

// The expected bytes are hand-derived from the Inkling-2 GGUF chat template.
func TestInkling2RenderGoldenToolTurn(t *testing.T) {
	messages := []Message{
		{Role: RoleSystem, Content: "You are terse."},
		{Role: RoleUser, Content: "List files."},
		{Role: RoleAssistant, ReasoningContent: "Need ls.", ToolCalls: []ToolCall{{
			ID: "c1", Type: "function",
			Function: Func{Name: "Bash", Arguments: `{"timeout": 5, "command": "ls -la", "opts": {"z": 1, "a": [1, 2]}}`},
		}}},
		{Role: RoleTool, ToolCallID: "c1", Content: "a.go\nb.go"},
		{Role: RoleAssistant, Content: "Two files."},
		{Role: RoleUser, Content: "Thanks"},
	}
	want := `<|message_system|>tool_declare<|content_xml|>` +
		`[{"description":"Run a shell command","name":"Bash","parameters":{"properties":{"command":{"type":"string"}},"required":["command"],"type":"object"},"type":"function"},` +
		`{"description":"","name":"Read","parameters":{},"type":"function"}]<|end_message|>` +
		`<|message_system|><|content_text|>You are terse.<|end_message|>` +
		`<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>` +
		`<|message_user|><|content_text|>List files.<|end_message|>` +
		`<|message_model|><|content_thinking|>Need ls.<|end_message|>` +
		`<|message_model|>Bash<|content_invoke_tool_json|>{"name":"Bash","args":{"command":"ls -la","opts":{"z":1,"a":[1,2]},"timeout":5}}<|end_message|>` +
		`<|content_model_end_sampling|>` +
		"<|message_tool|>Bash<|content_text|>a.go\nb.go<|end_message|>" +
		`<|message_model|><|content_text|>Two files.<|end_message|><|content_model_end_sampling|>` +
		`<|message_user|><|content_text|>Thanks<|end_message|>`
	got := renderInkling2Chat(messages, inkling2TestTools(), SampleParams{})
	if got != want {
		t.Fatalf("render mismatch\n got: %q\nwant: %q", got, want)
	}
}

func TestInkling2RenderDispatchesOnConfigPredicate(t *testing.T) {
	messages := []Message{{Role: RoleUser, Content: "hi"}}
	cfg := model.Config{ModelType: "inkling2_loop"}
	if !cfg.IsInkling2Loop() {
		t.Fatal("fixture config is not Inkling-2")
	}
	got := inKernelRenderPrompt(messages, inkling2TestTools(), cfg, SampleParams{})
	if !strings.HasPrefix(got, "<|message_system|>tool_declare<|content_xml|>[") {
		t.Fatalf("Inkling-2 config did not render the Inkling-2 template: %q", got)
	}
	none := inKernelRenderPrompt(messages, inkling2TestTools(), cfg, SampleParams{ToolChoice: json.RawMessage(`"none"`)})
	want := `<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>hi<|end_message|>`
	if none != want {
		t.Fatalf("tool_choice=none render = %q, want %q", none, want)
	}
	if chatml := inKernelRenderPrompt(messages, nil, model.Config{ModelType: "qwen2"}, SampleParams{}); strings.Contains(chatml, "<|message_") {
		t.Fatalf("non-Inkling config rendered Inkling markers: %q", chatml)
	}
}

func TestInkling2RenderEffortPreamble(t *testing.T) {
	zero, two := 0, 2048
	for _, tc := range []struct {
		sp   SampleParams
		want string
	}{
		{SampleParams{}, "0"},
		{SampleParams{ReasoningEffort: "none"}, "0"},
		{SampleParams{ReasoningEffort: "minimal"}, "0.1"},
		{SampleParams{ReasoningEffort: "low"}, "0.2"},
		{SampleParams{ReasoningEffort: "Medium"}, "0.7"},
		{SampleParams{ReasoningEffort: "high"}, "0.9"},
		{SampleParams{ReasoningEffort: "xhigh"}, "0.99"},
		{SampleParams{ReasoningEffort: "max"}, "0.99"},
		{SampleParams{ReasoningEffort: "0.5"}, "0.5"},
		{SampleParams{ReasoningEffort: "1.5"}, "0"},
		{SampleParams{ReasoningEffort: "bogus"}, "0"},
		{SampleParams{ThinkingBudget: &zero}, "0"},
		{SampleParams{ThinkingBudget: &two}, "0.9"},
	} {
		if got := inkling2Effort(tc.sp); got != tc.want {
			t.Errorf("effort(%+v) = %q, want %q", tc.sp.ReasoningEffort, got, tc.want)
		}
	}
	// With only system messages the preamble trails them.
	got := renderInkling2Chat([]Message{{Role: RoleSystem, Content: "s"}}, nil, SampleParams{ReasoningEffort: "high"})
	want := `<|message_system|><|content_text|>s<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>`
	if got != want {
		t.Fatalf("system-only render = %q, want %q", got, want)
	}
}

func TestInkling2RenderToolNamesAndEscaping(t *testing.T) {
	tools := []ToolDef{{Function: ToolDefFunction{Name: "Note", Description: "a<b & \"ü\"\n", Parameters: json.RawMessage(`{}`)}}}
	messages := []Message{
		{Role: RoleUser, Content: ""},
		{Role: RoleAssistant, Content: "ok", ToolCalls: []ToolCall{{ID: "x", Function: Func{Name: "Note", Arguments: `{"text":"<é>\t"}`}}}},
		{Role: RoleTool, Name: "Note", Content: "done"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "y", Function: Func{Name: "Note"}}}},
		{Role: RoleTool, ToolCallID: "missing", Content: "?"},
	}
	want := `<|message_system|>tool_declare<|content_xml|>[{"description":"a<b & \"ü\"\n","name":"Note","parameters":{},"type":"function"}]<|end_message|>` +
		`<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>` +
		`<|message_user|><|content_text|><|end_message|>` +
		`<|message_model|><|content_text|>ok<|end_message|>` +
		`<|message_model|>Note<|content_invoke_tool_json|>{"name":"Note","args":{"text":"<é>\t"}}<|end_message|><|content_model_end_sampling|>` +
		`<|message_tool|>Note<|content_text|>done<|end_message|>` +
		`<|message_model|>Note<|content_invoke_tool_json|>{"name":"Note","args":{}}<|end_message|><|content_model_end_sampling|>` +
		`<|message_tool|><|content_text|>?<|end_message|>`
	if got := renderInkling2Chat(messages, tools, SampleParams{}); got != want {
		t.Fatalf("render mismatch\n got: %q\nwant: %q", got, want)
	}
}

func TestInkling2RenderedCallRoundTripsThroughDialect(t *testing.T) {
	var b strings.Builder
	inkling2WriteToolCall(&b, Func{Name: "Bash", Arguments: `{"command":"go test ./...","timeout":60}`})
	blocks := extractInkling2ToolBlocks(b.String())
	if len(blocks) != 1 || blocks[0].start != 0 || blocks[0].end != b.Len() {
		t.Fatalf("blocks = %+v for %q", blocks, b.String())
	}
	if fn := blocks[0].call.Function; fn.Name != "Bash" || fn.Arguments != `{"command":"go test ./...","timeout":60}` {
		t.Fatalf("round trip = %+v", fn)
	}
}

func TestInkling2SplitOutput(t *testing.T) {
	raw := `<|message_model|><|content_thinking|>Check the tree first.<|end_message|>` +
		`<|message_model|><|content_text|>Listing.<|end_message|>` +
		`<|message_model|>Bash<|content_invoke_tool_json|>{"name":"Bash","args":{"command":"ls"}}<|end_message|>` +
		`<|message_model|>Read<|content_invoke_tool_json|>{"name":"Read","args":{"path":"go.mod"}}<|end_message|>` +
		Inkling2EndOfTurnMarker
	reasoning, content := SplitInkling2Output(raw)
	if reasoning != "Check the tree first." {
		t.Fatalf("reasoning = %q", reasoning)
	}
	m, rejected, dialect := liftTextToolCallsWithDialect(Message{Role: RoleAssistant, Content: content}, NewOfferedTools(inkling2TestTools()))
	if rejected || dialect != "inkling2_invoke_tool_json" {
		t.Fatalf("rejected=%v dialect=%q content=%q", rejected, dialect, content)
	}
	if m.Content != "Listing." || len(m.ToolCalls) != 2 {
		t.Fatalf("lifted = %+v", m)
	}
	if m.ToolCalls[0].Function.Name != "Bash" || m.ToolCalls[0].Function.Arguments != `{"command":"ls"}` ||
		m.ToolCalls[1].Function.Name != "Read" || m.ToolCalls[1].ID != "call_text_1" {
		t.Fatalf("calls = %+v", m.ToolCalls)
	}

	if r, c := SplitInkling2Output("<|message_model|><|content_thinking|>still going"); r != "still going" || c != "" {
		t.Fatalf("truncated thinking split = %q / %q", r, c)
	}
	if r, c := SplitInkling2Output("plain answer"); r != "" || c != "plain answer" {
		t.Fatalf("marker-free split = %q / %q", r, c)
	}
	if r, c := SplitInkling2Output("<|message_model|><|content_text|>Hi there.<|end_message|>" + Inkling2EndOfTurnMarker); r != "" || c != "Hi there." {
		t.Fatalf("text-only split = %q / %q", r, c)
	}
}

func TestInkling2DialectOnlyMatchesOwnMarkers(t *testing.T) {
	offered := NewOfferedTools(inkling2TestTools())
	hermes := `<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>`
	if blocks := extractInkling2ToolBlocks(hermes); blocks != nil {
		t.Fatalf("inkling2 dialect claimed a hermes call: %+v", blocks)
	}
	if _, _, dialect := liftTextToolCallsWithDialect(Message{Content: hermes}, offered); dialect != "hermes" {
		t.Fatalf("hermes call lifted by %q", dialect)
	}

	for name, content := range map[string]string{
		"header mismatch": `<|message_model|>Read<|content_invoke_tool_json|>{"name":"Bash","args":{}}<|end_message|>`,
		"truncated json":  `<|message_model|>Bash<|content_invoke_tool_json|>{"name":"Bash","args":{"command":"l`,
	} {
		if blocks := extractInkling2ToolBlocks(content); len(blocks) != 0 {
			t.Errorf("%s: lifted %+v", name, blocks)
		}
	}

	unoffered := `<|message_model|>Delete<|content_invoke_tool_json|>{"name":"Delete","args":{}}<|end_message|>`
	m, rejected, _ := liftTextToolCallsWithDialect(Message{Content: unoffered}, offered)
	if !rejected || len(m.ToolCalls) != 0 || m.Content != unoffered {
		t.Fatalf("unoffered call: rejected=%v %+v", rejected, m)
	}

	headerless, rejected, _ := liftTextToolCallsWithDialect(Message{Content: `<|content_invoke_tool_json|>{"name":"Bash","args":"{\"command\":\"pwd\"}"}`}, offered)
	if rejected || len(headerless.ToolCalls) != 1 || headerless.ToolCalls[0].Function.Arguments != `{"command":"pwd"}` || headerless.Content != "" {
		t.Fatalf("headerless stringified-args call = %+v", headerless)
	}
}

// Calls sit behind special tokens, so prose words or example-looking args in
// earlier blocks must not drop a real call.
func TestInkling2CallsSurviveExampleLikeProseAndArgs(t *testing.T) {
	offered := NewOfferedTools(inkling2TestTools())
	raw := `<|message_model|><|content_text|>Let me check the mock server, e.g. the sample call for instance.<|end_message|>` +
		`<|message_model|>Read<|content_invoke_tool_json|>{"name":"Read","args":{"path":"a.go"}}<|end_message|>` +
		Inkling2EndOfTurnMarker
	_, content := SplitInkling2Output(raw)
	if blocks := extractInkling2ToolBlocks(content); len(blocks) != 1 || blocks[0].call.Function.Name != "Read" {
		t.Fatalf("call after mock/example prose: blocks = %+v for %q", blocks, content)
	}
	m, rejected, dialect := liftTextToolCallsWithDialect(Message{Role: RoleAssistant, Content: content}, offered)
	if rejected || dialect != "inkling2_invoke_tool_json" || len(m.ToolCalls) != 1 {
		t.Fatalf("lift after prose: rejected=%v dialect=%q %+v", rejected, dialect, m)
	}

	parallel := `<|message_model|>Read<|content_invoke_tool_json|>{"name":"Read","args":{"path":"examples/a.go"}}<|end_message|>` +
		`<|message_model|>Bash<|content_invoke_tool_json|>{"name":"Bash","args":{"command":"ls examples"}}<|end_message|>` +
		Inkling2EndOfTurnMarker
	_, content = SplitInkling2Output(parallel)
	blocks := extractInkling2ToolBlocks(content)
	if len(blocks) != 2 || blocks[0].call.Function.Name != "Read" || blocks[1].call.Function.Name != "Bash" {
		t.Fatalf("parallel calls with examples/ args: blocks = %+v for %q", blocks, content)
	}
	m, rejected, _ = liftTextToolCallsWithDialect(Message{Role: RoleAssistant, Content: content}, offered)
	if rejected || len(m.ToolCalls) != 2 || m.Content != "" {
		t.Fatalf("parallel lift: rejected=%v %+v", rejected, m)
	}
}

const inkling2StopFixture = `{
  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": ["a b"]},
  "decoder": {"type": "ByteLevel"},
  "added_tokens": [
    {"id": 200, "content": "<|end_message|>", "special": true},
    {"id": 201, "content": "<|content_model_end_sampling|>", "special": true},
    {"id": 202, "content": "<|message_model|>", "special": true}
  ]
}`

func TestInkling2StopIDsEndOfTurnNotEndMessage(t *testing.T) {
	tok, err := tokenizer.ParseJSON([]byte(inkling2StopFixture))
	if err != nil {
		t.Fatalf("parse fixture tokenizer: %v", err)
	}
	stops := StopIDs(tok, model.Config{ModelType: "inkling2_loop", EOSTokenID: 200})
	if !stops[201] || stops[200] || stops[202] || len(stops) != 1 {
		t.Fatalf("Inkling-2 stops = %v, want only the end-of-turn marker", stops)
	}
	if other := StopIDs(tok, model.Config{EOSTokenID: 200}); !other[200] || other[201] {
		t.Fatalf("non-Inkling stops = %v, want config EOS only", other)
	}
}
