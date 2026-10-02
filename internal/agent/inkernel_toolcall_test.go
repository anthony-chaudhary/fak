package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// These tests exercise Complete's real post-decode tool-call pipeline with a tiny
// synthetic CPU model. The forced literal token isolates completion handling from
// model quality; this is not a weighted-model or hardware qualification witness.
func inKernelNormalize(t *testing.T, content, finishReason string) *Completion {
	t.Helper()
	return inKernelDecodeToCompletion(t, content, finishReason, nil)
}

// Drive the existing Complete API so the same test file also builds on the
// pre-classifier source. No post-decode guard is reproduced in test code.
func inKernelDecodeToCompletion(t *testing.T, raw, finishReason string, tools []ToolDef, opts ...SampleOpt) *Completion {
	t.Helper()
	const outputID = 259 // first slot after buildByteVocab's byte and ChatML tokens
	const stop = "<|toolcall_test_stop|>"
	piece := raw
	switch finishReason {
	case "length":
	case "stop":
		piece += stop
		opts = append(opts, WithStop([]string{stop}))
	default:
		t.Fatalf("unsupported test finish reason %q", finishReason)
	}
	var doc map[string]any
	if err := json.Unmarshal([]byte(buildByteVocab()), &doc); err != nil {
		t.Fatal(err)
	}
	doc["added_tokens"] = append(doc["added_tokens"].([]any), map[string]any{
		"id": outputID, "content": piece, "special": true,
	})
	encoded, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.ParseJSON(encoded)
	if err != nil {
		t.Fatal(err)
	}
	p := NewInKernelPlanner(model.NewSynthetic(tinyConcurrencyConfig()), tok, "toolcall-seam-test", false, nil, false)
	p.quant = false
	var pieces []string
	var decoded string
	opts = append(opts, WithMaxTokens(1), WithLogitBias(map[int]float64{outputID: 100}),
		WithDecodeTokenObserver(func(tokenPiece, rawText string) {
			if tokenPiece != "" {
				pieces = append(pieces, tokenPiece)
			} else {
				decoded = rawText
			}
		}))
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "probe"}}, tools, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if len(pieces) != 1 || pieces[0] != piece || decoded != raw || comp.Usage.CompletionTokens != 1 {
		t.Fatalf("decode witness: pieces=%q raw=%q tokens=%d, want one %q token and raw %q", pieces, decoded, comp.Usage.CompletionTokens, piece, raw)
	}
	wantFinish := finishReason
	if len(comp.Message.ToolCalls) > 0 {
		wantFinish = "tool_calls"
	}
	if comp.FinishReason != wantFinish {
		t.Fatalf("finish reason=%q, want %q", comp.FinishReason, wantFinish)
	}
	return comp
}

// TestCompleteLiftsTextToolCall: a well-formed Hermes <tool_call> (Qwen2.5's native
// dialect) decoded by the in-kernel forward is lifted to a structured ToolCall and the
// finish reason becomes tool_calls — the signal the gateway adjudicator + Anthropic wire
// read to emit a tool_use block.
func TestCompleteLiftsTextToolCall(t *testing.T) {
	comp := inKernelNormalize(t, `<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>`, "stop")
	if len(comp.Message.ToolCalls) != 1 {
		t.Fatalf("want 1 lifted tool call, got %d (content=%q)", len(comp.Message.ToolCalls), comp.Message.Content)
	}
	if comp.Message.ToolCalls[0].Function.Name != "Bash" {
		t.Fatalf("lifted call name = %q, want Bash", comp.Message.ToolCalls[0].Function.Name)
	}
	if comp.FinishReason != "tool_calls" {
		t.Fatalf("finish reason = %q, want tool_calls", comp.FinishReason)
	}
	if comp.ToolCallsDropped {
		t.Fatalf("a well-formed call must not be flagged dropped")
	}
}

// TestCompleteLiftsMultipleToolCalls: two <tool_call> blocks in one turn both lift.
func TestCompleteLiftsMultipleToolCalls(t *testing.T) {
	comp := inKernelNormalize(t,
		`<tool_call>{"name": "Read", "arguments": {"path": "a"}}</tool_call>`+
			"\n"+`<tool_call>{"name": "Read", "arguments": {"path": "b"}}</tool_call>`, "stop")
	if len(comp.Message.ToolCalls) != 2 {
		t.Fatalf("want 2 lifted tool calls, got %d", len(comp.Message.ToolCalls))
	}
}

// TestCompleteMalformedToolCallFailsClosed: a TRUNCATED/unclosed <tool_call> the lift
// cannot recover sets ToolCallsDropped so the conformance gate refuses the turn rather
// than leaking a half-formed call into Claude Code's context. The content is preserved.
func TestCompleteMalformedToolCallFailsClosed(t *testing.T) {
	comp := inKernelNormalize(t, `sure, calling it: <tool_call>{"name": "Bash", "argum`, "length")
	if len(comp.Message.ToolCalls) != 0 {
		t.Fatalf("a truncated call must lift 0 structured calls, got %d", len(comp.Message.ToolCalls))
	}
	if !comp.ToolCallsDropped {
		t.Fatalf("a truncated <tool_call> must set ToolCallsDropped (fail closed)")
	}
	if !strings.Contains(comp.Message.Content, "<tool_call>") {
		t.Fatalf("content must be preserved for the operator to see the truncation")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCompleteDroppedToolCallReason(t *testing.T) {
	for _, tc := range []struct {
		name, content, finishReason string
		want                        abi.ReasonCode
	}{
		{"unclosed at length", `<tool_call>{"name":"Bash","argum`, "length", abi.ReasonOversize},
		{"unclosed at stop", `<tool_call>{"name":"Bash","argum`, "stop", abi.ReasonMalformed},
		{"closed malformed at stop", `<tool_call>{"name":}</tool_call>`, "stop", abi.ReasonMalformed},
		{"closed malformed at length", `<tool_call>{"name":}</tool_call>`, "length", abi.ReasonMalformed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			comp := inKernelNormalize(t, tc.content, tc.finishReason)
			if !comp.ToolCallsDropped || len(comp.Message.ToolCalls) != 0 || comp.ToolCallsDroppedReason != tc.want {
				t.Fatalf("dropped=%v calls=%d reason=%v, want dropped with no calls and %v", comp.ToolCallsDropped, len(comp.Message.ToolCalls), comp.ToolCallsDroppedReason, tc.want)
			}
			if comp.Message.Content != tc.content || comp.FinishReason != tc.finishReason {
				t.Fatalf("drop classification changed content or finish reason: %+v", comp)
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestInKernelDroppedToolCallPreservesForcedReason(t *testing.T) {
	choice := json.RawMessage(`{"type":"function","function":{"name":"record_probe"}}`)
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "record_probe"}}}
	for _, finishReason := range []string{"stop", "length"} {
		t.Run(finishReason, func(t *testing.T) {
			content := `<tool_call>{"name":}</tool_call>`
			forced := enforceForcedToolChoice(&Completion{
				Message: Message{Role: RoleAssistant, Content: content}, FinishReason: finishReason,
			}, choice, tools, nil)
			want := forced.ToolCallsDroppedReason
			if !forced.ToolCallsDropped || want == abi.ReasonNone {
				t.Fatal("forced-tool enforcement must establish a drop reason")
			}
			comp := inKernelDecodeToCompletion(t, content, finishReason, tools, WithToolChoice(choice))
			if !comp.ToolCallsDropped || comp.ToolCallsDroppedReason != want || comp.Message.Content != content {
				t.Fatalf("drop guard changed the forced-tool classification or content: %+v", comp)
			}
		})
	}
}

// TestCompletePlainChatUnaffected: a turn with no tool call is unchanged — plain chat
// still works on the in-kernel path.
func TestCompletePlainChatUnaffected(t *testing.T) {
	comp := inKernelNormalize(t, "2 + 2 is 4.", "stop")
	if len(comp.Message.ToolCalls) != 0 {
		t.Fatalf("plain chat must lift 0 tool calls, got %d", len(comp.Message.ToolCalls))
	}
	if comp.ToolCallsDropped {
		t.Fatalf("plain chat must not be flagged dropped")
	}
	if comp.ToolCallsDroppedReason != abi.ReasonNone {
		t.Fatalf("plain chat reason = %v, want NONE", comp.ToolCallsDroppedReason)
	}
	if comp.FinishReason != "stop" {
		t.Fatalf("plain chat finish reason changed to %q", comp.FinishReason)
	}
	if comp.Message.Content != "2 + 2 is 4." {
		t.Fatalf("plain chat content changed: %q", comp.Message.Content)
	}
}

// TestReasoningThenToolCallOrdering: a reasoning model (Ornith / Qwen3.5) that opens the
// turn with a <think> block and THEN emits a tool call must have the reasoning stripped to
// ReasoningContent and the tool call lifted from the post-</think> content — the
// reasoning-before-call ordering issue #1059's acceptance names. The think text must NOT
// leak into Content (and thus into Claude Code's context), and the lift must still see the
// tool call that follows it. This is the composition splitReasoning → lift, in Complete's
// order.
func TestReasoningThenToolCallOrdering(t *testing.T) {
	raw := "<think>The user wants the directory listing. I'll run ls.</think>" +
		`<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>`
	comp := inKernelDecodeToCompletion(t, raw, "stop", nil)

	// The reasoning is split off into ReasoningContent, not left in Content.
	if !strings.Contains(comp.Message.ReasoningContent, "directory listing") {
		t.Fatalf("reasoning not captured in ReasoningContent: %q", comp.Message.ReasoningContent)
	}
	if strings.Contains(comp.Message.Content, "<think>") || strings.Contains(comp.Message.Content, "directory listing") {
		t.Fatalf("reasoning leaked into Content: %q", comp.Message.Content)
	}
	// The tool call that FOLLOWS the reasoning still lifts.
	if len(comp.Message.ToolCalls) != 1 {
		t.Fatalf("want 1 tool call lifted after the reasoning block, got %d (content=%q)",
			len(comp.Message.ToolCalls), comp.Message.Content)
	}
	if comp.Message.ToolCalls[0].Function.Name != "Bash" {
		t.Fatalf("lifted call name = %q, want Bash", comp.Message.ToolCalls[0].Function.Name)
	}
	if comp.FinishReason != "tool_calls" {
		t.Fatalf("finish reason = %q, want tool_calls", comp.FinishReason)
	}
	if comp.ToolCallsDropped {
		t.Fatalf("a well-formed reasoning+call turn must not be flagged dropped")
	}
}

// TestReasoningThenMultipleToolCalls: reasoning followed by MORE THAN ONE tool call — the
// multi-call + reasoning-before-call combination — lifts every call and still strips the
// reasoning. Guards the composition against a regression that only handles a single call
// after a think block.
func TestReasoningThenMultipleToolCalls(t *testing.T) {
	raw := "<think>Read both files to compare them.</think>" +
		`<tool_call>{"name": "Read", "arguments": {"path": "a"}}</tool_call>` + "\n" +
		`<tool_call>{"name": "Read", "arguments": {"path": "b"}}</tool_call>`
	comp := inKernelDecodeToCompletion(t, raw, "stop", nil)
	if !strings.Contains(comp.Message.ReasoningContent, "compare them") {
		t.Fatalf("reasoning not captured: %q", comp.Message.ReasoningContent)
	}
	if len(comp.Message.ToolCalls) != 2 {
		t.Fatalf("want 2 tool calls after the reasoning block, got %d", len(comp.Message.ToolCalls))
	}
}

// TestReasoningWithTruncatedToolCallFailsClosed: a <think> block followed by a TRUNCATED
// tool call must still fail closed (ToolCallsDropped) — the reasoning split must not mask
// the unclosed-call refuse path. The reasoning is captured; the dropped flag fires on the
// post-reasoning content.
func TestReasoningWithTruncatedToolCallFailsClosed(t *testing.T) {
	raw := "<think>I'll call Bash now.</think>" +
		`calling it: <tool_call>{"name": "Bash", "argum`
	comp := inKernelDecodeToCompletion(t, raw, "length", nil)
	if !strings.Contains(comp.Message.ReasoningContent, "call Bash") {
		t.Fatalf("reasoning not captured ahead of the truncated call: %q", comp.Message.ReasoningContent)
	}
	if len(comp.Message.ToolCalls) != 0 {
		t.Fatalf("a truncated call after reasoning must lift 0 calls, got %d", len(comp.Message.ToolCalls))
	}
	if !comp.ToolCallsDropped {
		t.Fatalf("a truncated <tool_call> after a reasoning block must still set ToolCallsDropped")
	}
	if comp.ToolCallsDroppedReason != abi.ReasonOversize {
		t.Fatalf("truncated call after reasoning has reason %v, want OVERSIZE", comp.ToolCallsDroppedReason)
	}
}

// TestToolSpecBlockMatchesQwen25TemplateContract pins toolSpecBlock to the tools branch
// of Qwen/Qwen2.5-Coder-7B-Instruct's chat_template (issue #10600): the usage preamble,
// the tojson-spaced signature, the JSON-object <tool_call> instruction, and the flush
// </tool_call> ending the system turn. Qwen3's template repeats this branch verbatim.
// The antl (<function=…>) instruction this block used to teach is the Ornith contract —
// it belongs to ornithToolSpecPrefix/Suffix only, and teaching it here drove Qwen2.5-Coder
// away from its trained dialect so native tool_calls never engaged.
func TestToolSpecBlockMatchesQwen25TemplateContract(t *testing.T) {
	got := toolSpecBlock([]ToolDef{{Type: "function", Function: ToolDefFunction{Name: "record_probe", Description: "record", Parameters: json.RawMessage(`{"type":"object","properties":{"probe":{"type":"string"}},"required":["probe"]}`)}}})
	for _, want := range []string{
		"# Tools\n\nYou may call one or more functions to assist with the user query.",
		"You are provided with function signatures within <tools></tools> XML tags:",
		`<tools>
{"type": "function", "function": {"name": "record_probe", "description": "record", "parameters": {"type": "object", "properties": {"probe": {"type": "string"}}, "required": ["probe"]}}}`,
		"For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
		"<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("tool preamble missing %q:\n%s", want, got)
		}
	}
	for _, banned := range []string{
		"If you choose to call a function ONLY reply in the following format",
		"<IMPORTANT>",
		`{"type":"function"`, // the compact json.Marshal form, not the template's tojson spacing
	} {
		if strings.Contains(got, banned) {
			t.Fatalf("tool preamble carries divergent %q:\n%s", banned, got)
		}
	}
	if strings.HasSuffix(got, "\n") {
		t.Fatalf("spec must end flush against <|im_end|> (template: </tool_call><|im_end|>):\n%s", got)
	}
}

func TestRenderInKernelChatMLRequestCarriesJSONSchema(t *testing.T) {
	raw := json.RawMessage(`{"type":"json_schema","json_schema":{"name":"probe","strict":true,"schema":{"type":"object","properties":{"model":{"type":"string"},"ok":{"type":"boolean"}},"required":["model","ok"],"additionalProperties":false}}}`)
	got := renderInKernelChatMLRequest([]Message{{Role: RoleUser, Content: "Return the probe."}}, nil, model.Config{}, raw, nil)
	for _, want := range []string{
		"Return only one valid JSON object matching this schema exactly.",
		`"required":["model","ok"]`,
		"Do not use Markdown fences or explanatory prose.",
		"<|im_start|>user\nReturn the probe.<|im_end|>",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered prompt missing %q:\n%s", want, got)
		}
	}
	if strings.Index(got, "JSON schema:") > strings.Index(got, "<|im_start|>user") {
		t.Fatalf("response-format instruction must precede user turn:\n%s", got)
	}
}

func TestRenderInKernelChatMLRequestIgnoresMalformedResponseFormat(t *testing.T) {
	messages := []Message{{Role: RoleUser, Content: "hello"}}
	got := renderInKernelChatMLRequest(messages, nil, model.Config{}, json.RawMessage(`{"type":`), nil)
	want := renderInKernelChatMLTools(messages, nil, model.Config{})
	if got != want {
		t.Fatalf("malformed response_format changed prompt:\n got %q\nwant %q", got, want)
	}
}

func TestLiftTextToolCallsQwenFunctionParameterDialect(t *testing.T) {
	m := LiftTextToolCalls(Message{Role: RoleAssistant, Content: `<tool_call>
<function=record_probe>
<parameter=hardware>
A100-SXM4-40GB
</parameter>
<parameter=passed>
true
</parameter>
</function>
</tool_call>`})
	if len(m.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1; content=%q", len(m.ToolCalls), m.Content)
	}
	got := m.ToolCalls[0].Function
	if got.Name != "record_probe" || got.Arguments != `{"hardware":"A100-SXM4-40GB","passed":true}` {
		t.Fatalf("tool call = %#v", got)
	}
	if strings.TrimSpace(m.Content) != "" {
		t.Fatalf("lifted tool call remained in content: %q", m.Content)
	}
}

func TestRenderInKernelChatMLRequestPinsForcedTool(t *testing.T) {
	choice := json.RawMessage(`{"type":"function","function":{"name":"record_probe"}}`)
	got := renderInKernelChatMLRequest([]Message{{Role: RoleUser, Content: "run it"}}, []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "record_probe"}}}, model.Config{}, nil, choice)
	want := `"const":"record_probe"`
	if !strings.Contains(got, want) {
		t.Fatalf("forced tool instruction missing:\n%s", got)
	}
}

func TestRenderInKernelChatMLRequestPinsForcedWriteArtifact(t *testing.T) {
	choice := json.RawMessage(`{"type":"function","function":{"name":"Write"}}`)
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Write"}}}
	got := renderInKernelChatMLRequest([]Message{{Role: RoleUser, Content: "Create index.html."}}, tools, model.Config{}, nil, choice)
	if !strings.Contains(got, "Return only the complete file contents") || strings.Contains(got, `"const":"Write"`) {
		t.Fatalf("forced Write should request artifact bytes for fail-closed wrapping:\n%s", got)
	}
}

func TestForcedToolArgumentsFromMessages(t *testing.T) {
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "record_probe", Parameters: json.RawMessage(`{"type":"object","properties":{"hardware":{"type":"string"},"passed":{"type":"boolean"}},"required":["hardware","passed"]}`)}}}
	got, ok := forcedToolArgumentsFromMessages("record_probe", tools, []Message{{Role: RoleUser, Content: "Set hardware to A100-SXM4-40GB. Set passed to the boolean true."}})
	if !ok || got != `{"hardware":"A100-SXM4-40GB","passed":true}` {
		t.Fatalf("got %q, %v", got, ok)
	}
}

func TestForcedWriteArgumentsCreateCapturedArtifact(t *testing.T) {
	root := t.TempDir()
	defer DisarmCodeTools()
	tools, err := ArmFocusedCodeTools(root)
	if err != nil {
		t.Fatalf("arm code tools: %v", err)
	}
	assistant := "```html\n<!doctype html><html><body>palette</body></html>\n```"
	args, ok := forcedToolArgumentsFromMessages("Write", tools, []Message{{Role: RoleUser, Content: "Create index.html with the complete color palette generator."}}, assistant)
	if !ok {
		t.Fatal("forced Write arguments were not grounded")
	}
	metrics, _ := runCodeToolLoop(t, root, []codeToolScript{{tool: "Write", args: args}})
	body, err := os.ReadFile(filepath.Join(root, "index.html"))
	if err != nil {
		t.Fatalf("read captured artifact: %v", err)
	}
	want := "<!doctype html><html><body>palette</body></html>\n"
	if string(body) != want {
		t.Fatalf("artifact bytes = %q, want %q", body, want)
	}
	if metrics.EngineCalls != 1 {
		t.Fatalf("EngineCalls=%d, want 1", metrics.EngineCalls)
	}
}

func TestForcedWriteArgumentsRejectUngroundedSuccessClaim(t *testing.T) {
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Write", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string"}},"required":["file_path","content","mode"]}`)}}}
	if args, ok := forcedToolArgumentsFromMessages("Write", tools, []Message{{Role: RoleUser, Content: "Create index.html."}}, "Index.html contains a complete color palette generator. Read to verify."); ok {
		t.Fatalf("ungrounded success claim synthesized Write args %q", args)
	}
}

func TestEnforceForcedWriteRejectsTextualSuccessWithoutArtifactCall(t *testing.T) {
	choice := json.RawMessage(`{"type":"function","function":{"name":"Write"}}`)
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Write", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string"}},"required":["file_path","content","mode"]}`)}}}
	comp := &Completion{Message: Message{Role: RoleAssistant, Content: "Index.html contains a complete color palette generator. Read to verify."}}
	got := enforceForcedToolChoice(comp, choice, tools, []Message{{Role: RoleUser, Content: "Create index.html."}})
	if !got.ToolCallsDropped || len(got.Message.ToolCalls) != 0 {
		t.Fatalf("forced Write failure was not closed: %+v", got)
	}
}

func TestEnforceForcedWriteLiftsGeneratedArtifact(t *testing.T) {
	choice := json.RawMessage(`{"type":"function","function":{"name":"Write"}}`)
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Write", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string"}},"required":["file_path","content","mode"]}`)}}}
	comp := &Completion{Message: Message{Role: RoleAssistant, Content: "```html\n<!doctype html><html></html>\n```"}}
	got := enforceForcedToolChoice(comp, choice, tools, []Message{{Role: RoleUser, Content: "Create index.html."}})
	if got.ToolCallsDropped || len(got.Message.ToolCalls) != 1 || got.Message.ToolCalls[0].Function.Name != "Write" {
		t.Fatalf("generated artifact was not lifted: %+v", got)
	}
}

func TestEnforceForcedWriteRejectsTruncatedArtifact(t *testing.T) {
	choice := json.RawMessage(`{"type":"function","function":{"name":"Write"}}`)
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Write", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string"}},"required":["file_path","content","mode"]}`)}}}
	comp := &Completion{Message: Message{Role: RoleAssistant, Content: "<!doctype html><html>"}, FinishReason: "length"}
	got := enforceForcedToolChoice(comp, choice, tools, []Message{{Role: RoleUser, Content: "Create index.html."}})
	if !got.ToolCallsDropped || len(got.Message.ToolCalls) != 0 {
		t.Fatalf("truncated artifact was not refused: %+v", got)
	}
}

func TestEnforceRequiredSingleToolChoice(t *testing.T) {
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "record_probe", Parameters: json.RawMessage(`{"type":"object","properties":{"probe":{"type":"string"}},"required":["probe"]}`)}}}
	comp := &Completion{Message: Message{Role: RoleAssistant, Content: `{"probe":"alpha"}`}}
	got := enforceForcedToolChoice(comp, json.RawMessage(`"required"`), tools, []Message{{Role: RoleUser, Content: "Set probe to alpha."}})
	if len(got.Message.ToolCalls) != 1 || got.Message.ToolCalls[0].Function.Name != "record_probe" {
		t.Fatalf("required tool call not enforced: %#v", got.Message)
	}
}
func TestRenderInKernelRequiredSingleTool(t *testing.T) {
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{
		Name: "record_probe", Parameters: json.RawMessage(`{"type":"object","properties":{"probe":{"type":"string"}},"required":["probe"]}`),
	}}}
	got := renderInKernelChatMLRequest([]Message{{Role: RoleUser, Content: "run it"}}, tools, model.Config{}, nil, json.RawMessage(`"required"`))
	for _, want := range []string{"Return only one valid JSON object", `"const":"record_probe"`, `"required":["probe"]`} {
		if !strings.Contains(got, want) {
			t.Fatalf("required single-tool prompt missing %q:\n%s", want, got)
		}
	}
}
