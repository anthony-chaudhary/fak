package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// These tests exercise the in-kernel planner's tool-call SEAM — the normalization that
// InKernelPlanner.Complete runs on its raw text completion — without booting a weighted
// model (which OOMs under WSL). They pin the exact transformation Complete applies after
// decode: lift the model's text-form <tool_call> into structured Message.ToolCalls, set
// the finish reason, and fail closed on a truncated call.

// inKernelNormalize mirrors the post-decode steps of InKernelPlanner.Complete (the two
// lines after the raw Completion is built), so the seam is testable on a synthesized
// completion. Keep it in lockstep with Complete.
func inKernelNormalize(content, finishReason string) *Completion {
	comp := &Completion{
		Message:      Message{Role: "assistant", Content: content},
		FinishReason: finishReason,
	}
	comp = normalizeCompletionToolCalls(comp)
	if len(comp.Message.ToolCalls) == 0 && strings.Contains(comp.Message.Content, "<tool_call>") {
		comp.ToolCallsDropped = true
	}
	return comp
}

// inKernelDecodeToCompletion mirrors the FULL post-decode pipeline of
// InKernelPlanner.Complete in the exact order Complete runs it: splitReasoning FIRST
// (Qwen3.5/Ornith <think> block → ReasoningContent, post-</think> answer → Content),
// THEN the tool-call lift over Content, THEN the truncated-call fail-closed flag. Unlike
// inKernelNormalize (which only covers the tool-call steps), this exercises the
// reasoning+tool-call COMPOSITION — the ordering issue #1059 names — so a regression that
// lifts the tool call before stripping reasoning (or feeds <think> text into the lift)
// would fail here. Keep it in lockstep with Complete (inkernel_planner.go lines ~667-687).
func inKernelDecodeToCompletion(raw, finishReason string) *Completion {
	reasoning, content := splitReasoning(raw)
	comp := &Completion{
		Message:      Message{Role: "assistant", Content: content, ReasoningContent: reasoning},
		FinishReason: finishReason,
	}
	comp = normalizeCompletionToolCalls(comp)
	if len(comp.Message.ToolCalls) == 0 && strings.Contains(comp.Message.Content, "<tool_call>") {
		comp.ToolCallsDropped = true
	}
	return comp
}

// TestCompleteLiftsTextToolCall: a well-formed Hermes <tool_call> (Qwen2.5's native
// dialect) decoded by the in-kernel forward is lifted to a structured ToolCall and the
// finish reason becomes tool_calls — the signal the gateway adjudicator + Anthropic wire
// read to emit a tool_use block.
func TestCompleteLiftsTextToolCall(t *testing.T) {
	comp := inKernelNormalize(`<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>`, "stop")
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
	comp := inKernelNormalize(
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
	comp := inKernelNormalize(`sure, calling it: <tool_call>{"name": "Bash", "argum`, "length")
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

// TestCompletePlainChatUnaffected: a turn with no tool call is unchanged — plain chat
// still works on the in-kernel path.
func TestCompletePlainChatUnaffected(t *testing.T) {
	comp := inKernelNormalize("2 + 2 is 4.", "stop")
	if len(comp.Message.ToolCalls) != 0 {
		t.Fatalf("plain chat must lift 0 tool calls, got %d", len(comp.Message.ToolCalls))
	}
	if comp.ToolCallsDropped {
		t.Fatalf("plain chat must not be flagged dropped")
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
	comp := inKernelDecodeToCompletion(raw, "stop")

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
	comp := inKernelDecodeToCompletion(raw, "stop")
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
	comp := inKernelDecodeToCompletion(raw, "length")
	if !strings.Contains(comp.Message.ReasoningContent, "call Bash") {
		t.Fatalf("reasoning not captured ahead of the truncated call: %q", comp.Message.ReasoningContent)
	}
	if len(comp.Message.ToolCalls) != 0 {
		t.Fatalf("a truncated call after reasoning must lift 0 calls, got %d", len(comp.Message.ToolCalls))
	}
	if !comp.ToolCallsDropped {
		t.Fatalf("a truncated <tool_call> after a reasoning block must still set ToolCallsDropped")
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
