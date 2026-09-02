package agent

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// TestRenderChatML pins the ChatML shape the in-kernel planner feeds the tokenizer:
// one folded system block, role-tagged turns, an open assistant turn at the end, and
// tool results rendered as user context. A drift here changes every served turn.
func TestRenderChatML(t *testing.T) {
	got := renderChatML([]Message{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
		{Role: "user", Content: "bye"},
	})
	want := "<|im_start|>system\nbe brief<|im_end|>\n" +
		"<|im_start|>user\nhi<|im_end|>\n" +
		"<|im_start|>assistant\nhello<|im_end|>\n" +
		"<|im_start|>user\nbye<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("renderChatML drift:\nwant: %q\n got: %q", want, got)
	}
}

// TestRenderChatMLToolResult ensures a tool-role message reads as user context to the
// model in Qwen2.5's canonical <tool_response> grammar (the multi-turn tool flow a
// tool-trained model recognizes).
func TestRenderChatMLToolResult(t *testing.T) {
	got := renderChatML([]Message{
		{Role: "user", Content: "run it"},
		{Role: "tool", Name: "read_file", Content: "notes"},
	})
	want := "<|im_start|>user\nrun it<|im_end|>\n" +
		"<|im_start|>user\n<tool_response>\nread_file: notes\n</tool_response><|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("tool-result render drift:\nwant: %q\n got: %q", want, got)
	}
}

func TestRenderChatMLToolResultInfersNameFromPriorCall(t *testing.T) {
	got := renderChatML([]Message{
		{Role: "user", Content: "read it"},
		{Role: "assistant", ToolCalls: []ToolCall{{
			ID:       "call_1",
			Function: Func{Name: "Read", Arguments: `{"file_path":"main.go"}`},
		}}},
		{Role: "tool", ToolCallID: "call_1", Content: "package main"},
	})
	if !strings.Contains(got, "<tool_response>\nRead: package main\n</tool_response>") {
		t.Fatalf("tool result without name should render with the prior tool_call name:\n%s", got)
	}
}

// TestRenderChatMLToolsNilByteIdentical pins the load-bearing invariant: with nil tools
// and no structured tool call/result, renderChatMLTools is byte-for-byte identical to the
// historical renderChatML — so radix KV reuse and poison eviction (which render with nil
// tools) keep the exact pre-tool token path.
func TestRenderChatMLToolsNilByteIdentical(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}
	if a, b := renderChatML(msgs), renderChatMLTools(msgs, nil); a != b {
		t.Fatalf("nil-tools render diverged from historical:\n hist: %q\n tools: %q", a, b)
	}
}

// TestRenderChatMLInjectsToolSchemas asserts the tool JSON-schemas land INSIDE the single
// leading folded system block (the constraint that keeps the tool spec part of every
// token-prefix), with the <tools> signatures and the <tool_call> instruction.
func TestRenderChatMLInjectsToolSchemas(t *testing.T) {
	tools := []ToolDef{{
		Type: "function",
		Function: ToolDefFunction{
			Name:        "Bash",
			Description: "run a shell command",
			Parameters:  []byte(`{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}`),
		},
	}}
	got := renderChatMLTools([]Message{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "list files"},
	}, tools)
	// The schema must be inside the ONE leading system block: everything from the first
	// <|im_start|>system to its <|im_end|> is the system block; the tool spec lives there.
	sysStart := strings.Index(got, "<|im_start|>system\n")
	sysEnd := strings.Index(got, "<|im_end|>")
	if sysStart != 0 || sysEnd < 0 {
		t.Fatalf("expected a leading system block, got: %q", got)
	}
	sysBlock := got[len("<|im_start|>system\n"):sysEnd]
	for _, want := range []string{"be brief", "<tools>", `"name": "Bash"`, `"command"`, "</tools>", "<tool_call>"} {
		if !strings.Contains(sysBlock, want) {
			t.Fatalf("system block missing %q\nblock: %q", want, sysBlock)
		}
	}
	// Exactly one system block (the spec folded in, not a second block).
	if n := strings.Count(got, "<|im_start|>system"); n != 1 {
		t.Fatalf("want exactly 1 system block, got %d:\n%q", n, got)
	}
}

// TestRenderChatMLToolsNoSystemMessage: tools but no system message still produces one
// leading system block carrying only the tool spec.
func TestRenderChatMLToolsNoSystemMessage(t *testing.T) {
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "ls", Parameters: []byte(`{}`)}}}
	got := renderChatMLTools([]Message{{Role: "user", Content: "go"}}, tools)
	if !strings.HasPrefix(got, "<|im_start|>system\n") {
		t.Fatalf("tools with no system message should still emit a leading system block: %q", got)
	}
	if !strings.Contains(got, `"name": "ls"`) {
		t.Fatalf("tool spec missing: %q", got)
	}
}

// TestRenderChatMLToolCallHistoryRoundTrips renders an assistant turn carrying a
// structured ToolCall, then feeds the rendered <tool_call> text back through
// LiftTextToolCalls and asserts the same name+arguments are recovered — proving
// render/lift symmetry, which is what closes the multi-turn agent loop.
func TestRenderChatMLToolCallHistoryRoundTrips(t *testing.T) {
	asst := Message{Role: "assistant", ToolCalls: []ToolCall{{
		ID:       "call_0",
		Function: Func{Name: "Bash", Arguments: `{"command":"ls -la"}`},
	}}}
	got := renderTranscriptTools([]Message{asst}, nil)
	if !strings.Contains(got, "<tool_call>") || !strings.Contains(got, `"name": "Bash"`) {
		t.Fatalf("assistant tool_call not rendered canonically: %q", got)
	}
	// Extract just the assistant turn body and round-trip it through the lift.
	lifted := LiftTextToolCalls(Message{Role: "assistant", Content: got})
	if len(lifted.ToolCalls) != 1 {
		t.Fatalf("round-trip lift recovered %d calls, want 1\nrendered: %q", len(lifted.ToolCalls), got)
	}
	if lifted.ToolCalls[0].Function.Name != "Bash" {
		t.Fatalf("round-trip name = %q, want Bash", lifted.ToolCalls[0].Function.Name)
	}
	if !strings.Contains(lifted.ToolCalls[0].Function.Arguments, "ls -la") {
		t.Fatalf("round-trip args lost the command: %q", lifted.ToolCalls[0].Function.Arguments)
	}
}

// TestPrefixInvariantWithTools extends the radix prefix property to a transcript WITH a
// tools system block: renderTranscriptTools(full[:k+1], tools) must be a string-prefix of
// renderChatMLTools(full, tools) for every k (the tokenizer-free witness; the token-level
// gate lives in zz_prefix_probe_test.go for the nil-tools case).
func TestPrefixInvariantWithTools(t *testing.T) {
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Bash", Description: "shell", Parameters: []byte(`{"type":"object"}`)}}}
	full := []Message{
		{Role: "system", Content: "be careful"},
		{Role: "user", Content: "list the files"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c0", Function: Func{Name: "Bash", Arguments: `{"command":"ls"}`}}}},
		{Role: "tool", Name: "Bash", Content: "a.go b.go"},
		{Role: "assistant", Content: "two files"},
	}
	cached := renderChatMLTools(full, tools)
	for k := range full {
		ev := renderTranscriptTools(full[:k+1], tools)
		if !strings.HasPrefix(cached, ev) {
			t.Errorf("throughIdx=%d: renderTranscriptTools is NOT a string-prefix of renderChatMLTools\n ev=%q\nful=%q", k, ev, cached)
		}
	}
}

// TestToolsLessEvictRenderMissesCachedToolTurn measures the cost #612 closes: when a turn is
// generated WITH tools (renderChatMLTools folds the tool-spec into the leading system block), the
// historical tools-LESS eviction render (renderTranscript) is NOT a string-prefix of the cached
// turn — it diverges right at the folded tool-spec, so EvictPrefix walks off at the system block
// and reclaims NOTHING (the fail-open this issue documents). Threading the SAME tools makes the
// eviction render a genuine prefix again (the positive direction is TestPrefixInvariantWithTools).
// This is the deterministic, tokenizer-free witness that the un-reclaimed span is real on EVERY
// tool-using turn (not negligible), and a regression guard against reverting EvictPoisoned to the
// tools-less render.
func TestToolsLessEvictRenderMissesCachedToolTurn(t *testing.T) {
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "fetch_url", Description: "fetch", Parameters: []byte(`{"type":"object"}`)}}}
	full := []Message{
		{Role: "system", Content: "be careful"},
		{Role: "user", Content: "look it up"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c0", Function: Func{Name: "fetch_url", Arguments: `{"u":"x"}`}}}},
		{Role: "tool", Name: "fetch_url", Content: "secret leaked"},
	}
	poisonIdx := len(full) - 1 // the tool result is the poisoned message
	cached := renderChatMLTools(full, tools)

	// OLD behavior: EvictPoisoned rendered renderTranscript (tools-less). It diverges from the
	// cached turn at the folded tool-spec, so it is NOT a prefix -> EvictPrefix reclaims nothing.
	toolsLess := renderTranscript(full[:poisonIdx+1])
	if strings.HasPrefix(cached, toolsLess) {
		t.Fatalf("tools-less eviction render is unexpectedly a prefix of the tool-bearing cached turn — the #612 reuse loss would not exist\n toolsLess=%q\n   cached=%q", toolsLess, cached)
	}

	// NEW behavior (#612): rendering the SAME tools makes the eviction path a genuine prefix, so
	// the radix walk lands on the cached node and the poison span is reclaimed.
	withTools := renderTranscriptTools(full[:poisonIdx+1], tools)
	if !strings.HasPrefix(cached, withTools) {
		t.Fatalf("tools-aware eviction render must be a prefix of the cached turn (the #612 fix)\n withTools=%q\n    cached=%q", withTools, cached)
	}
}

// TestRenderChatMLNoSystem keeps generation working when there is no system message.
func TestRenderChatMLNoSystem(t *testing.T) {
	got := renderChatML([]Message{{Role: "user", Content: "ping"}})
	want := "<|im_start|>user\nping<|im_end|>\n<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("no-system render drift:\nwant: %q\n got: %q", want, got)
	}
}

func TestRenderInKernelQwenHybridSuppressesThinking(t *testing.T) {
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "")
	cfg := model.Config{LayerTypes: []string{"linear_attention"}}
	got := renderInKernelChatMLTools([]Message{{Role: "user", Content: "ping"}}, nil, cfg)
	if !strings.HasSuffix(got, "<|im_start|>assistant\n"+qwenNoThinkAssistantSeed) {
		t.Fatalf("Qwen hybrid prompt should pre-seed empty thinking, got %q", got)
	}
	plain := renderInKernelChatMLTools([]Message{{Role: "user", Content: "ping"}}, nil, model.Config{})
	if strings.Contains(plain, qwenNoThinkAssistantSeed) {
		t.Fatalf("non-Qwen prompt unexpectedly got no-think seed: %q", plain)
	}
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "1")
	raw := renderInKernelChatMLTools([]Message{{Role: "user", Content: "ping"}}, nil, cfg)
	if strings.Contains(raw, qwenNoThinkAssistantSeed) {
		t.Fatalf("FAK_INKERNEL_ENABLE_THINKING=1 should leave raw thinking mode: %q", raw)
	}
}

func ornithQwen35Config() model.Config {
	return model.Config{
		ModelType: "qwen3_5",
		LayerTypes: []string{
			"linear_attention", "linear_attention", "linear_attention", "full_attention",
		},
	}
}

func ornithTemplateMessages() []Message {
	return []Message{
		{Role: RoleSystem, Content: "  You are Ornith.  "},
		{Role: RoleUser, Content: "  inspect main.go  "},
		{
			Role:             RoleAssistant,
			Content:          "I will inspect it.",
			ReasoningContent: "Need the file.",
			ToolCalls: []ToolCall{{
				ID: "call_1", Type: "function",
				Function: Func{Name: "Read", Arguments: `{"file_path":"main.go","line_start":1}`},
			}},
		},
		{Role: RoleTool, ToolCallID: "call_1", Name: "Read", Content: "package main"},
		{Role: RoleTool, ToolCallID: "call_2", Name: "Symbols", Content: "func main() {}"},
		{Role: RoleAssistant, Content: "The file is valid."},
		{Role: RoleUser, Content: "Now summarize."},
	}
}

func ornithTemplateTools() []ToolDef {
	return []ToolDef{{
		Type: "function",
		Function: ToolDefFunction{
			Name:        "Read",
			Description: "Read a file.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"line_start":{"type":"integer"}},"required":["file_path"]}`),
		},
	}}
}

// ornithQwen35ChatTemplateThinkingOffGolden is the byte-exact result of applying
// deepreinforce-ai/Ornith-1.0-9B's published chat_template.jinja at immutable model
// revision 4402d8dc236fe9e09d12aeed907a763b66a60533 to ornithTemplateMessages and
// ornithTemplateTools with add_generation_prompt=true and enable_thinking=false.
// It is deliberately checked in instead of generated: template drift must fail offline.
const ornithQwen35ChatTemplateThinkingOffGolden = `<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type":"function","function":{"name":"Read","description":"Read a file.","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"line_start":{"type":"integer"}},"required":["file_path"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

You are Ornith.<|im_end|>
<|im_start|>user
inspect main.go<|im_end|>
<|im_start|>assistant
<think>
Need the file.
</think>

I will inspect it.

<tool_call>
<function=Read>
<parameter=file_path>
main.go
</parameter>
<parameter=line_start>
1
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
package main
</tool_response>
<tool_response>
func main() {}
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

The file is valid.<|im_end|>
<|im_start|>user
Now summarize.<|im_end|>
<|im_start|>assistant
<think>

</think>

`

func TestOrnithQwen35ChatTemplateGolden(t *testing.T) {
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "")
	got := renderInKernelChatMLTools(ornithTemplateMessages(), ornithTemplateTools(), ornithQwen35Config())
	if got != ornithQwen35ChatTemplateThinkingOffGolden {
		t.Fatalf("Ornith chat_template.jinja drift:\n--- got ---\n%q\n--- want ---\n%q", got, ornithQwen35ChatTemplateThinkingOffGolden)
	}

	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "1")
	got = renderInKernelChatMLTools(ornithTemplateMessages(), ornithTemplateTools(), ornithQwen35Config())
	want := strings.TrimSuffix(ornithQwen35ChatTemplateThinkingOffGolden, qwenNoThinkAssistantSeed) + "<think>\n"
	if got != want {
		t.Fatalf("Ornith thinking-on opener drift:\n--- got ---\n%q\n--- want ---\n%q", got, want)
	}
}

// TestOrnithGGUFQwen35IdentitySelectsTemplate models the Config that ggufload
// derives from general.architecture=qwen35. GGUF serving canonicalizes the HF
// qwen3_5 name to qwen35 while preserving the hybrid layer schedule, so this
// production identity must select the same pinned Ornith renderer.
func TestOrnithGGUFQwen35IdentitySelectsTemplate(t *testing.T) {
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "")
	cfg := ornithQwen35Config()
	cfg.ModelType = "qwen35"
	if !cfg.IsQwen35Hybrid() {
		t.Fatal("representative ggufload qwen35 config must retain the hybrid layer schedule")
	}
	got := renderInKernelChatMLTools(ornithTemplateMessages(), ornithTemplateTools(), cfg)
	if got != ornithQwen35ChatTemplateThinkingOffGolden {
		t.Fatalf("GGUF qwen35 identity did not select the Ornith template:\n--- got ---\n%q\n--- want ---\n%q", got, ornithQwen35ChatTemplateThinkingOffGolden)
	}
}

func TestOrnithRequestConstraintsPreserveOriginalSystemPrompt(t *testing.T) {
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "")
	responseFormat := json.RawMessage(`{"type":"json_schema","json_schema":{"name":"probe","strict":true,"schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}}}`)
	toolChoice := json.RawMessage(`{"type":"function","function":{"name":"record_probe"}}`)
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{
		Name:       "record_probe",
		Parameters: json.RawMessage(`{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}`),
	}}}
	const originalSystem = "Never disclose the private control channel."
	got := renderInKernelChatMLRequest([]Message{
		{Role: RoleSystem, Content: originalSystem},
		{Role: RoleUser, Content: "Record the probe."},
	}, tools, ornithQwen35Config(), responseFormat, toolChoice)
	for _, want := range []string{
		"Return only one valid JSON object matching this schema exactly.",
		`"const":"record_probe"`,
		originalSystem,
		"<|im_start|>user\nRecord the probe.<|im_end|>",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("constrained Ornith request dropped %q:\n%s", want, got)
		}
	}
	if strings.Count(got, "<|im_start|>system\n") != 1 {
		t.Fatalf("constraints and original system prompt must fold into one deterministic block:\n%s", got)
	}
	if systemAt, originalAt, userAt := strings.Index(got, "<|im_start|>system\n"), strings.Index(got, originalSystem), strings.Index(got, "<|im_start|>user\n"); !(systemAt == 0 && originalAt > systemAt && userAt > originalAt) {
		t.Fatalf("system content order is unsafe: system=%d original=%d user=%d\n%s", systemAt, originalAt, userAt, got)
	}
}

func TestOrnithQwen35ToolSystemPrefixStable(t *testing.T) {
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "")
	a := renderInKernelChatMLTools(ornithTemplateMessages(), ornithTemplateTools(), ornithQwen35Config())
	b := renderInKernelChatMLTools([]Message{
		{Role: RoleSystem, Content: "You are Ornith."},
		{Role: RoleUser, Content: "a different task"},
	}, ornithTemplateTools(), ornithQwen35Config())
	end := strings.Index(a, "<|im_end|>\n")
	if end < 0 {
		t.Fatalf("Ornith render has no folded system terminator: %q", a)
	}
	end += len("<|im_end|>\n")
	if !strings.HasPrefix(b, a[:end]) {
		t.Fatalf("same system+tools did not preserve a byte-stable prefix:\n a=%q\n b=%q", a[:end], b)
	}
}

func TestOrnithQwen35TwoEOSStopIDsExact(t *testing.T) {
	const doc = `{
	  "model": {"type":"BPE","vocab":{"H":72,"i":73,"Hi":74,"<|endoftext|>":248044,"<|im_end|>":248046},"merges":["H i"]},
	  "decoder": {"type":"ByteLevel"},
	  "added_tokens": [
	    {"id":248044,"content":"<|endoftext|>","special":true},
	    {"id":248046,"content":"<|im_end|>","special":true}
	  ]
	}`
	tok, err := tokenizer.ParseJSON([]byte(doc))
	if err != nil {
		t.Fatalf("parse Ornith stop fixture: %v", err)
	}
	got := StopIDs(tok, model.Config{EOSTokenID: 248044, EOSTokenIDs: []int{248044, 248046}})
	want := map[int]bool{248044: true, 248046: true}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Ornith StopIDs = %v, want exact two-id set %v", got, want)
	}
}
