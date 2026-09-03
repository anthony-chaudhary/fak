package agent

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Bonsai (prism-ml Ternary-Bonsai-27B) chat-template + thinking + tool-call
// conformance witness — issue #4878, epic #4867 (serving conformance).
//
// Bonsai IS Qwen3.6-27B: the architecture is unchanged from the qwen35-family
// hybrid (only the weights are re-quantized to ternary), so a Bonsai checkpoint
// declares HF model_type "qwen3_5_text" and carries linear_attention layers —
// which is exactly what model.Config.IsQwen35Hybrid() keys on
// (internal/model/qwen35.go, internal/ggufload/gguf_bonsai_arch_test.go). The
// in-kernel prompt/render/parse surface is therefore ALREADY Bonsai's surface:
// renderInKernelChatMLTools mirrors apply_chat_template(enable_thinking=false),
// splitReasoning is the in-kernel --reasoning-parser qwen3, and LiftTextToolCalls
// lifts the emitted <tool_call>. This file PINS that reuse against a checked-in
// reference transcript so the in-kernel path is trustworthy for Bonsai without a
// per-model Jinja engine.
//
// Confusion risk (per the issue): fak's template is HARDCODED ChatML (Qwen/SmolLM2
// grammar), not a general Jinja engine. The one divergence this file used to document —
// omitting the upstream preamble line "You may call one or more functions to assist
// with the user query." — CLOSED with issue #10600: toolSpecBlock now ports the tools
// branch shared by the upstream Qwen3 / Qwen2.5-Coder templates byte-for-byte, and
// TestBonsaiChatMLToolPreambleDivergenceCaptured pins the closure. Everything here is
// OFFLINE: no weights, no GPU, no network — the render/parse functions are pure over
// the transcript.

// bonsaiConfig returns a model.Config that IsQwen35Hybrid() recognizes as a Bonsai /
// Qwen3.6 hybrid checkpoint: at least one linear_attention (Gated-DeltaNet) layer, the
// single axis IsQwen35Hybrid keys on. The 3:1 linear:full layer ratio mirrors the real
// qwen35 family so the config reads as a representative Bonsai head, not a synthetic stub.
func bonsaiConfig() model.Config {
	return model.Config{LayerTypes: []string{
		"linear_attention", "linear_attention", "linear_attention", "full_attention",
	}}
}

// bonsaiRequestMessages / bonsaiRequestTools are the fixed messages+tools request the
// golden transcript is rendered from. One system message, one user turn, one tool.
func bonsaiRequestMessages() []Message {
	return []Message{
		{Role: RoleSystem, Content: "You are Bonsai, a helpful assistant."},
		{Role: RoleUser, Content: "What is the weather in Paris?"},
	}
}

func bonsaiRequestTools() []ToolDef {
	return []ToolDef{{
		Type: "function",
		Function: ToolDefFunction{
			Name:        "get_weather",
			Description: "Get the current temperature for a city, in Celsius.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string","description":"City name, e.g. Paris"}},"required":["city"]}`),
		},
	}}
}

// bonsaiChatMLReferenceThinkingOff is the checked-in reference transcript: the exact
// ChatML fak renders for the bonsaiRequest with the thinking default (enable_thinking
// == false, which the qwen35 hybrid takes unless FAK_INKERNEL_ENABLE_THINKING=1). The
// leading system block folds the tool schema into a <tools>…</tools> advertisement plus
// the <tool_call> emission instruction; the assistant turn is pre-seeded with the empty
// <think>\n\n</think> block that mirrors apply_chat_template(enable_thinking=false).
// The system block is a byte-faithful port of the tools branch shared by the upstream
// Qwen3 and Qwen2.5-Coder chat templates (verified against Qwen/Qwen3-32B and
// Qwen/Qwen2.5-Coder-7B-Instruct tokenizer_config.json renders via jinja2) — issue
// #10600 closed the previously documented preamble/grammar divergence.
const bonsaiChatMLReferenceThinkingOff = `<|im_start|>system
You are Bonsai, a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get the current temperature for a city, in Celsius.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}}, "required": ["city"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What is the weather in Paris?<|im_end|>
<|im_start|>assistant
<think>

</think>

`

// TestBonsaiChatMLRenderConformance is the golden render witness. It pins that fak's
// live in-kernel renderer produces, byte-for-byte, the checked-in reference ChatML for a
// Bonsai messages+tools request — both with the thinking default (the empty <think> block
// pre-seeded) and with FAK_INKERNEL_ENABLE_THINKING=1 (no seed). The thinking-on transcript
// is the reference minus the seed suffix, so a single checked-in golden pins both surfaces.
func TestBonsaiChatMLRenderConformance(t *testing.T) {
	cfg := bonsaiConfig()
	if !cfg.IsQwen35Hybrid() {
		t.Fatal("bonsaiConfig must satisfy IsQwen35Hybrid() — Bonsai is a qwen35-family hybrid; the whole render path is gated on it")
	}
	msgs, tools := bonsaiRequestMessages(), bonsaiRequestTools()

	// Thinking default: enable_thinking=false is mirrored by pre-seeding <think>\n\n</think>.
	gotOff := renderInKernelChatMLTools(msgs, tools, cfg)
	if gotOff != bonsaiChatMLReferenceThinkingOff {
		t.Errorf("thinking-off render diverges from the reference transcript.\n--- got ---\n%q\n--- want ---\n%q", gotOff, bonsaiChatMLReferenceThinkingOff)
	}

	// The empty seed must be exactly the enable_thinking=false mirror, appended once.
	if !strings.HasSuffix(gotOff, qwenNoThinkAssistantSeed) {
		t.Fatalf("thinking-off render must end with the enable_thinking=false seed %q", qwenNoThinkAssistantSeed)
	}
	wantOn := strings.TrimSuffix(bonsaiChatMLReferenceThinkingOff, qwenNoThinkAssistantSeed)

	// FAK_INKERNEL_ENABLE_THINKING=1 keeps the raw reasoning mode: no seed, so the assistant
	// turn is left open for the model to emit its own <think>…</think>.
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "1")
	gotOn := renderInKernelChatMLTools(msgs, tools, cfg)
	if gotOn != wantOn {
		t.Errorf("thinking-on render diverges from the reference (reference minus seed).\n--- got ---\n%q\n--- want ---\n%q", gotOn, wantOn)
	}
	if strings.Contains(gotOn, qwenNoThinkAssistantSeed) {
		t.Errorf("thinking-on render must NOT pre-seed a think block, got:\n%q", gotOn)
	}
}

// TestBonsaiChatMLThinkingDefaultMatchesEnableThinkingFalse pins the thinking DEFAULT
// itself: an unset FAK_INKERNEL_ENABLE_THINKING on a Bonsai (qwen35 hybrid) config seeds
// the empty <think> block (enable_thinking=false), while a NON-hybrid config never does —
// so the seed is Bonsai-scoped, not a global behavior change. This is the control that
// proves the default is keyed on the checkpoint family, matching Bonsai's expected behavior.
func TestBonsaiChatMLThinkingDefaultMatchesEnableThinkingFalse(t *testing.T) {
	msgs, tools := bonsaiRequestMessages(), bonsaiRequestTools()

	bonsai := renderInKernelChatMLTools(msgs, tools, bonsaiConfig())
	if !strings.HasSuffix(bonsai, qwenNoThinkAssistantSeed) {
		t.Errorf("Bonsai default must seed enable_thinking=false block %q; got tail:\n%q", qwenNoThinkAssistantSeed, bonsai)
	}

	// A plain (non-hybrid) checkpoint carries no linear_attention layer, so IsQwen35Hybrid()
	// is false and the render leaves the assistant turn open — the seed must not leak to it.
	nonHybrid := model.Config{LayerTypes: []string{"full_attention", "full_attention"}}
	if nonHybrid.IsQwen35Hybrid() {
		t.Fatal("control config must NOT be a qwen35 hybrid")
	}
	plain := renderInKernelChatMLTools(msgs, tools, nonHybrid)
	if strings.Contains(plain, "<think>") {
		t.Errorf("non-hybrid render must not seed a think block, got:\n%q", plain)
	}
	if !strings.HasSuffix(plain, "<|im_start|>assistant\n") {
		t.Errorf("non-hybrid render must end on an open assistant turn, got tail:\n%q", plain)
	}
}

// TestBonsaiChatMLToolPreambleDivergenceCaptured used to CAPTURE a deliberate divergence
// (fak omitted the upstream "You may call one or more functions…" preamble and taught the
// antl emission format). Issue #10600 closed it: toolSpecBlock now ports the tools branch
// shared by the upstream Qwen3 / Qwen2.5-Coder templates byte-for-byte, so this test
// flipped into a closure witness — the preamble must be present, the antl instruction and
// the compact json.Marshal signature spacing must be gone, and the system turn must end
// flush with </tool_call><|im_end|> the way both upstream templates end it.
func TestBonsaiChatMLToolPreambleDivergenceCaptured(t *testing.T) {
	const upstreamPreamble = "You may call one or more functions to assist with the user query."
	got := renderInKernelChatMLTools(bonsaiRequestMessages(), bonsaiRequestTools(), bonsaiConfig())
	if !strings.Contains(got, upstreamPreamble) {
		t.Fatalf("render no longer carries the upstream preamble %q — the template diverged again:\n%q", upstreamPreamble, got)
	}
	// The structural tool advertisement fak renders is still present and complete.
	for _, want := range []string{
		"# Tools",
		"<tools>",
		`"name": "get_weather"`, // tojson spacing, not compact json.Marshal
		"</tools>",
		"For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("tool advertisement missing %q in render:\n%q", want, got)
		}
	}
	// The old antl emission format and the <IMPORTANT> reminder are the Ornith contract,
	// never the Qwen3/Qwen2.5 one; and the turn ends flush against <|im_end|>.
	for _, banned := range []string{
		"If you choose to call a function ONLY reply in the following format",
		"<IMPORTANT>",
		`"name":"get_weather"`,
		"</tool_call>\n<|im_end|>", // the template ends the turn flush, no newline between
	} {
		if strings.Contains(got, banned) {
			t.Errorf("render carries divergent %q:\n%q", banned, got)
		}
	}
}

// TestBonsaiReasoningSplitConformance pins the reasoning split for Bonsai's two emission
// shapes. With FAK_INKERNEL_ENABLE_THINKING=1 the model opens <think>…</think> then the
// answer, and splitReasoning (the in-kernel --reasoning-parser qwen3) routes the reasoning
// to ReasoningContent and only the post-</think> answer to Content. With the thinking
// default the assistant turn is pre-seeded with a CLOSED empty <think> block, so the model
// decodes only the answer (no tags) — splitReasoning must return it byte-identical, never
// leaking a half block. This is the parse half of the render golden above.
func TestBonsaiReasoningSplitConformance(t *testing.T) {
	// Thinking enabled: Bonsai emits an open+close reasoning block then the answer.
	reasoning, content := splitReasoning("<think>Paris is temperate; check the tool result.</think>It is 15°C in Paris.")
	if reasoning != "Paris is temperate; check the tool result." {
		t.Errorf("reasoning = %q, want the <think> block", reasoning)
	}
	if content != "It is 15°C in Paris." {
		t.Errorf("content = %q, want only the post-</think> answer", content)
	}

	// Thinking default (empty block pre-seeded into the PROMPT, not the decode): the model
	// decodes only the answer with no think tags, so the split is a byte-identical no-op.
	const plainAnswer = "It is 15°C in Paris."
	r2, c2 := splitReasoning(plainAnswer)
	if r2 != "" {
		t.Errorf("suppressed-thinking decode must yield no reasoning, got %q", r2)
	}
	if c2 != plainAnswer {
		t.Errorf("suppressed-thinking decode must pass content byte-identical, got %q want %q", c2, plainAnswer)
	}
}

// TestBonsaiToolCallLiftConformance pins the tool-call round-trip. First: a Bonsai model
// that emits its call as TEXT in the native Qwen <tool_call> dialect is lifted into a
// structured Message.ToolCalls (an unlifted text call is a silent adjudication bypass —
// the gateway adjudicates only structured calls). Second: rendering a completed assistant
// tool call plus its tool result back into the transcript produces the canonical Qwen
// <tool_call> / <tool_response> grammar, so a multi-turn Bonsai tool flow round-trips.
func TestBonsaiToolCallLiftConformance(t *testing.T) {
	// Lift: text-form <tool_call> -> structured ToolCalls, content stripped.
	emitted := "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"
	lifted := LiftTextToolCalls(Message{Role: RoleAssistant, Content: emitted})
	if len(lifted.ToolCalls) != 1 {
		t.Fatalf("want exactly one lifted tool call, got %d (%+v)", len(lifted.ToolCalls), lifted.ToolCalls)
	}
	tc := lifted.ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Errorf("lifted call name = %q, want get_weather", tc.Function.Name)
	}
	if !strings.Contains(tc.Function.Arguments, `"city"`) || !strings.Contains(tc.Function.Arguments, "Paris") {
		t.Errorf("lifted call arguments = %q, want the {\"city\":\"Paris\"} object", tc.Function.Arguments)
	}
	if strings.Contains(lifted.Content, "<tool_call>") {
		t.Errorf("lifted content must have the <tool_call> span stripped, got %q", lifted.Content)
	}

	// Round-trip: render the assistant call + tool result back into ChatML and assert the
	// canonical <tool_call>/<tool_response> grammar a tool-trained Bonsai recognizes.
	transcript := renderTranscriptTools([]Message{
		{Role: RoleUser, Content: "What is the weather in Paris?"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{
			ID:       "call_0",
			Type:     "function",
			Function: Func{Name: "get_weather", Arguments: `{"city":"Paris"}`},
		}}},
		{Role: RoleTool, ToolCallID: "call_0", Name: "get_weather", Content: "15"},
	}, bonsaiRequestTools())

	if !strings.Contains(transcript, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>") {
		t.Errorf("assistant tool call did not render as canonical <tool_call> block:\n%q", transcript)
	}
	if !strings.Contains(transcript, "<tool_response>\nget_weather: 15\n</tool_response>") {
		t.Errorf("tool result did not render as canonical <tool_response> block:\n%q", transcript)
	}
}
