package agent

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// loadQwen38GGUFFixture locates, verifies, and parses the pinned Qwen3.8 GGUF header fixture.
func loadQwen38GGUFFixture(t *testing.T) (*ggufload.File, model.Config) {
	t.Helper()

	candidates := []string{
		filepath.Join("..", "ggufload", "testdata", "qwen38_ud_q2kxl_header.gguf.gz"),
		filepath.Join("internal", "ggufload", "testdata", "qwen38_ud_q2kxl_header.gguf.gz"),
	}
	var fixturePath string
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			fixturePath = c
			break
		}
	}
	if fixturePath == "" {
		t.Fatalf("could not locate qwen38_ud_q2kxl_header.gguf.gz in candidate paths: %v", candidates)
	}

	compressed, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read compressed fixture %q: %v", fixturePath, err)
	}

	zr, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatalf("gzip.NewReader: %v", err)
	}
	defer zr.Close()

	raw, err := io.ReadAll(io.LimitReader(zr, 16<<20))
	if err != nil {
		t.Fatalf("read decompressed bytes: %v", err)
	}

	const (
		wantLen  = 10996640
		wantHash = "1fe82fda85430cca654a156e9ec2915baf460752197013563b426db2581dcc0f"
	)
	if len(raw) != wantLen {
		t.Fatalf("unexpected decompressed byte length: got %d, want %d", len(raw), wantLen)
	}
	gotHash := fmt.Sprintf("%x", sha256.Sum256(raw))
	if gotHash != wantHash {
		t.Fatalf("unexpected decompressed SHA256:\ngot:  %s\nwant: %s", gotHash, wantHash)
	}

	gg, err := ggufload.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("ggufload.Read: %v", err)
	}

	cfg, err := gg.Config()
	if err != nil {
		t.Fatalf("gg.Config: %v", err)
	}

	return gg, cfg
}

func TestQwen38GGUFHeaderIdentityAndConfig(t *testing.T) {
	_, cfg := loadQwen38GGUFFixture(t)

	if cfg.ModelType != "qwen35" {
		t.Errorf("cfg.ModelType = %q, want %q", cfg.ModelType, "qwen35")
	}
	if !cfg.IsQwen35Hybrid() {
		t.Errorf("cfg.IsQwen35Hybrid() = false, want true")
	}
	if cfg.Name != "Qwen3.8-27B" {
		t.Errorf("cfg.Name = %q, want %q", cfg.Name, "Qwen3.8-27B")
	}
	if cfg.EOSTokenID != 248046 {
		t.Errorf("cfg.EOSTokenID = %d, want 248046", cfg.EOSTokenID)
	}
	if !inKernelUsesOrnithQwen35Template(cfg) {
		t.Errorf("inKernelUsesOrnithQwen35Template(cfg) = false, want true")
	}
}

func TestQwen38ChatTemplateExtractedFromGGUF(t *testing.T) {
	gg, _ := loadQwen38GGUFFixture(t)

	tmpl, ok := gg.String("tokenizer.chat_template")
	if !ok {
		t.Fatal("tokenizer.chat_template key not found in GGUF metadata")
	}
	if tmpl == "" {
		t.Fatal("tokenizer.chat_template is empty")
	}

	// In the Jinja template source, escaped string literals carry literal \n sequences.
	// We verify both raw substrings and the normalized template text.
	normalized := strings.ReplaceAll(tmpl, `\n`, "\n")

	requiredSubstrings := []string{
		"# Tools\n\nYou have access to the following functions:\n\n<tools>",
		"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>",
		"<IMPORTANT>\nReminder:",
		"<tool_response>",
		"<think>",
		"add_generation_prompt",
	}
	for _, sub := range requiredSubstrings {
		rawEscaped := strings.ReplaceAll(sub, "\n", `\n`)
		if !strings.Contains(tmpl, sub) && !strings.Contains(tmpl, rawEscaped) && !strings.Contains(normalized, sub) {
			t.Errorf("chat template missing required substring: %q", sub)
		}
	}
}

func TestQwen38ChatTemplateAndToolSerializationParity(t *testing.T) {
	_, cfg := loadQwen38GGUFFixture(t)

	tools := []ToolDef{{
		Type: "function",
		Function: ToolDefFunction{
			Name:        "inspect_file",
			Description: "Read file contents from the workspace.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"lines":{"type":"integer"},"path":{"type":"string"}},"required":["path"]}`),
		},
	}}

	messages := []Message{
		{Role: RoleSystem, Content: "You are Qwen3.8, an expert implementation assistant."},
		{Role: RoleSystem, Content: "Adhere to the requested output constraints."},
		{Role: RoleUser, Content: "Inspect internal/agent/main.go"},
		{
			Role:             RoleAssistant,
			Content:          "I will inspect the file.",
			ReasoningContent: "Checking file path and line count.",
			ToolCalls: []ToolCall{{
				ID:   "call_inspect_1",
				Type: "function",
				Function: Func{
					Name:      "inspect_file",
					Arguments: `{"lines":50,"path":"internal/agent/main.go"}`,
				},
			}},
		},
		{
			Role:       RoleTool,
			ToolCallID: "call_inspect_1",
			Name:       "inspect_file",
			Content:    "package main\n\nfunc main() {}",
		},
		{
			Role:    RoleAssistant,
			Content: "The file is a Go entry point.",
		},
		{
			Role:    RoleUser,
			Content: "Now summarize the architecture.",
		},
	}

	// 1. Thinking off (default): assistant generation prompt ends with <think>\n\n</think>\n\n
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "")
	renderedOff := renderInKernelChatMLTools(messages, tools, cfg)

	if !strings.HasSuffix(renderedOff, "<|im_start|>assistant\n"+qwenNoThinkAssistantSeed) {
		t.Fatalf("thinking-off render must end with assistant seed %q, got tail:\n%q", qwenNoThinkAssistantSeed, renderedOff)
	}

	// Verify folded system messages:
	wantSystemFold := "You are Qwen3.8, an expert implementation assistant.\nAdhere to the requested output constraints."
	if !strings.Contains(renderedOff, wantSystemFold) {
		t.Errorf("folded system message not found in render:\n%s", renderedOff)
	}

	// Verify tool definitions in <tools>...</tools>
	if !strings.Contains(renderedOff, "<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"inspect_file\"") {
		t.Errorf("tools definition not found in render:\n%s", renderedOff)
	}

	// Verify user turn
	if !strings.Contains(renderedOff, "<|im_start|>user\nInspect internal/agent/main.go<|im_end|>\n") {
		t.Errorf("user turn not formatted as expected in render:\n%s", renderedOff)
	}

	// Verify assistant turn with reasoning and tool call
	wantAssistantTurn := "<|im_start|>assistant\n<think>\nChecking file path and line count.\n</think>\n\nI will inspect the file.\n\n<tool_call>\n<function=inspect_file>\n<parameter=lines>\n50\n</parameter>\n<parameter=path>\ninternal/agent/main.go\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
	if !strings.Contains(renderedOff, wantAssistantTurn) {
		t.Errorf("assistant turn with reasoning and tool call missing or mismatched:\n%s", renderedOff)
	}

	// Verify tool response (<tool_response>)
	wantToolResponse := "<|im_start|>user\n<tool_response>\npackage main\n\nfunc main() {}\n</tool_response><|im_end|>\n"
	if !strings.Contains(renderedOff, wantToolResponse) {
		t.Errorf("tool response turn missing or mismatched:\n%s", renderedOff)
	}

	// Verify multi-turn continuation
	wantContinuation := "<|im_start|>assistant\n<think>\n\n</think>\n\nThe file is a Go entry point.<|im_end|>\n<|im_start|>user\nNow summarize the architecture.<|im_end|>\n"
	if !strings.Contains(renderedOff, wantContinuation) {
		t.Errorf("multi-turn continuation missing or mismatched:\n%s", renderedOff)
	}

	// 2. Thinking on: assistant generation prompt ends with <think>\n
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "1")
	renderedOn := renderInKernelChatMLTools(messages, tools, cfg)

	if !strings.HasSuffix(renderedOn, "<|im_start|>assistant\n"+qwenThinkAssistantSeed) {
		t.Fatalf("thinking-on render must end with %q, got tail:\n%q", qwenThinkAssistantSeed, renderedOn)
	}
}

func TestQwen38JSONGrammarAndRequestConstraints(t *testing.T) {
	_, cfg := loadQwen38GGUFFixture(t)

	tools := []ToolDef{{
		Type: "function",
		Function: ToolDefFunction{
			Name:        "record_metric",
			Description: "Record a diagnostic metric.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"key":{"type":"string"},"value":{"type":"number"}},"required":["key","value"]}`),
		},
	}}

	messages := []Message{
		{Role: RoleSystem, Content: "You are Qwen3.8."},
		{Role: RoleUser, Content: "Emit the diagnostic result."},
	}

	// 1. response_format with json_object
	t.Run("response_format_json_object", func(t *testing.T) {
		rf := json.RawMessage(`{"type":"json_object"}`)
		got := renderInKernelChatMLRequest(messages, tools, cfg, rf, nil)
		const want = "Return only one valid JSON object. Do not use Markdown fences or explanatory prose."
		if !strings.Contains(got, want) {
			t.Errorf("json_object constraint missing %q:\n%s", want, got)
		}
	})

	// 2. response_format with json_schema
	t.Run("response_format_json_schema", func(t *testing.T) {
		rf := json.RawMessage(`{"type":"json_schema","json_schema":{"name":"status_report","schema":{"type":"object","properties":{"status":{"type":"string"}},"required":["status"]}}}`)
		got := renderInKernelChatMLRequest(messages, tools, cfg, rf, nil)
		const want = "Return only one valid JSON object matching this schema exactly. Do not use Markdown fences or explanatory prose. JSON schema: "
		if !strings.Contains(got, want) {
			t.Errorf("json_schema constraint missing %q:\n%s", want, got)
		}
	})

	// 3. Forced tool_choice
	t.Run("forced_tool_choice", func(t *testing.T) {
		tc := json.RawMessage(`{"type":"function","function":{"name":"record_metric"}}`)
		got := renderInKernelChatMLRequest(messages, tools, cfg, nil, tc)
		for _, want := range []string{
			"Return only one valid JSON object matching this schema exactly.",
			`"const":"record_metric"`,
			`"required":["name","arguments"]`,
		} {
			if !strings.Contains(got, want) {
				t.Errorf("forced tool_choice missing %q in render:\n%s", want, got)
			}
		}
	})

	// 4. Prefix stability across requests sharing system prompt and tool definitions
	t.Run("prefix_stability", func(t *testing.T) {
		reqA := []Message{
			{Role: RoleSystem, Content: "You are Qwen3.8."},
			{Role: RoleUser, Content: "First inquiry regarding system state."},
		}
		reqB := []Message{
			{Role: RoleSystem, Content: "You are Qwen3.8."},
			{Role: RoleUser, Content: "Subsequent completely different query."},
		}
		t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "")
		renderA := renderInKernelChatMLTools(reqA, tools, cfg)
		renderB := renderInKernelChatMLTools(reqB, tools, cfg)

		endSys := strings.Index(renderA, "<|im_end|>\n")
		if endSys < 0 {
			t.Fatalf("no system turn terminator found in renderA:\n%s", renderA)
		}
		endSys += len("<|im_end|>\n")

		sysPrefixA := renderA[:endSys]
		if !strings.HasPrefix(renderB, sysPrefixA) {
			t.Fatalf("prefix stability violated: shared system+tools block differs across requests.\nPrefix:\n%q\nRenderB:\n%q", sysPrefixA, renderB)
		}
	})
}

func TestQwen38OutputToolCallExtractionAndReasoningSplit(t *testing.T) {
	raw := `<think>
Evaluating workspace health and checking file sizes.
</think>I will inspect the specified file.
<tool_call>
<function=inspect_file>
<parameter=path>
internal/agent/main.go
</parameter>
<parameter=lines>
25
</parameter>
</function>
</tool_call>`

	// 1. splitReasoning separates <think>...</think> from visible content
	reasoning, visible := splitReasoning(raw)
	const wantReasoning = "Evaluating workspace health and checking file sizes."
	if reasoning != wantReasoning {
		t.Errorf("splitReasoning reasoning mismatch: got %q, want %q", reasoning, wantReasoning)
	}

	wantVisible := "I will inspect the specified file.\n<tool_call>\n<function=inspect_file>\n<parameter=path>\ninternal/agent/main.go\n</parameter>\n<parameter=lines>\n25\n</parameter>\n</function>\n</tool_call>"
	if visible != wantVisible {
		t.Errorf("splitReasoning visible content mismatch: got %q, want %q", visible, wantVisible)
	}

	// 2. LiftTextToolCalls lifts structured tool call
	lifted := LiftTextToolCalls(Message{Role: RoleAssistant, Content: visible})
	if len(lifted.ToolCalls) != 1 {
		t.Fatalf("expected exactly 1 lifted tool call, got %d (%+v)", len(lifted.ToolCalls), lifted.ToolCalls)
	}

	tc := lifted.ToolCalls[0]
	if tc.Function.Name != "inspect_file" {
		t.Errorf("lifted tool call name = %q, want inspect_file", tc.Function.Name)
	}
	if !strings.Contains(tc.Function.Arguments, `"path":"internal/agent/main.go"`) || !strings.Contains(tc.Function.Arguments, `"lines":25`) {
		t.Errorf("lifted tool call arguments mismatch: %s", tc.Function.Arguments)
	}
	if strings.TrimSpace(lifted.Content) != "I will inspect the specified file." {
		t.Errorf("lifted content retained tool_call markup: %q", lifted.Content)
	}

	// 3. Truncated and malformed tool calls are handled fail-closed
	malformedCases := []struct {
		name    string
		content string
	}{
		{
			name:    "truncated_function_close_tag",
			content: "<tool_call>\n<function=inspect_file>\n<parameter=path>\ninternal/agent/main.go\n</parameter>\n",
		},
		{
			name:    "truncated_mid_parameter",
			content: "<tool_call>\n<function=inspect_file>\n<parameter=path>\ninternal/agent/main.go",
		},
		{
			name:    "unclosed_parameter_tag",
			content: "<tool_call>\n<function=inspect_file>\n<parameter=path>internal/agent/main.go\n</function>\n</tool_call>",
		},
		{
			name:    "empty_function_name",
			content: "<tool_call>\n<function=>\n<parameter=path>\ninternal/agent/main.go\n</parameter>\n</function>\n</tool_call>",
		},
	}

	for _, mc := range malformedCases {
		t.Run(mc.name, func(t *testing.T) {
			m := LiftTextToolCalls(Message{Role: RoleAssistant, Content: mc.content})
			if len(m.ToolCalls) != 0 {
				t.Fatalf("malformed tool call must not be lifted (fail-closed); got %d calls: %+v", len(m.ToolCalls), m.ToolCalls)
			}
		})
	}
}

func TestQwen38StopTokens(t *testing.T) {
	_, cfg := loadQwen38GGUFFixture(t)

	// Construct a tokenizer declaring the Qwen special tokens <|im_end|> and <|endoftext|>.
	const tokenizerJSON = `{
	  "model": {
	    "type": "BPE",
	    "vocab": {
	      "Hello": 1,
	      "<|endoftext|>": 248044,
	      "<|im_end|>": 248046
	    },
	    "merges": []
	  },
	  "decoder": {"type": "ByteLevel"},
	  "added_tokens": [
	    {"id": 248044, "content": "<|endoftext|>", "special": true},
	    {"id": 248046, "content": "<|im_end|>", "special": true}
	  ]
	}`

	tok, err := tokenizer.ParseJSON([]byte(tokenizerJSON))
	if err != nil {
		t.Fatalf("tokenizer.ParseJSON: %v", err)
	}

	stops := StopIDs(tok, cfg)

	if !stops[248046] {
		t.Errorf("StopIDs missing 248046 (<|im_end|>): %v", stops)
	}
	if !stops[248044] {
		t.Errorf("StopIDs missing 248044 (<|endoftext|>): %v", stops)
	}
}

// TestQwen38UDQ2KXLChatTemplateParity satisfies the exact issue #11948 witness selector.
func TestQwen38UDQ2KXLChatTemplateParity(t *testing.T) {
	t.Run("HeaderIdentityAndConfig", TestQwen38GGUFHeaderIdentityAndConfig)
	t.Run("ChatTemplateExtractedFromGGUF", TestQwen38ChatTemplateExtractedFromGGUF)
	t.Run("ChatTemplateAndToolSerializationParity", TestQwen38ChatTemplateAndToolSerializationParity)
	t.Run("JSONGrammarAndRequestConstraints", TestQwen38JSONGrammarAndRequestConstraints)
	t.Run("OutputToolCallExtractionAndReasoningSplit", TestQwen38OutputToolCallExtractionAndReasoningSplit)
	t.Run("StopTokens", TestQwen38StopTokens)
}
