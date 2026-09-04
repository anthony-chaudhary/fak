package qwenflashnext

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestPinnedFixtureProvenance(t *testing.T) {
	data, err := os.ReadFile("testdata/chat_template.jinja")
	if err != nil {
		t.Fatal(err)
	}
	data = bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
	sum := sha256.Sum256(data)
	if got, want := hex.EncodeToString(sum[:]), "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"; got != want {
		t.Fatalf("template hash = %s, want %s", got, want)
	}
	var tokens struct {
		Revision string `json:"revision"`
		EOS      struct {
			Content string `json:"content"`
			ID      int    `json:"id"`
		} `json:"eos_token"`
	}
	raw, err := os.ReadFile("testdata/special_tokens.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &tokens); err != nil {
		t.Fatal(err)
	}
	if tokens.Revision != "f5d08274bafd880402bd16f5e3e6c514136ec06c" || tokens.EOS.Content != IMEnd || tokens.EOS.ID != StopTokenIDs[0] {
		t.Fatalf("unexpected token provenance: %+v", tokens)
	}
	if !reflect.DeepEqual(StopTokens, []string{"<|im_end|>"}) {
		t.Fatalf("stop tokens: %q", StopTokens)
	}
}

func TestRenderByteExactRolesAndThinking(t *testing.T) {
	messages := []Message{{Role: "system", Content: "  Be exact. "}, {Role: "user", Content: " Question? "}, {Role: "assistant", ReasoningContent: " inspect ", Content: " Answer. "}}
	tests := []struct {
		name string
		opts RenderOptions
		want string
	}{
		{"analysis-final", RenderOptions{PreserveThinking: true}, "<|im_start|>system\nBe exact.<|im_end|>\n<|im_start|>user\nQuestion?<|im_end|>\n<|im_start|>assistant\n<think>\ninspect\n</think>\n\nAnswer.<|im_end|>\n"},
		{"non-thinking", RenderOptions{}, "<|im_start|>system\nBe exact.<|im_end|>\n<|im_start|>user\nQuestion?<|im_end|>\n<|im_start|>assistant\nAnswer.<|im_end|>\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Render(messages, tt.opts)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("bytes:\n%q\nwant:\n%q", got, tt.want)
			}
		})
	}
}

func TestGenerationPromptThinkingModes(t *testing.T) {
	messages := []Message{{Role: "user", Content: "go"}}
	for _, tt := range []struct {
		name     string
		thinking bool
		want     string
	}{
		{"thinking", true, "<|im_start|>user\ngo<|im_end|>\n<|im_start|>assistant\n<think>\n"},
		{"non-thinking", false, "<|im_start|>user\ngo<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Render(messages, RenderOptions{AddGenerationPrompt: true, EnableThinking: tt.thinking})
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("got %q want %q", got, tt.want)
			}
		})
	}
}

func TestRecipientToolCallRenderAndParse(t *testing.T) {
	messages := []Message{{Role: "assistant", ReasoningContent: "Need weather.", Content: "I'll check.", ToolCalls: []ToolCall{{Name: "weather", Arguments: map[string]any{"days": 2, "city": "Paris"}}}}}
	want := "<|im_start|>assistant\n<think>\nNeed weather.\n</think>\n\nI'll check.\n\n<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
	got, err := Render(messages, RenderOptions{PreserveThinking: true})
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
	parsed, err := ParseResponse(got)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Analysis != "Need weather." || parsed.Commentary != "I'll check." || parsed.Final != "" || !parsed.Stopped {
		t.Fatalf("channels: %+v", parsed)
	}
	wantCalls := []ToolCall{{Name: "weather", Arguments: map[string]any{"city": "Paris", "days": float64(2)}}}
	if !reflect.DeepEqual(parsed.ToolCalls, wantCalls) {
		t.Fatalf("calls: %#v want %#v", parsed.ToolCalls, wantCalls)
	}
}

func TestToolOutputReentryByteExact(t *testing.T) {
	messages := []Message{{Role: "user", Content: "weather"}, {Role: "assistant", ToolCalls: []ToolCall{{Name: "weather", Arguments: map[string]any{"city": "Paris"}}}}, {Role: "tool", Content: `{"temp":21}`}, {Role: "tool", Content: `{"unit":"C"}`}, {Role: "assistant", Content: "21 C"}}
	got, err := Render(messages, RenderOptions{})
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>user\nweather<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{\"temp\":21}\n</tool_response>\n<tool_response>\n{\"unit\":\"C\"}\n</tool_response><|im_end|>\n<|im_start|>assistant\n21 C<|im_end|>\n"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestParseFinalAndStopToken(t *testing.T) {
	parsed, err := ParseResponse("<think>\nbrief\n</think>\n\nDone.<|im_end|>ignored")
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Analysis != "brief" || parsed.Final != "Done." || parsed.Commentary != "" || !parsed.Stopped {
		t.Fatalf("parsed: %+v", parsed)
	}
}

func TestPinnedTemplateNamesContractMarkers(t *testing.T) {
	raw, err := os.ReadFile("testdata/chat_template.jinja")
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	for _, marker := range []string{"reasoning_content", "preserve_thinking", "enable_thinking", "<tool_call>", "<tool_response>", "<|im_end|>"} {
		if !strings.Contains(text, marker) {
			t.Errorf("missing %q", marker)
		}
	}
}
