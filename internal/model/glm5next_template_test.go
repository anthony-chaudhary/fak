package model

import (
	"strings"
	"testing"
)

func TestFormatGLM5NextPrompt(t *testing.T) {
	msgs := []GLM5NextMessage{
		{Role: "system", Content: "You are a helpful coding assistant."},
		{Role: "user", Content: "Write hello world in Go."},
		{Role: "assistant", Thinking: "The user asks for a simple Go program.", Content: "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"Hello, World!\")\n}"},
		{Role: "user", Content: "Now run tests."},
		{Role: "observation", Content: "PASS: all tests green"},
	}

	prompt := FormatGLM5NextPrompt(msgs, true)

	// Must begin with [gMASK]<sop>
	if !strings.HasPrefix(prompt, "[gMASK]<sop>") {
		t.Fatalf("prompt must start with [gMASK]<sop>, got: %q", prompt[:20])
	}

	// Must contain role tokens
	for _, expected := range []string{"<|system|>", "<|user|>", "<|assistant|>", "<|thought|>", "<|observation|>"} {
		if !strings.Contains(prompt, expected) {
			t.Fatalf("prompt missing expected token %s: %s", expected, prompt)
		}
	}

	// Must end with open <|assistant|>\n
	if !strings.HasSuffix(prompt, "<|assistant|>\n") {
		t.Fatalf("prompt with addAssistantPrefix must end with <|assistant|>\\n, got suffix: %q", prompt[len(prompt)-20:])
	}
}

func TestParseGLM5NextAssistantResponse(t *testing.T) {
	t.Run("with closed thinking", func(t *testing.T) {
		raw := "<|thought|>\nThinking step 1\nStep 2\n<|thought|>\nFinal answer: 42"
		thinking, content := ParseGLM5NextAssistantResponse(raw)
		if thinking != "Thinking step 1\nStep 2" {
			t.Fatalf("unexpected thinking: %q", thinking)
		}
		if content != "Final answer: 42" {
			t.Fatalf("unexpected content: %q", content)
		}
	})

	t.Run("without thinking", func(t *testing.T) {
		raw := "Just the direct answer here."
		thinking, content := ParseGLM5NextAssistantResponse(raw)
		if thinking != "" {
			t.Fatalf("expected empty thinking, got %q", thinking)
		}
		if content != "Just the direct answer here." {
			t.Fatalf("unexpected content: %q", content)
		}
	})

	t.Run("unclosed thinking", func(t *testing.T) {
		raw := "<|thought|>\nStill generating thought..."
		thinking, content := ParseGLM5NextAssistantResponse(raw)
		if thinking != "Still generating thought..." {
			t.Fatalf("unexpected thinking: %q", thinking)
		}
		if content != "" {
			t.Fatalf("expected empty content, got %q", content)
		}
	})
}
