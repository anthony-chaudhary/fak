package main

import (
	"bytes"
	"strings"
	"testing"
)

func newPromptTestServeFlags() *serveFlags {
	_, sf := newServeFlagSet()
	return sf
}

func TestResolveServeModelOrPrompt_InteractivePromptSuccess(t *testing.T) {
	sf := newPromptTestServeFlags()
	var in bytes.Buffer
	var out bytes.Buffer
	in.WriteString("smollm2\n")

	err := resolveServeModelOrPrompt(sf, nil, &in, &out, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if *sf.ggufPath != "smollm2" {
		t.Fatalf("expected ggufPath to be %q, got %q", "smollm2", *sf.ggufPath)
	}
	if !strings.Contains(out.String(), "Enter model path or alias") {
		t.Fatalf("expected prompt in out, got %q", out.String())
	}
}

func TestResolveServeModelOrPrompt_InteractivePromptQuotedPath(t *testing.T) {
	sf := newPromptTestServeFlags()
	var in bytes.Buffer
	var out bytes.Buffer
	in.WriteString("\"models/my model.gguf\"\n")

	err := resolveServeModelOrPrompt(sf, nil, &in, &out, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if *sf.ggufPath != "models/my model.gguf" {
		t.Fatalf("expected ggufPath to be %q, got %q", "models/my model.gguf", *sf.ggufPath)
	}
}

func TestResolveServeModelOrPrompt_InteractivePromptURL(t *testing.T) {
	sf := newPromptTestServeFlags()
	var in bytes.Buffer
	var out bytes.Buffer
	in.WriteString("http://127.0.0.1:11434/v1\n")

	err := resolveServeModelOrPrompt(sf, nil, &in, &out, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if *sf.baseURL != "http://127.0.0.1:11434/v1" {
		t.Fatalf("expected baseURL to be %q, got %q", "http://127.0.0.1:11434/v1", *sf.baseURL)
	}
}

func TestResolveServeModelOrPrompt_InteractivePromptMock(t *testing.T) {
	sf := newPromptTestServeFlags()
	var in bytes.Buffer
	var out bytes.Buffer
	in.WriteString("mock\n")

	err := resolveServeModelOrPrompt(sf, nil, &in, &out, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !*sf.mock {
		t.Fatal("expected mock to be true")
	}
}

func TestResolveServeModelOrPrompt_InteractivePromptEmptyFails(t *testing.T) {
	sf := newPromptTestServeFlags()
	var in bytes.Buffer
	var out bytes.Buffer
	in.WriteString("   \n")

	err := resolveServeModelOrPrompt(sf, nil, &in, &out, true)
	if err == nil {
		t.Fatal("expected error on empty input, got nil")
	}
	if !strings.Contains(err.Error(), "no model provided") {
		t.Fatalf("expected 'no model provided' error, got %v", err)
	}
}

func TestResolveServeModelOrPrompt_NonInteractiveFailsWhenNoModel(t *testing.T) {
	sf := newPromptTestServeFlags()
	var in bytes.Buffer
	var out bytes.Buffer

	err := resolveServeModelOrPrompt(sf, nil, &in, &out, false)
	if err == nil {
		t.Fatal("expected error on non-interactive with no model, got nil")
	}
	if !strings.Contains(err.Error(), "no model provided") {
		t.Fatalf("expected 'no model provided' error, got %v", err)
	}
}

func TestResolveServeModelOrPrompt_ExplicitMockAllowedNonInteractive(t *testing.T) {
	sf := newPromptTestServeFlags()
	*sf.mock = true

	err := resolveServeModelOrPrompt(sf, nil, nil, nil, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestResolveServeModelOrPrompt_ExplicitModelMockAllowedNonInteractive(t *testing.T) {
	sf := newPromptTestServeFlags()
	explicit := map[string]bool{"model": true}
	*sf.model = "mock"

	err := resolveServeModelOrPrompt(sf, explicit, nil, nil, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !*sf.mock {
		t.Fatal("expected mock to be set true")
	}
}

func TestResolveServeModelOrPrompt_ExplicitModelTakesGGUF(t *testing.T) {
	sf := newPromptTestServeFlags()
	explicit := map[string]bool{"model": true}
	*sf.model = "smollm2"

	err := resolveServeModelOrPrompt(sf, explicit, nil, nil, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if *sf.ggufPath != "smollm2" {
		t.Fatalf("expected ggufPath = %q, got %q", "smollm2", *sf.ggufPath)
	}
}

func TestParseServeArgsModelGGUF(t *testing.T) {
	fs, sf := newServeFlagSet()
	argv := []string{"model", "--gguf", "qwen38:27b-q4"}
	if err := parseServeArgs(fs, sf, argv); err != nil {
		t.Fatalf("parseServeArgs failed: %v", err)
	}
	if *sf.ggufPath != "qwen38:27b-q4" {
		t.Fatalf("ggufPath = %q, want %q", *sf.ggufPath, "qwen38:27b-q4")
	}
}

func TestParseServeArgsModelPositional(t *testing.T) {
	fs, sf := newServeFlagSet()
	argv := []string{"model", "qwen38:27b-q4"}
	if err := parseServeArgs(fs, sf, argv); err != nil {
		t.Fatalf("parseServeArgs failed: %v", err)
	}
	if *sf.ggufPath != "qwen38:27b-q4" {
		t.Fatalf("ggufPath = %q, want %q", *sf.ggufPath, "qwen38:27b-q4")
	}
}

func TestParseServeArgsBarePositional(t *testing.T) {
	fs, sf := newServeFlagSet()
	argv := []string{"qwen38:27b-q4"}
	if err := parseServeArgs(fs, sf, argv); err != nil {
		t.Fatalf("parseServeArgs failed: %v", err)
	}
	if *sf.ggufPath != "qwen38:27b-q4" {
		t.Fatalf("ggufPath = %q, want %q", *sf.ggufPath, "qwen38:27b-q4")
	}
}

func TestParseServeArgsPositionalWithFlags(t *testing.T) {
	fs, sf := newServeFlagSet()
	argv := []string{"qwen38:27b-q4", "--addr", "127.0.0.1:9090"}
	if err := parseServeArgs(fs, sf, argv); err != nil {
		t.Fatalf("parseServeArgs failed: %v", err)
	}
	if *sf.ggufPath != "qwen38:27b-q4" {
		t.Fatalf("ggufPath = %q, want %q", *sf.ggufPath, "qwen38:27b-q4")
	}
	if *sf.addr != "127.0.0.1:9090" {
		t.Fatalf("addr = %q, want %q", *sf.addr, "127.0.0.1:9090")
	}

	fs2, sf2 := newServeFlagSet()
	argv2 := []string{"--addr", "127.0.0.1:9090", "qwen38:27b-q4"}
	if err := parseServeArgs(fs2, sf2, argv2); err != nil {
		t.Fatalf("parseServeArgs failed: %v", err)
	}
	if *sf2.ggufPath != "qwen38:27b-q4" {
		t.Fatalf("ggufPath = %q, want %q", *sf2.ggufPath, "qwen38:27b-q4")
	}
	if *sf2.addr != "127.0.0.1:9090" {
		t.Fatalf("addr = %q, want %q", *sf2.addr, "127.0.0.1:9090")
	}
}

func TestParseServeArgsModelFlagWithoutGGUF(t *testing.T) {
	fs, sf := newServeFlagSet()
	argv := []string{"--model", "qwen38:27b-q4"}
	if err := parseServeArgs(fs, sf, argv); err != nil {
		t.Fatalf("parseServeArgs failed: %v", err)
	}
	if *sf.ggufPath != "qwen38:27b-q4" {
		t.Fatalf("ggufPath = %q, want %q", *sf.ggufPath, "qwen38:27b-q4")
	}
}

func TestParseServeArgsRejectsUnexpectedArgs(t *testing.T) {
	fs, sf := newServeFlagSet()
	argv := []string{"foo", "bar"}
	err := parseServeArgs(fs, sf, argv)
	if err == nil || !strings.Contains(err.Error(), "unexpected argument") {
		t.Fatalf("expected unexpected argument error, got %v", err)
	}

	fs2, sf2 := newServeFlagSet()
	argv2 := []string{"model", "--gguf", "qwen38:27b-q4", "stray"}
	err2 := parseServeArgs(fs2, sf2, argv2)
	if err2 == nil {
		t.Fatal("expected error for stray argument after --gguf, got nil")
	}
}

func TestParseServeArgs_PositionalMock(t *testing.T) {
	fs, sf := newServeFlagSet()
	err := parseServeArgs(fs, sf, []string{"mock"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !*sf.mock {
		t.Fatal("expected mock to be set true")
	}
}

func TestParseServeArgs_PositionalURL(t *testing.T) {
	fs, sf := newServeFlagSet()
	err := parseServeArgs(fs, sf, []string{"https://api.openai.com/v1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if *sf.baseURL != "https://api.openai.com/v1" {
		t.Fatalf("expected baseURL = %q, got %q", "https://api.openai.com/v1", *sf.baseURL)
	}
}

func TestResolveServeModelOrPrompt_EarlyExitsIgnored(t *testing.T) {
	sf := newPromptTestServeFlags()
	*sf.stdio = true

	// stdio mode with no model should return nil (no prompt, falls back to MCP mock)
	err := resolveServeModelOrPrompt(sf, nil, nil, nil, false)
	if err != nil {
		t.Fatalf("unexpected error for stdio: %v", err)
	}

	sf2 := newPromptTestServeFlags()
	*sf2.printEffectiveConfig = true
	err = resolveServeModelOrPrompt(sf2, nil, nil, nil, false)
	if err != nil {
		t.Fatalf("unexpected error for printEffectiveConfig: %v", err)
	}
}
