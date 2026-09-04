package gateway

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentopt"
)

func TestModulateAnthropicThinking(t *testing.T) {
	// Initial Anthropic request with thinking enabled
	raw := []byte(`{"model":"claude-3-7-sonnet-20250219","max_tokens":1000,"thinking":{"type":"enabled","budget_tokens":2048},"messages":[{"role":"user","content":"read file"}]}`)

	// Modulate to none (disable thinking)
	modNone, ok := ModulateAnthropicThinking(raw, agentopt.EffortNone)
	if !ok {
		t.Fatalf("expected modification to succeed")
	}

	var parsedNone map[string]any
	if err := json.Unmarshal(modNone, &parsedNone); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	thNone, _ := parsedNone["thinking"].(map[string]any)
	if thNone["type"] != "disabled" {
		t.Errorf("expected thinking.type=disabled, got %v", thNone["type"])
	}

	// Verify messages prefix was preserved byte-identical
	if !bytes.Contains(modNone, []byte(`"messages":[{"role":"user","content":"read file"}]`)) {
		t.Errorf("messages prefix not preserved in modNone: %s", modNone)
	}

	// Modulate to low
	modLow, ok := ModulateAnthropicThinking(raw, agentopt.EffortLow)
	if !ok {
		t.Fatalf("expected modification to succeed")
	}
	var parsedLow map[string]any
	if err := json.Unmarshal(modLow, &parsedLow); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	thLow, _ := parsedLow["thinking"].(map[string]any)
	if thLow["type"] != "enabled" || thLow["budget_tokens"] != float64(256) {
		t.Errorf("expected thinking budget 256, got %v", thLow)
	}
}

func TestModulateOpenAIThinking(t *testing.T) {
	raw := []byte(`{"model":"o3-mini","messages":[{"role":"user","content":"hello"}],"reasoning_effort":"high"}`)

	modNone, ok := ModulateOpenAIThinking(raw, agentopt.EffortNone)
	if !ok {
		t.Fatalf("expected modification to succeed")
	}

	var parsed map[string]any
	if err := json.Unmarshal(modNone, &parsed); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	if parsed["reasoning_effort"] != "low" {
		t.Errorf("expected reasoning_effort low, got %v", parsed["reasoning_effort"])
	}

	// Verify messages prefix was preserved
	if !bytes.Contains(modNone, []byte(`"messages":[{"role":"user","content":"hello"}]`)) {
		t.Errorf("messages prefix not preserved: %s", modNone)
	}
}

func TestModulateGeminiThinking(t *testing.T) {
	raw := []byte(`{"contents":[{"role":"user","parts":[{"text":"list"}]}],"generationConfig":{"temperature":0.2}}`)

	modLow, ok := ModulateGeminiThinking(raw, agentopt.EffortLow)
	if !ok {
		t.Fatalf("expected modification to succeed")
	}

	var parsed map[string]any
	if err := json.Unmarshal(modLow, &parsed); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	gc, _ := parsed["generationConfig"].(map[string]any)
	tc, _ := gc["thinkingConfig"].(map[string]any)
	if tc["thinkingBudget"] != float64(256) {
		t.Errorf("expected thinkingBudget 256, got %v", tc["thinkingBudget"])
	}

	// Verify contents prefix was preserved
	if !bytes.Contains(modLow, []byte(`"contents":[{"role":"user","parts":[{"text":"list"}]}]`)) {
		t.Errorf("contents prefix not preserved: %s", modLow)
	}
}
