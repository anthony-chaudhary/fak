package tb4bench

import (
	"context"
	"testing"
)

func TestInKernelDeterministicDecoding(t *testing.T) {
	adapter, err := NewInKernelModelAdapter("", "")
	if err != nil {
		t.Fatalf("failed to initialize in-kernel adapter: %v", err)
	}

	ctx := context.Background()
	req := CompletionRequest{
		Messages: []Message{
			{Role: "system", Content: "You are a coding assistant."},
			{Role: "user", Content: "Inspect the repository and list files."},
		},
		Determinism: DefaultDeterminismEnvelope(),
	}

	// First execution
	resp1, err := adapter.Complete(ctx, req)
	if err != nil {
		t.Fatalf("first completion failed: %v", err)
	}

	// Second execution with identical prompt and seed
	resp2, err := adapter.Complete(ctx, req)
	if err != nil {
		t.Fatalf("second completion failed: %v", err)
	}

	if resp1.Text != resp2.Text {
		t.Fatalf("deterministic decoding failed: non-identical text:\nRun 1: %q\nRun 2: %q", resp1.Text, resp2.Text)
	}
	if resp1.CompletionTokens != resp2.CompletionTokens {
		t.Errorf("token count mismatch: %d vs %d", resp1.CompletionTokens, resp2.CompletionTokens)
	}

	// Verify non-zero temperature is refused under strict greedy contract
	badReq := req
	badReq.Determinism.Temperature = 0.5
	if _, err := adapter.Complete(ctx, badReq); err == nil {
		t.Errorf("expected error when temperature != 0.0, got nil")
	}

	// Verify telemetry tracking
	telemetry := adapter.Telemetry()
	if telemetry.PromptTokens <= 0 || telemetry.CompletionTokens <= 0 {
		t.Errorf("expected positive token telemetry, got %+v", telemetry)
	}
}

func TestParseToolCalls(t *testing.T) {
	raw := "Thinking...\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"cmd\": \"ls -la\"}}\n</tool_call>\nDone."
	calls, clean := ParseToolCalls(raw)
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Name != "bash" {
		t.Errorf("expected bash tool, got %s", calls[0].Name)
	}
	if clean != "Thinking...\n\nDone." {
		t.Errorf("expected cleaned text, got %q", clean)
	}
}
