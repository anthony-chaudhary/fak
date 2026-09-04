package qwenflashnext

import (
	"testing"
)

// BenchmarkEvaluatePrompt benchmarks rendering representative multi-turn conversations
// spanning system instructions, user queries, reasoning traces, and recipient tool calls.
func BenchmarkEvaluatePrompt(b *testing.B) {
	messages := []Message{
		{
			Role:    "system",
			Content: "You are an autonomous agent kernel with deterministic verification capabilities.",
		},
		{
			Role:    "user",
			Content: "Inspect system metrics and dispatch appropriate remediation worker.",
		},
		{
			Role:             "assistant",
			ReasoningContent: "Checking CPU and memory pressure before selecting execution strategy. Need telemetry probe.",
			Content:          "I will query runtime telemetry to verify system thresholds.",
			ToolCalls: []ToolCall{
				{
					Name: "system_telemetry",
					Arguments: map[string]any{
						"subsystem": "scheduler",
						"window_ms": 5000,
					},
				},
			},
		},
		{
			Role:    "tool",
			Content: `{"status":"nominal","utilization_pct":42.5}`,
		},
	}

	opts := RenderOptions{
		AddGenerationPrompt: true,
		EnableThinking:      true,
		PreserveThinking:    true,
		ReasoningEffort:     "xhigh",
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		rendered, err := Render(messages, opts)
		if err != nil {
			b.Fatalf("Render failed: %v", err)
		}
		if len(rendered) == 0 {
			b.Fatal("unexpected empty rendered output")
		}
	}
}

// BenchmarkParseResponse benchmarks decomposing generated raw model outputs into structured channels.
func BenchmarkParseResponse(b *testing.B) {
	payload := "<|im_start|>assistant\n<think>\nEvaluating telemetry metrics across cluster.\n</think>\n\nTelemetry verified.<tool_call>\n<function=dispatch_worker>\n<parameter=worker_id>\nworker-42\n</parameter>\n</function>\n</tool_call><|im_end|>\n"

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		parsed, err := ParseResponse(payload)
		if err != nil {
			b.Fatalf("ParseResponse failed: %v", err)
		}
		if len(parsed.ToolCalls) == 0 {
			b.Fatal("expected at least one parsed tool call")
		}
	}
}
