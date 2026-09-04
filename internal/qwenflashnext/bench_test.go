package qwenflashnext

import (
	"fmt"
	"testing"
)

func BenchmarkRender(b *testing.B) {
	messages := []Message{
		{Role: "system", Content: "You are a helpful coding assistant adhering strictly to instructions."},
		{Role: "user", Content: "Optimize this algorithm for bounded memory consumption."},
		{Role: "assistant", ReasoningContent: "Analyzing memory limits and computational bounds.", Content: "Here is the memory-bounded solution."},
	}
	opts := RenderOptions{
		PreserveThinking: true,
		EnableThinking:   true,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := Render(messages, opts)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRenderSubtests(b *testing.B) {
	b.Run("PlainConversation", func(b *testing.B) {
		msgs := []Message{
			{Role: "user", Content: "Hello world."},
			{Role: "assistant", Content: "Hello! How can I assist you today?"},
		}
		opts := RenderOptions{}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := Render(msgs, opts); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("ToolInteractionFlow", func(b *testing.B) {
		msgs := []Message{
			{Role: "user", Content: "Check server status."},
			{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{
						Name: "query_metrics",
						Arguments: map[string]any{
							"service": "gateway",
							"window":  "5m",
						},
					},
				},
			},
			{Role: "tool", Content: `{"status":"ok","latency_ms":12}`},
			{Role: "assistant", Content: "Server is healthy with 12ms latency."},
		}
		opts := RenderOptions{}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := Render(msgs, opts); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkParseResponse(b *testing.B) {
	raw := "<think>\nCarefully analyzing assumptions and evaluating execution bounds.\n</think>\n\nProceeding with implementation.<|im_end|>"

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		parsed, err := ParseResponse(raw)
		if err != nil {
			b.Fatal(err)
		}
		if !parsed.Stopped {
			b.Fatal("expected stopped response")
		}
	}
}

func BenchmarkParseResponseWithToolCalls(b *testing.B) {
	raw := fmt.Sprintf(
		"%sassistant\n<think>\nEvaluating system state.\n</think>\n\nExecuting requested check.\n\n<tool_call>\n<function=inspect_node>\n<parameter=node_id>\nworker-42\n</parameter>\n<parameter=verbose>\ntrue\n</parameter>\n</function>\n</tool_call>%s",
		IMStart,
		IMEnd,
	)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		parsed, err := ParseResponse(raw)
		if err != nil {
			b.Fatal(err)
		}
		if len(parsed.ToolCalls) == 0 {
			b.Fatal("expected tool calls")
		}
	}
}
