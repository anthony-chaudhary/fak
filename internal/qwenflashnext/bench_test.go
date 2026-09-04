package qwenflashnext_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwenflashnext"
)

func BenchmarkRender(b *testing.B) {
	messages := []qwenflashnext.Message{
		{
			Role:    "system",
			Content: "You are a helpful coding assistant operating in a repository.",
		},
		{
			Role:    "user",
			Content: "Please refactor the database connector and implement retry logic.",
		},
		{
			Role:             "assistant",
			ReasoningContent: "We need to check the interface contract and wrap transient network errors.",
			Content:          "I will examine the current database interface implementation.",
			ToolCalls: []qwenflashnext.ToolCall{
				{
					Name: "read_file",
					Arguments: map[string]any{
						"path":  "internal/db/connector.go",
						"limit": 50,
					},
				},
			},
		},
		{
			Role:    "tool",
			Content: "package db\n\ntype Connector struct{}\n",
		},
	}
	opts := qwenflashnext.RenderOptions{
		AddGenerationPrompt: true,
		EnableThinking:      true,
		ReasoningEffort:     "xhigh",
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := qwenflashnext.Render(messages, opts)
		if err != nil {
			b.Fatalf("Render failed: %v", err)
		}
	}
}

func BenchmarkParseResponse(b *testing.B) {
	raw := `<|im_start|>assistant
<think>
Analyzing the issue requirements and verifying that retry logic handles timeout errors.
Confirming parameter values before emitting the tool call.
</think>

I have updated the connector to support automatic retry backoff.
<tool_call>
<function=edit_file>
<parameter=content>
func (c *Connector) Connect() error { return nil }
</parameter>
<parameter=path>
internal/db/connector.go
</parameter>
</function>
</tool_call><|im_end|>
`

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := qwenflashnext.ParseResponse(raw)
		if err != nil {
			b.Fatalf("ParseResponse failed: %v", err)
		}
	}
}
