package schemaadapter

import (
	"encoding/json"
	"testing"
)

func BenchmarkSchemaAdapter(b *testing.B) {
	raw := json.RawMessage(`{
		"type": "object",
		"properties": {
			"query": {"type": "string", "description": "search query"},
			"limit": {"type": "integer", "description": "max results"},
			"filters": {
				"type": "object",
				"properties": {
					"tag": {"type": "string"}
				},
				"required": ["tag"]
			}
		},
		"required": ["query"]
	}`)

	dialects := []Dialect{
		DialectGemini,
		DialectOpenAI,
		DialectOpenAIStrict,
		DialectAnthropic,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, d := range dialects {
			if _, err := Normalize(raw, d); err != nil {
				b.Fatalf("Normalize failed for %s: %v", d, err)
			}
		}
	}
}
