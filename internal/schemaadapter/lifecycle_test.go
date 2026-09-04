package schemaadapter

import (
	"encoding/json"
	"testing"
)

// Invariant: Schema adaptation must convert standard JSON schemas to model-provider specific formats without losing properties.
// Guard: ToGemini rejects nil or invalid JSON inputs fail-closed.

func TestSchemaAdapterLifecycle(t *testing.T) {
	t.Parallel()

	input := json.RawMessage(`{"type":"object","properties":{"test":{"type":"string"}}}`)
	out, err := ToGemini(input)
	if err != nil {
		t.Fatalf("ToGemini failed: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("failed unmarshaling converted schema: %v", err)
	}
	if m["type"] != "OBJECT" {
		t.Fatalf("expected uppercase OBJECT type for Gemini, got %v", m["type"])
	}
}
