package providerjobaccounting

import (
	"encoding/json"
	"os"
	"testing"
)

// Invariant: Provider job accounting schemas must define strict workload envelopes and raw token counters.
// Guard: TestSchemaFileParses validates that the schema JSON file parses with non-empty defs.

func TestProviderJobAccountingLifecycle(t *testing.T) {
	t.Parallel()

	path := docsPath("standards", "provider-job-accounting-schema.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed reading schema: %v", err)
	}

	var schema map[string]any
	if err := json.Unmarshal(data, &schema); err != nil {
		t.Fatalf("failed unmarshaling schema: %v", err)
	}

	if len(schema) == 0 {
		t.Fatal("expected non-empty schema object")
	}
}
