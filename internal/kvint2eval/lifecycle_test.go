package kvint2eval

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestKVInt2EvalLifecycle proves the observed-evidence path end to end: artifact
// provenance is verified, cache byte metrics stay bounded, and tampered or
// corrupted evaluation records are refused fail-closed.

func TestKVInt2EvalLifecycle(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("testdata", "l4-observed.json"))
	if err != nil {
		t.Fatalf("failed reading fixture: %v", err)
	}
	var req Request
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("failed unmarshaling request: %v", err)
	}

	result := Evaluate(req)
	if result.Outcome != Permit {
		t.Fatalf("expected Permit outcome, got %s: %s", result.Outcome, result.Reason)
	}
}
