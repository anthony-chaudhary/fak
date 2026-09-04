package kvint2eval

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Invariant: KV INT2 quantization evaluation must verify artifact provenance and bound cache byte metrics.
// Guard: Evaluate refuses tampered or corrupted evaluation records fail-closed.

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
