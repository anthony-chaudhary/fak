package cubicquanteval

import (
	"os"
	"testing"
)

// Invariant: Cubic quantization evaluation must verify artifact provenance and bound RMSE reconstruction errors.
// Guard: Evaluate refuses evaluations with mismatched provenance hashes or invalid fixture schemas.

func TestCubicQuantEvalLifecycle(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile("testdata/evaluation-v1.json")
	if err != nil {
		t.Fatalf("failed reading fixture: %v", err)
	}

	result := Evaluate(Request{Scope: ScopeReconstruction, FixtureJSON: raw})
	if result.Outcome != Supported {
		t.Fatalf("expected Supported outcome, got %s: %s", result.Outcome, result.Reason)
	}
	if len(result.Rows) == 0 {
		t.Fatal("expected non-empty evaluation rows")
	}
}
