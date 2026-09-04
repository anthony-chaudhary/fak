package fp4runtime

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Invariant: FP4 runtime negotiation must verify matrix compatibility and delegate to supported runtimes.
// Guard: Negotiate returns OutcomeDelegate for verified hardware configurations.

func TestFP4RuntimeLifecycle(t *testing.T) {
	t.Parallel()

	matrixRaw, err := os.ReadFile(filepath.Join("testdata", "compatibility-matrix-v1.json"))
	if err != nil {
		t.Fatalf("failed reading matrix: %v", err)
	}
	reqRaw, err := os.ReadFile(filepath.Join("testdata", "nvfp4-blackwell-delegate.json"))
	if err != nil {
		t.Fatalf("failed reading request: %v", err)
	}

	var matrix Matrix
	if err := json.Unmarshal(matrixRaw, &matrix); err != nil {
		t.Fatalf("failed unmarshaling matrix: %v", err)
	}
	var req Request
	if err := json.Unmarshal(reqRaw, &req); err != nil {
		t.Fatalf("failed unmarshaling request: %v", err)
	}

	dec := Negotiate(req, matrix)
	if dec.Outcome != OutcomeDelegate {
		t.Fatalf("expected OutcomeDelegate, got %s: %s", dec.Outcome, dec.Reason)
	}
}
