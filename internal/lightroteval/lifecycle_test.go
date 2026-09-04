package lightroteval

import (
	"testing"
)

// Invariant: LightRot evaluation must preserve modeled quantization error metrics and provenance boundaries.
// Guard: Evaluate refuses invalid inputs, mismatched dimensions, and unpinned provenance records.

func TestLightRotEvalLifecycle(t *testing.T) {
	t.Parallel()

	r := fixture([][]float64{{8, .2, -.1, .1}, {7.5, .1, -.2, .2}, {-7.8, -.2, .1, -.1}, {.2, .1, -.2, 7}})
	got := Evaluate(r)
	if got.Outcome != OutcomeSupported {
		t.Fatalf("expected OutcomeSupported, got %s: %s", got.Outcome, got.Reason)
	}
	if len(got.Candidates) == 0 {
		t.Fatal("expected non-empty evaluation candidates")
	}
}
