package mixedprecision

import (
	"testing"
)

// Invariant: Mixed precision rule evaluation must verify supported artifact formats and runtime execution matrices.
// Guard: Adjudicate refuses combinations absent from the support table.

func TestMixedPrecisionLifecycle(t *testing.T) {
	t.Parallel()

	d := baseDescriptor()
	res := Evaluate(d, testSupport)
	if res.Outcome != OutcomeSupported {
		t.Fatalf("expected OutcomeSupported, got %s: %s", res.Outcome, res.Reason)
	}
}
