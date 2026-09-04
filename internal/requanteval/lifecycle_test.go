package requanteval

import (
	"testing"
)

// Invariant: Requantization evaluation must preserve deterministic refinement grids and artifact provenance.
// Guard: Evaluate refuses invalid contracts, mismatched versions, or non-positive definite Hessians.

func TestRequantEvalLifecycle(t *testing.T) {
	t.Parallel()

	req := fixture(6253, []float64{-.8, -.4}, [][]float64{{1, -.8}, {-.8, 1}})
	got := Evaluate(req)
	if got.Outcome != OutcomeSupported {
		t.Fatalf("expected OutcomeSupported, got %s: %s", got.Outcome, got.Reason)
	}
	if got.Metrics.FinalMSE >= got.Metrics.InitialMSE {
		t.Fatalf("expected MSE improvement, got initial=%f final=%f", got.Metrics.InitialMSE, got.Metrics.FinalMSE)
	}
}
