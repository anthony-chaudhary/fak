package estimatecal

import (
	"testing"
)

// Invariant: Token estimate calibration must accurately update moving ratios across distinct model and provider keys.
// Guard: Ratio returns false until MinSamples observations are recorded.

func TestEstimateCalLifecycle(t *testing.T) {
	t.Parallel()

	var store Store
	for i := 0; i < MinSamples; i++ {
		store.Observe("provider-a", "model-a", 100, 150)
	}

	ratio, ok := store.Ratio("provider-a", "model-a")
	if !ok {
		t.Fatal("expected Ratio to return true after MinSamples observations")
	}
	if ratio != 1.5 {
		t.Fatalf("expected ratio 1.5, got %f", ratio)
	}
}
