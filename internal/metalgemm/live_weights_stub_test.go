//go:build !darwin || !arm64 || !cgo

package metalgemm

import "testing"

func TestLiveWeightsStubZero(t *testing.T) {
	if got := LiveQ6KWeights(); got != 0 {
		t.Fatalf("LiveQ6KWeights() = %d, want 0", got)
	}
	if got := LiveQ8Weights(); got != 0 {
		t.Fatalf("LiveQ8Weights() = %d, want 0", got)
	}
}
