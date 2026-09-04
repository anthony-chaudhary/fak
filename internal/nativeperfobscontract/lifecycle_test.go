package nativeperfobscontract

import (
	"testing"
)

// Invariant: Native performance observability contracts must validate fak-native engine constraints.
// Guard: Validate rejects contracts referencing non-native engines or unbounded label dimensions.

func TestNativePerfObsContractLifecycle(t *testing.T) {
	t.Parallel()

	c := Frozen()
	if err := Validate(c); err != nil {
		t.Fatalf("expected Frozen contract to pass validation: %v", err)
	}
}
