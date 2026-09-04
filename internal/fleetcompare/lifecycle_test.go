package fleetcompare

import (
	"testing"
)

// Invariant: Fleet comparison metrics slicing must preserve monotonic coordinate sorting and isolated uplift math.
// Guard: SliceFixed validates non-empty columns and consistent observation lengths.

func TestFleetCompareLifecycle(t *testing.T) {
	t.Parallel()

	cols := fixtureCols()
	slice := SliceFixed(cols, "agents", 50)
	if len(slice.Xs) != 3 {
		t.Fatalf("expected 3 points in slice, got %d", len(slice.Xs))
	}
	if len(slice.Shared) != 3 || len(slice.Cross) != 3 || len(slice.Isolated) != 3 {
		t.Fatalf("slice arrays length mismatch: %+v", slice)
	}
}
