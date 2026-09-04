package resulttier

import (
	"testing"
)

// Invariant: Result tier pagination slicing must guarantee cursor determinism and bound result counts.
// Guard: Slice returns truncated continuation responses when items exceed requested limits.

func TestResultTierLifecycle(t *testing.T) {
	t.Parallel()

	items := []string{"a", "b", "c", "d", "e", "f"}
	sliced, cont, err := Slice(items, PaginationRequest{Limit: 3})
	if err != nil {
		t.Fatalf("Slice failed: %v", err)
	}
	if len(sliced) != 3 {
		t.Fatalf("expected 3 items, got %d", len(sliced))
	}
	if !cont.HasMore || !cont.Truncated {
		t.Fatalf("expected HasMore and Truncated, got cont: %+v", cont)
	}
}
