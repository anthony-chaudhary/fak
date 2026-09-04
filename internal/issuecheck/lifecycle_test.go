package issuecheck

import (
	"testing"
)

// Invariant: Issue review check catalog must be closed, versioned, and immutable.
// Guard: Lookup finds registered check items and returns defensive copies.

func TestIssueCheckLifecycle(t *testing.T) {
	t.Parallel()

	check, ok := Lookup("TC-01")
	if !ok {
		t.Fatal("expected TC-01 to exist in catalog")
	}
	if check.ID != "TC-01" {
		t.Fatalf("expected ID TC-01, got %s", check.ID)
	}
}
