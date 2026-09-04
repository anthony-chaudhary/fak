package edgequal

import (
	"testing"
)

// Invariant: Edge qualification receipts must validate hardware class, memory bounds, and execution constraints.
// Guard: Validate rejects unverified hardware classes and inconsistent quality metrics.

func TestEdgeQualLifecycle(t *testing.T) {
	t.Parallel()

	r := validReceipt("laptop_8gib")
	if err := Validate(r); err != nil {
		t.Fatalf("expected valid receipt to pass validation: %v", err)
	}

	// Invalid schema version
	bad := r
	bad.Schema = "fak.edgequal.v0"
	if err := Validate(bad); err == nil {
		t.Fatal("expected error on invalid schema")
	}
}
