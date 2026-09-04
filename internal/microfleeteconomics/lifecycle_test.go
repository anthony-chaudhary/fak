package microfleeteconomics

import (
	"testing"
)

// Invariant: Microfleet economics evaluation must account for all operational cost categories.
// Guard: Evaluate refuses receipts with mismatched quality dimensions or invalid branch counts.

func TestMicrofleetEconomicsLifecycle(t *testing.T) {
	t.Parallel()

	r := validReceipt()
	evaluated, err := Evaluate(r, Rates{})
	if err != nil {
		t.Fatalf("Evaluate failed: %v", err)
	}
	if evaluated.Name != "valid" {
		t.Fatalf("expected evaluated receipt name 'valid', got %q", evaluated.Name)
	}
}
