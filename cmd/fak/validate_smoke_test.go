package main

import (
	"slices"
	"testing"
)

func TestValidatePhaseOrderIncludesSmoke(t *testing.T) {
	withoutSmoke := validatePhaseOrder(false, false, false)
	if slices.Contains(withoutSmoke, "smoke") {
		t.Fatalf("expected validatePhaseOrder without smoke to omit 'smoke', got %v", withoutSmoke)
	}

	withSmoke := validatePhaseOrder(false, false, true)
	if !slices.Contains(withSmoke, "smoke") {
		t.Fatalf("expected validatePhaseOrder with smoke to contain 'smoke', got %v", withSmoke)
	}
	if withSmoke[len(withSmoke)-1] != "smoke" {
		t.Fatalf("expected 'smoke' phase to be at the end, got %v", withSmoke)
	}
}
