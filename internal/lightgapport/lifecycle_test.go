package lightgapport

import (
	"testing"
)

// Invariant: Lightgap port contracts must define non-empty swap matrices and explicit fak witness paths.
// Guard: Contract returns five standard swaps with populated fak witness paths.

func TestLightgapPortLifecycle(t *testing.T) {
	t.Parallel()

	c := Contract()
	if len(c.Swaps) != 5 {
		t.Fatalf("expected 5 swaps, got %d", len(c.Swaps))
	}
	for _, s := range c.Swaps {
		if s.Fak.Path == "" || s.Fak.Test == "" {
			t.Fatalf("swap %s missing fak witness", s.ID)
		}
	}
}
