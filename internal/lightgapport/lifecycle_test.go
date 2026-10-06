package lightgapport

import (
	"testing"
)

// Invariant: Lightgap port contracts must define non-empty swap matrices and explicit fak witness paths.
// Guard: Contract returns five standard swaps with populated fak witness paths.

func TestLightgapPortLifecycle(t *testing.T) {
	t.Parallel()

	wantIDs := map[string]bool{"agent": true, "provider": true, "model": true, "host": true, "policy": true}
	c := Contract()
	if len(c.Swaps) != len(wantIDs) {
		t.Fatalf("expected %d swaps, got %d", len(wantIDs), len(c.Swaps))
	}
	for _, s := range c.Swaps {
		if !wantIDs[s.ID] {
			t.Fatalf("unexpected swap %q", s.ID)
		}
		if s.Fak.Path == "" || s.Fak.Test == "" {
			t.Fatalf("swap %s missing fak witness", s.ID)
		}
	}
}
