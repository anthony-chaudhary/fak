package subtractiveprofile

import (
	"testing"
)

// Invariant: Subtractive profile resolution must guarantee deterministic capability inclusions and transitive dependency checks.
// Guard: Resolve returns an actionable error if any required capability dependency is missing.

func TestSubtractiveProfileLifecycle(t *testing.T) {
	t.Parallel()

	profiles := []Profile{
		{Include: []Capability{cap("tools"), cap("agent", "tools")}},
	}
	resolved, err := Resolve(profiles, Report{})
	if err != nil {
		t.Fatalf("expected clean resolution, got: %v", err)
	}
	if len(resolved.Capabilities) != 2 {
		t.Fatalf("expected 2 included capabilities, got %d", len(resolved.Capabilities))
	}
}
