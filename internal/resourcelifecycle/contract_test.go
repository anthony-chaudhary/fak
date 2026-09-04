package resourcelifecycle

import (
	"testing"
)

// Invariant: Resource lifecycle management must ensure that all acquired resources are cleanly torn down without leaks.
// Guard: Teardown releases all resources owned by the specified session.

func TestResourceLifecycleContract(t *testing.T) {
	t.Parallel()

	m := New()
	c := claim("tool_artifact", "session-contract", "tool/v1", true, false)
	alloc, err := m.Resolve(c, "host", "device")
	if err != nil {
		t.Fatalf("Resolve failed: %v", err)
	}

	res, ok := m.Get(alloc.Ref)
	if !ok || res.Released {
		t.Fatalf("expected active resource, got: %+v", res)
	}

	released := m.Teardown("session-contract")
	if len(released) != 1 {
		t.Fatalf("expected 1 released resource, got %d", len(released))
	}

	resAfter, ok := m.Get(alloc.Ref)
	if !ok || !resAfter.Released {
		t.Fatalf("expected released resource after teardown, got: %+v", resAfter)
	}
}
