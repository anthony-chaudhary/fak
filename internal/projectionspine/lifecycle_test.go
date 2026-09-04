package projectionspine

import (
	"testing"
)

// Invariant: Projection spine state snapshots must preserve authority and projection PID roles.
// Guard: Snapshot serializes role and address properties consistently.

func TestProjectionSpineLifecycle(t *testing.T) {
	t.Parallel()

	auth, err := NewAuthority(1234, "sess-1", 1, "marker-1")
	if err != nil {
		t.Fatalf("NewAuthority failed: %v", err)
	}

	snap := auth.Snapshot()
	if snap.AuthorityPID != 1234 || snap.SessionID != "sess-1" {
		t.Fatalf("unexpected snapshot properties: %+v", snap)
	}
}
