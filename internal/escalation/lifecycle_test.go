package escalation

import (
	"testing"
)

// Invariant: Escalation packets must contain valid schemas, positive revisions, and non-empty actions.
// Guard: ValidatePacket returns an error if packet fields violate structural requirements.

func TestEscalationLifecycle(t *testing.T) {
	t.Parallel()

	p := fixturePacket()
	if err := p.Validate(); err != nil {
		t.Fatalf("Validate failed for fixture packet: %v", err)
	}

	bad := p
	bad.Rev = 0
	if err := bad.Validate(); err == nil {
		t.Fatal("expected error on zero rev")
	}
}
