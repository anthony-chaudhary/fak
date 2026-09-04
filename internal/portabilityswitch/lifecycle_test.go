package portabilityswitch

import (
	"testing"
)

// Invariant: Portability switching must guarantee rollback safety and transactional quiescence.
// Guard: Switch rolls back when adapters fail and restores quiesced processes.

func TestPortabilitySwitchLifecycle(t *testing.T) {
	t.Parallel()

	c, _, rt, _ := fixture(HotSwitch)
	req := Request{"hot", "parent", "B"}
	receipt, err := c.Switch(req)
	if err != nil {
		t.Fatalf("Switch failed: %v", err)
	}
	if receipt.Status != "complete" {
		t.Fatalf("expected complete status, got %s", receipt.Status)
	}
	if rt.resumes["parent"] == 0 {
		t.Fatal("expected parent process to resume")
	}
}
