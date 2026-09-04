package fleetfreeze

import (
	"testing"
)

// Invariant: Fleet freeze state must hold new spawn operations while permitting ongoing status and harvest actions.
// Guard: Allowed returns Allow=false for OpSpawn when frozen, and Allow=true for harvesting operations.

func TestFleetFreezeLifecycle(t *testing.T) {
	t.Parallel()

	st := Freeze("test-freeze-reason", testFreezeUnix)
	if !st.Frozen {
		t.Fatal("expected Frozen to be true")
	}

	spawnDecision := Allowed(st, OpSpawn)
	if spawnDecision.Allow {
		t.Fatal("expected OpSpawn to be held when frozen")
	}

	harvestDecision := Allowed(st, OpWitnessClose)
	if !harvestDecision.Allow {
		t.Fatal("expected OpWitnessClose to remain allowed when frozen")
	}
}
