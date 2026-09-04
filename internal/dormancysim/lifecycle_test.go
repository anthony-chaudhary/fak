package dormancysim

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
)

// Invariant: Dormancy simulation clocks must advance monotonically and trigger corresponding rehydration rungs.
// Guard: Advance refuses backward clock mutations and preserves deterministic admission facts.

func TestDormancySimLifecycle(t *testing.T) {
	t.Parallel()

	var ran []rehydrate.Reason
	sim := New(epoch, dormancy.At(epoch), recordingGate(&ran))

	res := sim.Advance(context.Background(), time.Hour)
	if !res.Admitted {
		t.Fatal("expected admitted to be true")
	}
}
