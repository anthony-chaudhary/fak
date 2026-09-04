package epochbridge

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// Invariant: Epoch bridge mappings must preserve session trace generation and lineage edges.
// Guard: SpecContextFor returns zero epoch for generation 0 and non-zero distinct epochs for generations > 0.

func TestEpochBridgeLifecycle(t *testing.T) {
	t.Parallel()

	st0 := session.State{TraceID: "trace-0", Generation: 0}
	sc0 := SpecContextFor(st0)
	if sc0.Epoch != 0 || sc0.ParentEpoch != 0 {
		t.Fatalf("expected zero epoch for gen 0, got %+v", sc0)
	}

	c1 := session.ContinuationID("trace-0", 1)
	st1 := session.State{TraceID: c1, ParentTrace: "trace-0", Generation: 1}
	sc1 := SpecContextFor(st1)
	if sc1.Epoch == 0 {
		t.Fatal("expected non-zero epoch for gen 1")
	}
}
