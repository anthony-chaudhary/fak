package epochbridge

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestSpecContextForGenerationZero proves an original trace that never came from a
// re-continuation maps to abi's zero value (Speculative=false, Epoch/ParentEpoch=0) —
// "generation 0 reads as epoch 0", the default the #914 fence preserves.
func TestSpecContextForGenerationZero(t *testing.T) {
	sc := SpecContextFor(session.State{TraceID: "root-trace"})
	if sc.Speculative || sc.Epoch != 0 || sc.ParentEpoch != 0 {
		t.Fatalf("gen0 SpecContext = %+v, want {Speculative:false Epoch:0 ParentEpoch:0}", sc)
	}
}

// TestRecontinueLineageMapsToEpochLineage is the acceptance test: a real Recontinue
// parent->child generation maps to a parent->child EDGE in the shared epoch space —
// the child's ParentEpoch is exactly the parent's Epoch. It walks two real
// re-continuations (gen0 -> gen1 -> gen2) so the asserted edge is non-trivially
// non-zero, not 0 == 0.
func TestRecontinueLineageMapsToEpochLineage(t *testing.T) {
	tbl := session.NewTable()
	unbounded := session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded}

	t0 := "root-trace" // generation 0: an original trace, epoch 0
	c1 := session.ContinuationID(t0, 1)
	gen1 := tbl.Recontinue(t0, c1, unbounded) // {TraceID:c1, ParentTrace:t0, Generation:1}
	c2 := session.ContinuationID(c1, 1)
	gen2 := tbl.Recontinue(c1, c2, unbounded) // {TraceID:c2, ParentTrace:c1, Generation:2}

	sc1 := SpecContextFor(gen1)
	sc2 := SpecContextFor(gen2)

	// gen1 is a child of the original gen0: its own epoch is non-zero, its parent is 0.
	if sc1.Epoch == 0 {
		t.Fatalf("gen1 epoch = 0, want a non-zero continuation epoch")
	}
	if sc1.ParentEpoch != 0 {
		t.Errorf("gen1 parent epoch = %d, want 0 (its parent is the original gen0)", sc1.ParentEpoch)
	}
	// The lineage edge: gen2's parent epoch IS gen1's epoch — one parent->child edge in
	// the shared id space, exactly mirroring the Recontinue parent->child generation.
	if sc2.ParentEpoch != sc1.Epoch {
		t.Errorf("gen2 parent epoch = %d, want gen1 epoch %d (broken lineage edge)", sc2.ParentEpoch, sc1.Epoch)
	}
	if sc2.Epoch == sc1.Epoch {
		t.Errorf("gen2 epoch == gen1 epoch (%d); a new generation must mint a distinct epoch", sc2.Epoch)
	}
	// A session continuation is a durable generation, never a provisional branch.
	if sc1.Speculative || sc2.Speculative {
		t.Error("a Recontinue generation must map to Speculative=false (committed lineage)")
	}
	if got := GenerationOutcome(); got != abi.OutcomeCommitted {
		t.Errorf("GenerationOutcome() = %v, want OutcomeCommitted (a re-continuation always commits)", got)
	}
}
