package memq

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/memoryread"
)

// TestOverflowReasonLabelPinned keeps the two MEMORY_INDEX_OVERFLOW wire labels in sync.
// memq imports memoryread (notesbackend), so memoryread cannot import memq to share the
// constant; this pin catches any drift between the two literals.
func TestOverflowReasonLabelPinned(t *testing.T) {
	if OverflowReason != memoryread.OverflowReason {
		t.Fatalf("overflow label drift: memq=%q memoryread=%q", OverflowReason, memoryread.OverflowReason)
	}
}

// fixedBackend is a deterministic in-memory Backend for the overflow tests: it returns
// exactly the cells it was seeded with, in order, so the budget prefix is predictable
// without depending on the demo store's contents.
type fixedBackend struct{ cells []Cell }

func (b fixedBackend) Cells(context.Context) ([]Cell, error) { return b.cells, nil }
func (b fixedBackend) Materialize(_ context.Context, id string) ([]byte, error) {
	for _, c := range b.cells {
		if c.ID == id {
			return []byte(c.Descriptor), nil
		}
	}
	return nil, ErrSealed
}

// TestIndexOverflowTyped is the #2430 witness: an index 2x over budget loads the
// in-budget prefix and emits a TYPED MEMORY_INDEX_OVERFLOW verdict NAMING every entry
// that fell past the line (never an anonymous tail-drop), while a zero-overflow index
// emits nothing (no advisory spam).
func TestIndexOverflowTyped(t *testing.T) {
	ctx := context.Background()
	// Four 100-byte entries = 400 bytes; a 200-byte cap is exactly 2x over budget.
	seed := []Cell{
		{ID: "m1", Descriptor: "note-1", Bytes: 100, Durability: "durable"},
		{ID: "m2", Descriptor: "note-2", Bytes: 100, Durability: "durable"},
		{ID: "m3", Descriptor: "note-3", Bytes: 100, Durability: "durable"},
		{ID: "m4", Descriptor: "note-4", Bytes: 100, Durability: "durable"},
	}
	backend := fixedBackend{cells: seed}
	q := Query{Ops: []Op{{Kind: OpScan}, {Kind: OpBudget, Bytes: 200}}}

	res, err := Run(ctx, backend, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}

	// In-budget prefix: the first two entries fit exactly under 200 bytes.
	if got := len(res.Working); got != 2 {
		t.Fatalf("in-budget prefix = %d entries, want 2", got)
	}
	if res.Working[0].ID != "m1" || res.Working[1].ID != "m2" {
		t.Fatalf("in-budget prefix = %s,%s, want m1,m2", res.Working[0].ID, res.Working[1].ID)
	}

	// Typed verdict present, keyed by the closed-vocabulary reason.
	if res.Overflow == nil {
		t.Fatal("over-budget index produced no IndexOverflow verdict")
	}
	ov := res.Overflow
	if ov.Reason != OverflowReason {
		t.Errorf("overflow reason = %q, want %q", ov.Reason, OverflowReason)
	}
	if ov.Budget != 200 {
		t.Errorf("overflow budget = %d, want 200", ov.Budget)
	}
	if ov.Kept != 2 {
		t.Errorf("overflow kept = %d, want 2", ov.Kept)
	}

	// The dropped entries are NAMED, in working-set order, with their handles intact.
	if len(ov.Dropped) != 2 {
		t.Fatalf("overflow dropped = %d entries, want 2 (m3,m4)", len(ov.Dropped))
	}
	wantIDs := []string{"m3", "m4"}
	for i, e := range ov.Dropped {
		if e.ID != wantIDs[i] {
			t.Errorf("dropped[%d].ID = %q, want %q", i, e.ID, wantIDs[i])
		}
		if e.Descriptor == "" {
			t.Errorf("dropped[%d] (%s) has no descriptor — the entry is unnamed", i, e.ID)
		}
		if e.Bytes != 100 {
			t.Errorf("dropped[%d].Bytes = %d, want 100", i, e.Bytes)
		}
	}

	// The step note carries the typed reason and names, not an anonymous count.
	var budgetNote string
	for _, s := range res.Steps {
		if s.Kind == OpBudget {
			budgetNote = s.Note
		}
	}
	if budgetNote == "" {
		t.Fatal("OpBudget step recorded no note")
	}
	if !contains(budgetNote, OverflowReason) || !contains(budgetNote, "m3") || !contains(budgetNote, "m4") {
		t.Errorf("budget note = %q, want it to name %s and the dropped entries", budgetNote, OverflowReason)
	}
}

// TestIndexOverflowZeroEmitsNothing pins the no-spam fence: an index that fits under
// budget yields no verdict and no over-budget note.
func TestIndexOverflowZeroEmitsNothing(t *testing.T) {
	ctx := context.Background()
	seed := []Cell{
		{ID: "m1", Descriptor: "note-1", Bytes: 100, Durability: "durable"},
		{ID: "m2", Descriptor: "note-2", Bytes: 100, Durability: "durable"},
	}
	q := Query{Ops: []Op{{Kind: OpScan}, {Kind: OpBudget, Bytes: 1000}}}

	res, err := Run(ctx, fixedBackend{cells: seed}, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Overflow != nil {
		t.Fatalf("in-budget index emitted an overflow verdict: %+v", res.Overflow)
	}
	for _, s := range res.Steps {
		if s.Kind == OpBudget && s.Note != "" {
			t.Errorf("in-budget OpBudget recorded a note: %q", s.Note)
		}
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
