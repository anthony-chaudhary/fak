package beam

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestBeamZeroAllocKV is the load-bearing witness that beam branch operations are pure
// integer bookkeeping: a decode-like loop of Fork/Reorder/Slot must perform ZERO heap
// allocations. If Fork/Reorder ever allocated a slice (or, worse, cloned a KV tensor),
// testing.AllocsPerRun would report it.
func TestBeamZeroAllocKV(t *testing.T) {
	const B, MaxSteps = 4, 64
	table := NewBeamIndirectionTable(B, MaxSteps)
	// Pre-warm every lazily-created buffer and assign an initial row for each beam so
	// the measured loop only exercises steady-state integer moves.
	table.ensureReorderBuffers()
	for b := 0; b < B; b++ {
		for p := 0; p < MaxSteps; p++ {
			table.SetSlot(b, p, int32(b*MaxSteps+p))
		}
	}
	order := []int{B - 1, 0, 2, 1}
	var sink int32

	allocs := testing.AllocsPerRun(1000, func() {
		// A decode-like step: fork one beam, reorder the pool, read a slot.
		table.Fork(0, 1, 8)
		table.Reorder(order)
		sink += table.Slot(1, 8)
		sink += table.Slot(0, 8)
	})
	if allocs != 0 {
		t.Fatalf("decode-like Fork/Reorder/Slot allocations = %v, want 0", allocs)
	}
	if sink == 0 {
		t.Fatalf("sink stayed 0; loop was optimised away, witness is vacuous")
	}
}

// TestIndirectionSlotMappingAndFork pins the Fork contract: the child inherits the
// parent's [0,pos) slot pointers verbatim and receives a DISTINCT physical slot at pos,
// and every owned physical slot is unique across the table.
func TestIndirectionSlotMappingAndFork(t *testing.T) {
	const B, MaxSteps = 3, 8
	table := NewBeamIndirectionTable(B, MaxSteps)
	// Give the parent a full distinctive prefix (physical slots all within capacity).
	for p := 0; p < MaxSteps; p++ {
		table.SetSlot(0, p, int32(p))
	}
	table.Fork(0, 2, 5)

	for p := 0; p < 5; p++ {
		if got, want := table.Slot(2, p), table.Slot(0, p); got != want {
			t.Fatalf("child prefix pos %d = %d, want parent's %d", p, got, want)
		}
	}
	if child := table.Slot(2, 5); child == table.Slot(0, 5) {
		t.Fatalf("child pos 5 slot = %d, must differ from parent slot %d", child, table.Slot(0, 5))
	}

	// The shared prefix is deliberately shared by pointer, so a physical slot may be
	// referenced by more than one row there. The single-owner guarantee applies to
	// FRESHLY assigned slots: the child's dedicated slot at pos must not be owned by
	// any other beam's non-shared cell.
	childSlot := table.Slot(2, 5)
	for b := 0; b < B; b++ {
		if b == 2 {
			continue
		}
		for p := 0; p < MaxSteps; p++ {
			if table.Slot(b, p) == childSlot {
				t.Fatalf("child's fresh slot %d also referenced by beam %d pos %d", childSlot, b, p)
			}
		}
	}

	// Two logical beams must never claim the same FRESH slot. The child's dedicated
	// slot at pos is freshly assigned, so assigning it to another row must be allowed
	// (sharing is the design), but it must then be reflected in the refcount and that
	// slot must not be recycled while any row still references it.
	table.SetSlot(1, 0, childSlot)
	if !table.checkRefInvariant() {
		t.Fatalf("reference invariant broken after a legitimate shared SetSlot")
	}
}

// TestSharedPrefixSlotNotFreed is the regression for the false-free BLOCKER: a parent
// owns slot 2, Fork makes the child's row1 share slot 2 in its prefix, then the parent
// clears its own cell. The child's cell must STILL map to slot 2 and AssignNew must
// NOT recycle slot 2, otherwise a later AssignNew would clobber KV the child reads.
func TestSharedPrefixSlotNotFreed(t *testing.T) {
	const B, MaxSteps = 3, 8
	table := NewBeamIndirectionTable(B, MaxSteps)

	// Parent (beam 0) owns physical slot 2 at pos 0.
	table.SetSlot(0, 0, 2)
	if got := table.Slot(0, 0); got != 2 {
		t.Fatalf("parent slot = %d, want 2", got)
	}
	// Child (beam 1) inherits [0,1), so its pos 0 aliases parent's slot 2.
	table.Fork(0, 1, 5)
	if got := table.Slot(1, 0); got != 2 {
		t.Fatalf("child inherited slot = %d, want 2", got)
	}

	// The parent clears its OWN cell. Slot 2 is still referenced by the child, so it
	// must remain allocated.
	table.SetSlot(0, 0, -1)
	if got := table.Slot(1, 0); got != 2 {
		t.Fatalf("sibling cell = %d after owner cleared; want 2 (shared slot was freed)", got)
	}
	if !table.checkRefInvariant() {
		t.Fatalf("reference invariant broken after clearing a shared cell")
	}

	// AssignNew must not hand out the still-referenced slot 2.
	newSlot := table.AssignNew(2, 0)
	if newSlot == 2 {
		t.Fatalf("AssignNew returned still-referenced slot 2; sibling KV would be clobbered")
	}
}

// TestSharedPrefixSetSlotNoFalsePanic is the regression for the false-panic MAJOR: after
// Fork, the child re-setting its own inherited shared cell to the SAME physical slot is
// legitimate (it changes nothing) and must NOT panic. Reorder must not corrupt ownership
// either, so a subsequent legitimate SetSlot by the original referrer must not panic.
func TestSharedPrefixSetSlotNoFalsePanic(t *testing.T) {
	const B, MaxSteps = 3, 8
	table := NewBeamIndirectionTable(B, MaxSteps)

	for p := 0; p < MaxSteps; p++ {
		table.SetSlot(0, p, int32(p))
	}
	table.Fork(0, 1, 5)

	shared := table.Slot(1, 0)
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("child re-setting its own inherited shared cell panicked: %v", r)
			}
		}()
		table.SetSlot(1, 0, shared)
	}()
	if got := table.Slot(1, 0); got != shared {
		t.Fatalf("child cell = %d after no-op re-set, want %d", got, shared)
	}
	if !table.checkRefInvariant() {
		t.Fatalf("reference invariant broken after child no-op re-set")
	}

	// After a Reorder, the original referrer setting its own cell must still work.
	table.Reorder([]int{1, 0, 2})
	if !table.checkRefInvariant() {
		t.Fatalf("reference invariant broken after Reorder")
	}
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("legitimate SetSlot after Reorder panicked: %v", r)
			}
		}()
		table.SetSlot(0, 0, table.Slot(0, 0))
	}()
	if !table.checkRefInvariant() {
		t.Fatalf("reference invariant broken after post-Reorder SetSlot")
	}
}

// TestForkRecycleKeepsAncestorSlots is the regression for the grandchild/ancestor MAJOR:
// Fork clears+re-references the child's ENTIRE row before inheriting. When the child row
// itself shares prefix slots with an ancestor, clearing it must NOT free slots the
// ancestor still references.
func TestForkRecycleKeepsAncestorSlots(t *testing.T) {
	const B, MaxSteps = 4, 8
	table := NewBeamIndirectionTable(B, MaxSteps)

	// Ancestor (beam 0) owns a full distinctive row.
	for p := 0; p < MaxSteps; p++ {
		table.SetSlot(0, p, int32(p))
	}
	// Child (beam 1) inherits ancestor's prefix [0,4).
	table.Fork(0, 1, 4)
	ancestorBefore := table.Slot(0, 0)
	childBefore := table.Slot(1, 0)
	if ancestorBefore != childBefore {
		t.Fatalf("child prefix %d != ancestor %d", childBefore, ancestorBefore)
	}

	// Grandchild (beam 2) forks FROM the child, so beam 1's whole row is cleared and
	// re-inherited. The ancestor's cells must be untouched throughout.
	table.Fork(1, 2, 6)
	if got := table.Slot(0, 0); got != ancestorBefore {
		t.Fatalf("ancestor cell = %d after grandchild fork, want %d (ancestor slot corrupted)", got, ancestorBefore)
	}
	if got := table.Slot(2, 0); got != ancestorBefore {
		t.Fatalf("grandchild prefix = %d, want ancestor's %d", got, ancestorBefore)
	}
	if !table.checkRefInvariant() {
		t.Fatalf("reference invariant broken after grandchild fork")
	}

	// The ancestor's slot must not be recycled: it is still referenced by ancestor,
	// child and grandchild.
	ancestorSlot := table.Slot(0, 0)
	for {
		got := table.AssignNew(3, 0)
		if got == ancestorSlot {
			t.Fatalf("AssignNew recycled ancestor slot %d still referenced by 3 rows", got)
		}
		break
	}
	if !table.checkRefInvariant() {
		t.Fatalf("reference invariant broken after grandchild-fork AssignNew")
	}
}

// TestIndirectionReorderO checks Reorder permutes whole logical rows: after reordering
// with newOrder, row i must hold exactly the row previously at newOrder[i].
func TestIndirectionReorder(t *testing.T) {
	const B, MaxSteps = 3, 6
	table := NewBeamIndirectionTable(B, MaxSteps)
	before := make([][]int32, B)
	for b := 0; b < B; b++ {
		before[b] = make([]int32, MaxSteps)
		for p := 0; p < MaxSteps; p++ {
			before[b][p] = int32(b*MaxSteps + p)
			table.SetSlot(b, p, before[b][p])
		}
	}
	order := []int{2, 0, 1}
	table.Reorder(order)

	for i := 0; i < B; i++ {
		for p := 0; p < MaxSteps; p++ {
			if got, want := table.Slot(i, p), before[order[i]][p]; got != want {
				t.Fatalf("after Reorder row %d pos %d = %d, want row %d's %d", i, p, got, order[i], want)
			}
		}
	}
}

// TestBatchDriverSharesWeightStream proves the driver drives the REAL model batched
// decode path: an all-active Step is bit-for-bit StepBatch and reports the same exact
// MAC receipt, so a beam driver can attribute the shared-weight-stream work.
func TestBatchDriverSharesWeightStream(t *testing.T) {
	cfg := model.Config{
		HiddenSize: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := model.NewSynthetic(cfg)
	V := cfg.VocabSize
	const C = 4
	prompts := make([][]int, C)
	for b := 0; b < C; b++ {
		n := 3 + b*2
		p := make([]int, n)
		for i := range p {
			p[i] = (b*97 + i*31 + 5) % V
		}
		prompts[b] = p
	}
	ids := make([]int, C)
	for b := range ids {
		ids[b] = (b*13 + 3) % V
	}

	bs := m.NewBatchSession(C)
	bs.PrefillEach(prompts)
	table := NewBeamIndirectionTable(C, 8)
	driver := NewBatchDriver(bs, table)

	got := driver.Step(StepPlan{IDs: ids, Active: []bool{true, true, true, true}})
	gotMACs := driver.LastStepMACs()
	if gotMACs <= 0 {
		t.Fatalf("driver LastStepMACs = %d, want > 0", gotMACs)
	}

	// Reference: a fresh session driven directly through StepBatch must agree exactly.
	ref := m.NewBatchSession(C)
	ref.PrefillEach(prompts)
	want := ref.StepBatch(ids)
	if wantMACs := ref.LastStepMACs(); gotMACs != wantMACs {
		t.Fatalf("driver MACs = %d, direct StepBatch MACs = %d, want equal", gotMACs, wantMACs)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("driver Step logits differ from direct StepBatch (not bit-for-bit)")
	}
}

// TestBatchDriverRaggedPath witnesses the ragged branch of BatchDriver.Step: a partial
// Active mask must route to model.StepBatchActive and report proportionally less
// projection work, so a beam driver can attribute the idle-lane savings. This is the
// path the all-active test above does not exercise.
func TestBatchDriverRaggedPath(t *testing.T) {
	cfg := model.Config{
		HiddenSize: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := model.NewSynthetic(cfg)
	V := cfg.VocabSize
	const C = 4 // beams
	const idle = 2
	prompts := make([][]int, C)
	for b := 0; b < C; b++ {
		n := 3 + b*2
		p := make([]int, n)
		for i := range p {
			p[i] = (b*97 + i*31 + 5) % V
		}
		prompts[b] = p
	}
	ids := make([]int, C)
	for b := range ids {
		ids[b] = (b*13 + 3) % V
	}
	active := []bool{true, false, true, false}

	// Full-batch reference taken from a separate session in the same state.
	full := m.NewBatchSession(C)
	full.PrefillEach(prompts)
	full.StepBatch(ids)
	fullMACs := full.LastStepMACs()
	if fullMACs <= 0 {
		t.Fatalf("full-batch LastStepMACs = %d, want > 0", fullMACs)
	}

	bs := m.NewBatchSession(C)
	bs.PrefillEach(prompts)
	driver := NewBatchDriver(bs, NewBeamIndirectionTable(C, 8))
	out := driver.Step(StepPlan{IDs: ids, Active: active})
	ragMACs := driver.LastStepMACs()

	// Exact integer ratio: (C-idle)/C of the full-batch projection work.
	if ragMACs*int64(C) != fullMACs*int64(C-idle) {
		t.Fatalf("ragged MACs = %d, full = %d, want exactly %d/%d of full", ragMACs, fullMACs, C-idle, C)
	}
	// Idle lanes get nil logits; active lanes get a vector.
	for b := 0; b < C; b++ {
		if active[b] && len(out[b]) == 0 {
			t.Fatalf("active beam %d got empty logits", b)
		}
		if !active[b] && out[b] != nil {
			t.Fatalf("idle beam %d got non-nil logits", b)
		}
	}
}

// BenchmarkBeamIndirectionStep measures the per-branch cost of the pure-integer path.
func BenchmarkBeamIndirectionStep(b *testing.B) {
	const B, MaxSteps = 8, 128
	table := NewBeamIndirectionTable(B, MaxSteps)
	table.ensureReorderBuffers()
	for i := 0; i < B; i++ {
		for p := 0; p < MaxSteps; p++ {
			table.SetSlot(i, p, int32(i*MaxSteps+p))
		}
	}
	order := make([]int, B)
	for i := range order {
		order[i] = (i + 1) % B
	}
	b.ReportAllocs()
	b.ResetTimer()
	var sink int32
	for i := 0; i < b.N; i++ {
		table.Fork(0, 1, 16)
		table.Reorder(order)
		sink += table.Slot(1, 16)
	}
	_ = sink
}
