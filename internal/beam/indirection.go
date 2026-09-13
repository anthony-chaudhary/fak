// Package beam implements beam-search support primitives over the batched decode
// session in internal/model. Issue #12409 ("KV cache beam indirection slot table")
// extends #12407, but the beam-search driver package does not exist yet, so this
// package deliberately contains ONLY the dependency-free, independently-verifiable
// slice: the BeamIndirectionTable slot-mapping primitive plus a thin BatchDriver
// that calls the real model.BatchSession.StepBatch / StepBatchActive API.
//
// INVARIANT (the whole point of the indirection table): branching and reordering a
// beam search is a PURE INTEGER operation over slot pointers. The table never moves,
// copies, or clones float32/float64 KV tensors. Fork/Reorder/Reset touch only the
// flat []int32 mapping; the (potentially large) KV tensors stay put in the model's
// BatchSession. A physical slot may be referenced by MULTIPLE logical rows via a
// deliberately shared prefix (Fork aliases the parent's prefix pointers), so slot
// ownership is tracked as a per-slot REFERENCE COUNT rather than a single owner.
// This is what makes beam decode O(B*MaxSteps) integer moves with ZERO per-step heap
// allocations (witness: TestBeamZeroAllocKV with testing.AllocsPerRun).
package beam

import "fmt"

// BeamIndirectionTable is a flat, pre-allocated [beams][maxSteps]int32 mapping from a
// logical (beam, position) coordinate to a physical KV slot index. It is stored as one
// contiguous []int32 of length beams*maxSteps so the whole table is a single allocation
// and no per-step slice is ever created. A physical slot may be referenced by more than
// one logical cell (see slotRefs), so recycling is decided by reference count.
type BeamIndirectionTable struct {
	beams    int
	maxSteps int
	flat     []int32

	// physSlots is the physical-slot capacity. slotRefs[p] records the number of
	// logical cells ACROSS ALL ROWS whose flat entry currently equals physical slot p;
	// it is zero iff the slot is free. Multiple rows may reference the same physical
	// slot via a deliberately shared prefix (Fork aliases the parent's prefix
	// pointers), so this is a REFERENCE COUNT, not a scalar owner. The mapping itself
	// is the flat table above; slotRefs exists to decide when a slot may be recycled.
	physSlots int
	slotRefs  []int32

	// reorderBuf and reorderSeen are lazily materialised reusable buffers so
	// Reorder performs zero heap allocations on repeated calls. reorderBuf holds the
	// full flat table (a double buffer making row permutation overlap-proof).
	reorderBuf  []int32
	reorderSeen []bool
}

// NewBeamIndirectionTable builds a table of beams logical rows, each with maxSteps
// positions. Physical-slot capacity defaults to beams*maxSteps (every logical cell may
// own a distinct physical slot). All entries start at -1 (unassigned).
//
// Panics if beams or maxSteps is negative.
func NewBeamIndirectionTable(beams, maxSteps int) *BeamIndirectionTable {
	return NewBeamIndirectionTableWithSlots(beams, maxSteps, beams*maxSteps)
}

// NewBeamIndirectionTableWithSlots is NewBeamIndirectionTable with an explicit physical
// slot capacity. It exists so a caller can allocate a compact physical slab (e.g. one
// shared by a prefill prefix) while still tracking per-slot reference counts.
//
// CONFLICT POLICY: the table does NOT reject assigning a physical slot that other
// rows already reference. That is intentional: sharing a prefix slot by pointer
// between parent and child is the whole point of Fork, so a shared reference is a
// legitimate state, not a programmer error. SetSlot therefore only range-checks phys
// and maintains the reference count; a slot becomes recyclable (AssignNew returns it)
// exactly when its count drops to zero. Distinct ownership for a freshly decoded
// suffix position is guaranteed by construction because AssignNew only ever returns a
// zero-count slot.
//
// Panics if beams or maxSteps is negative, or physSlots is negative.
func NewBeamIndirectionTableWithSlots(beams, maxSteps, physSlots int) *BeamIndirectionTable {
	if beams < 0 {
		panic(fmt.Sprintf("beam: negative beams %d", beams))
	}
	if maxSteps < 0 {
		panic(fmt.Sprintf("beam: negative maxSteps %d", maxSteps))
	}
	if physSlots < 0 {
		panic(fmt.Sprintf("beam: negative physSlots %d", physSlots))
	}
	t := &BeamIndirectionTable{
		beams:     beams,
		maxSteps:  maxSteps,
		flat:      make([]int32, beams*maxSteps),
		physSlots: physSlots,
		slotRefs:  make([]int32, physSlots),
	}
	for i := range t.flat {
		t.flat[i] = -1
	}
	for i := range t.slotRefs {
		t.slotRefs[i] = 0
	}
	return t
}

// Beams returns the number of logical beam rows.
func (t *BeamIndirectionTable) Beams() int { return t.beams }

// MaxSteps returns the number of logical positions per beam.
func (t *BeamIndirectionTable) MaxSteps() int { return t.maxSteps }

// Slot returns the physical slot index mapped to logical (beam, pos), or -1 if the
// cell has not been assigned. Out-of-range coordinates panic.
func (t *BeamIndirectionTable) Slot(beam, pos int) int32 {
	t.checkCoord(beam, pos)
	return t.flat[beam*t.maxSteps+pos]
}

// SetSlot maps logical (beam, pos) to physical slot phys. It maintains the physical
// slot reference count exactly: the cell's previous slot (if any) is decremented and
// phys is incremented. Assigning a slot that other rows already reference is allowed
// and expected for a shared prefix; phys == -1 clears the assignment.
func (t *BeamIndirectionTable) SetSlot(beam, pos int, phys int32) {
	t.checkCoord(beam, pos)
	idx := beam*t.maxSteps + pos
	if phys == -1 {
		t.decRef(t.flat[idx])
		t.flat[idx] = -1
		return
	}
	if phys < 0 || int(phys) >= t.physSlots {
		panic(fmt.Sprintf("beam: physical slot %d out of range [0,%d)", phys, t.physSlots))
	}
	t.decRef(t.flat[idx])
	t.flat[idx] = phys
	t.incRef(phys)
}

// AssignNew maps logical (beam, pos) to the first free physical slot (refs == 0) and
// returns it. It panics if the physical slab is exhausted.
func (t *BeamIndirectionTable) AssignNew(beam, pos int) int32 {
	t.checkCoord(beam, pos)
	for p := 0; p < t.physSlots; p++ {
		if t.slotRefs[p] == 0 {
			t.SetSlot(beam, pos, int32(p))
			return int32(p)
		}
	}
	panic(fmt.Sprintf("beam: physical slab exhausted (%d slots)", t.physSlots))
}

// incRef increments the reference count of a physical slot.
func (t *BeamIndirectionTable) incRef(phys int32) {
	if phys >= 0 && int(phys) < t.physSlots {
		t.slotRefs[phys]++
	}
}

// decRef decrements the reference count of a physical slot, never below zero.
func (t *BeamIndirectionTable) decRef(phys int32) {
	if phys >= 0 && int(phys) < t.physSlots && t.slotRefs[phys] > 0 {
		t.slotRefs[phys]--
	}
}

// checkCoord panics on an out-of-range (beam, pos).
func (t *BeamIndirectionTable) checkCoord(beam, pos int) {
	if beam < 0 || beam >= t.beams {
		panic(fmt.Sprintf("beam: beam %d out of range [0,%d)", beam, t.beams))
	}
	if pos < 0 || pos >= t.maxSteps {
		panic(fmt.Sprintf("beam: pos %d out of range [0,%d)", pos, t.maxSteps))
	}
}

// Fork copies the parent beam's physical-slot pointers for positions [0,pos) into the
// child row (the child inherits the shared prefix), then assigns the child a FRESH
// dedicated physical slot at position pos. It is O(pos) integer moves and allocates
// nothing. The child must be a different, in-range beam and pos must be in range.
//
// The parent's cells in [0,pos) are NOT cleared: the prefix KV slots are deliberately
// shared by pointer between parent and child (zero-copy) until one branch overwrites
// its own cell for a later position.
func (t *BeamIndirectionTable) Fork(parent, child, pos int) {
	t.checkCoord(parent, 0)
	t.checkCoord(child, 0)
	t.checkCoord(child, pos)
	if parent == child {
		panic(fmt.Sprintf("beam: Fork parent == child (%d)", parent))
	}
	// Release anything the child row currently owned so its slots can be recycled.
	// This decrements each old cell's reference count and never frees a slot another
	// row (e.g. an ancestor) still references.
	base := child * t.maxSteps
	for p := 0; p < t.maxSteps; p++ {
		t.decRef(t.flat[base+p])
		t.flat[base+p] = -1
	}
	// Inherit the shared prefix [0,pos) from the parent. Each inherited cell aliases
	// the parent's physical slot from one more row, so increment that slot's count.
	src := parent * t.maxSteps
	for p := 0; p < pos; p++ {
		ps := t.flat[src+p]
		t.flat[base+p] = ps
		t.incRef(ps)
	}
	// The child gets a distinct physical slot for pos.
	t.AssignNew(child, pos)
}

// Reorder permutes the logical beam rows according to newOrder: after Reorder, logical
// row i holds the row that was previously at newOrder[i]. It is O(B*MaxSteps) integer
// moves into a scratch row (kept as a reusable field so the call itself allocates
// nothing) and never touches a float KV tensor.
//
// newOrder must be a permutation of [0,beams).
func (t *BeamIndirectionTable) Reorder(newOrder []int) {
	if len(newOrder) != t.beams {
		panic(fmt.Sprintf("beam: Reorder order length %d != beams %d", len(newOrder), t.beams))
	}
	t.ensureReorderBuffers()
	seen := t.reorderSeen
	for i := range seen {
		seen[i] = false
	}
	for _, src := range newOrder {
		if src < 0 || src >= t.beams || seen[src] {
			panic(fmt.Sprintf("beam: Reorder argument is not a permutation (bad index %d)", src))
		}
		seen[src] = true
	}
	// Snapshot the ENTIRE table into the reusable double buffer, then scatter rows
	// back. A row-at-a-time move is NOT safe: with newOrder = {2,0,1}, writing row 0
	// from row 2 clobbers row 0 before row 1 reads it. The full snapshot makes the
	// permutation overlap-proof while still allocating nothing after first use.
	buf := t.reorderBuf[:len(t.flat)]
	copy(buf, t.flat)
	for i := 0; i < t.beams; i++ {
		src := newOrder[i] * t.maxSteps
		dst := i * t.maxSteps
		copy(t.flat[dst:dst+t.maxSteps], buf[src:src+t.maxSteps])
	}
	// Rebuild the reference counts from the permuted table so they stay exact. Every
	// cell contributes exactly one reference to its physical slot.
	for i := range t.slotRefs {
		t.slotRefs[i] = 0
	}
	for b := 0; b < t.beams; b++ {
		row := b * t.maxSteps
		for p := 0; p < t.maxSteps; p++ {
			if ps := t.flat[row+p]; ps >= 0 && int(ps) < t.physSlots {
				t.slotRefs[ps]++
			}
		}
	}
}

// ensureReorderBuffers lazily materialises the reusable Reorder scratch buffers.
func (t *BeamIndirectionTable) ensureReorderBuffers() {
	if t.reorderBuf == nil {
		t.reorderBuf = make([]int32, len(t.flat))
	}
	if t.reorderSeen == nil {
		t.reorderSeen = make([]bool, t.beams)
	}
}

// Reset clears every logical cell and every reference count, leaving the backing
// arrays in place: reuse costs no allocation.
func (t *BeamIndirectionTable) Reset() {
	for i := range t.flat {
		t.flat[i] = -1
	}
	for i := range t.slotRefs {
		t.slotRefs[i] = 0
	}
}

// checkRefInvariant verifies that slotRefs[p] equals the number of logical cells whose
// flat entry is p, for every physical slot. It is a debug/verification helper used by
// tests; it never mutates the table.
func (t *BeamIndirectionTable) checkRefInvariant() bool {
	var seen []int32
	if t.physSlots > 0 {
		seen = make([]int32, t.physSlots)
	}
	for _, ps := range t.flat {
		if ps >= 0 {
			if int(ps) >= t.physSlots {
				return false
			}
			seen[ps]++
		}
	}
	for p := 0; p < t.physSlots; p++ {
		if seen[p] != t.slotRefs[p] {
			return false
		}
	}
	return true
}

// BytesOwned reports the bytes held by the indirection TABLE itself: the flat int32
// table plus the per-slot reference counts, plus any scratch rows already materialised.
// This is the ONLY memory the table costs; it is pure integer bookkeeping and is
// deliberately orders of magnitude smaller than any KV tensor. It does not account for
// KV tensors, which live in the driver's model BatchSession, not in this table.
func (t *BeamIndirectionTable) BytesOwned() int {
	n := len(t.flat)*4 + len(t.slotRefs)*4
	n += len(t.reorderBuf) * 4
	n += len(t.reorderSeen)
	return n
}

// AllocBytes is an alias for BytesOwned, reporting heap bytes held by the table. It is
// named to make the "no KV tensors here" contract explicit at call sites and to give
// tests a single number to assert against.
func (t *BeamIndirectionTable) AllocBytes() int { return t.BytesOwned() }
