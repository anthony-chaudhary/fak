package kvmmu_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestApplyPlanEvictsElidedToO1Residency is the load-bearing witness for the
// ctxplan->kvmmu residency bridge (issue #550). ctxplan produces an O(1) VIEW
// (a plan: some spans Selected resident, the rest Elided cold-but-recoverable);
// ApplyPlan evicts the elided segments' K/V so the kernel-owned cache's
// RESIDENCY shrinks to the resident view. The post-plan next-token distribution
// must be BIT-IDENTICAL to a reference session that only ever prefilled the
// RESIDENT spans + the query — true iff every elided span was removed AND the
// survivors were renumbered (re-RoPE'd) to the compacted positions. An O(1)
// view is thereby an O(1) KV residency, byte-for-byte.
func TestApplyPlanEvictsElidedToO1Residency(t *testing.T) {
	m := model.NewSynthetic(synthCfg())
	sys := []int{1, 2, 3, 4}
	a := []int{10, 11, 12}     // resident
	b := []int{20, 21, 22, 23} // elided (over budget)
	cc := []int{30, 31}        // elided (over budget)
	query := []int{40, 41}

	s := m.NewSession()
	c := kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", sys)
	c.Append("a", "read_notes", a)
	c.Append("b", "read_policy", b)
	c.Append("c", "read_log", cc)
	fullLen := c.CacheLen() // O(N) residency BEFORE the bridge

	// A ctxplan.Plan: sys+a resident, b+c elided (cold but recoverable). The IDs
	// match the segment IDs — the adapter contract the bridge keys on.
	plan := ctxplan.Plan{
		Objective: ctxplan.ObjGreedy,
		Selected: []ctxplan.Selection{
			{ID: "sys", Step: 0},
			{ID: "a", Step: 1},
		},
		Elided: []ctxplan.Elision{
			{ID: "b", Step: 2, Digest: "d-b", Reason: ctxplan.ElideOverBudget},
			{ID: "c", Step: 3, Digest: "d-c", Reason: ctxplan.ElideOverBudget},
		},
		Candidates: 4,
	}
	// The plan MUST be faithful (every elided span recoverable) for the bridge to
	// be honest: an unrecoverable elision is compaction, not a planned view, and
	// evicting it would destroy a fact instead of paging it out.
	if w := ctxplan.Audit(plan); !w.Faithful {
		t.Fatalf("test fixture plan is not faithful: %+v", w)
	}

	n := c.ApplyPlan(plan)
	if n != 2 {
		t.Fatalf("ApplyPlan evicted %d segments, want 2 (b and c)", n)
	}
	wantResident := len(sys) + len(a)
	if c.CacheLen() != wantResident {
		t.Fatalf("after ApplyPlan, KV residency = %d positions, want %d (an O(1) view must be an O(1) KV residency)",
			c.CacheLen(), wantResident)
	}
	if c.CacheLen() == fullLen {
		t.Fatal("KV residency did not shrink — the bridge evicted nothing")
	}
	if c.Evicted() != 2 {
		t.Fatalf("Evicted() = %d, want 2", c.Evicted())
	}

	// Non-vacuity control: a session that KEEPS the elided spans must give a
	// DIFFERENT distribution than the resident-only reference. If it did not, the
	// bit-exact claim below would be vacuous (the elided spans never mattered).
	lKept := m.NewSession().Prefill(cat(sys, a, b, cc, query))
	lResident := m.NewSession().Prefill(cat(sys, a, query))
	if d := maxAbsDiff(lKept, lResident); d == 0 {
		t.Fatalf("keeping the elided spans did not perturb the distribution — the bit-exact witness would be vacuous")
	}

	// BIT-EXACT witness: the bridged cache + query must equal a reference that
	// only ever prefilled the RESIDENT spans + query.
	lGot, _ := c.Append("usr", "user", query)
	if d := maxAbsDiff(lGot, lResident); d != 0 {
		t.Fatalf("post-ApplyPlan distribution != reference prefill(resident+query) (max|Δ|=%.3e); "+
			"the elision-to-eviction bridge is not bit-exact", d)
	}
}

// TestApplyPlanNoElisionKeepsFullResidency is the conservation control: a plan
// that elides NOTHING (everything selected) evicts nothing, so a faithful
// all-resident plan is a no-op on the KV cache — the bridge never shrinks
// residency below the view the plan asked for.
func TestApplyPlanNoElisionKeepsFullResidency(t *testing.T) {
	m := model.NewSynthetic(synthCfg())
	sys := []int{1, 2, 3}
	a := []int{10, 11}
	b := []int{20, 21, 22}

	s := m.NewSession()
	c := kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", sys)
	c.Append("a", "read_notes", a)
	c.Append("b", "read_policy", b)
	want := c.CacheLen()

	plan := ctxplan.Plan{
		Objective: ctxplan.ObjGreedy,
		Selected: []ctxplan.Selection{
			{ID: "sys", Step: 0},
			{ID: "a", Step: 1},
			{ID: "b", Step: 2},
		},
		Candidates: 3,
	}
	if w := ctxplan.Audit(plan); !w.Faithful {
		t.Fatalf("all-resident plan should be faithful: %+v", w)
	}
	if n := c.ApplyPlan(plan); n != 0 {
		t.Fatalf("an all-selected plan evicted %d segments, want 0", n)
	}
	if c.CacheLen() != want {
		t.Fatalf("after an all-selected plan, KV residency = %d, want %d (nothing should evict)", c.CacheLen(), want)
	}
	if c.Evicted() != 0 {
		t.Fatalf("Evicted() = %d, want 0", c.Evicted())
	}
}

// TestApplyPlanSelectedBeatsElided is the malformed-plan defense: a span listed
// as BOTH selected and elided is a partition violation (ctxplan.Audit would flag
// it), but the bridge must still keep it resident — defense in depth, so a
// malformed plan can never evict a span the view wanted.
func TestApplyPlanSelectedBeatsElided(t *testing.T) {
	m := model.NewSynthetic(synthCfg())
	sys := []int{1, 2, 3}
	a := []int{10, 11}

	s := m.NewSession()
	c := kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", sys)
	c.Append("a", "read_notes", a)
	want := c.CacheLen()

	// Malformed: "a" is both selected and elided.
	plan := ctxplan.Plan{
		Objective: ctxplan.ObjGreedy,
		Selected:  []ctxplan.Selection{{ID: "a", Step: 1}},
		Elided:    []ctxplan.Elision{{ID: "a", Step: 1, Digest: "d-a", Reason: ctxplan.ElideOverBudget}},
	}
	if n := c.ApplyPlan(plan); n != 0 {
		t.Fatalf("a selected-but-also-elided span was evicted (%d); selected must win", n)
	}
	if c.CacheLen() != want {
		t.Fatalf("KV residency = %d, want %d (the selected span must stay resident)", c.CacheLen(), want)
	}
}

// TestApplyPlanIgnoresUnknownIds proves the bridge never evicts a segment the
// plan did not speak to: an elision id with no matching segment is a no-op, and
// a ledger segment whose id is in neither set is left in place. The bridge
// shrinks residency by the plan's ELISION only, never by guessing.
func TestApplyPlanIgnoresUnknownIds(t *testing.T) {
	m := model.NewSynthetic(synthCfg())
	sys := []int{1, 2, 3}
	a := []int{10, 11}

	s := m.NewSession()
	c := kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", sys)
	c.Append("a", "read_notes", a)
	want := c.CacheLen()

	// Elides a nonexistent id; the real segments are neither selected nor elided.
	plan := ctxplan.Plan{
		Objective: ctxplan.ObjGreedy,
		Elided:    []ctxplan.Elision{{ID: "ghost", Step: 9, Digest: "d-ghost", Reason: ctxplan.ElideOverBudget}},
	}
	if n := c.ApplyPlan(plan); n != 0 {
		t.Fatalf("ApplyPlan evicted %d for an all-unknown elision set, want 0", n)
	}
	if c.CacheLen() != want {
		t.Fatalf("KV residency = %d, want %d (unknown elision ids must not evict real segments)", c.CacheLen(), want)
	}
}
