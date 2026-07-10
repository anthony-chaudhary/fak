package memq

import (
	"context"
	"reflect"
	"testing"
)

// hasRef reports whether id is in refs (order-independent membership).
func hasRef(refs []string, id string) bool {
	for _, r := range refs {
		if r == id {
			return true
		}
	}
	return false
}

// TestMergeOnEvictPreservesDroppedRefsOnSurvivor is the required witness for #4015:
// a below-budget cell that is a near-duplicate of a surviving cell is FOLDED into that
// survivor — its Refs (and its own ID) preserved on the survivor — instead of being
// tail-dropped and lost. The two cells share almost all their body text, so their
// simhash cosine clears the floor; the evicted cell leaves the overflow list entirely.
func TestMergeOnEvictPreservesDroppedRefsOnSurvivor(t *testing.T) {
	ctx := context.Background()
	// Bodies are the Descriptors (fixedBackend.Materialize returns them). The two are
	// near-identical (differ only by a trailing word) so cosine is high; `dup` carries
	// distinctive refs that must not vanish when the budget evicts it.
	seed := []Cell{
		{ID: "surv", Step: 1, Role: "user", Descriptor: "the quick brown fox jumps over the lazy dog near duplicate body", Bytes: 100, Durability: DurabilityDurable},
		{ID: "dup", Step: 2, Role: "tool", Descriptor: "the quick brown fox jumps over the lazy dog near duplicate body extra", Bytes: 100, Durability: DurabilityTurn, Refs: []string{"fact-x", "fact-y"}},
	}
	// scan -> budget(100): keeps `surv` (100), evicts `dup` (would be 200). The floor
	// 0.5 is well under the near-identical pair's cosine, so `dup` folds into `surv`.
	q := Query{Ops: []Op{
		{Kind: OpScan},
		{Kind: OpBudget, Bytes: 100, MergeFloor: 0.5},
	}}
	res, err := Run(ctx, fixedBackend{cells: seed}, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}

	// The evicted near-duplicate was folded, not lost: nothing reaches the overflow.
	if res.Overflow != nil {
		t.Fatalf("overflow = %+v, want nil (the evicted near-dup was folded, not dropped)", res.Overflow)
	}
	// The survivor is the sole working-set cell and now carries the evicted cell's refs
	// plus its provenance ID.
	if got := ids(res.Working); !reflect.DeepEqual(got, []string{"surv"}) {
		t.Fatalf("working set = %v, want [surv]", got)
	}
	surv := res.Working[0]
	for _, want := range []string{"fact-x", "fact-y", "dup"} {
		if !hasRef(surv.Refs, want) {
			t.Fatalf("survivor refs %v missing %q (evicted cell's provenance lost)", surv.Refs, want)
		}
	}
	if res.Stats.MergedOnEvict != 1 {
		t.Fatalf("MergedOnEvict = %d, want 1", res.Stats.MergedOnEvict)
	}
	// Exactly one merge Effect, naming [survivor, evicted], proposal-only.
	var effects int
	for _, e := range res.Effects {
		if e.Kind != MergeOnEvictKind {
			continue
		}
		effects++
		if e.Applied {
			t.Fatalf("merge Effect Applied=true, want proposal-only (rung 2 write-back)")
		}
		if !reflect.DeepEqual(e.Cells, []string{"surv", "dup"}) {
			t.Fatalf("merge Effect cells = %v, want [surv dup]", e.Cells)
		}
	}
	if effects != 1 {
		t.Fatalf("merge Effects = %d, want 1", effects)
	}
}

// TestMergeBelowFloorStillDrops pins the guardrail: a below-budget cell with NO
// surviving near-twin above the floor is a genuine eviction — it still reaches the
// typed overflow verdict, the survivor's refs are untouched, and nothing is folded.
// The graceful path never hides a real loss.
func TestMergeBelowFloorStillDrops(t *testing.T) {
	ctx := context.Background()
	seed := []Cell{
		{ID: "surv", Step: 1, Role: "user", Descriptor: "alpha survivor content wholly about tabs and spacing", Bytes: 100, Durability: DurabilityDurable},
		{ID: "far", Step: 2, Role: "tool", Descriptor: "zulu unrelated eviction text concerning weather forecasts", Bytes: 100, Durability: DurabilityTurn, Refs: []string{"r1"}},
	}
	// A high floor (0.9) that the unrelated pair's cosine cannot clear.
	q := Query{Ops: []Op{
		{Kind: OpScan},
		{Kind: OpBudget, Bytes: 100, MergeFloor: 0.9},
	}}
	res, err := Run(ctx, fixedBackend{cells: seed}, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Overflow == nil || len(res.Overflow.Dropped) != 1 || res.Overflow.Dropped[0].ID != "far" {
		t.Fatalf("overflow = %+v, want exactly far dropped (no fold target above floor)", res.Overflow)
	}
	if res.Stats.MergedOnEvict != 0 {
		t.Fatalf("MergedOnEvict = %d, want 0", res.Stats.MergedOnEvict)
	}
	if got := res.Working[0].Refs; len(got) != 0 {
		t.Fatalf("survivor refs = %v, want empty (no fold happened)", got)
	}
	for _, e := range res.Effects {
		if e.Kind == MergeOnEvictKind {
			t.Fatalf("unexpected merge Effect below the floor: %+v", e)
		}
	}
}

// TestMergeDefaultOffByteIdentical pins the default-off contract for #4015: with
// MergeFloor unset the budget op tail-drops exactly as before (the evicted cell reaches
// overflow, no survivor mutation), the render driver with Params.MergeFloor unset builds
// a query DeepEqual to the pre-knob legacy shape, and Validate refuses an out-of-range
// floor.
func TestMergeDefaultOffByteIdentical(t *testing.T) {
	ctx := context.Background()
	seed := []Cell{
		{ID: "surv", Step: 1, Role: "user", Descriptor: "the quick brown fox jumps over the lazy dog near duplicate body", Bytes: 100, Durability: DurabilityDurable},
		{ID: "dup", Step: 2, Role: "tool", Descriptor: "the quick brown fox jumps over the lazy dog near duplicate body extra", Bytes: 100, Durability: DurabilityTurn, Refs: []string{"fact-x"}},
	}
	// No MergeFloor: the near-dup tail-drops the old way, survivor untouched.
	q := Query{Ops: []Op{
		{Kind: OpScan},
		{Kind: OpBudget, Bytes: 100},
	}}
	res, err := Run(ctx, fixedBackend{cells: seed}, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Overflow == nil || len(res.Overflow.Dropped) != 1 || res.Overflow.Dropped[0].ID != "dup" {
		t.Fatalf("unset-path overflow = %+v, want exactly dup dropped (unchanged tail-drop)", res.Overflow)
	}
	if res.Stats.MergedOnEvict != 0 {
		t.Fatalf("unset-path MergedOnEvict = %d, want 0", res.Stats.MergedOnEvict)
	}
	if got := res.Working[0].Refs; len(got) != 0 {
		t.Fatalf("unset-path survivor refs = %v, want empty (no fold)", got)
	}

	// The render driver with MergeFloor unset compiles to the exact legacy shape.
	d, ok := Get("render")
	if !ok {
		t.Fatal("render driver not registered")
	}
	got := d.Build(Params{Intent: "alpha", Budget: 300})
	want := Query{
		Intent: "alpha",
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpFilter, Pred: &Pred{Op: PredAnd, Args: []Pred{
				{Op: PredEq, Field: "sealed", Value: "false"},
				{Op: PredEq, Field: "tombstoned", Value: "false"},
				{Op: PredOr, Args: []Pred{
					{Op: PredMatch, Value: "alpha"},
					{Op: PredEq, Field: "durability", Value: DurabilityDurable},
				}},
			}}},
			{Kind: OpDedup},
			{Kind: OpRank, By: RankRelevance, Desc: true},
			{Kind: OpBudget, Bytes: 300},
			{Kind: OpRender},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("render driver with MergeFloor unset drifted from the legacy query:\n got %+v\nwant %+v", got, want)
	}

	// Fail-closed: a floor outside [0,1] never runs.
	for _, bad := range []float64{-0.1, 1.5} {
		q := Query{Ops: []Op{{Kind: OpBudget, Bytes: 10, MergeFloor: bad}}}
		if err := Validate(q); err == nil {
			t.Fatalf("Validate accepted merge_floor=%v outside [0,1]", bad)
		}
	}
}
