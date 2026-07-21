package cachemeta

import (
	"reflect"
	"testing"
)

func TestDiffBlockStatesNoChange(t *testing.T) {
	snap := map[string]BlockTierState{
		"aaa": {Tier: TierHBM, Priority: 5},
		"bbb": {Tier: TierDRAM, Priority: 2},
	}
	// A byte-identical current snapshot emits nothing.
	if got := DiffBlockStates(snap, snap); got != nil {
		t.Fatalf("no-change diff = %v, want nil", got)
	}
	// A distinct-but-equal-valued current snapshot also emits nothing.
	cur := map[string]BlockTierState{
		"aaa": {Tier: TierHBM, Priority: 5},
		"bbb": {Tier: TierDRAM, Priority: 2},
	}
	if got := DiffBlockStates(snap, cur); got != nil {
		t.Fatalf("equal-valued diff = %v, want nil", got)
	}
}

func TestDiffBlockStatesAdded(t *testing.T) {
	prev := map[string]BlockTierState{"aaa": {Tier: TierHBM, Priority: 1}}
	cur := map[string]BlockTierState{
		"aaa": {Tier: TierHBM, Priority: 1},
		"zzz": {Tier: TierDRAM, Priority: 7},
	}
	got := DiffBlockStates(prev, cur)
	want := []BlockDelta{{
		Hash:          "zzz",
		Kind:          BlockAdded,
		NewTier:       TierDRAM,
		NewPriority:   7,
		TierMoved:     true,
		PriorityMoved: true,
	}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("added diff = %#v, want %#v", got, want)
	}
}

func TestDiffBlockStatesRemoved(t *testing.T) {
	prev := map[string]BlockTierState{
		"aaa": {Tier: TierHBM, Priority: 1},
		"ddd": {Tier: TierDisk, Priority: 3},
	}
	cur := map[string]BlockTierState{"aaa": {Tier: TierHBM, Priority: 1}}
	got := DiffBlockStates(prev, cur)
	want := []BlockDelta{{
		Hash:          "ddd",
		Kind:          BlockRemoved,
		OldTier:       TierDisk,
		OldPriority:   3,
		TierMoved:     true,
		PriorityMoved: true,
	}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("removed diff = %#v, want %#v", got, want)
	}
}

func TestDiffBlockStatesTierChange(t *testing.T) {
	prev := map[string]BlockTierState{"aaa": {Tier: TierHBM, Priority: 4}}
	cur := map[string]BlockTierState{"aaa": {Tier: TierDRAM, Priority: 4}}
	got := DiffBlockStates(prev, cur)
	want := []BlockDelta{{
		Hash:          "aaa",
		Kind:          BlockChanged,
		OldTier:       TierHBM,
		NewTier:       TierDRAM,
		OldPriority:   4,
		NewPriority:   4,
		TierMoved:     true,
		PriorityMoved: false,
	}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tier-change diff = %#v, want %#v", got, want)
	}
}

func TestDiffBlockStatesPriorityChange(t *testing.T) {
	prev := map[string]BlockTierState{"aaa": {Tier: TierHBM, Priority: 4}}
	cur := map[string]BlockTierState{"aaa": {Tier: TierHBM, Priority: 9}}
	got := DiffBlockStates(prev, cur)
	want := []BlockDelta{{
		Hash:          "aaa",
		Kind:          BlockChanged,
		OldTier:       TierHBM,
		NewTier:       TierHBM,
		OldPriority:   4,
		NewPriority:   9,
		TierMoved:     false,
		PriorityMoved: true,
	}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("priority-change diff = %#v, want %#v", got, want)
	}
}

func TestDiffBlockStatesBothAxesChange(t *testing.T) {
	prev := map[string]BlockTierState{"aaa": {Tier: TierHBM, Priority: 4}}
	cur := map[string]BlockTierState{"aaa": {Tier: TierDisk, Priority: 1}}
	got := DiffBlockStates(prev, cur)
	if len(got) != 1 {
		t.Fatalf("both-axes diff len = %d, want 1", len(got))
	}
	if !got[0].TierMoved || !got[0].PriorityMoved {
		t.Fatalf("both-axes diff = %#v, want both moved flags set", got[0])
	}
}

func TestDiffBlockStatesOrderedDeterministic(t *testing.T) {
	prev := map[string]BlockTierState{
		"m": {Tier: TierHBM, Priority: 1},
		"a": {Tier: TierHBM, Priority: 1},
	}
	cur := map[string]BlockTierState{
		"z": {Tier: TierDRAM, Priority: 2}, // added
		"a": {Tier: TierDisk, Priority: 1}, // tier change
		// "m" removed
	}
	first := DiffBlockStates(prev, cur)
	// Ordered by hash: a (change), m (remove), z (add).
	wantHashes := []string{"a", "m", "z"}
	if len(first) != len(wantHashes) {
		t.Fatalf("diff len = %d, want %d (%#v)", len(first), len(wantHashes), first)
	}
	for i, h := range wantHashes {
		if first[i].Hash != h {
			t.Fatalf("event[%d].Hash = %q, want %q", i, first[i].Hash, h)
		}
	}
	// Determinism: repeated calls with the same inputs are identical.
	for i := 0; i < 5; i++ {
		again := DiffBlockStates(prev, cur)
		if !reflect.DeepEqual(first, again) {
			t.Fatalf("non-deterministic diff on run %d: %#v vs %#v", i, first, again)
		}
	}
}

func TestDiffBlockStatesEmptyFailsClosed(t *testing.T) {
	if got := DiffBlockStates(nil, nil); got != nil {
		t.Fatalf("nil/nil diff = %v, want nil", got)
	}
	if got := DiffBlockStates(map[string]BlockTierState{}, map[string]BlockTierState{}); got != nil {
		t.Fatalf("empty/empty diff = %v, want nil", got)
	}
	// nil previous with a populated current = every block added.
	cur := map[string]BlockTierState{"aaa": {Tier: TierHBM, Priority: 1}}
	got := DiffBlockStates(nil, cur)
	if len(got) != 1 || got[0].Kind != BlockAdded {
		t.Fatalf("nil-prev diff = %#v, want one add", got)
	}
}
