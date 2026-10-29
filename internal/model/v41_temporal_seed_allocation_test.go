package model

import (
	"math"
	"reflect"
	"testing"
)

// The cold producer must construct only the final retained ring, without
// changing public empty-state readiness, validation, or continuation positions.
func TestV41TemporalSeedTarget(t *testing.T) {
	rows := v41DecodeRows(141, 2)
	carriers := [][]float32{{3, 4, 5}}
	old, err := NewV41AttentionState(2, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(old.window) != v41WindowSize || len(old.window[0]) != 2 || old.kvPublications == nil || old.indexPublications == nil {
		t.Fatal("public empty state is no longer ready")
	}
	target, err := newV41TemporalSeedTarget(2, 8)
	if err != nil {
		t.Fatal(err)
	}
	if target.window != nil || target.kvPublications != nil || target.indexPublications != nil {
		t.Fatal("private seed target preallocated discarded storage")
	}
	for _, state := range []*V41AttentionState{old, target} {
		if err := state.seedTemporalWithInputs(rows, carriers, 4, 7); err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(old, target) {
		t.Fatal("cold seed changed final state")
	}
	if target.nextWindowPos != 141 || target.nextCompressRow != 141 || !reflect.DeepEqual(target.PartialGroupPositions(), []int{140}) {
		t.Fatal("absolute continuation positions changed")
	}
	// The final ring and partial carriers still own their bytes.
	rows[140][0], carriers[0][0] = -99, -99
	if target.window[140%v41WindowSize][0] == -99 || target.partialInputs[0][0] == -99 {
		t.Fatal("seed retained borrowed input")
	}
	before := target.clone()
	bad := v41DecodeRows(141, 2)
	bad[140][1] = float32(math.NaN())
	if err := target.seedTemporalWithInputs(bad, [][]float32{{1, 2, 3}}, 4, 7); err == nil {
		t.Fatal("late invalid row accepted")
	}
	if !reflect.DeepEqual(before, target) {
		t.Fatal("failed reseed mutated existing state")
	}
	fresh, _ := newV41TemporalSeedTarget(2, 8)
	if err := fresh.seedTemporalWithInputs(bad, nil, 4, 7); err == nil {
		t.Fatal("invalid fresh seed accepted")
	}
	if fresh.window != nil || fresh.nextWindowPos != 0 || fresh.kvPublications != nil {
		t.Fatal("failed fresh seed published half state")
	}
	for _, dimensions := range [][2]int{{0, 0}, {2, 0}} {
		_, oldErr := NewV41AttentionState(dimensions[0], dimensions[1])
		got, newErr := newV41TemporalSeedTarget(dimensions[0], dimensions[1])
		if got != nil || oldErr == nil || newErr == nil || oldErr.Error() != newErr.Error() {
			t.Fatal("geometry refusal changed")
		}
	}
}

func TestV41TemporalSeedAvoidsDiscardedRingAllocations(t *testing.T) {
	rows := v41DecodeRows(141, 2)
	seed := func(public bool) {
		var state *V41AttentionState
		var err error
		if public {
			state, err = NewV41AttentionState(2, 8)
		} else {
			state, err = newV41TemporalSeedTarget(2, 8)
		}
		if err != nil {
			panic(err)
		}
		if err := state.seedTemporal(rows, 4, 7); err != nil {
			panic(err)
		}
		if state.nextWindowPos != len(rows) {
			panic("seed missing")
		}
	}
	legacy := testing.AllocsPerRun(5, func() { seed(true) })
	fresh := testing.AllocsPerRun(5, func() { seed(false) })
	if fresh >= legacy {
		t.Fatalf("fresh allocations %v, want less than legacy %v", fresh, legacy)
	}
}
