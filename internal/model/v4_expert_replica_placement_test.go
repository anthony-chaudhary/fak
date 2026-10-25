package model

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestV4ExpertReplicaPlacementPinnedPhysicalVectors(t *testing.T) {
	cases := []struct {
		name     string
		history  [][]int64
		physical []int
		capacity int
		want     []int
	}{
		{"zero-stable", [][]int64{{0, 0, 0, 0, 0, 0}}, []int{0, 1, 2, 3, 4, 5}, 3, []int{0, 1, 2, 3, 4, 5}},
		{"shuffled-zero-stable", [][]int64{{0, 0, 0}}, []int{2, 0, 1, 0, 2, 1}, 3, []int{0, 1, 2, 3, 4, 5}},
		{"tie-low-rank", [][]int64{{10, 0, 0, 0, 0, 0}, {0, 6, 4, 0, 0, 0}}, []int{0, 1, 2, 3, 4, 5}, 3, []int{0, 1, 3, 2, 4, 5}},
		{"flat-complement", [][]int64{{1, 1, 1, 1, 1, 1}}, []int{0, 1, 2, 3, 4, 5}, 3, []int{0, 2, 4, 1, 3, 5}},
		{"multi-step-complement", [][]int64{{10, 0, 6, 0, 4, 0}, {0, 10, 0, 6, 0, 4}}, []int{0, 1, 2, 3, 4, 5}, 3, []int{0, 1, 5, 2, 3, 4}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			counts := make([]int, len(tc.history[0]))
			for _, expert := range tc.physical {
				counts[expert]++
			}
			if got := v4ExpertPhysicalOrder(tc.history, tc.physical, counts, tc.capacity); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("order=%v want=%v", got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV4ExpertReplicaPlacementPinnedReplicaTies(t *testing.T) {
	for _, tc := range []struct {
		name         string
		history      [][]int64
		wantCount    []int
		wantPhysical []int
	}{
		{"all-zero", [][]int64{{0, 0, 0}, {0, 0, 0}}, []int{3, 1, 2}, []int{0, 1, 2, 2, 0, 0}},
		{"equal-nonzero", [][]int64{{5, 5, 5}, {5, 5, 5}}, []int{2, 2, 2}, []int{0, 2, 1, 1, 2, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NewV4ExpertReplicaPlacement(tc.history, 6, 2)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.LogicalCount, tc.wantCount) || !reflect.DeepEqual(got.PhysicalToLogical, tc.wantPhysical) {
				t.Fatalf("placement=%v counts=%v, want placement=%v counts=%v", got.PhysicalToLogical, got.LogicalCount, tc.wantPhysical, tc.wantCount)
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV4ExpertReplicaPlacementDeterministicInverseOwnersAndScoring(t *testing.T) {
	history := [][]int64{{5, 5, 5}, {5, 5, 5}}
	first, err := NewV4ExpertReplicaPlacement(history, 6, 2)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewV4ExpertReplicaPlacement(history, 6, 2)
	if err != nil || !reflect.DeepEqual(first, second) {
		t.Fatalf("non-deterministic: first=%+v second=%+v err=%v", first, second, err)
	}
	for expert, slots := range first.LogicalToPhysical {
		for _, slot := range slots {
			if first.PhysicalToLogical[slot] != expert {
				t.Fatalf("inverse expert=%d slot=%d maps to %d", expert, slot, first.PhysicalToLogical[slot])
			}
		}
	}
	owners, err := first.OwnerRanks(0)
	if err != nil || !reflect.DeepEqual(owners, []int{0, 2}) {
		t.Fatalf("owners=%v err=%v, want [0 2]", owners, err)
	}
	mask, err := first.RankResidentMask(0)
	if err != nil || !reflect.DeepEqual(mask, []bool{true, false, true}) {
		t.Fatalf("rank mask=%v err=%v", mask, err)
	}
	freq := []int64{10, 5, 1}
	got, err := first.ScoreRank(freq, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := ScoreExpertPlacement(freq, mask, 2)
	if got != want || got.Coverage != 11.0/16.0 || got.Drift != 0.5 {
		t.Fatalf("score=%+v want=%+v", got, want)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV4ExpertReplicaPlacementInvalidInputsRefuse(t *testing.T) {
	for _, tc := range []struct {
		name     string
		history  [][]int64
		physical int
		capacity int
	}{
		{"empty", nil, 2, 1},
		{"zero-logical", [][]int64{{}}, 2, 1},
		{"ragged", [][]int64{{1, 2}, {1}}, 2, 1},
		{"negative", [][]int64{{1, -1}}, 2, 1},
		{"too-few-slots", [][]int64{{1, 2, 3}}, 2, 1},
		{"zero-capacity", [][]int64{{1, 2}}, 2, 0},
		{"partial-rank", [][]int64{{1, 2}}, 3, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewV4ExpertReplicaPlacement(tc.history, tc.physical, tc.capacity); !errors.Is(err, ErrV4ExpertReplicaPlacement) {
				t.Fatalf("err=%v", err)
			}
		})
	}

	plan, err := NewV4ExpertReplicaPlacement([][]int64{{1, 2}}, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := plan.OwnerRanks(-1); !errors.Is(err, ErrV4ExpertReplicaPlacement) {
		t.Fatalf("negative expert err=%v", err)
	}
	if _, err := plan.RankResidentMask(2); !errors.Is(err, ErrV4ExpertReplicaPlacement) {
		t.Fatalf("rank err=%v", err)
	}
	if _, err := plan.ScoreRank([]int64{1}, 0, 1); !errors.Is(err, ErrV4ExpertReplicaPlacement) {
		t.Fatalf("width err=%v", err)
	}
	missing := V4ExpertReplicaPlacement{PhysicalToLogical: []int{0}, LogicalToPhysical: [][]int{{0}, {}}, LogicalCount: []int{1, 0}, Capacity: 1}
	if _, err := missing.OwnerRanks(1); !errors.Is(err, ErrV4ExpertReplicaPlacement) {
		t.Fatalf("missing logical expert err=%v", err)
	}
}
