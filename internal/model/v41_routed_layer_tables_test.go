package model

import (
	"math"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41RoutedLayerProjectionTables(t *testing.T) {
	legacy := func(picks int) (gates, ups, downs []compute.Tensor) {
		return make([]compute.Tensor, picks), make([]compute.Tensor, picks), make([]compute.Tensor, picks)
	}
	weights := make([]v41RoutedLayerWeights, compute.V41RoutedIDMaxPicks)
	for slot := range weights {
		newWeight := func(projection int) compute.Tensor {
			return compute.NewF32(compute.Default(), []int{1, 1}, []float32{float32(3*slot + projection + 1)})
		}
		weights[slot] = v41RoutedLayerWeights{newWeight(0), newWeight(1), newWeight(2)}
	}
	fill := func(gates, ups, downs []compute.Tensor) {
		for slot := range gates {
			gates[slot], ups[slot], downs[slot] = weights[slot].gate, weights[slot].up, weights[slot].down
		}
	}
	equal := func(got, want compute.Tensor) bool {
		return got.Buf() == want.Buf() && got.Backend() == want.Backend() && got.Dtype == want.Dtype &&
			got.Layout == want.Layout && got.Quant == want.Quant && slices.Equal(got.Shape, want.Shape) &&
			(len(got.Shape) == 0 || &got.Shape[0] == &want.Shape[0])
	}
	check := func(got, want [3][]compute.Tensor) {
		t.Helper()
		for projection := range got {
			if got[projection] == nil || len(got[projection]) != len(want[projection]) || cap(got[projection]) != len(want[projection]) {
				t.Fatalf("projection %d: nil=%t len/cap=%d/%d want=%d", projection, got[projection] == nil,
					len(got[projection]), cap(got[projection]), len(want[projection]))
			}
			for slot := range want[projection] {
				if !equal(got[projection][slot], want[projection][slot]) {
					t.Fatalf("projection %d slot %d descriptor or borrowed handle changed", projection, slot)
				}
			}
		}
	}
	sentinel := compute.NewF32(compute.Default(), []int{1}, []float32{-1})
	for _, picks := range []int{0, 1, 6, compute.V41RoutedIDMaxPicks} {
		gates, ups, downs := v41RoutedLayerProjectionTables(picks)
		oldGates, oldUps, oldDowns := legacy(picks)
		got, want := [3][]compute.Tensor{gates, ups, downs}, [3][]compute.Tensor{oldGates, oldUps, oldDowns}
		check(got, want) // All descriptors start zeroed, including the non-nil empty case.
		fill(gates, ups, downs)
		fill(oldGates, oldUps, oldDowns)
		check(got, want)
		nextGates, nextUps, nextDowns := v41RoutedLayerProjectionTables(picks)
		fill(nextGates, nextUps, nextDowns)
		next := [3][]compute.Tensor{nextGates, nextUps, nextDowns}
		for projection := range got {
			appended := append(got[projection], sentinel)
			if len(appended) != picks+1 || !equal(appended[picks], sentinel) {
				t.Fatalf("projection %d append lost its descriptor", projection)
			}
			check(got, want) // Append cannot overwrite the adjacent table.
			if picks > 0 {
				got[projection][0] = sentinel
				for other := range got {
					if other != projection && !equal(got[other][0], want[other][0]) {
						t.Fatalf("projection %d descriptor write changed projection %d", projection, other)
					}
				}
				check(next, want) // A retained result cannot alias a later call's tables.
				if !equal(weights[0].gate, oldGates[0]) || !equal(weights[0].up, oldUps[0]) || !equal(weights[0].down, oldDowns[0]) {
					t.Fatal("descriptor write changed the staged input weights")
				}
				got[projection][0] = want[projection][0]
			}
		}
	}

	// The production callers allocate only after these guards. Invalid pick
	// counts and mismatched weights must decline before invoking a backend.
	if compute.V41RoutedIDMaxPicks > math.MaxInt/3 {
		t.Fatal("admitted projection-table size can overflow")
	}
	for _, picks := range []int{math.MinInt, -1, 0, compute.V41RoutedIDMaxPicks + 1, math.MaxInt, 1} {
		if v41RoutedIDAdmits(nil, nil, picks) || v41RoutedFusedAdmits(nil, nil, picks) {
			t.Fatalf("invalid picks=%d or weights length admitted", picks)
		}
	}

	var gates, ups, downs []compute.Tensor
	measure := func(allocate func(int) ([]compute.Tensor, []compute.Tensor, []compute.Tensor)) float64 {
		return testing.AllocsPerRun(100, func() {
			gates, ups, downs = allocate(6)
			fill(gates, ups, downs)
		})
	}
	old, current := measure(legacy), measure(v41RoutedLayerProjectionTables)
	oldGates, oldUps, oldDowns := legacy(6)
	fill(oldGates, oldUps, oldDowns)
	check([3][]compute.Tensor{gates, ups, downs}, [3][]compute.Tensor{oldGates, oldUps, oldDowns})
	t.Logf("top-6 projection table allocs/op: three-make control=%g slab=%g", old, current)
	if old != 3 || current != 1 {
		t.Fatalf("projection-table allocation reduction missing: control=%g candidate=%g, want 3 and 1", old, current)
	}
}
