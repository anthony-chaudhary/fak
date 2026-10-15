package model

import (
	"math"
	"reflect"
	"sort"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41RoutedSumOrderStableAllocations(t *testing.T) {
	legacy := func(picks []routePick) []int {
		order := make([]int, len(picks))
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(i, j int) bool { return picks[order[i]].expert < picks[order[j]].expert })
		return order
	}
	cases := [][]routePick{
		nil,
		{},
		{{expert: 9}},
		{{expert: 301}, {expert: 7}, {expert: 199}, {expert: 42}, {expert: 383}, {expert: 0}},
		{{expert: 9}, {expert: 3}, {expert: 9}, {expert: 3}, {expert: 1}, {expert: 9}},
		{{expert: 0}, {expert: 1}, {expert: 2}, {expert: 3}, {expert: 4}, {expert: 5}},
		{{expert: 5}, {expert: 4}, {expert: 3}, {expert: 2}, {expert: 1}, {expert: 0}},
		{{expert: math.MaxInt}, {expert: math.MinInt}, {expert: 0}, {expert: math.MinInt}},
	}
	for _, picks := range cases {
		before := append([]routePick{}, picks...)
		want, got := legacy(picks), v41RoutedSumOrder(picks)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("picks=%v order=%v want=%v", picks, got, want)
		}
		if got == nil {
			t.Fatal("empty order lost the original non-nil slice contract")
		}
		// Every original slot appears once, ordered by expert and then by its
		// original position within each equal-expert group.
		if len(got) != len(picks) {
			t.Fatalf("order length=%d, want one slot per input pick (%d)", len(got), len(picks))
		}
		seen := make([]bool, len(picks))
		for pos, slot := range got {
			if slot < 0 || slot >= len(picks) || seen[slot] {
				t.Fatalf("order=%v is not a permutation of input slots: position=%d slot=%d", got, pos, slot)
			}
			seen[slot] = true
			if pos > 0 {
				previous := got[pos-1]
				if picks[previous].expert > picks[slot].expert {
					t.Fatalf("expert order decreased at position %d: order=%v picks=%v", pos, got, picks)
				}
				if picks[previous].expert == picks[slot].expert && previous >= slot {
					t.Fatalf("equal experts lost original slot order at position %d: order=%v", pos, got)
				}
			}
		}
		for i := range picks {
			if picks[i] != before[i] {
				t.Fatalf("input pick %d mutated", i)
			}
		}
		if len(got) > 0 {
			later := v41RoutedSumOrder(picks)
			got[0] = -1
			if !reflect.DeepEqual(later, want) {
				t.Fatal("returned orders share mutable storage")
			}
		}
	}

	picks := cases[3]
	var retained []int
	measure := func(order func([]routePick) []int) float64 {
		return testing.AllocsPerRun(100, func() { retained = order(picks) })
	}
	old, current := measure(legacy), measure(v41RoutedSumOrder)
	if !reflect.DeepEqual(retained, legacy(picks)) {
		t.Fatal("allocation witness did not retain the requested order")
	}
	t.Logf("top-6 order allocs/op: reflection control=%g typed stable sort=%g", old, current)
	if current != 1 || old-current < 2 {
		t.Fatalf("routed order allocation reduction missing: control=%g candidate=%g, want one owned-result allocation and at least two removed", old, current)
	}
}
