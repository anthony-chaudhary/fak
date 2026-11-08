package model

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestLoadBalancedExpertBandsMaxInt(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		load   []int
		ranks  int
		bounds []int
	}{
		{[]int{maxInt}, 1, []int{0, 1}},
		{[]int{maxInt, 0}, 2, []int{0, 1, 2}},
		{[]int{0, maxInt}, 2, []int{0, 1, 2}},
		{[]int{0, maxInt, 0}, 3, []int{0, 1, 2, 3}},
		{[]int{0, 0, maxInt}, 2, []int{0, 1, 3}},
		{[]int{maxInt - 1, 1}, 2, []int{0, 1, 2}},
	} {
		plan, err := LoadBalancedExpertBands(tc.load, tc.ranks)
		if err != nil {
			t.Fatalf("load=%v ranks=%d: %v", tc.load, tc.ranks, err)
		}
		if err := plan.Validate(); err != nil {
			t.Fatal(err)
		}
		bounds := []int{plan.Shards[0].Lo}
		for _, shard := range plan.Shards {
			bounds = append(bounds, shard.Hi)
		}
		if !reflect.DeepEqual(bounds, tc.bounds) {
			t.Fatalf("load=%v ranks=%d bounds=%v want %v", tc.load, tc.ranks, bounds, tc.bounds)
		}
	}
	for _, load := range [][]int{{maxInt, 1}, {maxInt - 1, 1, 1}} {
		before := append([]int(nil), load...)
		for ranks := 1; ranks <= len(load); ranks++ {
			plan, err := LoadBalancedExpertBands(load, ranks)
			if err == nil || !reflect.DeepEqual(plan, TPPlan{}) {
				t.Fatalf("overflow load=%v ranks=%d returned plan=%+v err=%v", load, ranks, plan, err)
			}
		}
		if !reflect.DeepEqual(load, before) {
			t.Fatal("overflow refusal mutated loads")
		}
	}
}
