package routing

import (
	"fmt"
	"testing"
)

// fak-test:runtime fast est=5ms
func TestLeastLoadTieBandSpreadsNearEqualAndKeepsBusyBehind(t *testing.T) {
	g := BalanceGroup{Name: "g", Members: []string{"a", "b", "c"}, Strategy: BalanceStrategyLeastLoad}
	snap := BalanceSnapshot{Group: "g", Observed: true, Members: map[string]MemberScore{
		"a": {Load: 0.25, Measured: true},
		"b": {Load: 0.25 + LeastLoadTieBand/2, Measured: true},
		"c": {Load: 0.25 + LeastLoadTieBand + 0.01, Measured: true},
	}}
	heads := map[string]int{}
	for i := 0; i < 400; i++ {
		order := (BalancePolicy{}).Select(g, snap, fmt.Sprintf("req-%d", i))
		heads[order.Members[0]]++
		if order.Members[2] != "c" {
			t.Fatalf("order = %v, want c (outside the band) last", order.Members)
		}
	}
	if heads["a"] < 100 || heads["b"] < 100 {
		t.Fatalf("heads = %v, want a and b to share the head", heads)
	}
}

// fak-test:runtime fast est=5ms
func TestLeastLoadTieBandHonoursDeclaredWeights(t *testing.T) {
	g := BalanceGroup{Name: "g", Members: []string{"a", "b"}, Strategy: BalanceStrategyLeastLoad, Weights: map[string]int{"a": 3, "b": 1}}
	snap := BalanceSnapshot{Group: "g", Observed: true, Members: map[string]MemberScore{
		"a": {Load: 0.2, Measured: true},
		"b": {Load: 0.2, Measured: true},
	}}
	heads := map[string]int{}
	const n = 2000
	for i := 0; i < n; i++ {
		heads[(BalancePolicy{}).Select(g, snap, fmt.Sprintf("req-%d", i)).Members[0]]++
	}
	if share := float64(heads["a"]) / n; share < 0.68 || share > 0.82 {
		t.Fatalf("a won %.3f of ties, want ~0.75 from weights 3:1 (heads %v)", share, heads)
	}
}
