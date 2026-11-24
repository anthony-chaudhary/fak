package routing

import (
	"math"
	"strconv"
	"testing"
)

// TestRegressionWeightedHashColdMemberIsNotStarved pins the cold-start
// starvation LB-15 found on a live router: under weighted-hash a member is only
// measured by serving, so demoting an unmeasured member once its peer is
// measured left it at the tail forever and the group never split. A 3:1 group
// with only the head measured must still give the cold member its declared
// quarter of first picks, and never mark it excluded.
//
// fak-test:runtime fast
func TestRegressionWeightedHashColdMemberIsNotStarved(t *testing.T) {
	group := BalanceGroup{
		Name:     "tier",
		Members:  []string{"alpha", "beta"},
		Strategy: BalanceStrategyWeightedHash,
		Weights:  map[string]int{"alpha": 3, "beta": 1},
	}
	snap := BalanceSnapshot{Group: "tier", Members: map[string]MemberScore{
		"alpha": {Load: 0, Measured: true},
		"beta":  {Measured: false},
	}}
	var p BalancePolicy
	const n = 2000
	firsts := map[string]int{}
	for i := 0; i < n; i++ {
		o := p.Select(group, snap, "cold-"+strconv.Itoa(i))
		for _, line := range o.Plan.Members {
			if line.Member == "beta" && line.Excluded == BalanceExcludeUnmeasured {
				t.Fatalf("key %d: cold member beta excluded as %q; it can never earn the measurement it lacks", i, line.Excluded)
			}
		}
		firsts[o.Members[0]]++
	}
	if share := float64(firsts["beta"]) / n; math.Abs(share-0.25) > 0.05 {
		t.Fatalf("cold member beta first-pick share = %.3f, want 0.25 ±0.05 (counts %v)", share, firsts)
	}
}
