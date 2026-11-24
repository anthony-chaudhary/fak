package routing

import (
	"slices"
	"strings"
	"testing"
)

// haloQuietSnapshot is the 2026-10-05 live shape: a signal owner answered for
// every Halo, and none of them carries a fresh reading because nothing is in
// flight and the last answer aged past the cutoff.
func haloQuietSnapshot(observed bool) (BalanceGroup, BalanceSnapshot) {
	return BalanceGroup{Name: "halo-qwen38", Members: []string{"halo-strix3", "halo-strix1", "halo-strix2"}},
		BalanceSnapshot{Group: "halo-qwen38", Observed: observed, Members: map[string]MemberScore{
			"halo-strix3": {Measured: false},
			"halo-strix1": {Measured: false},
			"halo-strix2": {Measured: false},
		}}
}

// fak-test:runtime fast est=1s
func TestBalanceSelectQuietObservedGroupSpreadsAcrossMembers(t *testing.T) {
	for _, strategy := range []BalanceStrategy{BalanceStrategyLeastLoad, BalanceStrategyLeastLatency} {
		t.Run(string(strategy), func(t *testing.T) {
			group, snap := haloQuietSnapshot(true)
			group.Strategy = strategy
			var p BalancePolicy

			const n = 3000
			chosen := map[string]int{}
			for i := 0; i < n; i++ {
				o := p.Select(group, snap, "req-"+itoa(i))
				if o.Plan.Reason != "" {
					t.Fatalf("req-%d reason = %q, want a balanced decision for a quiet observed group", i, o.Plan.Reason)
				}
				if len(o.Members) != 3 || o.Plan.Chosen != o.Members[0] {
					t.Fatalf("req-%d order = %v chosen = %q, want all three members with the head chosen", i, o.Members, o.Plan.Chosen)
				}
				chosen[o.Plan.Chosen]++
			}
			for _, m := range group.Members {
				within(t, "share["+m+"]", float64(chosen[m])/n, 1.0/3.0, 0.05)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestBalanceSelectQuietGroupReceiptNamesTheExploration(t *testing.T) {
	group, snap := haloQuietSnapshot(true)
	group.Strategy = BalanceStrategyLeastLoad
	o := (BalancePolicy{}).Select(group, snap, "Qwen3.8-27B-UD-Q2_K_XL")
	if o.Plan.Chosen == "" {
		t.Fatal("a quiet observed group named no chosen member")
	}
	if !slices.ContainsFunc(o.Plan.Audit, func(l string) bool { return strings.Contains(l, "explored=all_unmeasured") }) {
		t.Fatalf("audit = %v, want the all-unmeasured exploration recorded", o.Plan.Audit)
	}
	again := (BalancePolicy{}).Select(group, snap, "Qwen3.8-27B-UD-Q2_K_XL")
	if !slices.Equal(o.Members, again.Members) {
		t.Fatalf("same key gave %v then %v; the quiet order must stay replayable", o.Members, again.Members)
	}
}

// fak-test:runtime fast est=1s
func TestBalanceSelectQuietGroupKeepsDeclaredOrderWithoutExploration(t *testing.T) {
	cases := []struct {
		name     string
		strategy BalanceStrategy
		observed bool
		group    string
	}{
		{"no signal owner", BalanceStrategyLeastLoad, false, "halo-qwen38"},
		{"weighted-hash does not explore", BalanceStrategyWeightedHash, true, "halo-qwen38"},
		{"priority does not explore", BalanceStrategyPriority, true, "halo-qwen38"},
		{"snapshot for another group", BalanceStrategyLeastLoad, true, "other"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			group, snap := haloQuietSnapshot(tc.observed)
			group.Strategy = tc.strategy
			snap.Group = tc.group
			for i := 0; i < 50; i++ {
				o := (BalancePolicy{}).Select(group, snap, "req-"+itoa(i))
				if o.Plan.Reason != BalanceExcludeNoSnapshot {
					t.Fatalf("req-%d reason = %q, want %q", i, o.Plan.Reason, BalanceExcludeNoSnapshot)
				}
				if !slices.Equal(o.Members, group.Members) {
					t.Fatalf("req-%d order = %v, want the declared order %v", i, o.Members, group.Members)
				}
			}
		})
	}
}
