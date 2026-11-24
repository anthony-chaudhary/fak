package routing

import "testing"

// halo is the 2026-10-05 shape: one member serving with a real load, two that
// have never served and so carry no measurement.
func haloColdSnapshot(warm MemberScore) (BalanceGroup, BalanceSnapshot) {
	return BalanceGroup{Name: "halo-qwen38", Members: []string{"halo-strix3", "halo-strix1", "halo-strix2"}},
		BalanceSnapshot{Group: "halo-qwen38", Members: map[string]MemberScore{
			"halo-strix3": warm,
			"halo-strix1": {Measured: false},
			"halo-strix2": {Measured: false},
		}}
}

// fak-test:runtime fast est=1s
func TestBalanceSelectLeastLoadExploresColdMember(t *testing.T) {
	group, snap := haloColdSnapshot(MemberScore{Load: 0.4, Measured: true})
	group.Strategy = BalanceStrategyLeastLoad
	var p BalancePolicy

	chosen := map[string]int{}
	for i := 0; i < 200; i++ {
		o := p.Select(group, snap, "k"+itoa(i))
		if o.Plan.Reason != "" {
			t.Fatalf("k%d failed open with %q; one measured member must balance the group", i, o.Plan.Reason)
		}
		if len(o.Members) != 3 {
			t.Fatalf("k%d order = %v, want all three members", i, o.Members)
		}
		chosen[o.Plan.Chosen]++
		for _, m := range o.Plan.Members {
			if m.Member != "halo-strix3" && m.Excluded != "" {
				t.Fatalf("k%d cold member %s excluded = %q, want it competing", i, m.Member, m.Excluded)
			}
		}
	}
	if chosen["halo-strix3"] != 0 {
		t.Fatalf("chosen = %v, want a loaded member never ahead of idle cold peers", chosen)
	}
	if chosen["halo-strix1"] == 0 || chosen["halo-strix2"] == 0 {
		t.Fatalf("chosen = %v, want both cold members explored", chosen)
	}
}

// fak-test:runtime fast est=1s
func TestBalanceSelectLeastLatencyExploresColdMember(t *testing.T) {
	group, snap := haloColdSnapshot(MemberScore{Latency: 0.5, Measured: true})
	group.Strategy = BalanceStrategyLeastLatency
	var p BalancePolicy

	for i := 0; i < 50; i++ {
		if o := p.Select(group, snap, "k"+itoa(i)); o.Plan.Chosen == "halo-strix3" {
			t.Fatalf("k%d chose the slower measured member over unsampled cold peers", i)
		}
	}
}

// fak-test:runtime fast est=1s
func TestBalanceSelectColdMemberDoesNotStarveIdleMeasuredPeer(t *testing.T) {
	group, snap := haloColdSnapshot(MemberScore{Load: 0, Measured: true})
	group.Strategy = BalanceStrategyLeastLoad
	var p BalancePolicy

	const n = 3000
	warm := 0
	for i := 0; i < n; i++ {
		if p.Select(group, snap, "k"+itoa(i)).Plan.Chosen == "halo-strix3" {
			warm++
		}
	}
	within(t, "share[idle measured member]", float64(warm)/n, 1.0/3.0, 0.05)
}
