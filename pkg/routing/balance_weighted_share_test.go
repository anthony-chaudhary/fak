package routing

import "testing"

// fak-test:runtime fast est=50ms
func TestWeightedShareUsesDeclaredRequestsWithoutMeasurements(t *testing.T) {
	g := BalanceGroup{Name: "share", Members: []string{"first", "second", "paid", "drained"}, Strategy: BalanceStrategyWeightedShare,
		Weights: map[string]int{"first": 2, "second": 2, "paid": 1, "drained": 0}}
	busy := loads(g.Name, map[string]float64{"first": 0.2, "second": 0.8, "paid": 1, "drained": 0}, g.Members...)
	busy.Members["paid"] = MemberScore{Load: 1, Latency: 1, Measured: true}
	snapshots := []BalanceSnapshot{{}, {Group: "foreign"}, {Group: g.Name, Observed: true}, busy}
	counts := map[string]int{}
	const requests = 5000
	for i := 0; i < requests; i++ {
		key := "request-" + itoa(i)
		baseline := (BalancePolicy{}).Select(g, snapshots[0], key)
		counts[baseline.Plan.Chosen]++
		for _, snap := range snapshots {
			order := (BalancePolicy{}).Select(g, snap, key)
			if order.Plan.Reason != "" || order.Plan.Chosen != baseline.Plan.Chosen || len(order.Members) != len(g.Members) {
				t.Fatalf("key %s order = %+v, want snapshot-independent declared allocation", key, order)
			}
			for _, member := range order.Plan.Members {
				if member.EffectiveWeight != float64(g.WeightOf(member.Member)) {
					t.Fatalf("member %s effective weight = %v, want declared weight", member.Member, member.EffectiveWeight)
				}
			}
		}
	}
	for member, want := range map[string]float64{"first": 0.4, "second": 0.4, "paid": 0.2, "drained": 0} {
		within(t, member, float64(counts[member])/requests, want, 0.04)
	}
	if counts["drained"] != 0 {
		t.Fatalf("drained member received %d requests", counts["drained"])
	}
}

// fak-test:runtime fast est=50ms
func TestWeightedShareRetunesDeclaredWeights(t *testing.T) {
	g := BalanceGroup{Name: "share", Members: []string{"first", "second", "paid"}, Strategy: BalanceStrategyWeightedShare,
		Weights: map[string]int{"first": 1, "second": 1, "paid": 8}}
	counts := map[string]int{}
	const requests = 3000
	for i := 0; i < requests; i++ {
		order := (BalancePolicy{}).Select(g, BalanceSnapshot{}, "request-"+itoa(i))
		counts[order.Plan.Chosen]++
	}
	within(t, "retuned paid allocation", float64(counts["paid"])/requests, 0.8, 0.04)
}
