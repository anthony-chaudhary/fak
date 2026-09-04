package dispatchauto

import "testing"

// BenchmarkDispatchAuto measures the throughput and allocation overhead of the
// pure automated dispatch sizing and placement algorithm.
func BenchmarkDispatchAuto(b *testing.B) {
	in := Input{
		EffectiveCap:        16,
		DistinctPools:       12,
		ReadyWork:           8,
		RequiredWorkers:     6,
		LiveWorkers:         2,
		SharedContextTokens: 64_000,
		Nodes: []Node{
			{Name: "worker-1", SeatCap: 4, Live: 1, Healthy: true},
			{Name: "worker-2", SeatCap: 4, Live: 1, Healthy: true},
			{Name: "worker-3", SeatCap: 4, Live: 0, Healthy: false},
			{Name: "worker-4", SeatCap: 4, Live: 0, Healthy: true},
		},
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		plan := PlanAuto(in)
		if plan.Target <= 0 {
			b.Fatalf("unexpected target %d", plan.Target)
		}
	}
}
