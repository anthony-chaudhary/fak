package guardroute

import "testing"

func TestCompareLocalKeepsGuardRouteAlternativesExplicit(t *testing.T) {
	r := CompareLocal()
	want := [][2]string{{"fak native guard-journal route", "native"}, {"count-threshold-only routing", "baseline"}, {"fak + DOS decisions", "integration"}, {"OPA decision policy", "external"}, {"Cedar policy evaluator", "external"}, {"Drools rule engine", "external"}, {"Prometheus Alertmanager routing", "external"}}
	if len(r.Arms) != len(want) {
		t.Fatal(len(r.Arms))
	}
	for i, w := range want {
		a := r.Arms[i]
		if a.Name != w[0] || a.Kind != w[1] {
			t.Fatalf("arm %d %+v", i, a)
		}
		if i >= 2 && (a.Available || a.Correct || a.Latency != 0 || a.Cases != 0 || a.CostUSD != 0) {
			t.Fatalf("unwitnessed measured: %+v", a)
		}
	}
	if a := r.Arms[0]; !a.Correct || a.Passed != 6 {
		t.Fatalf("native: %+v", a)
	}
	if a := r.Arms[1]; a.Correct || a.MissedRoutes != 4 {
		t.Fatalf("baseline: %+v", a)
	}
}
func BenchmarkGuardJournalRoute(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if a := runNativeComparison(); !a.Correct {
			b.Fatal(a)
		}
	}
}
