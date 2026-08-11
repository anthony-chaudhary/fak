package blastradius

import "testing"

func TestCompareLocalKeepsBlastRadiusAlternativesExplicit(t *testing.T) {
	r := CompareLocal()
	want := [][2]string{{"fak native dependency blast-radius estimator", "native"}, {"broken-tree intersection only", "baseline"}, {"fak + DOS leases", "integration"}, {"Bazel query reverse dependencies", "external"}, {"Pants dependents", "external"}, {"Nx affected graph", "external"}, {"Kubernetes Lease impact labels", "external"}}
	if len(r.Arms) != len(want) {
		t.Fatal(len(r.Arms))
	}
	for i, w := range want {
		a := r.Arms[i]
		if a.Name != w[0] || a.Kind != w[1] {
			t.Fatal(i, a)
		}
		if i >= 2 && (a.Available || a.Correct || a.Latency != 0 || a.Radius != 0 || a.CostUSD != 0) {
			t.Fatalf("unwitnessed %+v", a)
		}
	}
	if a := r.Arms[0]; !a.Correct || a.Radius != 3 || a.HeldLeases != 2 || a.HeldIssues != 1 {
		t.Fatal(a)
	}
	if a := r.Arms[1]; a.Correct || a.Misses != 2 {
		t.Fatal(a)
	}
}
func BenchmarkDependencyBlastRadius(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if a := nativeArm(); !a.Correct {
			b.Fatal(a)
		}
	}
}
