package testroute

import "testing"

func TestCompareLocalKeepsTestRouteAlternativesExplicit(t *testing.T) {
	r := CompareLocal()
	want := [][2]string{{"fak native test-route decision", "native"}, {"GOOS-only native-or-CI rule", "baseline"}, {"fak + GitHub Actions", "integration"}, {"Go toolchain native execution", "external"}, {"WSL test wrapper", "external"}, {"GitHub Actions workflow routing", "external"}, {"Bazel platform constraints", "external"}}
	if len(r.Arms) != len(want) {
		t.Fatal(len(r.Arms))
	}
	for i, w := range want {
		a := r.Arms[i]
		if a.Name != w[0] || a.Kind != w[1] {
			t.Fatal(i, a)
		}
		if i >= 2 && (a.Available || a.Correct || a.Latency != 0 || a.Cases != 0 || a.CostUSD != 0) {
			t.Fatalf("unwitnessed: %+v", a)
		}
	}
	if a := r.Arms[0]; !a.Correct || a.Passed != 4 {
		t.Fatal(a)
	}
	if a := r.Arms[1]; a.Correct || a.FalseRoutes == 0 {
		t.Fatal(a)
	}
}
func BenchmarkTestRouteDecision(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if a := nativeArm(); !a.Correct {
			b.Fatal(a)
		}
	}
}
