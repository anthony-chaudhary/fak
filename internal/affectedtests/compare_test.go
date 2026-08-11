package affectedtests

import "testing"

func TestCompareLocalKeepsAffectedTestAlternativesExplicit(t *testing.T) {
	r := CompareLocal()
	want := [][2]string{{"fak native reverse-dependency affected-test selection", "native"}, {"changed-package tests only", "baseline"}, {"fak + Go test", "integration"}, {"Bazel reverse-dependency query + test-work", "external"}, {"Pants changed-since test selection", "external"}, {"Nx affected", "external"}, {"Gradle configured task closure", "external"}}
	if len(r.Arms) != len(want) {
		t.Fatal(len(r.Arms))
	}
	for i, w := range want {
		a := r.Arms[i]
		if a.Name != w[0] || a.Kind != w[1] {
			t.Fatal(i, a)
		}
		if i >= 2 && i != 4 && (!a.Available || !a.Correct || a.Selected != 4 || a.Misses != 0 || a.FalseIncludes != 0) {
			t.Fatalf("witnessed arm %d: %+v", i, a)
		}
	}
	if a := r.Arms[0]; !a.Correct || a.Selected != 4 {
		t.Fatal(a)
	}
	if a := r.Arms[1]; a.Correct || a.Misses != 3 {
		t.Fatal(a)
	}
	if a := r.Arms[4]; a.Available || a.Correct || a.Selected != 0 || a.Misses != 4 || a.Latency != 0 || a.CostUSD != 0 {
		t.Fatalf("unavailable Pants arm must remain measurement-zero: %+v", a)
	}
}
func BenchmarkAffectedTestSelection(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if a := nativeArm(); !a.Correct {
			b.Fatal(a)
		}
	}
}
