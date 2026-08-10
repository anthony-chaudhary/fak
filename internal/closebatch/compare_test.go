package closebatch

import "testing"

func TestCompareLocalKeepsCloseBatchAlternativesExplicit(t *testing.T) {
	r := CompareLocal()
	want := [][2]string{{"fak native issue-close batch planner", "native"}, {"fixed-size chunking only", "baseline"}, {"fak + GitHub Issues", "integration"}, {"GitHub CLI issue close loop", "external"}, {"GitHub GraphQL mutation batching", "external"}, {"Jira bulk transition", "external"}, {"Linear bulk issue update", "external"}}
	if len(r.Arms) != len(want) {
		t.Fatal(len(r.Arms))
	}
	for i, w := range want {
		a := r.Arms[i]
		if a.Name != w[0] || a.Kind != w[1] {
			t.Fatal(i, a)
		}
		if i >= 2 && (a.Available || a.Correct || a.Latency != 0 || a.Issues != 0 || a.CostUSD != 0) {
			t.Fatalf("unwitnessed %+v", a)
		}
	}
	if a := r.Arms[0]; !a.Correct || a.Allowed != 2 || a.Held != 1 {
		t.Fatal(a)
	}
	if a := r.Arms[1]; a.Correct || a.FalsePlans == 0 {
		t.Fatal(a)
	}
}
func BenchmarkIssueCloseBatchPlan(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if a := nativeArm(); !a.Correct {
			b.Fatal(a)
		}
	}
}
