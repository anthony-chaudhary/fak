package issuededup

import "testing"

func TestCompareLocalKeepsIssueDedupAlternativesExplicit(t *testing.T) {
	r := CompareLocal()
	want := [][2]string{{"fak native issue near-duplicate gate", "native"}, {"normalized exact-title equality", "baseline"}, {"fak + GitHub issue search", "integration"}, {"GitHub duplicate issue detection", "external"}, {"Linear duplicate detection", "external"}, {"Jira similar requests", "external"}, {"sentence-transformer cosine retrieval", "external"}}
	if len(r.Arms) != len(want) {
		t.Fatalf("arms=%d want %d", len(r.Arms), len(want))
	}
	for i, w := range want {
		a := r.Arms[i]
		if a.Name != w[0] || a.Kind != w[1] {
			t.Fatalf("arm[%d]=%q/%q", i, a.Name, a.Kind)
		}
		if i >= 2 && (a.Available || a.Correct || a.Latency != 0 || a.Cases != 0 || a.CPUSeconds != 0 || a.PeakRSSBytes != 0 || a.CostUSD != 0) {
			t.Fatalf("unwitnessed arm measured: %+v", a)
		}
	}
	n := r.Arms[0]
	if !n.Correct || n.TruePositives != 2 || n.TrueNegatives != 1 || n.FalsePositives != 0 || n.FalseNegatives != 0 {
		t.Fatalf("native: %+v", n)
	}
	b := r.Arms[1]
	if b.Correct || b.FalseNegatives != 2 || b.TrueNegatives != 1 {
		t.Fatalf("baseline: %+v", b)
	}
}
func BenchmarkIssueNearDuplicateGate(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if a := runNativeComparison(); !a.Correct {
			b.Fatalf("native: %+v", a)
		}
	}
}
