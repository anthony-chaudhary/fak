package astquery

import "testing"

func TestCompareLocalKeepsStructuralSearchAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{{"fak native Go AST query", "native"}, {"literal text search", "baseline"}, {"Semgrep", "external"}, {"ast-grep", "external"}, {"Comby", "external"}, {"gogrep", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, a := range got.Arms {
		if a.Name != want[i].name || a.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, a)
		}
		if i < 2 {
			if !a.Available || a.InputBytes == 0 {
				t.Fatalf("local=%+v", a)
			}
			continue
		}
		if a.Available || a.Correct || a.Latency != 0 || a.TruePositives != 0 || a.FalsePositives != 0 || a.FalseNegatives != 0 || a.BindingErrors != 0 || a.LocationErrors != 0 || a.ParseFailures != 0 || a.InputBytes != 0 || a.CPUSeconds != 0 || a.PeakRSSBytes != 0 || a.NetworkBytes != 0 || a.OperatorSeconds != 0 || a.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", a)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].TruePositives != 2 {
		t.Fatalf("native=%+v", got.Arms[0])
	}
	if got.Arms[1].Correct || got.Arms[1].FalsePositives == 0 {
		t.Fatalf("baseline=%+v", got.Arms[1])
	}
}
func BenchmarkSearchRepeatedMetavariable(b *testing.B) {
	b.ReportAllocs()
	b.SetBytes(int64(len(comparisonSource)))
	for i := 0; i < b.N; i++ {
		ms, err := Search(comparisonSource, comparisonPattern)
		if err != nil || !correctMatches(ms) {
			b.Fatalf("matches=%+v err=%v", ms, err)
		}
	}
}
