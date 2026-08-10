package trigram

import "testing"

func TestCompareLocalKeepsIndexedSearchAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{{"fak native trigram indexed literal search", "native"}, {"optimized in-memory linear scan", "baseline"}, {"ripgrep", "external"}, {"git grep", "external"}, {"Zoekt", "external"}, {"livegrep", "external"}, {"Sourcegraph Search", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, a := range got.Arms {
		if a.Name != want[i].name || a.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, a)
		}
		if i < 2 {
			if !a.Available || !a.Correct || a.Queries != 4 || a.ExactQueries != 4 || a.CorpusBytes == 0 {
				t.Fatalf("local=%+v", a)
			}
			continue
		}
		if a.Available || a.Correct || a.BuildLatency != 0 || a.QueryLatency != 0 || a.Queries != 0 || a.ExactQueries != 0 || a.FalsePositives != 0 || a.FalseNegatives != 0 || a.LocationErrors != 0 || a.CPUSeconds != 0 || a.PeakRSSBytes != 0 || a.CorpusBytes != 0 || a.IndexBytes != 0 || a.NetworkBytes != 0 || a.StorageBytes != 0 || a.OperatorSeconds != 0 || a.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", a)
		}
	}
}
func BenchmarkBuildAndQueryLiteralCorpus(b *testing.B) {
	b.ReportAllocs()
	b.SetBytes(corpusBytes())
	for i := 0; i < b.N; i++ {
		ix := buildComparisonIndex()
		for _, q := range comparisonQueries {
			if !expectedQuery(q, ix.Search(q)) {
				b.Fatalf("q=%s", q)
			}
		}
	}
}
