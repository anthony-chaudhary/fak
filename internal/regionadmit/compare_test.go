package regionadmit

import "testing"

func TestCompareLocalKeepsRegionAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{
		{"fak native shared region admission", "native"},
		{"geometry-only region overlap", "baseline"},
		{"fak + DOS arbitrate", "integration"},
		{"fak + Git-ref leases", "integration"},
		{"Kubernetes Lease coordination", "external"},
		{"etcd concurrency mutex", "external"},
		{"GitHub Actions concurrency groups", "external"},
	}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d want %d", len(got.Arms), len(want))
	}
	for i, arm := range got.Arms {
		if arm.Name != want[i].name || arm.Kind != want[i].kind {
			t.Fatalf("arm[%d]=(%q,%q), want (%q,%q)", i, arm.Name, arm.Kind, want[i].name, want[i].kind)
		}
		if i < 2 {
			if !arm.Available || arm.Decisions != 6 {
				t.Fatalf("local arm[%d] lacks an executed six-decision measurement: %+v", i, arm)
			}
			continue
		}
		if arm.Available || arm.Correct || arm.Latency != 0 || arm.Decisions != 0 || arm.Bytes != 0 || arm.CostUSD != 0 {
			t.Fatalf("unwitnessed arm[%d] reports measurements: %+v", i, arm)
		}
	}
	if !got.Arms[0].Correct {
		t.Fatal("native region admission failed the common verdict oracle")
	}
	if got.Arms[1].Correct {
		t.Fatal("geometry-only baseline unexpectedly satisfies lane/exclusivity semantics")
	}
}

func BenchmarkDecideSharedRegionAdmission(b *testing.B) {
	tax, live, requests := comparisonFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, req := range requests {
			_ = Decide(req, live, tax)
		}
	}
}
