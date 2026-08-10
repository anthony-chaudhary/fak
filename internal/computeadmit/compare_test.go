package computeadmit

import "testing"

func TestCompareLocalKeepsSchedulerAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{"fak native compute-region admission": {"native", true}, "dispatch without region admission": {"baseline", true}, "Kubernetes scheduler": {"external", false}, "Slurm scheduler": {"external", false}, "Ray scheduler": {"external", false}, "AWS Batch": {"external", false}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d want %d: %#v", len(got.Arms), len(want), got.Arms)
	}
	for _, arm := range got.Arms {
		expected, ok := want[arm.Name]
		if !ok {
			t.Fatalf("unexpected arm %q", arm.Name)
		}
		if arm.Kind != expected.kind || arm.Available != expected.available {
			t.Errorf("arm %q=%q available=%v want %q/%v", arm.Name, arm.Kind, arm.Available, expected.kind, expected.available)
		}
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Decisions != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Decisions != 4 {
		t.Fatalf("native=%#v", got.Arms[0])
	}
}

func BenchmarkDecideComputeRegionAdmission(b *testing.B) {
	tax, live, requests := comparisonFixture()
	var got Decision
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, req := range requests {
			got = Decide(req, live, tax)
		}
	}
	if !got.Admit {
		b.Fatal(got)
	}
}
