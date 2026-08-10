package attemptbudget

import "testing"

func TestCompareLocalKeepsRetryAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{
		"fak native attempt budget": {"native", true},
		"unlimited retries":         {"baseline", true},
		"Envoy retry budget":        {"external", false},
		"gRPC retry policy":         {"external", false},
		"AWS SDK adaptive retry":    {"external", false},
	}
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
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Attempts != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims a result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Attempts != 3 {
		t.Fatalf("native=%#v", got.Arms[0])
	}
}

func BenchmarkDecideExhausted(b *testing.B) {
	in := Input{IssueID: "same-call", Budget: 3, Attempts: []Attempt{{FailureClass: "schema_mismatch", AtUnix: 1}, {FailureClass: "schema_mismatch", AtUnix: 2}, {FailureClass: "schema_mismatch", AtUnix: 3}}}
	var got Decision
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got = Decide(in)
	}
	if got.Verdict != VerdictStructuralBlock {
		b.Fatal(got)
	}
}
