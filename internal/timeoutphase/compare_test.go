package timeoutphase

import "testing"

func TestCompareLocalKeepsTracingAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{
		"fak native timeout-phase classifier": {"native", true}, "one undifferentiated timeout bucket": {"baseline", true},
		"OpenTelemetry spans": {"external", false}, "Datadog APM": {"external", false}, "AWS X-Ray": {"external", false}}
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
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Rows != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Rows != 6 {
		t.Fatalf("native=%#v", got.Arms[0])
	}
}
func BenchmarkClassifyTimeoutPhase(b *testing.B) {
	in := Attempt{ID: "test", Started: true, LastStage: StageTest}
	var got Row
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got = Classify(in)
	}
	if got.Phase != PhaseDuringTests {
		b.Fatal(got)
	}
}
