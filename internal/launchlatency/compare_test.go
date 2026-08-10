package launchlatency

import "testing"

func TestCompareLocalKeepsTelemetryAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{
		"fak native launch-latency summary": {"native", true}, "raw launch events without summary": {"baseline", true},
		"Prometheus histogram": {"external", false}, "OpenTelemetry metrics": {"external", false}, "Datadog distribution metric": {"external", false},
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
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Samples != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Samples != 6 {
		t.Fatalf("native=%#v", got.Arms[0])
	}
}

func BenchmarkLaunchLatencySummary(b *testing.B) {
	launches := []Launch{{DispatchSec: 10, HeartbeatSec: 11}, {DispatchSec: 10, HeartbeatSec: 12}, {DispatchSec: 10, HeartbeatSec: 15}, {DispatchSec: 10, HeartbeatSec: 20}, {DispatchSec: 10, HeartbeatSec: 40}, {DispatchSec: 20, HeartbeatSec: 19}}
	buckets := []float64{1, 2, 5, 10, 30}
	var hist []BucketCount
	var p50, p95 float64
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		hist = Histogram(launches, buckets)
		p50, p95 = P50P95(launches)
	}
	if len(hist) != 6 || p50 != 2 || p95 != 30 {
		b.Fatal(hist, p50, p95)
	}
}
