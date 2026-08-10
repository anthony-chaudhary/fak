package cacheobs

import "testing"

func TestCompareLocalKeepsTelemetryAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{
		"fak native cache observer": {"native", true},
		"no telemetry":              {"baseline", true},
		"Prometheus client":         {"external", false},
		"OpenTelemetry metrics":     {"external", false},
		"Datadog DogStatsD":         {"external", false},
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
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Events != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims a result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Events != 1000 {
		t.Fatalf("native result=%#v", got.Arms[0])
	}
}

func BenchmarkObserveCacheHit(b *testing.B) {
	obs := New()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		obs.Observe(128, 96)
	}
}
