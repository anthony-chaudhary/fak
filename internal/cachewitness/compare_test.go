package cachewitness

import "testing"

func TestCompareLocalKeepsDivergenceAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{{"fak native reuse-divergence fold", "native"}, {"raw reuse counters without divergence detection", "baseline"}, {"fak + Prometheus", "integration"}, {"fak + OpenTelemetry", "integration"}, {"Prometheus recording and alerting rules", "external"}, {"Datadog anomaly monitor", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, a := range got.Arms {
		if a.Name != want[i].name || a.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, a)
		}
		if i < 2 {
			if !a.Available || a.Records != 3 {
				t.Fatalf("local=%+v", a)
			}
			continue
		}
		if a.Available || a.Correct || a.Latency != 0 || a.Records != 0 || a.Alerts != 0 || a.Bytes != 0 || a.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", a)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Alerts != 1 || got.Arms[1].Correct {
		t.Fatalf("oracle=%+v", got.Arms)
	}
}
func BenchmarkFoldReuseDivergence(b *testing.B) {
	records := divergenceFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FoldReuseDivergence(records, DefaultReuseDivergenceTolerance)
	}
}
