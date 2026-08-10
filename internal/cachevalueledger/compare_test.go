package cachevalueledger

import "testing"

func TestCompareLocalKeepsTrendAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{{"fak native trailing-window trend gate", "native"}, {"raw JSONL ledger without trend gate", "baseline"}, {"fak + Prometheus", "integration"}, {"fak + OpenTelemetry", "integration"}, {"Prometheus recording and alerting rules", "external"}, {"Datadog change and anomaly monitor", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, a := range got.Arms {
		if a.Name != want[i].name || a.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, a)
		}
		if i < 2 {
			if !a.Available || a.Rows != 5 || a.PromptTokens != 1700 {
				t.Fatalf("local=%+v", a)
			}
			continue
		}
		if a.Available || a.Correct || a.Latency != 0 || a.Rows != 0 || a.Alerts != 0 || a.PromptTokens != 0 || a.Bytes != 0 || a.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", a)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Alerts != 1 || got.Arms[1].Correct {
		t.Fatalf("oracle=%+v", got.Arms)
	}
}
func BenchmarkFoldTrendGate(b *testing.B) {
	rows := trendFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FoldTrendGate(rows)
	}
}
