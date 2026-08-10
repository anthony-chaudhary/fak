package vcacheobserve

import "testing"

func TestCompareLocalKeepsEconomicsAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{{"fak native provider-cache economics fold", "native"}, {"raw provider usage without economics fold", "baseline"}, {"fak + Prometheus", "integration"}, {"fak + OpenTelemetry", "integration"}, {"Anthropic usage and cost reporting", "external"}, {"OpenAI usage and cost reporting", "external"}, {"Datadog LLM Observability", "external"}, {"LangSmith", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, a := range got.Arms {
		if a.Name != want[i].name || a.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, a)
		}
		if i < 2 {
			if !a.Available || a.Turns != 4 || a.InputTokens != 1700 || a.CacheWriteTokens != 900 || a.CacheReadTokens != 900 {
				t.Fatalf("local=%+v", a)
			}
			continue
		}
		if a.Available || a.Correct || a.Latency != 0 || a.Turns != 0 || a.InputTokens != 0 || a.CacheWriteTokens != 0 || a.CacheReadTokens != 0 || a.SavedTokenEquiv != 0 || a.Bytes != 0 || a.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", a)
		}
	}
	if !got.Arms[0].Correct || got.Arms[1].Correct {
		t.Fatalf("oracle=%+v", got.Arms)
	}
}
func BenchmarkObserveEconomics(b *testing.B) {
	turns := economicsFixture()
	m := DefaultMultipliers()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Observe(turns, m)
	}
}
