package vcachewarm

import "testing"

func TestCompareLocalKeepsWarmingAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{{"fak native dedicated warm planner and accounting", "native"}, {"demand-only fills without dedicated warming", "baseline"}, {"fak + Anthropic prompt caching", "integration"}, {"fak + Gemini CachedContent", "integration"}, {"fak + OpenAI automatic prefix caching", "integration"}, {"fak + LMCache", "integration"}, {"fak + Mooncake", "integration"}, {"vLLM automatic prefix caching", "external"}, {"SGLang HiCache", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, a := range got.Arms {
		if a.Name != want[i].name || a.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, a)
		}
		if i < 2 {
			if !a.Available || a.Cases != 5 {
				t.Fatalf("local=%+v", a)
			}
			continue
		}
		if a.Available || a.Correct || a.Latency != 0 || a.Cases != 0 || a.DedicatedWarms != 0 || a.ConfirmedWarms != 0 || a.WastedWarms != 0 || a.CacheReadTokens != 0 || a.Bytes != 0 || a.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", a)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].DedicatedWarms != 1 || got.Arms[0].ConfirmedWarms != 1 || got.Arms[0].CacheReadTokens != 2048 || got.Arms[1].Correct {
		t.Fatalf("oracle=%+v", got.Arms)
	}
}
func BenchmarkPlanAndReconcileWarm(b *testing.B) {
	reqs := warmingFixture()
	readbacks := []CacheReadback{{CacheReadTokens: 1024}, {CacheReadTokens: 1024}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var d Decision
		for _, r := range reqs {
			d = Plan(r)
		}
		d = Plan(reqs[0])
		_ = ReconcileWarm(d, true, readbacks)
	}
}
