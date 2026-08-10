package vcachegov

import "testing"

func TestCompareLocalKeepsWarmSchedulingAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{
		{"fak native warm-budget scheduler", "native"},
		{"demand-only fills without proactive warming", "baseline"},
		{"fak + LMCache", "integration"},
		{"fak + Mooncake", "integration"},
		{"fak + NIXL", "integration"},
		{"vLLM automatic prefix caching", "external"},
		{"SGLang HiCache and cache-aware scheduling", "external"},
	}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d want %d", len(got.Arms), len(want))
	}
	for i, arm := range got.Arms {
		if arm.Name != want[i].name || arm.Kind != want[i].kind {
			t.Fatalf("arm[%d]=(%q,%q), want (%q,%q)", i, arm.Name, arm.Kind, want[i].name, want[i].kind)
		}
		if i < 2 {
			if !arm.Available || arm.Candidates != 6 {
				t.Fatalf("local arm[%d] lacks an executed six-candidate measurement: %+v", i, arm)
			}
			continue
		}
		if arm.Available || arm.Correct || arm.Latency != 0 || arm.Candidates != 0 || arm.Warmed != 0 || arm.ValueCaptured != 0 || arm.Tokens != 0 || arm.Bytes != 0 || arm.CostUSD != 0 {
			t.Fatalf("unwitnessed arm[%d] reports measurements: %+v", i, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Warmed != 2 || got.Arms[0].ValueCaptured != 89000 || got.Arms[0].Tokens != 2000 {
		t.Fatalf("native scheduler failed the quality/resource oracle: %+v", got.Arms[0])
	}
	if got.Arms[1].Correct || got.Arms[1].Warmed != 0 || got.Arms[1].ValueCaptured != 0 {
		t.Fatalf("demand-only baseline unexpectedly captured proactive warm value: %+v", got.Arms[1])
	}
}

func BenchmarkWarmBudgetSchedule(b *testing.B) {
	rate, anchorTokens, ttlMillis, candidates := comparisonFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		budget := PlanWarmBudget(rate, anchorTokens, ttlMillis)
		_ = Schedule(candidates, budget)
	}
}
