package mutationbudget

import "testing"

func TestCompareLocalKeepsAPIBudgetAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{
		"fak native mutation reserve":        {"native", true},
		"direct API calls without reserve":   {"baseline", true},
		"GitHub Octokit rate-limit handling": {"external", false},
		"gh api rate-limit handling":         {"external", false},
		"Envoy global rate limit":            {"external", false},
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
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Calls != 0 || arm.Held || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims a result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || !got.Arms[0].Held || got.Arms[0].Calls != 15 {
		t.Fatalf("native=%#v", got.Arms[0])
	}
}

func BenchmarkEstimateMutationHour(b *testing.B) {
	budget := Budget{Remaining: 12, Limit: 5000, ResetAtUnix: 1_700_003_600}
	plan := HourlyPlan{Closes: 8, Comments: 5, Fetches: 2}
	var got HourlyEstimate
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got = EstimateHour(budget, plan, 5, 1_700_000_000)
	}
	if got.Allow || got.TotalCalls != 15 {
		b.Fatal(got)
	}
}
