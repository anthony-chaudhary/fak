package cachesweep

import "testing"

func TestCompareLocalKeepsCacheSimulatorAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{"fak native prefix-cache sweep": {"native", true}, "no prefix cache": {"baseline", true}, "libCacheSim": {"external", false}, "Caffeine simulator": {"external", false}, "Redis or Valkey maxmemory policies": {"external", false}}
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
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Accesses != 0 || arm.SavedTokens != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Accesses != 5 || got.Arms[0].SavedTokens <= 0 {
		t.Fatalf("native=%#v", got.Arms[0])
	}
}

func BenchmarkSweepPrefixCacheBudgets(b *testing.B) {
	trace := comparisonTrace()
	opts := Options{Budgets: []int{1, 2, 4, 8}, WriteDelayNs: 1, KneeFraction: DefaultKneeFraction}
	var got Result
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got = Sweep(trace, opts)
	}
	if got.Accesses != 5 || got.Ceiling.ReusedTokens <= 0 {
		b.Fatal(got)
	}
}
