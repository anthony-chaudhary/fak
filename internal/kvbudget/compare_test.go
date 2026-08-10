package kvbudget

import "testing"

func TestCompareLocalKeepsRuntimeProfilersExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{
		"fak native KV budget model": {"native", true}, "full-MHA closed form": {"baseline", true},
		"vLLM memory profiler": {"external", false}, "SGLang memory pool": {"external", false}, "NVIDIA GenAI-Perf": {"external", false},
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
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims a result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Bytes != 129536 {
		t.Fatalf("native=%#v", got.Arms[0])
	}
}

func BenchmarkKVBytesPerTokenModel(b *testing.B) {
	var got float64
	for i := 0; i < b.N; i++ {
		got = GLM52DSA.KVBytesPerToken(F16)
	}
	if got != 129536 {
		b.Fatal(got)
	}
}
