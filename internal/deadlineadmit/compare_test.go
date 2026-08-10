package deadlineadmit

import "testing"

func TestCompareLocalKeepsSchedulerAndIntegrationArmsExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{
		"fak native EDF admission":             {"native", true},
		"FIFO without predicted-miss shedding": {"baseline", true},
		"Mooncake deadline-aware admission":    {"external", false},
		"vLLM priority scheduling":             {"external", false},
		"SGLang priority scheduling":           {"external", false},
		"fak + vLLM priority scheduling":       {"integration", false},
		"fak + SGLang priority scheduling":     {"integration", false},
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
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Admitted != 0 || arm.Shed != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims a result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Admitted != 3 || got.Arms[0].Shed != 1 {
		t.Fatalf("native=%#v", got.Arms[0])
	}
}

func BenchmarkAdmitPredictedMiss(b *testing.B) {
	items := []Item{{ID: 4, Deadline: 85, PredictedCost: 20, Degradable: true}, {ID: 1, Deadline: 100, PredictedCost: 10}, {ID: 2, Deadline: 100, PredictedCost: 10}, {ID: 3, Deadline: 110, PredictedCost: 50}}
	var got Plan
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got = Admit(items, 80, 10)
	}
	if len(got.Order) != 3 || len(got.Shed) != 1 {
		b.Fatal(got)
	}
}
