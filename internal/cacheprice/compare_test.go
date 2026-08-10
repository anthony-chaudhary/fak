package cacheprice

import "testing"

func TestCompareLocalKeepsCloudCalculatorsExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{
		"fak native cache pricing":        {"native", true},
		"charge full prompt":              {"baseline", true},
		"AWS Pricing Calculator":          {"external", false},
		"Google Cloud Pricing Calculator": {"external", false},
		"Azure Pricing Calculator":        {"external", false},
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
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Tokens != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims a result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Tokens != 1024 {
		t.Fatalf("native result=%#v", got.Arms[0])
	}
	if got.Arms[1].Correct || got.Arms[1].Tokens != 4096 {
		t.Fatalf("baseline result=%#v", got.Arms[1])
	}
}

func BenchmarkAdmissionTokens(b *testing.B) {
	var got int
	for i := 0; i < b.N; i++ {
		got = AdmissionTokens(4096, 3072)
	}
	if got != 1024 {
		b.Fatal(got)
	}
}
