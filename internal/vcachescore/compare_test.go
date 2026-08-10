package vcachescore

import "testing"

func TestCompareDefaultReadinessLocalKeepsPolicyAlternativesExplicit(t *testing.T) {
	got := CompareDefaultReadinessLocal()
	want := []struct{ name, kind string }{{"fak native default-cache readiness gate", "native"}, {"usefulness-score threshold only", "baseline"}, {"fak + Prometheus", "integration"}, {"fak + OpenTelemetry", "integration"}, {"OPA/Rego", "external"}, {"Prometheus rules", "external"}, {"Datadog monitors", "external"}, {"LangSmith evaluations", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, arm := range got.Arms {
		if arm.Name != want[i].name || arm.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, arm)
		}
		if i < 2 {
			if !arm.Available || arm.Cases != 5 {
				t.Fatalf("local=%+v", arm)
			}
			continue
		}
		if arm.Available || arm.Correct || arm.Latency != 0 || arm.Cases != 0 || arm.TrueReady != 0 || arm.TrueBlocked != 0 || arm.FalseReady != 0 || arm.FalseBlocked != 0 || arm.ReasonMismatches != 0 || arm.CPUSeconds != 0 || arm.PeakRSSBytes != 0 || arm.InputBytes != 0 || arm.NetworkBytes != 0 || arm.StorageBytes != 0 || arm.OperatorSeconds != 0 || arm.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", arm)
		}
	}
	native, baseline := got.Arms[0], got.Arms[1]
	if !native.Correct || native.TrueReady != 1 || native.TrueBlocked != 4 || native.FalseReady != 0 || native.FalseBlocked != 0 || native.ReasonMismatches != 0 {
		t.Fatalf("native=%+v", native)
	}
	if baseline.Correct || baseline.FalseReady == 0 || baseline.ReasonMismatches == 0 {
		t.Fatalf("baseline=%+v", baseline)
	}
}

func BenchmarkDefaultReadinessFiveCases(b *testing.B) {
	cases := readinessComparisonCases()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		arm := runNativeReadiness(cases)
		if !arm.Correct {
			b.Fatalf("arm=%+v", arm)
		}
	}
}
