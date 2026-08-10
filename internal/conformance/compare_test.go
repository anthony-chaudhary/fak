package conformance

import "testing"

func TestCompareLocalKeepsContractFrameworkAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{{"fak native compiled conformance suite", "native"}, {"embedded JSON and schema equality only", "baseline"}, {"OPA test", "external"}, {"Conftest", "external"}, {"OpenAPI and JSON Schema contract tests", "external"}, {"Pact", "external"}, {"Cedar policy validator and tests", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, a := range got.Arms {
		if a.Name != want[i].name || a.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, a)
		}
		if i < 2 {
			if !a.Available || a.Checks != 2 || a.InputBytes == 0 {
				t.Fatalf("local=%+v", a)
			}
			continue
		}
		if a.Available || a.Correct || a.Latency != 0 || a.Checks != 0 || a.PassedChecks != 0 || a.MissedChecks != 0 || a.FalseFailures != 0 || a.MutationCases != 0 || a.MutationsCaught != 0 || a.ReasonErrors != 0 || a.CPUSeconds != 0 || a.PeakRSSBytes != 0 || a.InputBytes != 0 || a.NetworkBytes != 0 || a.OperatorSeconds != 0 || a.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", a)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].PassedChecks != 2 {
		t.Fatalf("native=%+v", got.Arms[0])
	}
	if got.Arms[1].Correct || got.Arms[1].MissedChecks != 1 {
		t.Fatalf("baseline=%+v", got.Arms[1])
	}
}
func BenchmarkCompiledConformanceSuite(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if a := runNativeComparison(); !a.Correct {
			b.Fatalf("arm=%+v", a)
		}
	}
}
