package claimcheck

import "testing"

func TestCompareLocalKeepsClaimEvaluationAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{{"fak native net-true claim grader", "native"}, {"accept claim when any witness exists", "baseline"}, {"fak + Prometheus", "integration"}, {"fak + OpenTelemetry", "integration"}, {"OPA/Rego", "external"}, {"OpenAI Evals graders", "external"}, {"LangSmith evaluators", "external"}, {"Braintrust scorers", "external"}, {"DeepEval metrics", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, a := range got.Arms {
		if a.Name != want[i].name || a.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, a)
		}
		if i < 2 {
			if !a.Available || a.Cases != len(Fixture()) {
				t.Fatalf("local=%+v", a)
			}
			continue
		}
		if a.Available || a.Correct || a.Latency != 0 || a.Cases != 0 || a.ExactVerdicts != 0 || a.WrongNetTrue != 0 || a.WrongStrawman != 0 || a.WrongNotYet != 0 || a.ReasonMismatches != 0 || a.CPUSeconds != 0 || a.PeakRSSBytes != 0 || a.InputBytes != 0 || a.ModelTokens != 0 || a.NetworkBytes != 0 || a.StorageBytes != 0 || a.OperatorSeconds != 0 || a.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", a)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].ExactVerdicts != len(Fixture()) {
		t.Fatalf("native=%+v", got.Arms[0])
	}
	if got.Arms[1].Correct || got.Arms[1].WrongNetTrue == 0 {
		t.Fatalf("baseline=%+v", got.Arms[1])
	}
}

func BenchmarkGradeFixture(b *testing.B) {
	cases := Fixture()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		a := runNativeComparison(cases)
		if !a.Correct {
			b.Fatalf("arm=%+v", a)
		}
	}
}
