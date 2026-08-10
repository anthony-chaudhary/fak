package canon

import "testing"

func TestCompareTokenUsageLocalKeepsTelemetryAlternativesExplicit(t *testing.T) {
	got := CompareTokenUsageLocal()
	want := []struct{ name, kind string }{{"fak native canonical token-usage adapter", "native"}, {"provider total fields only", "baseline"}, {"fak + OpenAI", "integration"}, {"fak + Anthropic", "integration"}, {"fak + local provider", "integration"}, {"fak + OpenTelemetry", "integration"}, {"OpenAI SDK usage models", "external"}, {"Anthropic SDK usage models", "external"}, {"LiteLLM usage normalization", "external"}, {"OpenTelemetry GenAI semantic conventions", "external"}, {"LangSmith token and cost tracking", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, a := range got.Arms {
		if a.Name != want[i].name || a.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, a)
		}
		if i < 2 {
			if !a.Available || a.Cases != 7 || a.InputBytes == 0 {
				t.Fatalf("local=%+v", a)
			}
			continue
		}
		if a.Available || a.Correct || a.Latency != 0 || a.Cases != 0 || a.CorrectCases != 0 || a.RejectionErrors != 0 || a.ClassErrors != 0 || a.RawLosses != 0 || a.InputBytes != 0 || a.RepresentedTokens != 0 || a.CPUSeconds != 0 || a.PeakRSSBytes != 0 || a.NetworkBytes != 0 || a.ModelTokens != 0 || a.OperatorSeconds != 0 || a.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", a)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].CorrectCases != 7 {
		t.Fatalf("native=%+v", got.Arms[0])
	}
	if got.Arms[1].Correct || got.Arms[1].ClassErrors == 0 || got.Arms[1].RawLosses == 0 {
		t.Fatalf("baseline=%+v", got.Arms[1])
	}
}
func BenchmarkAdaptTokenUsageCorpus(b *testing.B) {
	b.ReportAllocs()
	b.SetBytes(usageInputBytes())
	for i := 0; i < b.N; i++ {
		if a := runNativeUsage(); !a.Correct {
			b.Fatalf("arm=%+v", a)
		}
	}
}
