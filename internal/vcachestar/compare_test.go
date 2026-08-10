package vcachestar

import "testing"

func TestCompareTelemetryLocalKeepsReconciliationAlternativesExplicit(t *testing.T) {
	got := CompareTelemetryLocal()
	want := []struct{ name, kind string }{{"fak native cache telemetry reconciliation", "native"}, {"trust warm manifest and modeled savings", "baseline"}, {"fak + Anthropic prompt caching", "integration"}, {"fak + OpenAI prompt caching", "integration"}, {"fak + Gemini context caching", "integration"}, {"fak + Prometheus", "integration"}, {"fak + OpenTelemetry", "integration"}, {"Prometheus recording and alert rules", "external"}, {"Datadog monitors", "external"}, {"LangSmith traces", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, arm := range got.Arms {
		if arm.Name != want[i].name || arm.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, arm)
		}
		if i < 2 {
			if !arm.Available {
				t.Fatalf("local=%+v", arm)
			}
			continue
		}
		if arm.Available || arm.Correct || arm.Latency != 0 || arm.Demoted || arm.Alarmed || arm.FirstDivergeSegment != 0 || arm.FirstDivergeTokenOffset != 0 || arm.FirstDivergeByteOffset != 0 || arm.BookedUncachedTokens != 0 || arm.RebateTokens != 0 || arm.CPUSeconds != 0 || arm.PeakRSSBytes != 0 || arm.TelemetryBytes != 0 || arm.NetworkBytes != 0 || arm.StorageBytes != 0 || arm.OperatorSeconds != 0 || arm.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", arm)
		}
	}
	native, baseline := got.Arms[0], got.Arms[1]
	if !native.Correct || !native.Demoted || !native.Alarmed || native.FirstDivergeSegment != 1 || native.FirstDivergeTokenOffset != 100 || native.FirstDivergeByteOffset != 7 || native.BookedUncachedTokens != 150 || native.RebateTokens != 0 {
		t.Fatalf("native=%+v", native)
	}
	if baseline.Correct || baseline.Demoted || baseline.Alarmed || baseline.BookedUncachedTokens != 0 || baseline.RebateTokens != 150 {
		t.Fatalf("baseline=%+v", baseline)
	}
}

func BenchmarkFoldTelemetryDivergentZeroRead(b *testing.B) {
	belief, telemetry := reconciliationFixture()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if fold := FoldTelemetry(belief, telemetry); !correctReconciliation(fold) {
			b.Fatalf("fold=%+v", fold)
		}
	}
}
