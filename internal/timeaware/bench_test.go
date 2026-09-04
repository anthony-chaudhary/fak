package timeaware

import (
	"fmt"
	"testing"
)

// TestBenchmarkSuite verifies that the fixtures used in BenchmarkTimeAware remain valid.
func TestBenchmarkSuite(t *testing.T) {
	spans := makeBenchmarkSpans(20)
	edges := []Edge{
		{From: "span-0", To: "span-1", Kind: EdgeDependsOn},
		{From: "span-1", To: "span-2", Kind: EdgeDependsOn},
	}
	r := Aggregate(spans, edges)
	if r.SpanCount != 20 {
		t.Fatalf("expected 20 spans, got %d", r.SpanCount)
	}
	if r.Measures.EffortNS <= 0 {
		t.Fatalf("expected positive effort, got %d", r.Measures.EffortNS)
	}
}

func makeBenchmarkSpans(n int) []Span {
	spans := make([]Span, n)
	for i := 0; i < n; i++ {
		spans[i] = Span{
			Schema:  SpanSchema,
			ID:      fmt.Sprintf("span-%d", i),
			StartNS: int64(i * 100),
			EndNS:   int64((i + 1) * 100),
			Phase:   PhaseActive,
			Dimensions: Dimensions{
				SessionID: "bench-session",
				RunID:     "bench-run",
				AgentID:   "bench-worker",
				Retry:     i % 5,
				Poll:      i % 3,
			},
		}
	}
	return spans
}

// BenchmarkTimeAware exercises span aggregation and activity snapshot rendering in a loop.
func BenchmarkTimeAware(b *testing.B) {
	spans := makeBenchmarkSpans(64)
	edges := []Edge{
		{From: "span-0", To: "span-1", Kind: EdgeDependsOn},
		{From: "span-1", To: "span-2", Kind: EdgeDependsOn},
		{From: "span-2", To: "span-3", Kind: EdgeDependsOn},
	}
	snapshot := ActivitySnapshot{
		State:  StateWorking,
		Motion: MotionAdvancing,
		Scope: Scope{
			Completed:        32,
			Total:            KnownCount(64),
			DenominatorClass: DenominatorDeclaredWork,
			Revision:         2,
		},
		Queued:   KnownCount(4),
		InFlight: KnownCount(2),
		Current:  Summary{Text: "aggregating execution spans", Provenance: ProvenanceFact},
		Next:     Summary{Text: "render activity telemetry", Provenance: ProvenanceForecast},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = Aggregate(spans, edges)
		_ = FormatActivitySnapshot(snapshot, 80)
	}
}

// BenchmarkAggregateSpans benchmarks aggregation over concurrent and sequential spans.
func BenchmarkAggregateSpans(b *testing.B) {
	spans := makeBenchmarkSpans(100)
	edges := []Edge{
		{From: "span-0", To: "span-10", Kind: EdgeDependsOn},
		{From: "span-10", To: "span-20", Kind: EdgeDependsOn},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = Aggregate(spans, edges)
	}
}

// BenchmarkActivityRendering benchmarks deterministic single-line activity formatting.
func BenchmarkActivityRendering(b *testing.B) {
	snapshot := ActivitySnapshot{
		State:  StateWorking,
		Motion: MotionAdvancing,
		Scope: Scope{
			Completed:        10,
			Total:            KnownCount(20),
			DenominatorClass: DenominatorDiscoveredWork,
			Revision:         5,
		},
		Queued:   KnownCount(1),
		InFlight: KnownCount(3),
		Current:  Summary{Text: "optimizing timeline", Provenance: ProvenanceInference},
		Next:     Summary{Text: "commit telemetry snapshot", Provenance: ProvenanceForecast},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = FormatActivitySnapshot(snapshot, 72)
	}
}
