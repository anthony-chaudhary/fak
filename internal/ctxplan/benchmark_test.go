package ctxplan

import (
	"context"
	"fmt"
	"testing"
)

var (
	benchPlanSink         Plan
	benchIndexSink        *Index
	benchViewSink         View
	benchWitnessSink      Witness
	benchEvictionPlanSink EvictionPlan
)

func makeBenchmarkSpans(n int) []Span {
	spans := make([]Span, n)
	roles := []string{"system", "user", "assistant", "tool", "WebSearch", "Bash", "Read"}
	durabilities := []string{DurabilityDurable, DurabilitySession, DurabilityTurn, DurabilityBounded}
	for i := 0; i < n; i++ {
		role := roles[i%len(roles)]
		dur := durabilities[i%len(durabilities)]
		desc := fmt.Sprintf("%s turn %d: execution of task step with details on auth token cache and context management", role, i)
		cluster := ""
		kind := ""
		if i%10 == 0 {
			cluster = fmt.Sprintf("cluster-%d", i/10)
			kind = EvidenceDecisive
		} else if i%10 < 3 {
			cluster = fmt.Sprintf("cluster-%d", i/10)
			kind = EvidenceSupport
		}
		spans[i] = Span{
			ID:              fmt.Sprintf("span:%d", i),
			Step:            i,
			Role:            role,
			Descriptor:      desc,
			Digest:          fmt.Sprintf("d-%d", i),
			Bytes:           int64(40 + (i%50)*8),
			Durability:      dur,
			EvidenceCluster: cluster,
			EvidenceKind:    kind,
		}
	}
	return spans
}

func makeBenchmarkStore(n int) *MemStore {
	st := NewMemStore()
	roles := []string{"system", "user", "assistant", "WebSearch", "Bash", "Read"}
	durabilities := []string{DurabilityDurable, DurabilitySession, DurabilityTurn, DurabilityBounded}
	for i := 0; i < n; i++ {
		role := roles[i%len(roles)]
		dur := durabilities[i%len(durabilities)]
		body := []byte(fmt.Sprintf("%s turn %d: payload with tokens, cache status, and verification log for task execution", role, i))
		st.Add(role, dur, body, false)
	}
	return st
}

func makeBenchmarkPages(n int) []Page {
	pages := make([]Page, n)
	kinds := PageKinds()
	intents := []RetentionIntent{RetentionKeep, RetentionNeutral, RetentionDrop}
	for i := 0; i < n; i++ {
		kind := kinds[i%len(kinds)]
		intent := intents[i%len(intents)]
		pages[i] = Page{
			ID:     fmt.Sprintf("page:%d", i),
			Kind:   kind,
			Tokens: 20 + (i%30)*5,
			Retention: []RetentionAnnotation{
				{
					Intent:     intent,
					Source:     "rule:benchmark",
					ReasonCode: "BENCH",
				},
			},
		}
	}
	return pages
}

func BenchmarkPlanCells_Greedy(b *testing.B) {
	for _, n := range []int{50, 200, 500} {
		spans := makeBenchmarkSpans(n)
		forecast := Forecast{
			Intents:  []string{"auth token", "cache management", "task execution"},
			Pins:     []string{"span:0", "span:1"},
			Releases: []string{"span:4"},
			Horizon:  3,
		}
		budget := Budget{Tokens: 2000}

		b.Run(fmt.Sprintf("spans=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchPlanSink = PlanCells(spans, forecast, budget, nil)
			}
		})
	}
}

func BenchmarkOptimize_Greedy(b *testing.B) {
	spans := makeBenchmarkSpans(200)
	forecast := Forecast{
		Intents: []string{"auth token", "cache management", "task execution"},
		Pins:    []string{"span:0", "span:1"},
		Horizon: 3,
	}
	cands := Candidates(spans, forecast, nil)
	pins := pinSet(forecast.Pins)
	budget := Budget{Tokens: 2000}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchPlanSink = Optimize(cands, budget, pins, ObjGreedy)
	}
}

func BenchmarkOptimize_Coverage(b *testing.B) {
	spans := makeBenchmarkSpans(100)
	forecast := Forecast{
		Intents: []string{"auth token", "cache management", "task execution", "verification log"},
		Pins:    []string{"span:0"},
		Horizon: 3,
	}
	cands := Candidates(spans, forecast, nil)
	pins := pinSet(forecast.Pins)
	budget := Budget{Tokens: 1500}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchPlanSink = Optimize(cands, budget, pins, ObjCoverage)
	}
}

func BenchmarkIndex_PlanCells(b *testing.B) {
	for _, n := range []int{100, 500, 1000} {
		spans := makeBenchmarkSpans(n)
		ix := BuildIndex(spans)
		forecast := Forecast{
			Intents: []string{"auth token", "cache management"},
			Pins:    []string{"span:0", "span:1"},
			Horizon: 2,
		}
		budget := Budget{Tokens: 2000}

		b.Run(fmt.Sprintf("spans=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchPlanSink = ix.PlanCells(forecast, budget, nil, ProbeOptions{})
			}
		})
	}
}

func BenchmarkIndex_Build(b *testing.B) {
	for _, n := range []int{100, 500} {
		spans := makeBenchmarkSpans(n)

		b.Run(fmt.Sprintf("spans=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchIndexSink = BuildIndex(spans)
			}
		})
	}
}

func BenchmarkMaterialize(b *testing.B) {
	ctx := context.Background()
	store := makeBenchmarkStore(100)
	forecast := Forecast{
		Intents: []string{"auth token", "cache status"},
		Pins:    []string{"span:0"},
		Horizon: 2,
	}
	budget := Budget{Tokens: 2000}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := Materialize(ctx, store, forecast, budget, nil)
		if err != nil {
			b.Fatal(err)
		}
		benchViewSink = v
	}
}

func BenchmarkAudit(b *testing.B) {
	spans := makeBenchmarkSpans(200)
	forecast := Forecast{
		Intents: []string{"auth token", "cache management"},
		Pins:    []string{"span:0"},
		Horizon: 2,
	}
	plan := PlanCells(spans, forecast, Budget{Tokens: 2000}, nil)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchWitnessSink = Audit(plan)
	}
}

func BenchmarkPlanEviction(b *testing.B) {
	pages := makeBenchmarkPages(150)
	budgetTokens := 1000

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchEvictionPlanSink = PlanEviction(pages, budgetTokens)
	}
}
