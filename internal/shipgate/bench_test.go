package shipgate

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// BenchmarkShipGate exercises shipgate evaluation and canary checks in a loop.
func BenchmarkShipGate(b *testing.B) {
	b.Run("Evaluate", func(b *testing.B) {
		w := Witness{
			Class:       ClassFull,
			Metric:      "p50_ns",
			Before:      1000,
			After:       800,
			LowerBetter: true,
			SuiteGreen:  true,
			TruthClean:  true,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			d, _ := Evaluate(w)
			if d != KEEP {
				b.Fatalf("unexpected decision: %v", d)
			}
		}
	})

	b.Run("Adjudicate", func(b *testing.B) {
		adj := DefaultAdjudicator
		ctx := context.Background()
		call := &abi.ToolCall{
			Tool: "ship",
			Meta: map[string]string{"witness": "ancestor:HEAD~1"},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := adj.Adjudicate(ctx, call)
			if v.Kind != abi.VerdictRequireWitness {
				b.Fatalf("unexpected verdict: %v", v.Kind)
			}
		}
	})

	b.Run("CanaryAdjudicate", func(b *testing.B) {
		caseItem := CanaryCase{
			ID:         "bench-pr-gate",
			Tier:       CanaryTierPR,
			CostNote:   "~1ms CPU, no GPU",
			MinSamples: 10,
			Provenance: CanaryProvenance{
				Model:     "qwen3.8",
				Tokenizer: "qwen3.8-v1",
				Engine:    "fak-native",
				Seed:      "oracle-1",
				Revision:  "r100",
				Baseline:  "v1-gold",
			},
			Slices: []QualitySlice{
				{
					Name:      "accuracy",
					Critical:  true,
					Baseline:  0.95,
					Candidate: 0.96,
					Tolerance: 0.01,
					Samples:   100,
					Measured:  true,
				},
			},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := AdjudicateCanary(caseItem)
			if !res.Promoted {
				b.Fatalf("unexpected verdict: %s", res.Verdict)
			}
		}
	})
}
