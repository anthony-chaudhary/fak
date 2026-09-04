package cohort

import (
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/comm"
)

// BenchmarkCohort exercises cohort agreement folds and drift monitoring in loops.
func BenchmarkCohort(b *testing.B) {
	b.Run("Agree", func(b *testing.B) {
		const size = 5
		members := make([]comm.Member, size)
		outputs := make(map[string]string, size)
		for i := 0; i < size; i++ {
			id := "worker-" + strconv.Itoa(i)
			members[i] = comm.Member{ID: id, Weight: 1.0, Lane: "default"}
			outputs[id] = "decision-commit"
		}

		c, err := FromMembers("bench-wave", "bench-trace", members)
		if err != nil {
			b.Fatalf("FromMembers failed: %v", err)
		}
		quorum := c.MajorityFloor()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			agreement, err := c.Agree(outputs, quorum)
			if err != nil || !agreement.Agreed {
				b.Fatalf("Agree failed: err=%v agreed=%v", err, agreement.Agreed)
			}
		}
	})

	b.Run("Drift", func(b *testing.B) {
		obs := []CohortObservation{
			{
				Cohort:      "cohort-alpha",
				Tier:        DriftTierPR,
				CostSeconds: 1.2,
				Provenance: DriftProvenance{
					Model:     "qwen3-8b",
					Tokenizer: "qwen3-bpe",
					Engine:    "native",
					Oracle:    "exact-match",
					Revision:  "rev-100",
					Baseline:  "baseline@rev-99",
				},
				Baseline: map[string]float64{
					SignalMix:          0.50,
					SignalLength:       128.0,
					SignalDegeneration: 0.02,
					SignalRubric:       0.90,
				},
				Observed: map[string]float64{
					SignalMix:          0.51,
					SignalLength:       130.0,
					SignalDegeneration: 0.02,
					SignalRubric:       0.89,
				},
				Tolerance: map[string]float64{
					SignalMix:          0.05,
					SignalLength:       16.0,
					SignalDegeneration: 0.01,
					SignalRubric:       0.03,
				},
			},
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			report := MonitorDrift("rev-100", obs)
			if !report.Clean {
				b.Fatalf("MonitorDrift unexpectedly drifted: %+v", report)
			}
		}
	})
}
