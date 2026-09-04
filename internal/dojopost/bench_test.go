package dojopost

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

// Invariant: benchmark exercises rollups and trends in a loop without external side effects.
func BenchmarkDojoPost(b *testing.B) {
	eps := make([]dojo.Episode, 0, 20)
	levers := []string{"resume-posture", "vcache-warmth", "compaction", "adjudication", "read-cache"}
	metrics := []string{"cold_write_share", "warm_recall", "token_shed_ratio", "decision_latency"}

	for i, lever := range levers {
		for j, metric := range metrics {
			claimed := 0.85
			realized := 0.40 + float64((i+j)%5)*0.10
			p := dojo.Prediction{
				Lever:         lever,
				Metric:        metric,
				Claimed:       claimed,
				Unit:          "ratio",
				Basis:         "bench",
				LowerIsBetter: false,
			}
			o := dojo.Outcome{
				Realized:   realized,
				Provenance: dojo.Witnessed,
				Source:     "bench",
				Measured:   true,
				Sample:     10,
			}
			eps = append(eps, dojo.Score(lever, p, o, dojo.DefaultCalibBand()))
		}
	}

	rep := dojo.Report{
		Commit:       "abcdef1234567890",
		LeverCount:   len(levers),
		EpisodeCount: len(eps),
		Measured:     len(eps),
		Calibrated:   len(eps) / 2,
		MeanCalibErr: 0.285,
		Grade:        "B",
		NextAction:   "inspect calibration drift before changing policy",
		Episodes:     eps,
	}

	rows := []dojo.LedgerRow{
		{Schema: dojo.LedgerSchema, Date: "2026-06-25", Commit: "c1", GeneratedAt: "2026-06-25T01:00:00Z", LeverCount: 3, EpisodeCount: 7, MeanCalibErr: 0.70, Grade: "F", Calibrated: 2, Measured: 6},
		{Schema: dojo.LedgerSchema, Date: "2026-06-26", Commit: "c2", GeneratedAt: "2026-06-26T01:00:00Z", LeverCount: 3, EpisodeCount: 7, MeanCalibErr: 0.50, Grade: "D", Calibrated: 2, Measured: 6},
		{Schema: dojo.LedgerSchema, Date: "2026-06-27", Commit: "c3", GeneratedAt: "2026-06-27T01:00:00Z", LeverCount: 3, EpisodeCount: 7, MeanCalibErr: 0.34, Grade: "C", Calibrated: 2, Measured: 6},
		{Schema: dojo.LedgerSchema, Date: "2026-06-28", Commit: "c4", GeneratedAt: "2026-06-28T01:00:00Z", LeverCount: 5, EpisodeCount: 20, MeanCalibErr: 0.28, Grade: "B", Calibrated: 10, Measured: 20},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		p1 := RollupFromReport(rep, 8)
		if p1.Text() == "" {
			b.Fatal("expected non-empty rollup text")
		}
		if len(p1.Blocks()) == 0 {
			b.Fatal("expected non-empty rollup blocks")
		}

		p2 := TrendFromLedger(rows, 4)
		if p2.Text() == "" {
			b.Fatal("expected non-empty trend text")
		}
		if len(p2.Blocks()) == 0 {
			b.Fatal("expected non-empty trend blocks")
		}
	}
}
