package dojocal

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

// BenchmarkDojoCal exercises candidate proposal and priority ranking in a loop.
func BenchmarkDojoCal(b *testing.B) {
	// Build a synthetic set of episodes across multiple levers.
	eps := make([]dojo.Episode, 0, 40)
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
				Sample:     5,
			}
			eps = append(eps, dojo.Score(lever, p, o, dojo.DefaultCalibBand()))
		}
	}

	rep := dojo.Fold(eps, dojo.FoldOpts{Workspace: "bench"})
	opts := SelectOptions{
		Now:         time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC),
		RecheckDays: DefaultCellRecheckDays,
	}

	rows := []JournalRow{
		{
			GeneratedAt: "2026-09-01T00:00:00Z",
			Date:        "2026-09-01",
			Lever:       "resume-posture",
			Metric:      "cold_write_share",
		},
		{
			GeneratedAt: "2026-08-20T00:00:00Z",
			Date:        "2026-08-20",
			Lever:       "vcache-warmth",
			Metric:      "warm_recall",
		},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		pl := ProposeRecals(rep)
		if len(pl.Candidates) == 0 {
			b.Fatal("expected candidates")
		}
		ranked := RankCandidates(pl.Candidates, rows, opts)
		if len(ranked) == 0 {
			b.Fatal("expected ranked cells")
		}
		_ = ScheduleWakeup(ranked, opts.Now)
	}
}

// BenchmarkRankCandidates isolates candidate ranking and saturation checking.
func BenchmarkRankCandidates(b *testing.B) {
	candidates := make([]Recal, 0, 50)
	for i := 0; i < 50; i++ {
		candidates = append(candidates, Recal{
			Lever:      fmt.Sprintf("lever-%d", i%5),
			Metric:     fmt.Sprintf("metric-%d", i%10),
			Kind:       RecalibrateKind,
			OldClaimed: 0.9,
			NewClaimed: 0.5,
			CalibErr:   0.4,
			Sample:     10,
		})
	}

	rows := []JournalRow{
		{
			GeneratedAt: "2026-09-02T10:00:00Z",
			Date:        "2026-09-02",
			Lever:       "lever-1",
			Metric:      "metric-1",
		},
	}
	opts := SelectOptions{
		Now:         time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC),
		RecheckDays: 7,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ranked := RankCandidates(candidates, rows, opts)
		if len(ranked) == 0 {
			b.Fatal("empty ranked output")
		}
	}
}
