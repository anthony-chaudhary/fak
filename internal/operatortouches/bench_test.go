package operatortouches

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

func BenchmarkOperatorTouches(b *testing.B) {
	drain := &resume.WatchdogDrainStatus{
		Verdict: resume.WatchdogDrainGreen,
		MTTRSessions: []resume.WatchdogMTTRRow{
			{
				Session:             "sid-fast",
				Status:              resume.WatchdogMTTRRecovered,
				DetectedAt:          1_000,
				ResumedAt:           1_100,
				ProgressWitnessedAt: 1_300,
			},
			{
				Session:             "sid-slow",
				Status:              resume.WatchdogMTTRRecovered,
				DetectedAt:          2_000,
				ResumedAt:           2_500,
				ProgressWitnessedAt: 3_000,
			},
			{
				Session:             "sid-slowest",
				Status:              resume.WatchdogMTTRRecovered,
				DetectedAt:          4_000,
				ResumedAt:           5_000,
				ProgressWitnessedAt: 6_000,
			},
		},
	}
	params := Params{
		AsOf:  time.Unix(10_000, 0),
		Drain: drain,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := Fold(nil, params)
		if report.MTTRSessions.Status != KPIMeasured {
			b.Fatalf("unexpected MTTRSessions status: %v", report.MTTRSessions.Status)
		}
	}
}
