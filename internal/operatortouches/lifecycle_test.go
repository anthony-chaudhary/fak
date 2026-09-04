package operatortouches

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// Invariant: Operator touches folding must distinguish measured watchdog recovery from unproven runs.
// Guard: Fold marks MTTRSessions as KPIMeasured only when recovered drain rows are present.

func TestOperatorTouchesLifecycle(t *testing.T) {
	t.Parallel()

	report := Fold(nil, Params{
		AsOf: time.Unix(10_000, 0),
		Drain: &resume.WatchdogDrainStatus{
			Verdict: resume.WatchdogDrainGreen,
			MTTRSessions: []resume.WatchdogMTTRRow{
				{
					Session:             "sess-recovered",
					Status:              resume.WatchdogMTTRRecovered,
					DetectedAt:          1_000,
					ResumedAt:           1_100,
					ProgressWitnessedAt: 1_200,
				},
			},
		},
	})

	if report.MTTRSessions.Status != KPIMeasured {
		t.Fatalf("expected KPIMeasured status, got %s", report.MTTRSessions.Status)
	}
	if report.MTTRSessions.Value != 200 {
		t.Fatalf("expected MTTR value 200, got %v", report.MTTRSessions.Value)
	}
}
