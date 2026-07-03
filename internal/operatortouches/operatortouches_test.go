package operatortouches

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

func TestFoldMTTRSessionsMeasuredFromRecoveredDrainRows(t *testing.T) {
	got := Fold(nil, Params{
		AsOf: time.Unix(10_000, 0),
		Drain: &resume.WatchdogDrainStatus{
			Verdict: resume.WatchdogDrainGreen,
			MTTRSessions: []resume.WatchdogMTTRRow{
				{Session: "sid-fast", Status: resume.WatchdogMTTRRecovered,
					DetectedAt: 1_000, ResumedAt: 1_100, ProgressWitnessedAt: 1_300},
				{Session: "sid-slow", Status: resume.WatchdogMTTRRecovered,
					DetectedAt: 2_000, ResumedAt: 2_500, ProgressWitnessedAt: 3_000},
				{Session: "sid-slowest", Status: resume.WatchdogMTTRRecovered,
					DetectedAt: 4_000, ResumedAt: 5_000, ProgressWitnessedAt: 6_000},
			},
		},
	})

	if got.MTTRSessions.Status != KPIMeasured {
		t.Fatalf("mttr_sessions = %+v, want measured", got.MTTRSessions)
	}
	// MTTRs are 300s, 1000s, 2000s; the p50 is the middle recovered row.
	if got.MTTRSessions.Value != 1000 || got.MTTRSessions.Unit != "seconds" {
		t.Fatalf("mttr_sessions = %+v, want p50 1000 seconds", got.MTTRSessions)
	}
}

func TestFoldMTTRSessionsLaunchedUnprovenNeverCounts(t *testing.T) {
	got := Fold(nil, Params{
		AsOf: time.Unix(10_000, 0),
		Drain: &resume.WatchdogDrainStatus{
			Verdict: resume.WatchdogDrainRed,
			MTTRSessions: []resume.WatchdogMTTRRow{
				{Session: "sid-queued", Status: resume.WatchdogMTTRQueued, DetectedAt: 1_000},
				{Session: "sid-launched", Status: resume.WatchdogMTTRLaunchedUnproven,
					DetectedAt: 1_000, ResumedAt: 1_100},
			},
		},
	})

	if got.MTTRSessions.Status != KPINotYet {
		t.Fatalf("mttr_sessions = %+v, want not_yet when no row recovered", got.MTTRSessions)
	}
	if !strings.Contains(got.MTTRSessions.Missing, "launched") {
		t.Fatalf("missing = %q, must name the launched≠took fence", got.MTTRSessions.Missing)
	}
}

func TestFoldMTTRSessionsNotYetWithoutDrainWitness(t *testing.T) {
	got := Fold(nil, Params{AsOf: time.Unix(10_000, 0)})

	if got.MTTRSessions.Status != KPINotYet {
		t.Fatalf("mttr_sessions = %+v, want not_yet without a drain report", got.MTTRSessions)
	}
	if !strings.Contains(got.MTTRSessions.Missing, "#2273") {
		t.Fatalf("missing = %q, must name the missing drain witness (#2273)", got.MTTRSessions.Missing)
	}
}
