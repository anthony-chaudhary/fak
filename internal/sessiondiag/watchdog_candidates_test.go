package sessiondiag

import (
	"fmt"
	"testing"
)

func TestProjectWatchdogCandidatesMatchesTwentySessionAbruptCohort(t *testing.T) {
	sessions := make([]SessionRecord, 0, 23)
	for i := 0; i < 20; i++ {
		id := fmt.Sprintf("10000000-0000-4000-8000-%012d", i)
		sessions = append(sessions, SessionRecord{Thread: &ThreadRecord{ID: id, CWD: "/repo"}, LatestTurn: &TurnRecord{Status: "inProgress"}, Kind: KindInteractiveTUI, Health: HealthUnknown, Reasons: []string{ReasonNoCurrentEvidence}})
	}
	sessions = append(sessions,
		SessionRecord{Thread: &ThreadRecord{ID: "active"}, LatestTurn: &TurnRecord{Status: "inProgress"}, Kind: KindInteractiveTUI, Health: HealthActive, Reasons: []string{ReasonTurnInProgress}},
		SessionRecord{Thread: &ThreadRecord{ID: "done"}, LatestTurn: &TurnRecord{Status: "completed"}, Kind: KindInteractiveTUI, Health: HealthCompleted},
		SessionRecord{Thread: &ThreadRecord{ID: "worker"}, LatestTurn: &TurnRecord{Status: "inProgress"}, Kind: KindHeadlessExec, Health: HealthUnknown, Reasons: []string{ReasonNoCurrentEvidence}},
	)
	report := ProjectWatchdogCandidates(InventoryReport{ObservedAt: "2026-08-18T00:00:00Z", Sessions: sessions})
	if len(report.Candidates) != 20 {
		t.Fatalf("candidates=%d want 20", len(report.Candidates))
	}
	if len(report.Exclusions) != 3 {
		t.Fatalf("exclusions=%d want 3", len(report.Exclusions))
	}
	if report.Counts[WatchdogIncludeAbruptInteractive] != 20 || report.Counts[WatchdogExcludeHealth] != 1 || report.Counts[WatchdogExcludeTurn] != 1 || report.Counts[WatchdogExcludeKind] != 1 {
		t.Fatalf("counts=%v", report.Counts)
	}
	for i, candidate := range report.Candidates {
		want := fmt.Sprintf("10000000-0000-4000-8000-%012d", i)
		if candidate.Session != want || candidate.Harness != "codex" {
			t.Fatalf("candidate[%d]=%+v want session %s", i, candidate, want)
		}
	}
}
