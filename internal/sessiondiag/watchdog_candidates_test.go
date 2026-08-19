package sessiondiag

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
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

func TestProjectWatchdogCandidatesPreservesRegisteredHarnessIdentity(t *testing.T) {
	now := time.Date(2026, 8, 18, 0, 0, 0, 0, time.UTC)
	harnesses := []string{"claude", "codex", "opencode"}
	in := InventoryInput{Window: time.Hour, StaleAfter: time.Minute}
	for i, harness := range harnesses {
		id := fmt.Sprintf("20000000-0000-4000-8000-%012d", i)
		in.Threads = append(in.Threads, ThreadEvidence{ThreadID: id, Source: "cli", UpdatedAt: now, CWD: "/repo"})
		in.Turns = append(in.Turns, TurnEvidence{ThreadID: id, TurnID: "turn", Status: "inProgress", StartedAt: now})
		in.Registrations = append(in.Registrations, sessionregistry.Record{Identity: sessionregistry.Identity{Runtime: harness, ThreadID: id}})
	}
	report := ReconcileInventory(in, now.Add(2*time.Minute))
	for _, session := range report.Sessions {
		if session.HarnessSource != "session_registration" {
			t.Fatalf("session %s harness source=%q", session.RecordID, session.HarnessSource)
		}
	}
	candidates := ProjectWatchdogCandidates(report)
	if len(candidates.Candidates) != len(harnesses) {
		t.Fatalf("candidates=%+v exclusions=%+v", candidates.Candidates, candidates.Exclusions)
	}
	for i, candidate := range candidates.Candidates {
		if candidate.Harness != harnesses[i] {
			t.Fatalf("candidate[%d].harness=%q want %q", i, candidate.Harness, harnesses[i])
		}
	}
}

func TestInventoryTypesLegacyCodexHarnessFallback(t *testing.T) {
	now := time.Date(2026, 8, 18, 0, 0, 0, 0, time.UTC)
	report := ReconcileInventory(InventoryInput{
		Threads: []ThreadEvidence{{ThreadID: "legacy", Source: "cli", UpdatedAt: now}},
		Turns:   []TurnEvidence{{ThreadID: "legacy", Status: "inProgress", StartedAt: now}},
		Window:  time.Hour,
	}, now)
	if len(report.Sessions) != 1 || report.Sessions[0].Harness != "codex" || report.Sessions[0].HarnessSource != "legacy_codex_inventory" {
		t.Fatalf("legacy session=%+v", report.Sessions)
	}
}
