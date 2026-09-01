package sessionregistry

import (
	"testing"
	"time"
)

func TestBuildWIPBindingsCountsRootChildrenAndResumesOnce(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	rows := []Record{
		{RegistrationID: "root", RootRegistrationID: "root", RootIssue: "10437", AttemptID: "a0", Identity: Identity{SessionID: "s0"}, State: StateActive, CreatedAt: now},
		{RegistrationID: "child", ParentRegistrationID: "root", RootRegistrationID: "root", RootIssue: "10437", AttemptID: "a1", Identity: Identity{SessionID: "s1"}, State: StateCompleted, CreatedAt: now},
		{RegistrationID: "resume", RootRegistrationID: "root", RootIssue: "10437", AttemptID: "a2", ResumeOfAttemptID: "a1", Identity: Identity{SessionID: "s2"}, State: StateActive, CreatedAt: now},
	}
	report := BuildWIPBindings(rows, "anthony-chaudhary/fak", time.Time{})
	if len(report.Bindings) != 1 {
		t.Fatalf("bindings = %d, want one logical unit", len(report.Bindings))
	}
	got := report.Bindings[0]
	if got.Status != WIPBindingJoined || got.Issue == nil || got.Issue.Number != 10437 {
		t.Fatalf("binding = %#v", got)
	}
	if len(got.AttemptIDs) != 3 || len(got.SessionIDs) != 3 {
		t.Fatalf("attempts=%v sessions=%v", got.AttemptIDs, got.SessionIDs)
	}
}

func TestBuildWIPBindingsReturnsTypedDebt(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	rows := []Record{
		{RegistrationID: "missing", RootRegistrationID: "missing", AttemptID: "m", State: StateActive, CreatedAt: now},
		{RegistrationID: "conflict-root", RootRegistrationID: "conflict-root", RootIssue: "10437", AttemptID: "c0", State: StateActive, CreatedAt: now},
		{RegistrationID: "conflict-child", RootRegistrationID: "conflict-root", RootIssue: "10438", AttemptID: "c1", State: StateActive, CreatedAt: now},
		{RegistrationID: "stale", RootRegistrationID: "stale", RootIssue: "10439", AttemptID: "s", State: StateActive, CreatedAt: now.Add(-2 * time.Hour)},
	}
	report := BuildWIPBindings(rows, "anthony-chaudhary/fak", now.Add(-time.Hour))
	statuses := map[string]WIPBindingStatus{}
	for _, binding := range report.Bindings {
		statuses[binding.RootRegistrationID] = binding.Status
	}
	if statuses["missing"] != WIPBindingMissing || statuses["conflict-root"] != WIPBindingConflicting || statuses["stale"] != WIPBindingStale {
		t.Fatalf("statuses = %#v", statuses)
	}
}

func TestBuildWIPBindingsDoesNotMergeSimilarTasksOrUnrelatedRoots(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	rows := []Record{
		{RegistrationID: "one", RootRegistrationID: "one", RootIssue: "10437", TaskID: "same title", AttemptID: "a", State: StateActive, CreatedAt: now},
		{RegistrationID: "two", RootRegistrationID: "two", RootIssue: "10438", TaskID: "same title", AttemptID: "b", State: StateActive, CreatedAt: now},
	}
	report := BuildWIPBindings(rows, "anthony-chaudhary/fak", time.Time{})
	if len(report.Bindings) != 2 || report.Bindings[0].Status != WIPBindingJoined || report.Bindings[1].Status != WIPBindingJoined {
		t.Fatalf("bindings = %#v", report.Bindings)
	}

	rows[1].RootIssue = "10437"
	report = BuildWIPBindings(rows, "anthony-chaudhary/fak", time.Time{})
	if report.Bindings[0].Status != WIPBindingAmbiguous || report.Bindings[1].Status != WIPBindingAmbiguous {
		t.Fatalf("same issue on unrelated roots must be ambiguous: %#v", report.Bindings)
	}
}
