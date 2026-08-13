package main

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/microagent"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

func TestRunRegistersEveryLogicalContextUnderStartingGoal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "child-registrations.jsonl")
	store := sessionregistry.Store{Path: path}
	now := time.Date(2026, 8, 13, 19, 0, 0, 0, time.UTC)
	root, err := sessionregistry.New(sessionregistry.NewInput{
		RegistrationID: "top-goal", RootIssue: "6583", TaskID: "trace-starting-goal",
		LaunchKind: "guard", Runtime: "codex", SessionID: "top-session", Now: now,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Register(root); err != nil {
		t.Fatal(err)
	}
	lineage := &microagent.Lineage{
		Store: store, ParentRegistrationID: root.RegistrationID, ParentAttemptID: root.AttemptID,
		RootRegistrationID: root.RootRegistrationID, RootIssue: root.RootIssue, TaskID: root.TaskID,
		Now: func() time.Time { return now.Add(time.Second) },
	}

	report, err := run(context.Background(), config{Contexts: 3, Workers: 2, Delay: 20 * time.Millisecond, Selfcheck: true, Lineage: lineage})
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "PASS" || report.Completed != 3 {
		t.Fatalf("report = %+v", report)
	}
	rows, err := store.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 4 {
		t.Fatalf("registrations = %d, want root + 3 contexts", len(rows))
	}
	seenSessions := map[string]bool{}
	for _, row := range rows[1:] {
		if row.ParentRegistrationID != root.RegistrationID || row.RootRegistrationID != root.RegistrationID {
			t.Fatalf("context escaped root: %+v", row)
		}
		if row.RootIssue != "6583" || row.TaskID != "trace-starting-goal" {
			t.Fatalf("context lost goal labels: %+v", row)
		}
		if row.LaunchKind != "in_process_microagent" || row.State != sessionregistry.StateCompleted {
			t.Fatalf("context lifecycle = %+v", row)
		}
		if row.Identity.SessionID == "" || seenSessions[row.Identity.SessionID] {
			t.Fatalf("non-unique context session %q", row.Identity.SessionID)
		}
		seenSessions[row.Identity.SessionID] = true
	}
}
