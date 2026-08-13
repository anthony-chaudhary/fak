package sessionregistry

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestRegistrationChainAndTerminalReadback(t *testing.T) {
	now := time.Date(2026, 8, 11, 12, 0, 0, 0, time.UTC)
	s := Store{Path: filepath.Join(t.TempDir(), "registry.jsonl")}
	root, err := New(NewInput{RegistrationID: "root", RootIssue: "6458", TaskID: "issue-6458", AttemptID: "attempt-root", LaunchKind: "guarded_tui", Runtime: "codex", Lane: "sessionregistry", Now: now})
	if err != nil {
		t.Fatal(err)
	}
	if err = s.Register(root); err != nil {
		t.Fatal(err)
	}
	child, err := New(NewInput{RegistrationID: "child", ParentRegistrationID: "root", RootRegistrationID: "root", RootIssue: "6458", TaskID: "issue-6458", AttemptID: "attempt-child", LaunchKind: "headless_worker", Runtime: "codex", Lane: "sessionregistry", LeaseID: "lease-1", Scope: []string{"internal/sessionregistry/**"}, Now: now.Add(time.Second)})
	if err != nil {
		t.Fatal(err)
	}
	if err = s.Register(child); err != nil {
		t.Fatal(err)
	}
	if _, err = s.Start("child", 42, now.Add(2*time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err = s.Terminal("child", StateCompleted, "", "commit:abc", now.Add(3*time.Second)); err != nil {
		t.Fatal(err)
	}
	rows, err := s.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	chain := Chain(rows, "child")
	if len(chain) != 2 || chain[0].RegistrationID != "root" || chain[1].State != StateCompleted || chain[1].Identity.PID != 42 || chain[1].WitnessRef != "commit:abc" {
		t.Fatalf("chain=%+v", chain)
	}
}

func TestRegistrationFailsClosedForMissingParentAndConflicts(t *testing.T) {
	s := Store{Path: filepath.Join(t.TempDir(), "registry.jsonl")}
	child, err := New(NewInput{RegistrationID: "child", ParentRegistrationID: "missing", RootRegistrationID: "root", AttemptID: "a", LaunchKind: "subagent", Runtime: "claude"})
	if err != nil {
		t.Fatal(err)
	}
	if err = s.Register(child); err == nil {
		t.Fatal("missing parent admitted")
	}
	root, _ := New(NewInput{RegistrationID: "root", AttemptID: "a", LaunchKind: "headless_worker", Runtime: "codex"})
	if err = s.Register(root); err != nil {
		t.Fatal(err)
	}
	if err = s.Register(root); err != nil {
		t.Fatalf("idempotent retry: %v", err)
	}
	conflict := root
	conflict.Identity.Runtime = "claude"
	if err = s.Register(conflict); err == nil {
		t.Fatal("conflicting replay admitted")
	}
}

func TestUnknownRequiresReason(t *testing.T) {
	r, _ := New(NewInput{RegistrationID: "root", AttemptID: "a", LaunchKind: "headless_worker", Runtime: "codex"})
	r.State = StateUnknown
	if err := Validate(r); err == nil {
		t.Fatal("unknown without reason admitted")
	}
}

func TestFilterCountsAndUnregisteredObserved(t *testing.T) {
	now := time.Now().UTC()
	rows := []Record{{Schema: Schema, RegistrationID: "a", RootRegistrationID: "a", RootIssue: "6458", AttemptID: "x", LaunchKind: "guarded_tui", Identity: Identity{Runtime: "codex", PID: 10, ProcessStartedAt: now}, State: StateActive, CreatedAt: now}, {Schema: Schema, RegistrationID: "b", ParentRegistrationID: "a", RootRegistrationID: "a", RootIssue: "6458", AttemptID: "y", LaunchKind: "subagent", Lane: "agent", LeaseID: "l", Identity: Identity{Runtime: "claude"}, State: StateCompleted, CreatedAt: now, TerminalAt: now, WitnessRef: "commit:x"}}
	if got := Filter(rows, Query{RootIssue: "6458", Lane: "agent", WitnessRef: "commit:x"}); len(got) != 1 || got[0].RegistrationID != "b" {
		t.Fatalf("filter=%+v", got)
	}
	missing := ReconcileObserved(rows, []ObservedProcess{{PID: 10, ProcessStartedAt: now, Runtime: "codex"}, {PID: 11, ProcessStartedAt: now, Runtime: "claude"}})
	if len(missing) != 1 || missing[0].Verdict != "UNREGISTERED_OBSERVED" || missing[0].Process.PID != 11 {
		t.Fatalf("missing=%+v", missing)
	}
	c := Summarize(rows, len(missing))
	if c.Active != 1 || c.Terminal != 1 || c.UnregisteredObserved != 1 || c.ByKind["subagent"] != 1 {
		t.Fatalf("counts=%+v", c)
	}
}

func TestResumeHasDistinctAttemptAndParentAttempt(t *testing.T) {
	now := time.Now().UTC()
	s := Store{Path: filepath.Join(t.TempDir(), "r.jsonl")}
	root, _ := New(NewInput{RegistrationID: "root", AttemptID: "parent-a", LaunchKind: "guarded_tui", Runtime: "codex", Now: now})
	if err := s.Register(root); err != nil {
		t.Fatal(err)
	}
	resume, _ := New(NewInput{RegistrationID: "resume", ParentRegistrationID: "root", ParentAttemptID: "parent-a", RootRegistrationID: "root", AttemptID: "child-b", ResumeOfAttemptID: "child-a", LaunchKind: "resume_wrapper", Runtime: "codex", Now: now.Add(time.Second)})
	if err := s.Register(resume); err != nil {
		t.Fatal(err)
	}
	rows, _ := s.ReadAll()
	got := rows[1]
	if got.AttemptID == got.ResumeOfAttemptID || got.ParentAttemptID != "parent-a" {
		t.Fatalf("resume=%+v", got)
	}
}

func TestChildCannotCompleteBeforeParentRegistrationPersists(t *testing.T) {
	s := Store{Path: filepath.Join(t.TempDir(), "r.jsonl")}
	child, _ := New(NewInput{RegistrationID: "child", ParentRegistrationID: "parent", ParentAttemptID: "p", RootRegistrationID: "parent", AttemptID: "c", LaunchKind: "subagent", Runtime: "codex"})
	if err := s.Register(child); err == nil {
		t.Fatal("child admitted before parent")
	}
	if _, err := s.Terminal("child", StateCompleted, "", "w", time.Now()); err == nil {
		t.Fatal("unregistered child terminalized")
	}
}

func TestPIDReuseRequiresProcessStartIdentity(t *testing.T) {
	n := time.Now().UTC()
	rows := []Record{{RegistrationID: "old", Identity: Identity{PID: 7, ProcessStartedAt: n}}}
	missing := ReconcileObserved(rows, []ObservedProcess{{PID: 7, ProcessStartedAt: n.Add(time.Hour), Runtime: "codex"}})
	if len(missing) != 1 {
		t.Fatalf("pid reuse matched old row: %+v", missing)
	}
}

func TestRegisterRepairsInterruptedFinalAppend(t *testing.T) {
	now := time.Date(2026, 8, 13, 12, 0, 0, 0, time.UTC)
	s := Store{Path: filepath.Join(t.TempDir(), "registry.jsonl")}
	root, err := New(NewInput{RegistrationID: "root", RootIssue: "6458", TaskID: "issue-6458", AttemptID: "attempt-root", LaunchKind: "guarded_tui", Runtime: "codex", Lane: "sessionregistry", Now: now})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Register(root); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(s.Path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(`{"schema":"fak-child-registration/1","record":`); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	rows, err := s.ReadAll()
	if err != nil {
		t.Fatalf("read should retain events before interrupted append: %v", err)
	}
	if len(rows) != 1 || rows[0].RegistrationID != "root" {
		t.Fatalf("rows = %#v, want surviving root", rows)
	}

	child, err := New(NewInput{RegistrationID: "child", ParentRegistrationID: "root", RootRegistrationID: "root", RootIssue: "6458", TaskID: "issue-6458", AttemptID: "attempt-child", LaunchKind: "headless_worker", Runtime: "codex", Lane: "sessionregistry", Now: now.Add(time.Second)})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Register(child); err != nil {
		t.Fatalf("register after interrupted append: %v", err)
	}
	rows, err = s.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want repaired root and child", len(rows))
	}
}
