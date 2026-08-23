package sessionregistry

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type schemaCollisionMarker interface {
	error
	schemaCollision()
}

func TestReadAllNamesDescriptorStoreSchemaCollision(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session-registry.json")
	before := []byte("{\n  \"version\": \"fak.session-descriptors.v1\",\n  \"descriptors\": []\n}\n")
	if err := os.WriteFile(path, before, 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := (Store{Path: path}).ReadAll()
	if err == nil {
		t.Fatal("ReadAll() adopted a descriptor store; want schema-collision refusal")
	}
	var collision schemaCollisionMarker
	if !errors.As(err, &collision) {
		t.Fatalf("ReadAll() error %T %v, want typed schema collision", err, err)
	}
	for _, want := range []string{"session registry schema collision", "fak.session-descriptors.v1", "fak-child-registration/1", "<UserConfigDir>/fak/session-registry.json", "<UserConfigDir>/fak/child-registrations.jsonl"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal message %q missing %q", err, want)
		}
	}
	record, newErr := New(NewInput{RegistrationID: "guard-child", AttemptID: "guard-attempt", LaunchKind: "headless_worker", Runtime: "codex"})
	if newErr != nil {
		t.Fatal(newErr)
	}
	registerErr := (Store{Path: path}).Register(record)
	if !errors.As(registerErr, &collision) {
		t.Fatalf("guard registration error %T %v, want typed schema collision", registerErr, registerErr)
	}
	if strings.Contains(registerErr.Error(), "unexpected end of JSON input") {
		t.Fatalf("guard registration misreported schema collision as truncation: %v", registerErr)
	}
	after, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if !bytes.Equal(after, before) {
		t.Fatalf("schema-collision read changed descriptor store\nbefore: %q\n after: %q", before, after)
	}
}

func TestIsTerminalCoversEveryState(t *testing.T) {
	cases := map[State]bool{
		StateRegistered: false, StateActive: false, StateCompleted: true, StateFailed: true,
		StateCancelled: true, StateLost: true, StateReaped: true, StateUnknown: false,
	}
	for state, want := range cases {
		if got := isTerminal(state); got != want {
			t.Errorf("isTerminal(%q) = %v, want %v", state, got, want)
		}
	}
	if isTerminal(State("future")) {
		t.Fatal("unknown future state must not be treated as terminal")
	}
}

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

func TestCanonicalGoalIdentityPropagatesWithoutInference(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registrations.jsonl")
	s := Store{Path: path}
	now := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	rootA := Record{Schema: Schema, RegistrationID: "root-a", RootRegistrationID: "root-a", TaskID: "same-title", GoalID: "goal_observe", AttemptID: "a", LaunchKind: "guarded_tui", Identity: Identity{Runtime: "claude"}, State: StateRegistered, CreatedAt: now}
	rootB := rootA
	rootB.RegistrationID, rootB.RootRegistrationID, rootB.AttemptID, rootB.Identity.Runtime = "root-b", "root-b", "b", "codex"
	for _, root := range []Record{rootA, rootB} {
		if err := s.Register(root); err != nil {
			t.Fatal(err)
		}
	}
	child := Record{Schema: Schema, RegistrationID: "child-a", ParentRegistrationID: "root-a", RootRegistrationID: "root-a", TaskID: "same-title", GoalID: "goal_observe", AttemptID: "c", LaunchKind: "subagent", Identity: Identity{Runtime: "codex"}, State: StateRegistered, CreatedAt: now}
	if err := s.Register(child); err != nil {
		t.Fatal(err)
	}
	rows, err := s.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 3 || rows[2].GoalID != "goal_observe" {
		t.Fatalf("rows = %+v", rows)
	}
}

func TestCanonicalGoalIdentityCannotChangeWithinExecutionLineage(t *testing.T) {
	for _, tc := range []struct{ name, parentGoal, childGoal string }{
		{"change", "goal_a", "goal_b"},
		{"drop", "goal_a", ""},
		{"inject_into_legacy", "", "goal_a"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := Store{Path: filepath.Join(t.TempDir(), "registrations.jsonl")}
			now := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
			root := Record{Schema: Schema, RegistrationID: "root", RootRegistrationID: "root", TaskID: "same-title", GoalID: tc.parentGoal, AttemptID: "a", LaunchKind: "guarded_tui", Identity: Identity{Runtime: "codex"}, State: StateRegistered, CreatedAt: now}
			if err := s.Register(root); err != nil {
				t.Fatal(err)
			}
			child := Record{Schema: Schema, RegistrationID: "child", ParentRegistrationID: "root", RootRegistrationID: "root", TaskID: "same-title", GoalID: tc.childGoal, AttemptID: "b", LaunchKind: "subagent", Identity: Identity{Runtime: "codex"}, State: StateRegistered, CreatedAt: now}
			if err := s.Register(child); err == nil || !strings.Contains(err.Error(), "goal_id differ") {
				t.Fatalf("want goal lineage refusal, got %v", err)
			}
		})
	}
}

func TestLegacyRegistrationOmitsCanonicalGoalIdentity(t *testing.T) {
	r := Record{Schema: Schema, RegistrationID: "legacy", RootRegistrationID: "legacy", TaskID: "same-title", AttemptID: "a", LaunchKind: "guarded_tui", Identity: Identity{Runtime: "codex"}, State: StateRegistered, CreatedAt: time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "goal_id") {
		t.Fatalf("legacy row changed shape: %s", b)
	}
}

func TestBindGoalRootUpdatesOnlyWitnessedExecutionRoot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.jsonl")
	s := Store{Path: path}
	now := time.Unix(1700000000, 0).UTC()
	rows := []Record{
		{Schema: Schema, RegistrationID: "root-a", RootRegistrationID: "root-a", TaskID: "same-title", AttemptID: "a", LaunchKind: "guarded_tui", Identity: Identity{Runtime: "claude"}, State: StateRegistered, CreatedAt: now},
		{Schema: Schema, RegistrationID: "child-a", ParentRegistrationID: "root-a", RootRegistrationID: "root-a", TaskID: "same-title", AttemptID: "b", LaunchKind: "subagent", Identity: Identity{Runtime: "codex"}, State: StateRegistered, CreatedAt: now.Add(time.Second)},
		{Schema: Schema, RegistrationID: "root-b", RootRegistrationID: "root-b", TaskID: "same-title", AttemptID: "c", LaunchKind: "guarded_tui", Identity: Identity{Runtime: "codex"}, State: StateRegistered, CreatedAt: now.Add(2 * time.Second)},
	}
	for _, row := range rows {
		if err := s.Register(row); err != nil {
			t.Fatal(err)
		}
	}
	preview, err := s.BindGoalRoot("root-a", "goal-observe", false)
	if err != nil || len(preview) != 2 {
		t.Fatalf("preview=%#v err=%v", preview, err)
	}
	before, _ := s.ReadAll()
	if before[0].GoalID != "" {
		t.Fatal("dry-run mutated registry")
	}
	if _, err := s.BindGoalRoot("root-a", "goal-observe", true); err != nil {
		t.Fatal(err)
	}
	after, err := s.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range after {
		if row.RootRegistrationID == "root-a" && row.GoalID != "goal-observe" {
			t.Fatalf("bound row = %#v", row)
		}
		if row.RootRegistrationID == "root-b" && row.GoalID != "" {
			t.Fatalf("same-title root inferred = %#v", row)
		}
	}
	if _, err := s.BindGoalRoot("root-a", "goal-other", true); err == nil || !strings.Contains(err.Error(), "conflicting") {
		t.Fatalf("conflict=%v", err)
	}
	if _, err := s.BindGoalRoot("missing", "goal-observe", true); err == nil {
		t.Fatal("missing root accepted")
	}
}

func TestGoalTopologyGroupsExactExplicitIdentity(t *testing.T) {
	s := Store{Path: filepath.Join(t.TempDir(), "registry.jsonl")}
	now := time.Unix(1700000000, 0).UTC()
	rows := []Record{
		{Schema: Schema, RegistrationID: "root-z", RootRegistrationID: "root-z", GoalID: "goal-one", TaskID: "same", AttemptID: "a", LaunchKind: "guarded_tui", Identity: Identity{Runtime: "claude", SessionID: "s1"}, State: StateRegistered, CreatedAt: now},
		{Schema: Schema, RegistrationID: "child-z", ParentRegistrationID: "root-z", RootRegistrationID: "root-z", GoalID: "goal-one", TaskID: "same", AttemptID: "b", LaunchKind: "subagent", Identity: Identity{Runtime: "codex", SessionID: "s2"}, State: StateRegistered, CreatedAt: now.Add(time.Second)},
		{Schema: Schema, RegistrationID: "root-a", RootRegistrationID: "root-a", GoalID: "goal-one", TaskID: "same", AttemptID: "c", LaunchKind: "guarded_tui", Identity: Identity{Runtime: "codex", SessionID: "s3"}, State: StateRegistered, CreatedAt: now},
		{Schema: Schema, RegistrationID: "root-unbound", RootRegistrationID: "root-unbound", TaskID: "same", AttemptID: "d", LaunchKind: "guarded_tui", Identity: Identity{Runtime: "claude"}, State: StateRegistered, CreatedAt: now},
		{Schema: Schema, RegistrationID: "root-other", RootRegistrationID: "root-other", GoalID: "goal-two", TaskID: "same", AttemptID: "e", LaunchKind: "guarded_tui", Identity: Identity{Runtime: "claude"}, State: StateRegistered, CreatedAt: now},
	}
	for _, row := range rows {
		if err := s.Register(row); err != nil {
			t.Fatal(err)
		}
	}
	groups, err := s.GoalTopology("goal-one")
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 || groups[0][0].RootRegistrationID != "root-a" || groups[1][0].RootRegistrationID != "root-z" || len(groups[1]) != 2 {
		t.Fatalf("groups=%#v", groups)
	}
}
