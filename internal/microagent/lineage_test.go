package microagent

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

type lineageTestAgent struct{ err error }

func (a lineageTestAgent) Step(context.Context, Gateway) (bool, error) { return a.err == nil, a.err }

type lineageTestGateway struct{}

func (lineageTestGateway) Model() string { return "test" }
func (lineageTestGateway) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, nil
}

func TestWithLineageRegistersNestedMicroagentUnderStartingGoal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registrations.jsonl")
	store := sessionregistry.Store{Path: path}
	now := time.Date(2026, 8, 13, 18, 0, 0, 0, time.UTC)
	root, err := sessionregistry.New(sessionregistry.NewInput{RegistrationID: "root", RootIssue: "6583", TaskID: "goal-fleet-lineage", GoalID: "goal_observe", LaunchKind: "guard", Runtime: "codex", SessionID: "top", Now: now})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Register(root); err != nil {
		t.Fatal(err)
	}
	lineage := &Lineage{Store: store, ParentRegistrationID: root.RegistrationID, ParentAttemptID: root.AttemptID, RootRegistrationID: root.RootRegistrationID, RootIssue: root.RootIssue, TaskID: root.TaskID, GoalID: root.GoalID, Now: func() time.Time { return now.Add(time.Second) }}

	wrapped := WithLineage("shard-007", lineageTestAgent{}, lineage)
	done, err := wrapped.Step(context.Background(), lineageTestGateway{})
	if err != nil || !done {
		t.Fatalf("Step() = (%v, %v), want done", done, err)
	}
	rows, err := store.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want root + microagent", len(rows))
	}
	child := rows[1]
	if child.ParentRegistrationID != root.RegistrationID || child.RootRegistrationID != root.RegistrationID {
		t.Fatalf("lineage = parent %q root %q", child.ParentRegistrationID, child.RootRegistrationID)
	}
	if child.RootIssue != "6583" || child.TaskID != "goal-fleet-lineage" || child.GoalID != "goal_observe" {
		t.Fatalf("goal labels = issue %q task %q", child.RootIssue, child.TaskID)
	}
	if child.LaunchKind != "in_process_microagent" || child.Identity.Runtime != "microagent" || child.Identity.SessionID == "" {
		t.Fatalf("identity = %+v launch=%q", child.Identity, child.LaunchKind)
	}
	if child.State != sessionregistry.StateCompleted || child.TerminalAt.IsZero() {
		t.Fatalf("terminal child = %+v", child)
	}
}

func TestWithLineageTerminalizesFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registrations.jsonl")
	store := sessionregistry.Store{Path: path}
	now := time.Now().UTC()
	root, _ := sessionregistry.New(sessionregistry.NewInput{RegistrationID: "root", LaunchKind: "guard", Runtime: "codex", Now: now})
	if err := store.Register(root); err != nil {
		t.Fatal(err)
	}
	want := errors.New("model failed")
	wrapped := WithLineage("bad-shard", lineageTestAgent{err: want}, &Lineage{Store: store, ParentRegistrationID: root.RegistrationID, RootRegistrationID: root.RootRegistrationID, Now: func() time.Time { return now }})
	if _, err := wrapped.Step(context.Background(), lineageTestGateway{}); !errors.Is(err, want) {
		t.Fatalf("error = %v, want %v", err, want)
	}
	rows, err := store.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	var got sessionregistry.Record
	for _, row := range rows {
		if row.ParentRegistrationID != "" {
			got = row
		}
	}
	if got.State != sessionregistry.StateFailed || got.Reason != want.Error() {
		t.Fatalf("failed child = %+v", got)
	}
}

func TestWithLineageFailsClosedBeforeInnerExecution(t *testing.T) {
	lineage := &Lineage{Store: sessionregistry.Store{Path: filepath.Join(t.TempDir(), "missing-parent.jsonl")}, ParentRegistrationID: "absent", RootRegistrationID: "root"}
	_, err := WithLineage("shard", lineageTestAgent{}, lineage).Step(context.Background(), lineageTestGateway{})
	if err == nil {
		t.Fatal("expected missing-parent registration failure")
	}
}

func TestLineageChildPreservesRootAcrossNestedFanout(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registrations.jsonl")
	store := sessionregistry.Store{Path: path}
	now := time.Now().UTC()
	root, _ := sessionregistry.New(sessionregistry.NewInput{RegistrationID: "root", LaunchKind: "guard", Runtime: "codex", Now: now})
	if err := store.Register(root); err != nil {
		t.Fatal(err)
	}
	lineage := &Lineage{Store: store, ParentRegistrationID: root.RegistrationID, RootRegistrationID: root.RootRegistrationID, Now: func() time.Time { return now }}
	parent := WithLineage("parent-micro", lineageTestAgent{}, lineage)
	if _, err := parent.Step(context.Background(), lineageTestGateway{}); err != nil {
		t.Fatal(err)
	}
	nested := WithLineage("nested-micro", lineageTestAgent{}, lineage.Child("parent-micro"))
	if _, err := nested.Step(context.Background(), lineageTestGateway{}); err != nil {
		t.Fatal(err)
	}
	rows, err := store.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	byParent := map[string]sessionregistry.Record{}
	for _, row := range rows {
		byParent[row.ParentRegistrationID] = row
	}
	first := byParent[root.RegistrationID]
	second := byParent[first.RegistrationID]
	if first.RegistrationID == "" || second.RegistrationID == "" {
		t.Fatalf("nested chain missing: %+v", rows)
	}
	if second.RootRegistrationID != root.RegistrationID {
		t.Fatalf("nested root = %q, want %q", second.RootRegistrationID, root.RegistrationID)
	}
}
