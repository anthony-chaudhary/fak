package harnesscompose

import (
	"errors"
	"reflect"
	"testing"
)

func TestDeriveWorkflowOrdersRecursiveGraphAndPreservesBoundaries(t *testing.T) {
	nodes := []TaskNode{
		{ID: "ship", DependsOn: []string{"test", "build"}, ActionRef: "action:ship", EvidenceRef: "receipt:ship", Verified: true, MaxAttempts: 1, DoneWhen: "release-green"},
		{ID: "test", DependsOn: []string{"build"}, ActionRef: "action:test", EvidenceRef: "receipt:test", Verified: true, MaxAttempts: 2, DoneWhen: "tests-green"},
		{ID: "build", ActionRef: "action:build", EvidenceRef: "receipt:build", Verified: true, MaxAttempts: 1, DoneWhen: "artifact-exists"},
	}
	stages, err := DeriveWorkflow(nodes)
	if err != nil {
		t.Fatalf("DeriveWorkflow: %v", err)
	}
	got := []string{stages[0].ID, stages[1].ID, stages[2].ID}
	if want := []string{"build", "test", "ship"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("stage order = %v, want %v", got, want)
	}
	if stages[1].MaxAttempts != 2 || stages[1].DoneWhen != "tests-green" || stages[1].EvidenceRef != "receipt:test" {
		t.Fatalf("test stage lost retry, gate, or provenance: %#v", stages[1])
	}
}

func TestDeriveWorkflowRefusesCycleAndUnboundedRetry(t *testing.T) {
	base := func(id string) TaskNode {
		return TaskNode{ID: id, ActionRef: "action:" + id, EvidenceRef: "receipt:" + id, Verified: true, MaxAttempts: 1, DoneWhen: id + "-done"}
	}
	a, b := base("a"), base("b")
	a.DependsOn, b.DependsOn = []string{"b"}, []string{"a"}
	if _, err := DeriveWorkflow([]TaskNode{a, b}); !errors.Is(err, ErrInvalidTaskGraph) {
		t.Fatalf("cycle error = %v, want ErrInvalidTaskGraph", err)
	}
	unbounded := base("unbounded")
	unbounded.MaxAttempts = 0
	if _, err := DeriveWorkflow([]TaskNode{unbounded}); !errors.Is(err, ErrInvalidTaskGraph) {
		t.Fatalf("retry error = %v, want ErrInvalidTaskGraph", err)
	}
}
