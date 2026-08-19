package microagent_test

import (
	"errors"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"testing"
)

func TestRequestChildUsesHostAdmissionAndPreservesSpawnInvariants(t *testing.T) {
	sink := &countingSink{}
	budget := &microagent.SpawnBudget{MaxDepth: 2, MaxChildren: 1, MaxDescendants: 1}
	h, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 1, Queue: 2, Audit: sink, SpawnBudget: budget})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer h.Close()
	if err := h.RequestChild(microagent.SpawnRequest{ParentID: "root", ChildID: "child", Depth: 1}, &turnAgent{id: "child", turns: 1}); err != nil {
		t.Fatalf("RequestChild: %v", err)
	}
	if err := h.RequestChild(microagent.SpawnRequest{ParentID: "root", ChildID: "sibling", Depth: 1}, &turnAgent{id: "child", turns: 1}); !errors.Is(err, microagent.ErrSpawnBudget) {
		t.Fatalf("second RequestChild = %v, want ErrSpawnBudget", err)
	}
	if got := sink.kind(microagent.EventSpawn); got != 1 {
		t.Fatalf("spawn audit events = %d, want one admitted child", got)
	}
	if got := budget.Descendants(); got != 1 {
		t.Fatalf("admitted descendants = %d, want one", got)
	}
}

func TestRequestChildRefusesWithoutHostBudget(t *testing.T) {
	h, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 1})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer h.Close()
	if err := h.RequestChild(microagent.SpawnRequest{ParentID: "root", ChildID: "child", Depth: 1}, &turnAgent{id: "child", turns: 1}); !errors.Is(err, microagent.ErrSpawnBudget) {
		t.Fatalf("RequestChild without host budget = %v, want ErrSpawnBudget", err)
	}
}
