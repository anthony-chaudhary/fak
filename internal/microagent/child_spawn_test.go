package microagent_test

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
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

func TestRequestChildReservesAndReconcilesAggregateLineageBudget(t *testing.T) {
	for _, test := range []struct {
		name   string
		second microagent.LineageBudget
	}{
		{name: "tokens", second: microagent.LineageBudget{Tokens: 41, OutputTokens: 10, CostMicrosUSD: 400}},
		{name: "output tokens", second: microagent.LineageBudget{Tokens: 40, OutputTokens: 11, CostMicrosUSD: 400}},
		{name: "cost", second: microagent.LineageBudget{Tokens: 40, OutputTokens: 10, CostMicrosUSD: 401}},
	} {
		t.Run(test.name, func(t *testing.T) {
			budget := &microagent.SpawnBudget{MaxDepth: 2, MaxChildren: 2, MaxDescendants: 2, MaxTokens: 100, MaxOutputTokens: 20, MaxCostMicrosUSD: 1_000}
			host, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 1, Queue: 2, SpawnBudget: budget})
			if err != nil {
				t.Fatal(err)
			}
			defer host.Close()
			first := microagent.SpawnRequest{ParentID: "root", ChildID: "first", Depth: 1, Budget: microagent.LineageBudget{Tokens: 60, OutputTokens: 10, CostMicrosUSD: 600}}
			if err := host.RequestChild(first, &turnAgent{id: first.ChildID, turns: 1}); err != nil {
				t.Fatal(err)
			}
			second := microagent.SpawnRequest{ParentID: "first", ChildID: "second", Depth: 2, Budget: test.second}
			if err := host.RequestChild(second, &turnAgent{id: second.ChildID, turns: 1}); !errors.Is(err, microagent.ErrSpawnBudget) {
				t.Fatalf("%s over-budget child = %v, want ErrSpawnBudget", test.name, err)
			}
		})
	}

	budget := &microagent.SpawnBudget{
		MaxDepth:         2,
		MaxChildren:      2,
		MaxDescendants:   3,
		MaxTokens:        100,
		MaxOutputTokens:  20,
		MaxCostMicrosUSD: 1_000,
	}
	host, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 1, Queue: 3, SpawnBudget: budget})
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()

	requests := []microagent.SpawnRequest{
		{ParentID: "root", ChildID: "parent", Depth: 1, Budget: microagent.LineageBudget{Tokens: 60, OutputTokens: 10, CostMicrosUSD: 600}},
		{ParentID: "parent", ChildID: "nested", Depth: 2, Budget: microagent.LineageBudget{Tokens: 40, OutputTokens: 10, CostMicrosUSD: 400}},
	}
	for _, request := range requests {
		if err := host.RequestChild(request, &turnAgent{id: request.ChildID, turns: 1}); err != nil {
			t.Fatalf("RequestChild(%s): %v", request.ChildID, err)
		}
	}
	if err := host.RequestChild(microagent.SpawnRequest{
		ParentID: "root", ChildID: "multiplied", Depth: 1,
		Budget: microagent.LineageBudget{Tokens: 1, OutputTokens: 1, CostMicrosUSD: 1},
	}, &turnAgent{id: "multiplied", turns: 1}); !errors.Is(err, microagent.ErrSpawnBudget) {
		t.Fatalf("aggregate over-budget child = %v, want ErrSpawnBudget", err)
	}
	if err := host.Drain(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := host.ReconcileChild("parent", microagent.LineageBudget{Tokens: 20, OutputTokens: 5, CostMicrosUSD: 200}); err != nil {
		t.Fatalf("ReconcileChild(parent): %v", err)
	}

	next, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 1, Queue: 1, SpawnBudget: budget})
	if err != nil {
		t.Fatal(err)
	}
	defer next.Close()
	request := microagent.SpawnRequest{
		ParentID: "root", ChildID: "replacement", Depth: 1,
		Budget: microagent.LineageBudget{Tokens: 40, OutputTokens: 5, CostMicrosUSD: 400},
	}
	if err := next.RequestChild(request, &turnAgent{id: request.ChildID, turns: 1}); err != nil {
		t.Fatalf("reconciled capacity should admit replacement: %v", err)
	}
	if err := next.Drain(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := next.ReconcileChild("replacement", microagent.LineageBudget{Tokens: 41, OutputTokens: 5, CostMicrosUSD: 400}); !errors.Is(err, microagent.ErrSpawnBudget) {
		t.Fatalf("over-reservation reconciliation = %v, want ErrSpawnBudget", err)
	}
}
