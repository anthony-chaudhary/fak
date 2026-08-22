package microagent_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

type goalCountingPlanner struct{ calls atomic.Int64 }

func (p *goalCountingPlanner) Model() string { return "goal-counting-fixture" }
func (p *goalCountingPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls.Add(1)
	return stubPlanner{}.Complete(ctx, messages, tools, opts...)
}

func TestRequestChildRefusesDuplicateAndCyclicGoalsBeforeModelTurn(t *testing.T) {
	planner := &goalCountingPlanner{}
	sink := &countingSink{}
	budget := &microagent.SpawnBudget{
		RootID: "root", RootGoal: "Deliver the release safely",
		MaxDepth: 2, MaxChildren: 3, MaxDescendants: 3,
	}
	host, err := microagent.NewHost(planner, microagent.Config{Workers: 1, Queue: 3, Audit: sink, SpawnBudget: budget})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer host.Close()

	if err := host.RequestChild(microagent.SpawnRequest{
		ParentID: "root", ChildID: "research", Goal: "Review release dependencies", Depth: 1,
	}, &turnAgent{id: "research", turns: 1}); err != nil {
		t.Fatalf("first RequestChild: %v", err)
	}
	if err := host.RequestChild(microagent.SpawnRequest{
		ParentID: "root", ChildID: "duplicate", Goal: "  REVIEW   release dependencies  ", Depth: 1,
	}, &turnAgent{id: "duplicate", turns: 1}); !errors.Is(err, microagent.ErrDuplicateGoal) {
		t.Fatalf("duplicate RequestChild = %v, want ErrDuplicateGoal", err)
	}
	if err := host.RequestChild(microagent.SpawnRequest{
		ParentID: "research", ChildID: "cycle", Goal: "deliver THE release safely", Depth: 2,
	}, &turnAgent{id: "cycle", turns: 1}); !errors.Is(err, microagent.ErrCyclicGoal) {
		t.Fatalf("cyclic RequestChild = %v, want ErrCyclicGoal", err)
	}
	if err := host.Drain(context.Background()); err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if err := host.ReconcileChild("research", microagent.LineageBudget{}); err != nil {
		t.Fatalf("ReconcileChild: %v", err)
	}
	if err := host.RequestChild(microagent.SpawnRequest{
		ParentID: "root", ChildID: "replayed", Goal: "review release dependencies", Depth: 1,
	}, &turnAgent{id: "replayed", turns: 1}); !errors.Is(err, microagent.ErrDuplicateGoal) {
		t.Fatalf("completed duplicate RequestChild = %v, want ErrDuplicateGoal", err)
	}
	if got := planner.calls.Load(); got != 1 {
		t.Fatalf("model calls = %d, want one admitted goal only", got)
	}
	if got := sink.kind(microagent.EventSpawn); got != 1 {
		t.Fatalf("spawn audit events = %d, want one admitted goal only", got)
	}
}

func TestRequestChildUsesHostAdmissionAndPreservesSpawnInvariants(t *testing.T) {
	sink := &countingSink{}
	budget := &microagent.SpawnBudget{RootID: "root", RootGoal: "complete the parent task", MaxDepth: 2, MaxChildren: 1, MaxDescendants: 1}
	h, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 1, Queue: 2, Audit: sink, SpawnBudget: budget})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer h.Close()
	if err := h.RequestChild(microagent.SpawnRequest{ParentID: "root", ChildID: "child", Goal: "implement the child", Depth: 1}, &turnAgent{id: "child", turns: 1}); err != nil {
		t.Fatalf("RequestChild: %v", err)
	}
	if err := h.RequestChild(microagent.SpawnRequest{ParentID: "root", ChildID: "sibling", Goal: "implement the sibling", Depth: 1}, &turnAgent{id: "child", turns: 1}); !errors.Is(err, microagent.ErrSpawnBudget) {
		t.Fatalf("second RequestChild = %v, want ErrSpawnBudget", err)
	}
	if got := sink.kind(microagent.EventSpawn); got != 1 {
		t.Fatalf("spawn audit events = %d, want one admitted child", got)
	}
	if got := budget.Descendants(); got != 1 {
		t.Fatalf("admitted descendants = %d, want one", got)
	}
}

func TestRequestChildDoesNotPoisonGoalFingerprintWhenSpawnFails(t *testing.T) {
	planner := &goalCountingPlanner{}
	budget := &microagent.SpawnBudget{
		RootID: "root", RootGoal: "complete the parent task",
		MaxDepth: 1, MaxChildren: 1, MaxDescendants: 1,
	}
	host, err := microagent.NewHost(planner, microagent.Config{Workers: 1, Queue: 1, SpawnBudget: budget})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer host.Close()
	request := microagent.SpawnRequest{ParentID: "root", ChildID: "child", Goal: "implement the child", Depth: 1}
	if err := host.RequestChild(request, nil); !errors.Is(err, microagent.ErrNilAgent) {
		t.Fatalf("nil child RequestChild = %v, want ErrNilAgent", err)
	}
	if err := host.RequestChild(request, &turnAgent{id: request.ChildID, turns: 1}); err != nil {
		t.Fatalf("retry after failed spawn: %v", err)
	}
	if err := host.Drain(context.Background()); err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if got := planner.calls.Load(); got != 1 {
		t.Fatalf("model calls = %d, want one successful retry", got)
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
			budget := &microagent.SpawnBudget{RootID: "root", RootGoal: "complete the parent task", MaxDepth: 2, MaxChildren: 2, MaxDescendants: 2, MaxTokens: 100, MaxOutputTokens: 20, MaxCostMicrosUSD: 1_000}
			host, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 1, Queue: 2, SpawnBudget: budget})
			if err != nil {
				t.Fatal(err)
			}
			defer host.Close()
			first := microagent.SpawnRequest{ParentID: "root", ChildID: "first", Goal: "implement the first child", Depth: 1, Budget: microagent.LineageBudget{Tokens: 60, OutputTokens: 10, CostMicrosUSD: 600}}
			if err := host.RequestChild(first, &turnAgent{id: first.ChildID, turns: 1}); err != nil {
				t.Fatal(err)
			}
			second := microagent.SpawnRequest{ParentID: "first", ChildID: "second", Goal: "implement the nested child", Depth: 2, Budget: test.second}
			if err := host.RequestChild(second, &turnAgent{id: second.ChildID, turns: 1}); !errors.Is(err, microagent.ErrSpawnBudget) {
				t.Fatalf("%s over-budget child = %v, want ErrSpawnBudget", test.name, err)
			}
		})
	}

	budget := &microagent.SpawnBudget{
		RootID:           "root",
		RootGoal:         "complete the parent task",
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
		{ParentID: "root", ChildID: "parent", Goal: "implement the parent child", Depth: 1, Budget: microagent.LineageBudget{Tokens: 60, OutputTokens: 10, CostMicrosUSD: 600}},
		{ParentID: "parent", ChildID: "nested", Goal: "implement the nested child", Depth: 2, Budget: microagent.LineageBudget{Tokens: 40, OutputTokens: 10, CostMicrosUSD: 400}},
	}
	for _, request := range requests {
		if err := host.RequestChild(request, &turnAgent{id: request.ChildID, turns: 1}); err != nil {
			t.Fatalf("RequestChild(%s): %v", request.ChildID, err)
		}
	}
	if err := host.RequestChild(microagent.SpawnRequest{
		ParentID: "root", ChildID: "multiplied", Depth: 1,
		Goal:   "implement an extra child",
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
		Goal:   "implement the replacement child",
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

func TestRequestChildFailsClosedOnUnverifiableGoalAncestry(t *testing.T) {
	tests := []struct {
		name    string
		budget  *microagent.SpawnBudget
		request microagent.SpawnRequest
		want    error
	}{
		{
			name:    "missing root id",
			budget:  &microagent.SpawnBudget{RootGoal: "root goal", MaxDepth: 2},
			request: microagent.SpawnRequest{ParentID: "root", ChildID: "child", Goal: "child goal", Depth: 1},
			want:    microagent.ErrInvalidAncestry,
		},
		{
			name:    "missing root goal",
			budget:  &microagent.SpawnBudget{RootID: "root", MaxDepth: 2},
			request: microagent.SpawnRequest{ParentID: "root", ChildID: "child", Goal: "child goal", Depth: 1},
			want:    microagent.ErrInvalidAncestry,
		},
		{
			name:    "missing child goal",
			budget:  &microagent.SpawnBudget{RootID: "root", RootGoal: "root goal", MaxDepth: 2},
			request: microagent.SpawnRequest{ParentID: "root", ChildID: "child", Depth: 1},
			want:    microagent.ErrSpawnBudget,
		},
		{
			name:    "unknown parent",
			budget:  &microagent.SpawnBudget{RootID: "root", RootGoal: "root goal", MaxDepth: 2},
			request: microagent.SpawnRequest{ParentID: "missing", ChildID: "child", Goal: "child goal", Depth: 1},
			want:    microagent.ErrInvalidAncestry,
		},
		{
			name:    "forged depth",
			budget:  &microagent.SpawnBudget{RootID: "root", RootGoal: "root goal", MaxDepth: 2},
			request: microagent.SpawnRequest{ParentID: "root", ChildID: "child", Goal: "child goal", Depth: 2},
			want:    microagent.ErrInvalidAncestry,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			planner := &goalCountingPlanner{}
			sink := &countingSink{}
			host, err := microagent.NewHost(planner, microagent.Config{Workers: 1, Queue: 1, Audit: sink, SpawnBudget: test.budget})
			if err != nil {
				t.Fatalf("NewHost: %v", err)
			}
			defer host.Close()
			if err := host.RequestChild(test.request, &turnAgent{id: test.request.ChildID, turns: 1}); !errors.Is(err, test.want) {
				t.Fatalf("RequestChild = %v, want %v", err, test.want)
			}
			if got := planner.calls.Load(); got != 0 {
				t.Fatalf("model calls = %d, want zero after fail-closed refusal", got)
			}
			if got := sink.kind(microagent.EventSpawn); got != 0 {
				t.Fatalf("spawn audit events = %d, want zero after fail-closed refusal", got)
			}
		})
	}
}

func TestRequestChildAdmitsThreeDistinctHarnessGoalClasses(t *testing.T) {
	planner := &goalCountingPlanner{}
	budget := &microagent.SpawnBudget{
		RootID: "root", RootGoal: "ship a safe change",
		MaxDepth: 1, MaxChildren: 3, MaxDescendants: 3,
	}
	host, err := microagent.NewHost(planner, microagent.Config{Workers: 3, Queue: 3, SpawnBudget: budget})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer host.Close()
	requests := []microagent.SpawnRequest{
		{ParentID: "root", ChildID: "research", Goal: "research the dependency evidence", Depth: 1},
		{ParentID: "root", ChildID: "implementation", Goal: "implement the bounded change", Depth: 1},
		{ParentID: "root", ChildID: "operations", Goal: "verify the rollout procedure", Depth: 1},
	}
	for _, request := range requests {
		if err := host.RequestChild(request, &turnAgent{id: request.ChildID, turns: 1}); err != nil {
			t.Fatalf("RequestChild(%s): %v", request.ChildID, err)
		}
	}
	if err := host.Drain(context.Background()); err != nil {
		t.Fatalf("Drain: %v", err)
	}
	results := host.Reap()
	if len(results) != len(requests) {
		t.Fatalf("retired results = %d, want %d", len(results), len(requests))
	}
	for _, result := range results {
		if !result.Done || result.Err != nil {
			t.Fatalf("result %q = done %v err %v", result.ID, result.Done, result.Err)
		}
	}
	if got := planner.calls.Load(); got != int64(len(requests)) {
		t.Fatalf("model calls = %d, want %d", got, len(requests))
	}
}
