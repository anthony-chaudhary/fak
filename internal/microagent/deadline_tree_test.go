package microagent

import (
	"context"
	"errors"
	"testing"
	"time"
)

type contextBoundaryAgent struct{ block bool }

func (a contextBoundaryAgent) Step(ctx context.Context, _ Gateway) (bool, error) {
	if !a.block {
		return true, nil
	}
	<-ctx.Done()
	return false, ctx.Err()
}

func TestRequestChildPropagatesParentDeadlineWithoutCancellingSiblingReceipt(t *testing.T) {
	h, err := NewHost(lineageTestGateway{}, Config{Workers: 2, Queue: 4, SpawnBudget: &SpawnBudget{RootID: "root", RootGoal: "complete parent", MaxChildren: 2, MaxDescendants: 2, MaxDepth: 2}})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	parent, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	base := SpawnRequest{ParentID: "root", Depth: 1, Budget: LineageBudget{Tokens: 1}}
	timed := base
	timed.ChildID, timed.Goal, timed.Context = "timed", "wait for deadline", parent
	if err := h.RequestChild(timed, contextBoundaryAgent{block: true}); err != nil {
		t.Fatal(err)
	}
	completed := base
	completed.ChildID, completed.Goal, completed.Context = "completed", "finish independently", context.Background()
	if err := h.RequestChild(completed, contextBoundaryAgent{}); err != nil {
		t.Fatal(err)
	}
	ctx, stop := context.WithTimeout(context.Background(), time.Second)
	defer stop()
	if err := h.Drain(ctx); err != nil {
		t.Fatal(err)
	}
	results := h.Reap()
	if len(results) != 2 {
		t.Fatalf("results = %+v", results)
	}
	byID := map[string]Result{results[0].ID: results[0], results[1].ID: results[1]}
	if !errors.Is(byID["timed"].Err, context.DeadlineExceeded) || byID["timed"].Done {
		t.Fatalf("timed = %+v", byID["timed"])
	}
	if !byID["completed"].Done || byID["completed"].Err != nil {
		t.Fatalf("completed sibling receipt = %+v", byID["completed"])
	}
}
