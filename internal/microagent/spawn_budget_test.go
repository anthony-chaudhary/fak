package microagent_test

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

func TestSpawnBudgetBoundsDepthFanoutAndAggregateLineage(t *testing.T) {
	budget := &microagent.SpawnBudget{RootID: "root", RootGoal: "complete the parent task", MaxDepth: 2, MaxChildren: 2, MaxDescendants: 3}
	admit := func(parent, child string, depth int) error {
		return budget.Admit(microagent.SpawnRequest{ParentID: parent, ChildID: child, Goal: "goal for " + child, Depth: depth})
	}
	if err := admit("root", "architecture", 1); err != nil {
		t.Fatalf("parent admission: %v", err)
	}
	if err := admit("architecture", "tools", 2); err != nil {
		t.Fatalf("child admission: %v", err)
	}
	if err := admit("tools", "great-grandchild", 3); !errors.Is(err, microagent.ErrSpawnBudget) {
		t.Fatalf("depth refusal = %v, want ErrSpawnBudget", err)
	}
	if err := admit("architecture", "proof", 2); err != nil {
		t.Fatalf("second child admission: %v", err)
	}
	if err := admit("architecture", "model", 2); !errors.Is(err, microagent.ErrSpawnBudget) {
		t.Fatalf("fanout refusal = %v, want ErrSpawnBudget", err)
	}
	if err := admit("root", "policy", 1); !errors.Is(err, microagent.ErrSpawnBudget) {
		t.Fatalf("aggregate refusal = %v, want ErrSpawnBudget", err)
	}
	if got := budget.Descendants(); got != 3 {
		t.Fatalf("descendants = %d, want three admitted children; refusals must not consume capacity", got)
	}
}

func TestSpawnBudgetAtomicallyAdmitsOneEquivalentGoal(t *testing.T) {
	const contenders = 16
	budget := &microagent.SpawnBudget{
		RootID: "root", RootGoal: "complete the parent task",
		MaxDepth: 1, MaxChildren: contenders, MaxDescendants: contenders,
	}
	start := make(chan struct{})
	results := make(chan error, contenders)
	var workers sync.WaitGroup
	for i := 0; i < contenders; i++ {
		workers.Add(1)
		go func(i int) {
			defer workers.Done()
			<-start
			results <- budget.Admit(microagent.SpawnRequest{
				ParentID: "root", ChildID: fmt.Sprintf("child-%02d", i),
				Goal: "  SHARED decomposition goal  ", Depth: 1,
			})
		}(i)
	}
	close(start)
	workers.Wait()
	close(results)

	admitted, duplicates := 0, 0
	for err := range results {
		switch {
		case err == nil:
			admitted++
		case errors.Is(err, microagent.ErrDuplicateGoal):
			duplicates++
		default:
			t.Fatalf("concurrent admission = %v, want nil or ErrDuplicateGoal", err)
		}
	}
	if admitted != 1 || duplicates != contenders-1 {
		t.Fatalf("concurrent admissions = %d admitted, %d duplicates; want 1/%d", admitted, duplicates, contenders-1)
	}
	if got := budget.Descendants(); got != 1 {
		t.Fatalf("descendants = %d, want one atomic winner", got)
	}
}
