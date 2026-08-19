package microagent_test

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

func TestSpawnBudgetBoundsDepthFanoutAndAggregateLineage(t *testing.T) {
	budget := &microagent.SpawnBudget{MaxDepth: 2, MaxChildren: 2, MaxDescendants: 3}
	admit := func(parent, child string, depth int) error {
		return budget.Admit(microagent.SpawnRequest{ParentID: parent, ChildID: child, Depth: depth})
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
