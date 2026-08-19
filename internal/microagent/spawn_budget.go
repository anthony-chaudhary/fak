package microagent

import (
	"errors"
	"fmt"
	"sync"
)

// SpawnBudget bounds one host-mediated recursive task tree.
type SpawnBudget struct {
	MaxDepth       int
	MaxChildren    int
	MaxDescendants int

	mu          sync.Mutex
	descendants int
	children    map[string]int
}

// SpawnRequest carries lineage metadata the host, not the child, adjudicates.
type SpawnRequest struct {
	ParentID string
	ChildID  string
	Depth    int
}

var ErrSpawnBudget = errors.New("microagent: recursive spawn budget refused")

// Admit reserves one child slot. A refusal never consumes aggregate capacity.
func (b *SpawnBudget) Admit(request SpawnRequest) error {
	if b == nil {
		return fmt.Errorf("%w: missing host budget", ErrSpawnBudget)
	}
	if request.ParentID == "" || request.ChildID == "" || request.Depth < 1 {
		return fmt.Errorf("%w: incomplete lineage", ErrSpawnBudget)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.MaxDepth > 0 && request.Depth > b.MaxDepth {
		return fmt.Errorf("%w: depth %d exceeds %d", ErrSpawnBudget, request.Depth, b.MaxDepth)
	}
	if b.MaxChildren > 0 && b.children[request.ParentID] >= b.MaxChildren {
		return fmt.Errorf("%w: parent %q exhausted child fanout %d", ErrSpawnBudget, request.ParentID, b.MaxChildren)
	}
	if b.MaxDescendants > 0 && b.descendants >= b.MaxDescendants {
		return fmt.Errorf("%w: lineage exhausted aggregate descendants %d", ErrSpawnBudget, b.MaxDescendants)
	}
	if b.children == nil {
		b.children = make(map[string]int)
	}
	b.children[request.ParentID]++
	b.descendants++
	return nil
}

// Descendants reports host-admitted children across the entire lineage.
func (b *SpawnBudget) Descendants() int {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.descendants
}
