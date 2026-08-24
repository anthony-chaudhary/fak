package harnesscompose

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

var ErrInvalidTaskGraph = errors.New("harnesscompose: invalid recursive task graph")

// TaskNode is a verified decomposition receipt. It contains only the bounded
// execution contract needed by the generated workflow, never child transcripts.
type TaskNode struct {
	ID          string
	DependsOn   []string
	ActionRef   string
	EvidenceRef string
	Verified    bool
	MaxAttempts int
	DoneWhen    string
}

// WorkflowStage is the deterministic execution contract derived from one node.
type WorkflowStage struct {
	ID          string
	DependsOn   []string
	ActionRef   string
	MaxAttempts int
	DoneWhen    string
	EvidenceRef string
}

// DeriveWorkflow compiles a verified recursive decomposition graph into
// dependency-ordered stages. Cycles, missing edges, unverified nodes, and
// unbounded retries are refused before an artifact is produced.
func DeriveWorkflow(nodes []TaskNode) ([]WorkflowStage, error) {
	byID := make(map[string]TaskNode, len(nodes))
	for i, node := range nodes {
		node.ID = strings.TrimSpace(node.ID)
		if node.ID == "" || strings.TrimSpace(node.ActionRef) == "" || strings.TrimSpace(node.EvidenceRef) == "" || strings.TrimSpace(node.DoneWhen) == "" || !node.Verified {
			return nil, fmt.Errorf("%w: node[%d] is incomplete or unverified", ErrInvalidTaskGraph, i)
		}
		if node.MaxAttempts < 1 {
			return nil, fmt.Errorf("%w: node %q has no bounded retry policy", ErrInvalidTaskGraph, node.ID)
		}
		if _, exists := byID[node.ID]; exists {
			return nil, fmt.Errorf("%w: duplicate node %q", ErrInvalidTaskGraph, node.ID)
		}
		node.DependsOn = append([]string(nil), node.DependsOn...)
		sort.Strings(node.DependsOn)
		byID[node.ID] = node
	}
	indegree := make(map[string]int, len(nodes))
	children := make(map[string][]string, len(nodes))
	for id, node := range byID {
		seen := make(map[string]bool, len(node.DependsOn))
		for _, dependency := range node.DependsOn {
			if dependency == id || seen[dependency] {
				return nil, fmt.Errorf("%w: invalid dependency %q on node %q", ErrInvalidTaskGraph, dependency, id)
			}
			if _, exists := byID[dependency]; !exists {
				return nil, fmt.Errorf("%w: node %q depends on missing node %q", ErrInvalidTaskGraph, id, dependency)
			}
			seen[dependency] = true
			indegree[id]++
			children[dependency] = append(children[dependency], id)
		}
	}
	ready := make([]string, 0)
	for id := range byID {
		if indegree[id] == 0 {
			ready = append(ready, id)
		}
	}
	sort.Strings(ready)
	stages := make([]WorkflowStage, 0, len(nodes))
	for len(ready) > 0 {
		id := ready[0]
		ready = ready[1:]
		node := byID[id]
		stages = append(stages, WorkflowStage{ID: id, DependsOn: node.DependsOn, ActionRef: node.ActionRef, MaxAttempts: node.MaxAttempts, DoneWhen: node.DoneWhen, EvidenceRef: node.EvidenceRef})
		sort.Strings(children[id])
		for _, child := range children[id] {
			indegree[child]--
			if indegree[child] == 0 {
				ready = append(ready, child)
				sort.Strings(ready)
			}
		}
	}
	if len(stages) != len(nodes) {
		return nil, fmt.Errorf("%w: dependency cycle", ErrInvalidTaskGraph)
	}
	return stages, nil
}
