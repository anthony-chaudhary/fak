package agentopt

import (
	"fmt"
	"strings"
)

// Family 2: Test-time compute scaling & search.
//
// MCTS branch pruning optimizes search budgets by terminating dead-end
// exploration paths immediately upon observing deterministic failures or
// sub-threshold scores, preventing unpromising branch expansion.

// MCTSNode models a node in the Monte Carlo Tree Search graph.
type MCTSNode struct {
	ID          string      `json:"id"`
	Action      string      `json:"action"`
	Score       float64     `json:"score"`
	Visits      int         `json:"visits"`
	Children    []*MCTSNode `json:"children,omitempty"`
	Pruned      bool        `json:"pruned"`
	PruneReason string      `json:"prune_reason,omitempty"`
	Depth       int         `json:"depth"`
}

// NewMCTSNode constructs a new unpruned search node.
func NewMCTSNode(id, action string) *MCTSNode {
	return &MCTSNode{
		ID:       id,
		Action:   action,
		Children: make([]*MCTSNode, 0),
	}
}

// CanExpand reports whether the node is eligible to spawn child branches.
// Pruned nodes cannot expand children.
func (n *MCTSNode) CanExpand() bool {
	if n == nil {
		return false
	}
	return !n.Pruned
}

// AddChild registers a child node under this node, updating the child's depth.
// If this node is pruned, child addition is rejected and returns false.
func (n *MCTSNode) AddChild(child *MCTSNode) bool {
	if n == nil || n.Pruned || child == nil {
		return false
	}
	child.Depth = n.Depth + 1
	n.Children = append(n.Children, child)
	return true
}

// BestChild returns the highest-scoring unpruned child, or nil if none exist.
func (n *MCTSNode) BestChild() *MCTSNode {
	if n == nil || len(n.Children) == 0 {
		return nil
	}
	var best *MCTSNode
	for _, child := range n.Children {
		if child.Pruned {
			continue
		}
		if best == nil || child.Score > best.Score {
			best = child
		}
	}
	return best
}

// ActiveChildren returns all immediate child nodes that have not been pruned.
func (n *MCTSNode) ActiveChildren() []*MCTSNode {
	if n == nil || len(n.Children) == 0 {
		return nil
	}
	var active []*MCTSNode
	for _, child := range n.Children {
		if !child.Pruned {
			active = append(active, child)
		}
	}
	return active
}

// BranchEvalResult encapsulates the evaluation outcome of an action or branch.
type BranchEvalResult struct {
	Score             float64        `json:"score"`
	CompilationError  bool           `json:"compilation_error,omitempty"`
	SyntaxViolation   bool           `json:"syntax_violation,omitempty"`
	DeterministicFail bool           `json:"deterministic_fail,omitempty"`
	FailureDetails    string         `json:"failure_details,omitempty"`
	Metadata          map[string]any `json:"metadata,omitempty"`
}

// HasDeterministicFailure reports whether the evaluation detected a deterministic failure.
func (r BranchEvalResult) HasDeterministicFailure() bool {
	return r.CompilationError || r.SyntaxViolation || r.DeterministicFail
}

// MCTSPruner configures and applies branch pruning policies over MCTS nodes.
type MCTSPruner struct {
	MaxDepth    int     `json:"max_depth"`
	MaxChildren int     `json:"max_children"`
	CutoffScore float64 `json:"cutoff_score"`
}

// NewMCTSPruner creates a pruner with the specified budget constraints and cutoff score.
func NewMCTSPruner(maxDepth, maxChildren int, cutoffScore float64) *MCTSPruner {
	return &MCTSPruner{
		MaxDepth:    maxDepth,
		MaxChildren: maxChildren,
		CutoffScore: cutoffScore,
	}
}

// CanExpand checks whether a node may be expanded with additional children
// under the pruner's budget constraints and pruning status.
func (p *MCTSPruner) CanExpand(node *MCTSNode) bool {
	if p == nil || node == nil || node.Pruned {
		return false
	}
	if p.MaxDepth > 0 && node.Depth >= p.MaxDepth {
		return false
	}
	if p.MaxChildren > 0 && len(node.Children) >= p.MaxChildren {
		return false
	}
	return true
}

// AddChild attaches a child node to a parent node if permitted by bounds and pruning state.
// It sets the child's depth to parent.Depth + 1. If the parent is pruned or at capacity,
// the operation returns false without modifying the parent.
func (p *MCTSPruner) AddChild(parent, child *MCTSNode) bool {
	if p == nil || parent == nil || child == nil {
		return false
	}
	if !p.CanExpand(parent) {
		return false
	}
	child.Depth = parent.Depth + 1
	parent.Children = append(parent.Children, child)
	return true
}

// EvaluateAndPrune applies deterministic failure checks, score cutoffs, and depth
// bounds to a node. It marks failing or sub-cutoff nodes as pruned and returns
// true if pruned, or false if the node remains viable.
func (p *MCTSPruner) EvaluateAndPrune(node *MCTSNode, evalResult BranchEvalResult) bool {
	if node == nil || p == nil {
		return false
	}

	node.Score = evalResult.Score
	node.Visits++

	// Check deterministic failures first:
	if evalResult.CompilationError {
		node.Pruned = true
		if evalResult.FailureDetails != "" {
			node.PruneReason = fmt.Sprintf("compilation error: %s", evalResult.FailureDetails)
		} else {
			node.PruneReason = "compilation error"
		}
		p.pruneDescendants(node, node.PruneReason)
		return true
	}

	if evalResult.SyntaxViolation {
		node.Pruned = true
		if evalResult.FailureDetails != "" {
			node.PruneReason = fmt.Sprintf("syntax violation: %s", evalResult.FailureDetails)
		} else {
			node.PruneReason = "syntax violation"
		}
		p.pruneDescendants(node, node.PruneReason)
		return true
	}

	if evalResult.DeterministicFail {
		node.Pruned = true
		if evalResult.FailureDetails != "" {
			node.PruneReason = fmt.Sprintf("deterministic failure: %s", evalResult.FailureDetails)
		} else {
			node.PruneReason = "deterministic failure"
		}
		p.pruneDescendants(node, node.PruneReason)
		return true
	}

	// Zero score cutoff:
	if evalResult.Score <= 0.0 {
		node.Pruned = true
		node.PruneReason = "zero score cutoff"
		p.pruneDescendants(node, node.PruneReason)
		return true
	}

	// Score cutoff threshold check:
	if p.CutoffScore > 0.0 && evalResult.Score < p.CutoffScore {
		node.Pruned = true
		node.PruneReason = fmt.Sprintf("score cutoff: %.3f below threshold %.3f", evalResult.Score, p.CutoffScore)
		p.pruneDescendants(node, node.PruneReason)
		return true
	}

	// Depth ceiling check:
	if p.MaxDepth > 0 && node.Depth > p.MaxDepth {
		node.Pruned = true
		node.PruneReason = fmt.Sprintf("depth ceiling exceeded: %d > %d", node.Depth, p.MaxDepth)
		p.pruneDescendants(node, node.PruneReason)
		return true
	}

	node.Pruned = false
	node.PruneReason = ""
	return false
}

// pruneDescendants marks any existing children of the pruned node as pruned.
func (p *MCTSPruner) pruneDescendants(node *MCTSNode, reason string) {
	if node == nil {
		return
	}
	for _, child := range node.Children {
		if child != nil {
			child.Pruned = true
			child.PruneReason = reason
			p.pruneDescendants(child, reason)
		}
	}
}

// PruneSubtree recursively marks a node and all its descendants as pruned.
func (p *MCTSPruner) PruneSubtree(node *MCTSNode, reason string) {
	if node == nil {
		return
	}
	node.Pruned = true
	node.PruneReason = reason
	p.pruneDescendants(node, reason)
}

// SearchTreeStats summarizes the size and pruning metrics of a search tree.
type SearchTreeStats struct {
	TotalNodes    int `json:"total_nodes"`
	ActiveNodes   int `json:"active_nodes"`
	PrunedNodes   int `json:"pruned_nodes"`
	MaxDepthSeen  int `json:"max_depth_seen"`
	PrunedByError int `json:"pruned_by_error"`
	PrunedByScore int `json:"pruned_by_score"`
	PrunedByDepth int `json:"pruned_by_depth"`
}

// CollectStats traverses a search tree rooted at node to compute pruning metrics.
func (p *MCTSPruner) CollectStats(root *MCTSNode) SearchTreeStats {
	stats := SearchTreeStats{}
	if root == nil {
		return stats
	}
	var walk func(n *MCTSNode)
	walk = func(n *MCTSNode) {
		if n == nil {
			return
		}
		stats.TotalNodes++
		if n.Depth > stats.MaxDepthSeen {
			stats.MaxDepthSeen = n.Depth
		}
		if n.Pruned {
			stats.PrunedNodes++
			if strings.Contains(n.PruneReason, "compilation") ||
				strings.Contains(n.PruneReason, "syntax") ||
				strings.Contains(n.PruneReason, "deterministic") {
				stats.PrunedByError++
			} else if strings.Contains(n.PruneReason, "score") {
				stats.PrunedByScore++
			} else if strings.Contains(n.PruneReason, "depth") {
				stats.PrunedByDepth++
			}
		} else {
			stats.ActiveNodes++
		}
		for _, ch := range n.Children {
			walk(ch)
		}
	}
	walk(root)
	return stats
}

// BestPath extracts the path of unpruned nodes with the highest score from root.
func (p *MCTSPruner) BestPath(root *MCTSNode) []*MCTSNode {
	if root == nil || root.Pruned {
		return nil
	}
	path := []*MCTSNode{root}
	curr := root
	for len(curr.Children) > 0 {
		best := curr.BestChild()
		if best == nil || best.Pruned {
			break
		}
		path = append(path, best)
		curr = best
	}
	return path
}
