package compute

import (
	"fmt"
	"math"
)

const defaultSCCPMaxVisits = 4096

// SCCPReceipt is the deterministic evidence for one bounded sparse conditional
// constant-propagation analysis. A budget hit returns the original graph exactly:
// no partial rewrite is allowed to escape the witnessed envelope.
type SCCPReceipt struct {
	MaxVisits   int  `json:"max_visits"`
	Visits      int  `json:"visits"`
	BudgetHit   bool `json:"budget_hit"`
	FoldedNodes int  `json:"folded_nodes"`
	DeadNodes   int  `json:"dead_nodes"`
}

// SparseConditionalConstantPropagationPass folds values and executable structured
// branches through the native graph-pass seam. MaxVisits is a hard analysis bound.
type SparseConditionalConstantPropagationPass struct {
	MaxVisits int
}

func (SparseConditionalConstantPropagationPass) Name() GraphPassName { return SCCPPassName }

// Apply satisfies GraphPass. Call ApplyWithReceipt when the bounded-analysis
// counters are part of the witness.
func (p SparseConditionalConstantPropagationPass) Apply(input Graph) (Graph, error) {
	result, _, err := p.ApplyWithReceipt(input)
	return result, err
}

// ApplyWithReceipt evaluates only values demanded by graph outputs and only the
// selected region of a proven branch. Runtime inputs are overdefined. An unknown
// branch is kept structurally intact, and a budget hit returns an exact clone of
// input rather than a partially optimized graph.
func (p SparseConditionalConstantPropagationPass) ApplyWithReceipt(input Graph) (Graph, SCCPReceipt, error) {
	receipt := SCCPReceipt{MaxVisits: p.MaxVisits}
	if p.MaxVisits <= 0 {
		return Graph{}, receipt, fmt.Errorf("compute graph sccp: MaxVisits must be positive")
	}
	if err := input.Validate(); err != nil {
		return Graph{}, receipt, fmt.Errorf("compute graph sccp: invalid input: %w", err)
	}

	analysis := newSCCPAnalysis(input, p.MaxVisits)
	for _, output := range input.Outputs {
		analysis.evaluate(output)
		if analysis.budgetHit {
			receipt.Visits = analysis.visits
			receipt.BudgetHit = true
			return cloneGraph(input), receipt, nil
		}
	}

	rewritten := cloneGraph(input)
	receipt.FoldedNodes = rewriteSCCPConstants(rewritten.Nodes, analysis.values)
	cleaned, err := (DeadCodeEliminationPass{}).Apply(rewritten)
	if err != nil {
		return Graph{}, SCCPReceipt{}, fmt.Errorf("compute graph sccp: remove newly dead nodes: %w", err)
	}
	receipt.Visits = analysis.visits
	receipt.DeadNodes = countGraphNodes(input.Nodes) - countGraphNodes(cleaned.Nodes)
	return cleaned, receipt, nil
}

type sccpLatticeKind uint8

const (
	sccpUnknown sccpLatticeKind = iota
	sccpConstant
	sccpOverdefined
)

type sccpLattice struct {
	kind  sccpLatticeKind
	value float64
}

type sccpAnalysis struct {
	nodes     map[NodeID]GraphNode
	values    map[NodeID]sccpLattice
	active    map[NodeID]bool
	maxVisits int
	visits    int
	budgetHit bool
}

func newSCCPAnalysis(graph Graph, maxVisits int) *sccpAnalysis {
	nodes := make(map[NodeID]GraphNode)
	indexGraphNodes(graph.Nodes, nodes)
	return &sccpAnalysis{
		nodes:     nodes,
		values:    make(map[NodeID]sccpLattice, len(nodes)),
		active:    make(map[NodeID]bool, len(nodes)),
		maxVisits: maxVisits,
	}
}

func indexGraphNodes(nodes []GraphNode, index map[NodeID]GraphNode) {
	for _, node := range nodes {
		index[node.ID] = node
		for _, region := range node.Regions {
			indexGraphNodes(region.Nodes, index)
		}
	}
}

func (a *sccpAnalysis) evaluate(id NodeID) sccpLattice {
	if value, ok := a.values[id]; ok {
		return value
	}
	if a.budgetHit {
		return sccpLattice{kind: sccpUnknown}
	}
	if a.visits == a.maxVisits {
		a.budgetHit = true
		return sccpLattice{kind: sccpUnknown}
	}
	if a.active[id] {
		// Validate rejects cycles. Keep this fail-closed guard so a future region
		// extension cannot turn a malformed recurrence into a constant claim.
		return sccpLattice{kind: sccpOverdefined}
	}
	node, ok := a.nodes[id]
	if !ok {
		return sccpLattice{kind: sccpOverdefined}
	}

	a.visits++
	a.active[id] = true
	value := a.evaluateNode(node)
	delete(a.active, id)
	if !a.budgetHit {
		a.values[id] = value
	}
	return value
}

func (a *sccpAnalysis) evaluateNode(node GraphNode) sccpLattice {
	switch node.Op {
	case GraphOpInput:
		return sccpLattice{kind: sccpOverdefined}
	case GraphOpConstant:
		return sccpLattice{kind: sccpConstant, value: node.Value}
	case GraphOpIdentity:
		return a.evaluate(node.Inputs[0])
	case GraphOpAdd, GraphOpMultiply, GraphOpDivide:
		left := a.evaluate(node.Inputs[0])
		if a.budgetHit {
			return sccpLattice{kind: sccpUnknown}
		}
		right := a.evaluate(node.Inputs[1])
		if left.kind == sccpOverdefined || right.kind == sccpOverdefined {
			return sccpLattice{kind: sccpOverdefined}
		}
		if left.kind != sccpConstant || right.kind != sccpConstant {
			return sccpLattice{kind: sccpUnknown}
		}
		var result float64
		switch node.Op {
		case GraphOpAdd:
			result = left.value + right.value
		case GraphOpMultiply:
			result = left.value * right.value
		case GraphOpDivide:
			if right.value == 0 {
				return sccpLattice{kind: sccpOverdefined}
			}
			result = left.value / right.value
		}
		if math.IsNaN(result) || math.IsInf(result, 0) {
			return sccpLattice{kind: sccpOverdefined}
		}
		return sccpLattice{kind: sccpConstant, value: result}
	case GraphOpSelect:
		condition := a.evaluate(node.Inputs[0])
		if a.budgetHit {
			return sccpLattice{kind: sccpUnknown}
		}
		if condition.kind == sccpConstant {
			if condition.value != 0 {
				return a.evaluate(node.Inputs[1])
			}
			return a.evaluate(node.Inputs[2])
		}
		trueVal := a.evaluate(node.Inputs[1])
		if a.budgetHit {
			return sccpLattice{kind: sccpUnknown}
		}
		falseVal := a.evaluate(node.Inputs[2])
		if trueVal.kind == sccpConstant && falseVal.kind == sccpConstant && trueVal.value == falseVal.value {
			return trueVal
		}
		return sccpLattice{kind: sccpOverdefined}
	case GraphOpIf:
		condition := a.evaluate(node.Inputs[0])
		if condition.kind != sccpConstant {
			return sccpLattice{kind: sccpOverdefined}
		}
		region := node.Regions[0]
		if condition.value == 0 {
			region = node.Regions[1]
		}
		return a.evaluate(region.Outputs[0])
	default:
		return sccpLattice{kind: sccpOverdefined}
	}
}

func rewriteSCCPConstants(nodes []GraphNode, values map[NodeID]sccpLattice) int {
	folded := 0
	for i := range nodes {
		node := &nodes[i]
		for regionIndex := range node.Regions {
			folded += rewriteSCCPConstants(node.Regions[regionIndex].Nodes, values)
		}
		value, ok := values[node.ID]
		if !ok || value.kind != sccpConstant || node.Op == GraphOpConstant {
			continue
		}
		node.Op = GraphOpConstant
		node.Inputs = []NodeID{}
		node.Value = value.value
		node.Regions = nil
		folded++
	}
	return folded
}

func countGraphNodes(nodes []GraphNode) int {
	count := len(nodes)
	for _, node := range nodes {
		for _, region := range node.Regions {
			count += countGraphNodes(region.Nodes)
		}
	}
	return count
}
