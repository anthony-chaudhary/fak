package compute

import (
	"fmt"
	"reflect"
	"testing"
)

func TestCanonicalGraphPipelineFoldsConstantChain(t *testing.T) {
	input := Graph{
		Nodes: []GraphNode{
			{ID: "two", Op: GraphOpConstant, Inputs: []NodeID{}, Value: 2},
			{ID: "three", Op: GraphOpConstant, Inputs: []NodeID{}, Value: 3},
			{ID: "sum", Op: GraphOpAdd, Inputs: []NodeID{"two", "three"}},
		},
		Outputs: []NodeID{"sum"},
	}

	got, _, err := CanonicalGraphPipeline().Run(input)
	if err != nil {
		t.Fatalf("pipeline run: %v", err)
	}
	if len(got.Nodes) != 1 || got.Nodes[0].ID != "sum" || got.Nodes[0].Op != GraphOpConstant || got.Nodes[0].Value != 5 {
		t.Fatalf("constant chain was not folded and cleaned: %#v", got)
	}
}

func TestSCCPFoldsConstantRegionAndMatchesIndependentOracle(t *testing.T) {
	input := sccpRegionFixture()
	pass := SparseConditionalConstantPropagationPass{MaxVisits: 64}

	optimized, receipt, err := pass.ApplyWithReceipt(input)
	if err != nil {
		t.Fatalf("ApplyWithReceipt: %v", err)
	}
	wantReceipt := SCCPReceipt{
		MaxVisits:   64,
		Visits:      9,
		FoldedNodes: 3,
		DeadNodes:   8,
	}
	if !reflect.DeepEqual(receipt, wantReceipt) {
		t.Fatalf("receipt = %#v, want %#v", receipt, wantReceipt)
	}

	choice := graphNodeByID(t, optimized.Nodes, "choice")
	if choice.Op != GraphOpConstant || choice.Value != 20 || len(choice.Inputs) != 0 || len(choice.Regions) != 0 {
		t.Fatalf("constant branch was not folded to choice=20: %#v", choice)
	}
	unknownBefore := graphNodeByID(t, input.Nodes, "runtime-choice")
	unknownAfter := graphNodeByID(t, optimized.Nodes, "runtime-choice")
	if !reflect.DeepEqual(unknownAfter, unknownBefore) {
		t.Fatalf("unknown branch changed:\nafter:  %#v\nbefore: %#v", unknownAfter, unknownBefore)
	}

	for _, runtimeCondition := range []float64{0, 1} {
		inputs := map[NodeID]float64{"runtime-condition": runtimeCondition}
		want, err := evaluateStructuredGraphOracle(input, inputs)
		if err != nil {
			t.Fatalf("oracle input condition %v: %v", runtimeCondition, err)
		}
		got, err := evaluateStructuredGraphOracle(optimized, inputs)
		if err != nil {
			t.Fatalf("oracle optimized condition %v: %v", runtimeCondition, err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("condition %v outputs = %v, independent eager oracle = %v", runtimeCondition, got, want)
		}
	}

	again, againReceipt, err := pass.ApplyWithReceipt(input)
	if err != nil {
		t.Fatalf("second ApplyWithReceipt: %v", err)
	}
	if !reflect.DeepEqual(again, optimized) || !reflect.DeepEqual(againReceipt, receipt) {
		t.Fatalf("SCCP is nondeterministic:\nfirst:  %#v %#v\nsecond: %#v %#v", optimized, receipt, again, againReceipt)
	}
}

func TestSCCPBudgetHitReturnsExactOriginalGraph(t *testing.T) {
	input := longSCCPRecurrence(64)
	pass := SparseConditionalConstantPropagationPass{MaxVisits: 11}

	got, receipt, err := pass.ApplyWithReceipt(input)
	if err != nil {
		t.Fatalf("ApplyWithReceipt: %v", err)
	}
	wantReceipt := SCCPReceipt{MaxVisits: 11, Visits: 11, BudgetHit: true}
	if !reflect.DeepEqual(receipt, wantReceipt) {
		t.Fatalf("receipt = %#v, want %#v", receipt, wantReceipt)
	}
	if !reflect.DeepEqual(got, input) {
		t.Fatal("budget fallback returned a partial rewrite instead of the exact original graph")
	}

	again, againReceipt, err := pass.ApplyWithReceipt(input)
	if err != nil {
		t.Fatalf("second ApplyWithReceipt: %v", err)
	}
	if !reflect.DeepEqual(again, got) || !reflect.DeepEqual(againReceipt, receipt) {
		t.Fatalf("budget fallback is nondeterministic:\nfirst:  %#v %#v\nsecond: %#v %#v", got, receipt, again, againReceipt)
	}
}

func sccpRegionFixture() Graph {
	return Graph{
		Nodes: []GraphNode{
			{ID: "condition", Op: GraphOpConstant, Inputs: []NodeID{}, Value: 1},
			{ID: "base", Op: GraphOpConstant, Inputs: []NodeID{}, Value: 2},
			{
				ID:     "choice",
				Op:     GraphOpIf,
				Inputs: []NodeID{"condition", "base"},
				Regions: []GraphRegion{
					{
						Nodes: []GraphNode{
							{ID: "then-three", Op: GraphOpConstant, Inputs: []NodeID{}, Value: 3},
							{ID: "then-sum", Op: GraphOpAdd, Inputs: []NodeID{"base", "then-three"}},
							{ID: "then-four", Op: GraphOpConstant, Inputs: []NodeID{}, Value: 4},
							{ID: "then-product", Op: GraphOpMultiply, Inputs: []NodeID{"then-sum", "then-four"}},
						},
						Outputs: []NodeID{"then-product"},
					},
					{
						Nodes: []GraphNode{
							{ID: "else-hundred", Op: GraphOpConstant, Inputs: []NodeID{}, Value: 100},
							{ID: "else-sum", Op: GraphOpAdd, Inputs: []NodeID{"base", "else-hundred"}},
						},
						Outputs: []NodeID{"else-sum"},
					},
				},
			},
			{ID: "runtime-condition", Op: GraphOpInput, Inputs: []NodeID{}},
			{
				ID:     "runtime-choice",
				Op:     GraphOpIf,
				Inputs: []NodeID{"runtime-condition", "choice"},
				Regions: []GraphRegion{
					{
						Nodes: []GraphNode{
							{ID: "runtime-one", Op: GraphOpConstant, Inputs: []NodeID{}, Value: 1},
							{ID: "runtime-add", Op: GraphOpAdd, Inputs: []NodeID{"choice", "runtime-one"}},
						},
						Outputs: []NodeID{"runtime-add"},
					},
					{
						Nodes: []GraphNode{
							{ID: "runtime-two", Op: GraphOpConstant, Inputs: []NodeID{}, Value: 2},
							{ID: "runtime-multiply", Op: GraphOpMultiply, Inputs: []NodeID{"choice", "runtime-two"}},
						},
						Outputs: []NodeID{"runtime-multiply"},
					},
				},
			},
		},
		Outputs: []NodeID{"choice", "runtime-choice"},
	}
}

func longSCCPRecurrence(length int) Graph {
	nodes := []GraphNode{
		{ID: "seed", Op: GraphOpConstant, Inputs: []NodeID{}, Value: 0},
		{ID: "one", Op: GraphOpConstant, Inputs: []NodeID{}, Value: 1},
	}
	previous := NodeID("seed")
	for i := 0; i < length; i++ {
		id := NodeID(fmt.Sprintf("step-%03d", i))
		nodes = append(nodes, GraphNode{ID: id, Op: GraphOpAdd, Inputs: []NodeID{previous, "one"}})
		previous = id
	}
	return Graph{Nodes: nodes, Outputs: []NodeID{previous}}
}

func graphNodeByID(t *testing.T, nodes []GraphNode, id NodeID) GraphNode {
	t.Helper()
	for _, node := range nodes {
		if node.ID == id {
			return node
		}
	}
	t.Fatalf("node %q is missing", id)
	return GraphNode{}
}

// evaluateStructuredGraphOracle is intentionally separate from SCCP: it executes both
// original and rewritten graphs using direct CPU arithmetic and runtime branch selection.
func evaluateStructuredGraphOracle(graph Graph, inputs map[NodeID]float64) ([]float64, error) {
	nodes := make(map[NodeID]GraphNode)
	indexOracleNodes(graph.Nodes, nodes)
	values := make(map[NodeID]float64, len(nodes))
	active := make(map[NodeID]bool, len(nodes))

	var evaluate func(NodeID) (float64, error)
	evaluate = func(id NodeID) (float64, error) {
		if value, ok := values[id]; ok {
			return value, nil
		}
		if active[id] {
			return 0, fmt.Errorf("oracle cycle at %q", id)
		}
		node, ok := nodes[id]
		if !ok {
			return 0, fmt.Errorf("oracle missing node %q", id)
		}
		active[id] = true
		defer delete(active, id)

		var value float64
		switch node.Op {
		case GraphOpInput:
			var exists bool
			value, exists = inputs[id]
			if !exists {
				return 0, fmt.Errorf("oracle missing input %q", id)
			}
		case GraphOpConstant:
			value = node.Value
		case GraphOpIdentity:
			var err error
			value, err = evaluate(node.Inputs[0])
			if err != nil {
				return 0, err
			}
		case GraphOpAdd, GraphOpMultiply:
			left, err := evaluate(node.Inputs[0])
			if err != nil {
				return 0, err
			}
			right, err := evaluate(node.Inputs[1])
			if err != nil {
				return 0, err
			}
			value = left + right
			if node.Op == GraphOpMultiply {
				value = left * right
			}
		case GraphOpIf:
			condition, err := evaluate(node.Inputs[0])
			if err != nil {
				return 0, err
			}
			region := node.Regions[0]
			if condition == 0 {
				region = node.Regions[1]
			}
			value, err = evaluate(region.Outputs[0])
			if err != nil {
				return 0, err
			}
		default:
			return 0, fmt.Errorf("oracle unsupported op %q", node.Op)
		}
		values[id] = value
		return value, nil
	}

	outputs := make([]float64, len(graph.Outputs))
	for i, output := range graph.Outputs {
		value, err := evaluate(output)
		if err != nil {
			return nil, err
		}
		outputs[i] = value
	}
	return outputs, nil
}

func indexOracleNodes(nodes []GraphNode, index map[NodeID]GraphNode) {
	for _, node := range nodes {
		index[node.ID] = node
		for _, region := range node.Regions {
			indexOracleNodes(region.Nodes, index)
		}
	}
}
