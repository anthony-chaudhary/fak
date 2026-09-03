package compute

import (
	"fmt"
	"reflect"
	"testing"
)

// evaluateBranchWitnessGraph evaluates a graph on inputs using an independent eager reference oracle.
func evaluateBranchWitnessGraph(graph Graph, inputs map[NodeID]float64) ([]float64, error) {
	nodes := make(map[NodeID]GraphNode)
	indexOracleNodes(graph.Nodes, nodes)
	values := make(map[NodeID]float64)
	active := make(map[NodeID]bool)

	var evaluate func(NodeID) (float64, error)
	evaluate = func(id NodeID) (float64, error) {
		if val, exists := values[id]; exists {
			return val, nil
		}
		if active[id] {
			return 0, fmt.Errorf("cycle detected on %q", id)
		}
		node, ok := nodes[id]
		if !ok {
			return 0, fmt.Errorf("oracle missing node %q", id)
		}
		active[id] = true
		defer delete(active, id)

		var val float64
		switch node.Op {
		case GraphOpInput:
			var exists bool
			val, exists = inputs[id]
			if !exists {
				return 0, fmt.Errorf("oracle missing input %q", id)
			}
		case GraphOpConstant:
			val = node.Value
		case GraphOpIdentity:
			var err error
			val, err = evaluate(node.Inputs[0])
			if err != nil {
				return 0, err
			}
		case GraphOpAdd:
			l, err := evaluate(node.Inputs[0])
			if err != nil {
				return 0, err
			}
			r, err := evaluate(node.Inputs[1])
			if err != nil {
				return 0, err
			}
			val = l + r
		case GraphOpMultiply:
			l, err := evaluate(node.Inputs[0])
			if err != nil {
				return 0, err
			}
			r, err := evaluate(node.Inputs[1])
			if err != nil {
				return 0, err
			}
			val = l * r
		case GraphOpDivide:
			l, err := evaluate(node.Inputs[0])
			if err != nil {
				return 0, err
			}
			r, err := evaluate(node.Inputs[1])
			if err != nil {
				return 0, err
			}
			if r == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			val = l / r
		case GraphOpSelect:
			c, err := evaluate(node.Inputs[0])
			if err != nil {
				return 0, err
			}
			if c != 0 {
				return evaluate(node.Inputs[1])
			}
			return evaluate(node.Inputs[2])
		case GraphOpIf:
			c, err := evaluate(node.Inputs[0])
			if err != nil {
				return 0, err
			}
			region := node.Regions[0]
			if c == 0 {
				region = node.Regions[1]
			}
			subNodes := make(map[NodeID]GraphNode)
			indexOracleNodes(region.Nodes, subNodes)
			for k, v := range nodes {
				if _, exists := subNodes[k]; !exists {
					subNodes[k] = v
				}
			}
			// evaluate region output
			var evaluateSub func(NodeID) (float64, error)
			evaluateSub = func(subID NodeID) (float64, error) {
				if v, exists := values[subID]; exists {
					return v, nil
				}
				subNode, ok := subNodes[subID]
				if !ok {
					return 0, fmt.Errorf("sub-node missing %q", subID)
				}
				switch subNode.Op {
				case GraphOpConstant:
					return subNode.Value, nil
				case GraphOpInput:
					return inputs[subID], nil
				case GraphOpAdd:
					l, err := evaluateSub(subNode.Inputs[0])
					if err != nil {
						return 0, err
					}
					r, err := evaluateSub(subNode.Inputs[1])
					if err != nil {
						return 0, err
					}
					return l + r, nil
				case GraphOpMultiply:
					l, err := evaluateSub(subNode.Inputs[0])
					if err != nil {
						return 0, err
					}
					r, err := evaluateSub(subNode.Inputs[1])
					if err != nil {
						return 0, err
					}
					return l * r, nil
				case GraphOpDivide:
					l, err := evaluateSub(subNode.Inputs[0])
					if err != nil {
						return 0, err
					}
					r, err := evaluateSub(subNode.Inputs[1])
					if err != nil {
						return 0, err
					}
					if r == 0 {
						return 0, fmt.Errorf("division by zero")
					}
					return l / r, nil
				default:
					return evaluate(subID)
				}
			}
			val, err = evaluateSub(region.Outputs[0])
			if err != nil {
				return 0, err
			}
		default:
			return 0, fmt.Errorf("oracle unsupported op %q", node.Op)
		}
		values[id] = val
		return val, nil
	}

	outputs := make([]float64, len(graph.Outputs))
	for i, out := range graph.Outputs {
		var err error
		outputs[i], err = evaluate(out)
		if err != nil {
			return nil, err
		}
	}
	return outputs, nil
}

func TestBranchCanonicalizationWitness(t *testing.T) {
	// First witness:
	// 1. Canonicalize one pure if to a select.
	// 2. Keep a potentially trapping division branch structured.
	// 3. Propagate true/false facts inside both regions.
	// 4. Prove stable graph digest.
	// 5. Exact eager parity.

	graph := Graph{
		Nodes: []GraphNode{
			{ID: "cond_pure", Op: GraphOpInput},
			{ID: "cond_trap", Op: GraphOpInput},
			{ID: "x", Op: GraphOpInput},
			{ID: "y", Op: GraphOpInput},
			{ID: "z", Op: GraphOpInput},
			// Pure IF: can be converted to select
			{
				ID:     "pure_if",
				Op:     GraphOpIf,
				Inputs: []NodeID{"cond_pure", "x"},
				Regions: []GraphRegion{
					{
						// then: x + 10
						Nodes: []GraphNode{
							{ID: "then_ten", Op: GraphOpConstant, Value: 10},
							{ID: "then_add", Op: GraphOpAdd, Inputs: []NodeID{"x", "then_ten"}},
						},
						Outputs: []NodeID{"then_add"},
					},
					{
						// else: x * 2
						Nodes: []GraphNode{
							{ID: "else_two", Op: GraphOpConstant, Value: 2},
							{ID: "else_mul", Op: GraphOpMultiply, Inputs: []NodeID{"x", "else_two"}},
						},
						Outputs: []NodeID{"else_mul"},
					},
				},
			},
			// Trapping IF: contains division, must remain structured; uses cond_trap in both regions so facts are propagated
			{
				ID:     "trapping_if",
				Op:     GraphOpIf,
				Inputs: []NodeID{"cond_trap", "y", "z"},
				Regions: []GraphRegion{
					{
						// then: (y / z) + cond_trap
						Nodes: []GraphNode{
							{ID: "then_div", Op: GraphOpDivide, Inputs: []NodeID{"y", "z"}},
							{ID: "then_res", Op: GraphOpAdd, Inputs: []NodeID{"then_div", "cond_trap"}},
						},
						Outputs: []NodeID{"then_res"},
					},
					{
						// else: y + cond_trap
						Nodes: []GraphNode{
							{ID: "else_res", Op: GraphOpAdd, Inputs: []NodeID{"y", "cond_trap"}},
						},
						Outputs: []NodeID{"else_res"},
					},
				},
			},
			{ID: "final_sum", Op: GraphOpAdd, Inputs: []NodeID{"pure_if", "trapping_if"}},
		},
		Outputs: []NodeID{"final_sum"},
	}

	pass := BranchCanonicalizeGraphPass{}
	got, receipt, err := pass.ApplyWithReceipt(graph)
	if err != nil {
		t.Fatalf("ApplyWithReceipt failed: %v", err)
	}

	// 1. Verify converted pure if to select
	if receipt.ConvertedSelects != 1 {
		t.Fatalf("expected 1 converted select, got %d", receipt.ConvertedSelects)
	}
	var foundSelect bool
	for _, n := range got.Nodes {
		if n.ID == "pure_if" && n.Op == GraphOpSelect {
			foundSelect = true
			if len(n.Inputs) != 3 || n.Inputs[0] != "cond_pure" || n.Inputs[1] != "then_add" || n.Inputs[2] != "else_mul" {
				t.Fatalf("unexpected select inputs: %v", n.Inputs)
			}
		}
	}
	if !foundSelect {
		t.Fatalf("pure_if was not converted to GraphOpSelect")
	}

	// 2. Verify trapping division branch remained structured
	if receipt.RetainedBranches != 1 {
		t.Fatalf("expected 1 retained structured branch, got %d", receipt.RetainedBranches)
	}
	var foundStructured bool
	var trappingNode GraphNode
	for _, n := range got.Nodes {
		if n.ID == "trapping_if" && n.Op == GraphOpIf {
			foundStructured = true
			trappingNode = n
		}
	}
	if !foundStructured {
		t.Fatalf("trapping_if did not remain structured GraphOpIf")
	}

	// 3. Verify true/false facts were propagated inside regions
	if receipt.SubstitutedConditions != 2 {
		t.Fatalf("expected 2 substituted conditions, got %d", receipt.SubstitutedConditions)
	}
	// then region should have then_cond constant with value 1.0
	var thenHasConstOne bool
	for _, n := range trappingNode.Regions[0].Nodes {
		if n.Op == GraphOpConstant && n.Value == 1.0 {
			thenHasConstOne = true
		}
	}
	if !thenHasConstOne {
		t.Fatalf("trapping_if then-region missing propagated constant 1.0")
	}
	// else region should have else_cond constant with value 0.0
	var elseHasConstZero bool
	for _, n := range trappingNode.Regions[1].Nodes {
		if n.Op == GraphOpConstant && n.Value == 0.0 {
			elseHasConstZero = true
		}
	}
	if !elseHasConstZero {
		t.Fatalf("trapping_if else-region missing propagated constant 0.0")
	}

	// 4. Stable graph digest
	digest1, err := got.Digest()
	if err != nil {
		t.Fatalf("Digest failed: %v", err)
	}
	digest2, err := got.Digest()
	if err != nil {
		t.Fatalf("second Digest failed: %v", err)
	}
	if digest1 != digest2 || digest1 == "" {
		t.Fatalf("graph digest is not stable: %q vs %q", digest1, digest2)
	}
	if receipt.FinalGraphDigest != digest1 {
		t.Fatalf("receipt digest %q != computed digest %q", receipt.FinalGraphDigest, digest1)
	}

	// 5. Exact eager parity on multiple test vectors
	testCases := []struct {
		inputs map[NodeID]float64
	}{
		{inputs: map[NodeID]float64{"cond_pure": 1, "cond_trap": 1, "x": 5, "y": 20, "z": 4}},
		{inputs: map[NodeID]float64{"cond_pure": 0, "cond_trap": 1, "x": 5, "y": 20, "z": 4}},
		{inputs: map[NodeID]float64{"cond_pure": 1, "cond_trap": 0, "x": 5, "y": 20, "z": 4}},
		{inputs: map[NodeID]float64{"cond_pure": 0, "cond_trap": 0, "x": 5, "y": 20, "z": 4}},
		// When cond_trap is 0, z is 0 (division by zero would happen if evaluated speculatively, but structured IF skips it!)
		{inputs: map[NodeID]float64{"cond_pure": 1, "cond_trap": 0, "x": 7, "y": 14, "z": 0}},
	}

	for i, tc := range testCases {
		origOut, err := evaluateBranchWitnessGraph(graph, tc.inputs)
		if err != nil {
			t.Fatalf("tc %d: evaluate original graph: %v", i, err)
		}
		gotOut, err := evaluateBranchWitnessGraph(got, tc.inputs)
		if err != nil {
			t.Fatalf("tc %d: evaluate optimized graph: %v", i, err)
		}
		if !reflect.DeepEqual(origOut, gotOut) {
			t.Fatalf("tc %d: outputs disagree: got=%v, want=%v", i, gotOut, origOut)
		}
	}
}
