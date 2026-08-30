package compute

import (
	"fmt"
	"reflect"
	"sort"
	"testing"
)

func TestGraphSROASplitsNestedAggregatePreservesOracleAndProvenance(t *testing.T) {
	scalar := GraphValueType{Kind: GraphValueScalar}
	pair := GraphValueType{Kind: GraphValueStruct, Fields: []GraphValueType{scalar, scalar}}
	window := GraphValueType{Kind: GraphValueArray, Length: 2, Element: &pair}
	frame := GraphValueType{Kind: GraphValueStruct, Fields: []GraphValueType{scalar, window}}
	input := AggregateGraph{
		Nodes: []AggregateGraphNode{
			{ID: "a", Op: GraphOpInput, Type: scalar},
			{ID: "b", Op: GraphOpInput, Type: scalar},
			{ID: "c", Op: GraphOpInput, Type: scalar},
			{ID: "d", Op: GraphOpInput, Type: scalar},
			{ID: "e", Op: GraphOpInput, Type: scalar},
			{ID: "tmp", Op: GraphOpAggregate, Inputs: []NodeID{"a", "b", "c", "d", "e"}, Type: frame},
			{ID: "window", Op: GraphOpProject, Inputs: []NodeID{"tmp"}, Path: []int{1}, Type: window},
			{ID: "left", Op: GraphOpProject, Inputs: []NodeID{"window"}, Path: []int{0}, Type: pair},
			{ID: "left-first", Op: GraphOpProject, Inputs: []NodeID{"left"}, Path: []int{0}, Type: scalar},
			{ID: "left-second", Op: GraphOpProject, Inputs: []NodeID{"left"}, Path: []int{1}, Type: scalar},
			{ID: "right", Op: GraphOpProject, Inputs: []NodeID{"window"}, Path: []int{1}, Type: pair},
			{ID: "right-first", Op: GraphOpProject, Inputs: []NodeID{"right"}, Path: []int{0}, Type: scalar},
			{ID: "sum", Op: GraphOpAdd, Inputs: []NodeID{"left-first", "left-second"}, Type: scalar},
			{ID: "product", Op: GraphOpMultiply, Inputs: []NodeID{"sum", "right-first"}, Type: scalar},
		},
		Outputs: []NodeID{"product", "right"},
		Provenance: []GraphProvenanceExpression{{
			Value:      "tmp",
			Variable:   "frame",
			Expression: "frame.tmp",
		}},
	}
	original := cloneAggregateGraph(input)

	optimized, receipt, err := (GraphSROAPass{MaxArrayElements: 4}).Apply(input)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	wantReceipt := GraphSROAReceipt{
		MaxArrayElements:      4,
		ReplacedAggregates:    1,
		ScalarSlots:           5,
		RebuiltAggregates:     1,
		ProvenanceExpressions: 5,
	}
	if !reflect.DeepEqual(receipt, wantReceipt) {
		t.Fatalf("receipt = %#v, want %#v", receipt, wantReceipt)
	}
	if !reflect.DeepEqual(input, original) {
		t.Fatal("SROA mutated its input graph")
	}

	for _, removed := range []NodeID{"tmp", "window", "left"} {
		if aggregateGraphHasNode(optimized, removed) {
			t.Fatalf("projection-only aggregate %q was rebuilt instead of removed", removed)
		}
	}
	right := aggregateGraphNodeByID(t, optimized, "right")
	if right.Op != GraphOpAggregateRebuild || !reflect.DeepEqual(right.Inputs, []NodeID{"tmp$sroa$1_1_0", "tmp$sroa$1_1_1"}) {
		t.Fatalf("residual aggregate right = %#v, want one rebuild from its two scalar slots", right)
	}
	if got := countAggregateGraphOp(optimized, GraphOpAggregateRebuild); got != 1 {
		t.Fatalf("aggregate rebuilds = %d, want exactly 1 residual rebuild", got)
	}
	for _, id := range []NodeID{"left-first", "left-second", "right-first"} {
		node := aggregateGraphNodeByID(t, optimized, id)
		if node.Op != GraphOpIdentity || len(node.Inputs) != 1 {
			t.Fatalf("scalar projection %q was not exposed as one scalar identity: %#v", id, node)
		}
	}

	wantPaths := []string{"0", "1.0.0", "1.0.1", "1.1.0", "1.1.1"}
	gotPaths := make([]string, 0, len(optimized.Provenance))
	seenValues := make(map[NodeID]bool)
	for _, expression := range optimized.Provenance {
		if expression.Variable != "frame" || expression.Expression != "frame.tmp" {
			t.Fatalf("provenance source was not retained: %#v", expression)
		}
		if seenValues[expression.Value] {
			t.Fatalf("scalar slot %q has duplicate provenance expressions", expression.Value)
		}
		seenValues[expression.Value] = true
		gotPaths = append(gotPaths, formatGraphPath(expression.Path))
	}
	sort.Strings(gotPaths)
	if !reflect.DeepEqual(gotPaths, wantPaths) {
		t.Fatalf("retained provenance paths = %v, want every original leaf exactly once as %v", gotPaths, wantPaths)
	}

	inputs := map[NodeID]float64{"a": 2, "b": 3, "c": 5, "d": 7, "e": 11}
	before, err := evaluateAggregateGraph(input, inputs)
	if err != nil {
		t.Fatalf("evaluate original: %v", err)
	}
	after, err := evaluateAggregateGraph(optimized, inputs)
	if err != nil {
		t.Fatalf("evaluate optimized: %v", err)
	}
	want := nestedAggregateCPUOracle(2, 3, 5, 7, 11)
	if !reflect.DeepEqual(before, want) || !reflect.DeepEqual(after, want) {
		t.Fatalf("outputs:\noriginal  = %#v\noptimized = %#v\nCPU oracle = %#v", before, after, want)
	}

	again, againReceipt, err := (GraphSROAPass{MaxArrayElements: 4}).Apply(input)
	if err != nil {
		t.Fatalf("second Apply: %v", err)
	}
	if !reflect.DeepEqual(again, optimized) || !reflect.DeepEqual(againReceipt, receipt) {
		t.Fatalf("SROA is nondeterministic:\nfirst:  %#v %#v\nsecond: %#v %#v", optimized, receipt, again, againReceipt)
	}
}

func TestGraphSROAKeepsDynamicAndOversizedArraysUnsplit(t *testing.T) {
	scalar := GraphValueType{Kind: GraphValueScalar}
	array2 := GraphValueType{Kind: GraphValueArray, Length: 2, Element: &scalar}
	array5 := GraphValueType{Kind: GraphValueArray, Length: 5, Element: &scalar}
	tests := []struct {
		name   string
		graph  AggregateGraph
		want   GraphSROAReceipt
		inputs map[NodeID]float64
		oracle []aggregateTestValue
	}{
		{
			name: "dynamic index",
			graph: AggregateGraph{
				Nodes: []AggregateGraphNode{
					{ID: "x", Op: GraphOpInput, Type: scalar},
					{ID: "y", Op: GraphOpInput, Type: scalar},
					{ID: "index", Op: GraphOpInput, Type: scalar},
					{ID: "array", Op: GraphOpAggregate, Inputs: []NodeID{"x", "y"}, Type: array2},
					{ID: "array-alias", Op: GraphOpIdentity, Inputs: []NodeID{"array"}, Type: array2},
					{ID: "pick", Op: GraphOpDynamicProject, Inputs: []NodeID{"array-alias", "index"}, Type: scalar},
				},
				Outputs:    []NodeID{"pick"},
				Provenance: []GraphProvenanceExpression{{Value: "array", Variable: "array", Expression: "array.tmp"}},
			},
			want:   GraphSROAReceipt{MaxArrayElements: 4, UnsplitDynamicIndices: 1},
			inputs: map[NodeID]float64{"x": 13, "y": 17, "index": 1},
			oracle: []aggregateTestValue{scalarAggregateTestValue(17)},
		},
		{
			name: "oversized array",
			graph: AggregateGraph{
				Nodes: []AggregateGraphNode{
					{ID: "a", Op: GraphOpInput, Type: scalar},
					{ID: "b", Op: GraphOpInput, Type: scalar},
					{ID: "c", Op: GraphOpInput, Type: scalar},
					{ID: "d", Op: GraphOpInput, Type: scalar},
					{ID: "e", Op: GraphOpInput, Type: scalar},
					{ID: "array", Op: GraphOpAggregate, Inputs: []NodeID{"a", "b", "c", "d", "e"}, Type: array5},
					{ID: "pick", Op: GraphOpProject, Inputs: []NodeID{"array"}, Path: []int{3}, Type: scalar},
				},
				Outputs: []NodeID{"pick"},
			},
			want:   GraphSROAReceipt{MaxArrayElements: 4, UnsplitOversizedArrays: 1},
			inputs: map[NodeID]float64{"a": 2, "b": 3, "c": 5, "d": 7, "e": 11},
			oracle: []aggregateTestValue{scalarAggregateTestValue(7)},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, receipt, err := (GraphSROAPass{MaxArrayElements: 4}).Apply(tc.graph)
			if err != nil {
				t.Fatalf("Apply: %v", err)
			}
			if !reflect.DeepEqual(got, tc.graph) {
				t.Fatalf("ineligible control changed:\ngot:  %#v\nwant: %#v", got, tc.graph)
			}
			if !reflect.DeepEqual(receipt, tc.want) {
				t.Fatalf("receipt = %#v, want %#v", receipt, tc.want)
			}
			outputs, err := evaluateAggregateGraph(got, tc.inputs)
			if err != nil {
				t.Fatalf("evaluate: %v", err)
			}
			if !reflect.DeepEqual(outputs, tc.oracle) {
				t.Fatalf("outputs = %#v, CPU oracle = %#v", outputs, tc.oracle)
			}
		})
	}
}

func aggregateGraphNodeByID(t *testing.T, graph AggregateGraph, id NodeID) AggregateGraphNode {
	t.Helper()
	for _, node := range graph.Nodes {
		if node.ID == id {
			return node
		}
	}
	t.Fatalf("node %q is missing", id)
	return AggregateGraphNode{}
}

func aggregateGraphHasNode(graph AggregateGraph, id NodeID) bool {
	for _, node := range graph.Nodes {
		if node.ID == id {
			return true
		}
	}
	return false
}

func countAggregateGraphOp(graph AggregateGraph, op GraphOp) int {
	count := 0
	for _, node := range graph.Nodes {
		if node.Op == op {
			count++
		}
	}
	return count
}

func formatGraphPath(path []int) string {
	if len(path) == 0 {
		return ""
	}
	result := fmt.Sprint(path[0])
	for _, index := range path[1:] {
		result += fmt.Sprintf(".%d", index)
	}
	return result
}

type aggregateTestValue struct {
	scalar *float64
	fields []aggregateTestValue
}

func scalarAggregateTestValue(value float64) aggregateTestValue {
	return aggregateTestValue{scalar: &value}
}

func nestedAggregateCPUOracle(a, b, c, d, e float64) []aggregateTestValue {
	_ = a
	return []aggregateTestValue{
		scalarAggregateTestValue((b + c) * d),
		{fields: []aggregateTestValue{scalarAggregateTestValue(d), scalarAggregateTestValue(e)}},
	}
}

// evaluateAggregateGraph is an IR interpreter, intentionally separate from the direct CPU
// oracle above so SROA cannot validate itself with its own projection or rebuild logic.
func evaluateAggregateGraph(graph AggregateGraph, inputs map[NodeID]float64) ([]aggregateTestValue, error) {
	nodes := make(map[NodeID]AggregateGraphNode, len(graph.Nodes))
	for _, node := range graph.Nodes {
		nodes[node.ID] = node
	}
	values := make(map[NodeID]aggregateTestValue, len(nodes))
	active := make(map[NodeID]bool, len(nodes))

	var evaluate func(NodeID) (aggregateTestValue, error)
	evaluate = func(id NodeID) (aggregateTestValue, error) {
		if value, ok := values[id]; ok {
			return value, nil
		}
		if active[id] {
			return aggregateTestValue{}, fmt.Errorf("cycle at %q", id)
		}
		node, ok := nodes[id]
		if !ok {
			return aggregateTestValue{}, fmt.Errorf("missing node %q", id)
		}
		active[id] = true
		defer delete(active, id)

		var value aggregateTestValue
		switch node.Op {
		case GraphOpInput:
			input, ok := inputs[id]
			if !ok {
				return aggregateTestValue{}, fmt.Errorf("missing input %q", id)
			}
			value = scalarAggregateTestValue(input)
		case GraphOpConstant:
			value = scalarAggregateTestValue(node.Value)
		case GraphOpIdentity, GraphOpSROASlot:
			var err error
			value, err = evaluate(node.Inputs[0])
			if err != nil {
				return aggregateTestValue{}, err
			}
		case GraphOpAdd, GraphOpMultiply:
			left, err := evaluate(node.Inputs[0])
			if err != nil {
				return aggregateTestValue{}, err
			}
			right, err := evaluate(node.Inputs[1])
			if err != nil {
				return aggregateTestValue{}, err
			}
			if left.scalar == nil || right.scalar == nil {
				return aggregateTestValue{}, fmt.Errorf("%s requires scalar inputs", node.Op)
			}
			result := *left.scalar + *right.scalar
			if node.Op == GraphOpMultiply {
				result = *left.scalar * *right.scalar
			}
			value = scalarAggregateTestValue(result)
		case GraphOpAggregate, GraphOpAggregateRebuild:
			leaves := make([]aggregateTestValue, len(node.Inputs))
			for i, input := range node.Inputs {
				leaf, err := evaluate(input)
				if err != nil {
					return aggregateTestValue{}, err
				}
				leaves[i] = leaf
			}
			var consumed int
			var err error
			value, consumed, err = buildAggregateTestValue(node.Type, leaves, 0)
			if err != nil {
				return aggregateTestValue{}, err
			}
			if consumed != len(leaves) {
				return aggregateTestValue{}, fmt.Errorf("aggregate %q consumed %d of %d leaves", id, consumed, len(leaves))
			}
		case GraphOpProject:
			root, err := evaluate(node.Inputs[0])
			if err != nil {
				return aggregateTestValue{}, err
			}
			value, err = projectAggregateTestValue(root, node.Path)
			if err != nil {
				return aggregateTestValue{}, err
			}
		case GraphOpDynamicProject:
			root, err := evaluate(node.Inputs[0])
			if err != nil {
				return aggregateTestValue{}, err
			}
			index, err := evaluate(node.Inputs[1])
			if err != nil {
				return aggregateTestValue{}, err
			}
			if index.scalar == nil {
				return aggregateTestValue{}, fmt.Errorf("dynamic index is aggregate")
			}
			value, err = projectAggregateTestValue(root, []int{int(*index.scalar)})
			if err != nil {
				return aggregateTestValue{}, err
			}
		default:
			return aggregateTestValue{}, fmt.Errorf("unsupported op %q", node.Op)
		}
		values[id] = value
		return value, nil
	}

	outputs := make([]aggregateTestValue, len(graph.Outputs))
	for i, output := range graph.Outputs {
		value, err := evaluate(output)
		if err != nil {
			return nil, err
		}
		outputs[i] = value
	}
	return outputs, nil
}

func buildAggregateTestValue(typ GraphValueType, leaves []aggregateTestValue, offset int) (aggregateTestValue, int, error) {
	if typ.Kind == GraphValueScalar {
		if offset >= len(leaves) || leaves[offset].scalar == nil {
			return aggregateTestValue{}, offset, fmt.Errorf("missing scalar leaf at %d", offset)
		}
		return leaves[offset], offset + 1, nil
	}
	children := typ.Fields
	if typ.Kind == GraphValueArray {
		children = make([]GraphValueType, typ.Length)
		for i := range children {
			children[i] = *typ.Element
		}
	}
	value := aggregateTestValue{fields: make([]aggregateTestValue, len(children))}
	next := offset
	for i, childType := range children {
		child, consumed, err := buildAggregateTestValue(childType, leaves, next)
		if err != nil {
			return aggregateTestValue{}, next, err
		}
		value.fields[i] = child
		next = consumed
	}
	return value, next, nil
}

func projectAggregateTestValue(value aggregateTestValue, path []int) (aggregateTestValue, error) {
	for _, index := range path {
		if index < 0 || index >= len(value.fields) {
			return aggregateTestValue{}, fmt.Errorf("projection index %d out of range", index)
		}
		value = value.fields[index]
	}
	return value, nil
}
