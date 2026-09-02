package compute

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"
)

// GraphValueKind distinguishes scalar values from fixed aggregate values in the SROA leaf IR.
type GraphValueKind string

const (
	GraphValueScalar GraphValueKind = "scalar"
	GraphValueStruct GraphValueKind = "struct"
	GraphValueArray  GraphValueKind = "array"
)

// GraphValueType describes a scalar, fixed struct, or fixed array. Struct fields are ordered;
// arrays have Length copies of Element.
type GraphValueType struct {
	Kind    GraphValueKind   `json:"kind"`
	Fields  []GraphValueType `json:"fields,omitempty"`
	Element *GraphValueType  `json:"element,omitempty"`
	Length  int              `json:"length,omitempty"`
}

const (
	GraphOpAggregate        GraphOp = "aggregate"
	GraphOpProject          GraphOp = "project"
	GraphOpDynamicProject   GraphOp = "dynamic-project"
	GraphOpSROASlot         GraphOp = "sroa-slot"
	GraphOpAggregateRebuild GraphOp = "aggregate-rebuild"
)

// AggregateGraphNode extends the scalar graph primitives with fixed aggregate construction and
// projection. Aggregate and rebuild inputs are scalar leaves in depth-first field order.
type AggregateGraphNode struct {
	ID     NodeID         `json:"id"`
	Op     GraphOp        `json:"op"`
	Inputs []NodeID       `json:"inputs"`
	Value  float64        `json:"value"`
	Type   GraphValueType `json:"type"`
	Path   []int          `json:"path,omitempty"`
}

// GraphProvenanceExpression binds one IR value to a source variable expression. Path identifies
// the original source field represented by a scalarized value.
type GraphProvenanceExpression struct {
	Value      NodeID `json:"value"`
	Variable   string `json:"variable"`
	Expression string `json:"expression"`
	Path       []int  `json:"path,omitempty"`
}

// AggregateGraph is the package-local fixed-aggregate extension of the backend-neutral graph IR.
type AggregateGraph struct {
	Nodes      []AggregateGraphNode        `json:"nodes"`
	Outputs    []NodeID                    `json:"outputs"`
	Provenance []GraphProvenanceExpression `json:"provenance,omitempty"`
}

// GraphSROAReceipt records the conservative eligibility decisions and rewrites from one pass.
type GraphSROAReceipt struct {
	MaxArrayElements       int `json:"max_array_elements"`
	ReplacedAggregates     int `json:"replaced_aggregates"`
	ScalarSlots            int `json:"scalar_slots"`
	RebuiltAggregates      int `json:"rebuilt_aggregates"`
	ProvenanceExpressions  int `json:"provenance_expressions"`
	UnsplitDynamicIndices  int `json:"unsplit_dynamic_indices"`
	UnsplitOversizedArrays int `json:"unsplit_oversized_arrays"`
}

// GraphSROAPass scalar-replaces eligible aggregate temporaries. Arrays larger than
// MaxArrayElements, including nested arrays, stay on the original aggregate path.
type GraphSROAPass struct {
	MaxArrayElements int
}

// Apply recursively exposes scalar leaves when every projection chain is static. Aggregate
// aliases are rebuilt only for a graph output or a non-projection consumer.
func (p GraphSROAPass) Apply(input AggregateGraph) (AggregateGraph, GraphSROAReceipt, error) {
	receipt := GraphSROAReceipt{MaxArrayElements: p.MaxArrayElements}
	if p.MaxArrayElements <= 0 {
		return AggregateGraph{}, receipt, fmt.Errorf("compute graph sroa: MaxArrayElements must be positive")
	}
	if err := input.Validate(); err != nil {
		return AggregateGraph{}, receipt, fmt.Errorf("compute graph sroa: invalid input: %w", err)
	}

	graph := cloneAggregateGraph(input)
	index := indexAggregateGraph(graph)
	usedIDs := make(map[NodeID]bool, len(index.nodes))
	for id := range index.nodes {
		usedIDs[id] = true
	}

	plans := make(map[NodeID]*graphSROAPlan)
	owners := make(map[NodeID]*graphSROAPlan)
	for _, node := range graph.Nodes {
		if node.Op != GraphOpAggregate {
			continue
		}
		if graphTypeHasOversizedArray(node.Type, p.MaxArrayElements) {
			receipt.UnsplitOversizedArrays++
			continue
		}
		plan := analyzeGraphSROARoot(node, index)
		if plan.dynamicIndex {
			receipt.UnsplitDynamicIndices++
			continue
		}
		leaves := graphValueLeaves(node.Type, nil)
		plan.slots = make([]graphSROASlot, len(leaves))
		for i, leaf := range leaves {
			id := freshSROASlotID(node.ID, leaf.path, usedIDs)
			plan.slots[i] = graphSROASlot{id: id, path: leaf.path, source: node.Inputs[i]}
		}
		plans[node.ID] = plan
		for alias := range plan.aliases {
			owners[alias] = plan
		}
		receipt.ReplacedAggregates++
		receipt.ScalarSlots += len(plan.slots)
	}

	if len(plans) == 0 {
		return graph, receipt, nil
	}

	result := AggregateGraph{
		Nodes:   make([]AggregateGraphNode, 0, len(graph.Nodes)),
		Outputs: cloneNodeIDs(graph.Outputs),
	}
	for _, node := range graph.Nodes {
		plan := owners[node.ID]
		if plan == nil {
			result.Nodes = append(result.Nodes, cloneAggregateGraphNode(node))
			continue
		}

		alias := plan.aliases[node.ID]
		if node.ID == plan.root {
			for _, slot := range plan.slots {
				result.Nodes = append(result.Nodes, AggregateGraphNode{
					ID:     slot.id,
					Op:     GraphOpSROASlot,
					Inputs: []NodeID{slot.source},
					Type:   scalarGraphValueType(),
				})
			}
		}
		if alias.typ.Kind == GraphValueScalar {
			slot, ok := plan.slotForPath(alias.path)
			if !ok {
				return AggregateGraph{}, GraphSROAReceipt{}, fmt.Errorf("compute graph sroa: no scalar slot for projection %q", node.ID)
			}
			node.Op = GraphOpIdentity
			node.Inputs = []NodeID{slot.id}
			node.Path = nil
			result.Nodes = append(result.Nodes, node)
			continue
		}
		if !alias.residual {
			continue
		}
		node.Op = GraphOpAggregateRebuild
		node.Inputs = plan.slotIDsUnder(alias.path)
		node.Path = nil
		result.Nodes = append(result.Nodes, node)
		receipt.RebuiltAggregates++
	}

	for _, expression := range graph.Provenance {
		plan := owners[expression.Value]
		if plan == nil {
			result.Provenance = append(result.Provenance, cloneGraphProvenanceExpression(expression))
			continue
		}
		alias := plan.aliases[expression.Value]
		if alias.typ.Kind == GraphValueScalar {
			slot, ok := plan.slotForPath(alias.path)
			if !ok {
				return AggregateGraph{}, GraphSROAReceipt{}, fmt.Errorf("compute graph sroa: no scalar slot for provenance value %q", expression.Value)
			}
			expression.Value = slot.id
			expression.Path = cloneIntPath(expression.Path)
			result.Provenance = append(result.Provenance, expression)
			receipt.ProvenanceExpressions++
			continue
		}
		for _, leaf := range graphValueLeaves(alias.typ, nil) {
			absolutePath := appendGraphPath(alias.path, leaf.path)
			slot, ok := plan.slotForPath(absolutePath)
			if !ok {
				return AggregateGraph{}, GraphSROAReceipt{}, fmt.Errorf("compute graph sroa: no scalar slot for provenance path %v", absolutePath)
			}
			result.Provenance = append(result.Provenance, GraphProvenanceExpression{
				Value:      slot.id,
				Variable:   expression.Variable,
				Expression: expression.Expression,
				Path:       appendGraphPath(expression.Path, leaf.path),
			})
			receipt.ProvenanceExpressions++
		}
	}

	if err := result.Validate(); err != nil {
		return AggregateGraph{}, GraphSROAReceipt{}, fmt.Errorf("compute graph sroa: produced invalid graph: %w", err)
	}
	return result, receipt, nil
}

// Validate rejects malformed aggregate graphs before projection analysis or rewriting.
func (g AggregateGraph) Validate() error {
	if len(g.Nodes) == 0 {
		return fmt.Errorf("aggregate graph: must contain at least one node")
	}
	if len(g.Outputs) == 0 {
		return fmt.Errorf("aggregate graph: must contain at least one output")
	}

	nodes := make(map[NodeID]AggregateGraphNode, len(g.Nodes))
	for i, node := range g.Nodes {
		if node.ID == "" {
			return fmt.Errorf("aggregate graph: node %d has empty ID", i)
		}
		if _, exists := nodes[node.ID]; exists {
			return fmt.Errorf("aggregate graph: duplicate node ID %q", node.ID)
		}
		if math.IsNaN(node.Value) || math.IsInf(node.Value, 0) {
			return fmt.Errorf("aggregate graph: node %q has non-finite value", node.ID)
		}
		if err := validateGraphValueType(node.Type, make(map[*GraphValueType]bool)); err != nil {
			return fmt.Errorf("aggregate graph: node %q type: %w", node.ID, err)
		}
		nodes[node.ID] = cloneAggregateGraphNode(node)
	}
	for _, node := range g.Nodes {
		for _, input := range node.Inputs {
			if _, ok := nodes[input]; !ok {
				return fmt.Errorf("aggregate graph: node %q references unknown input %q", node.ID, input)
			}
		}
		if err := validateAggregateGraphNode(node, nodes); err != nil {
			return err
		}
	}
	for i, output := range g.Outputs {
		if _, ok := nodes[output]; !ok {
			return fmt.Errorf("aggregate graph: output %d references unknown node %q", i, output)
		}
	}
	for i, expression := range g.Provenance {
		if _, ok := nodes[expression.Value]; !ok {
			return fmt.Errorf("aggregate graph: provenance %d references unknown node %q", i, expression.Value)
		}
		for _, index := range expression.Path {
			if index < 0 {
				return fmt.Errorf("aggregate graph: provenance %d has negative path index", i)
			}
		}
	}
	if !aggregateGraphIsAcyclic(g.Nodes) {
		return fmt.Errorf("aggregate graph: dependency cycle detected")
	}
	return nil
}

type aggregateGraphIndex struct {
	nodes     map[NodeID]AggregateGraphNode
	consumers map[NodeID][]aggregateGraphUse
	outputs   map[NodeID]bool
}

type aggregateGraphUse struct {
	consumer NodeID
	input    int
}

func indexAggregateGraph(graph AggregateGraph) aggregateGraphIndex {
	index := aggregateGraphIndex{
		nodes:     make(map[NodeID]AggregateGraphNode, len(graph.Nodes)),
		consumers: make(map[NodeID][]aggregateGraphUse, len(graph.Nodes)),
		outputs:   make(map[NodeID]bool, len(graph.Outputs)),
	}
	for _, node := range graph.Nodes {
		index.nodes[node.ID] = node
		for inputIndex, input := range node.Inputs {
			index.consumers[input] = append(index.consumers[input], aggregateGraphUse{consumer: node.ID, input: inputIndex})
		}
	}
	for _, output := range graph.Outputs {
		index.outputs[output] = true
	}
	return index
}

type graphSROAAlias struct {
	path     []int
	typ      GraphValueType
	residual bool
}

type graphSROASlot struct {
	id     NodeID
	path   []int
	source NodeID
}

type graphSROAPlan struct {
	root         NodeID
	aliases      map[NodeID]graphSROAAlias
	slots        []graphSROASlot
	dynamicIndex bool
}

func analyzeGraphSROARoot(root AggregateGraphNode, index aggregateGraphIndex) *graphSROAPlan {
	plan := &graphSROAPlan{
		root:    root.ID,
		aliases: map[NodeID]graphSROAAlias{root.ID: {typ: cloneGraphValueType(root.Type)}},
	}
	work := []NodeID{root.ID}
	for len(work) > 0 {
		id := work[0]
		work = work[1:]
		alias := plan.aliases[id]
		alias.residual = index.outputs[id]
		for _, use := range index.consumers[id] {
			consumer := index.nodes[use.consumer]
			if use.input == 0 && consumer.Op == GraphOpDynamicProject {
				plan.dynamicIndex = true
				continue
			}
			if use.input == 0 && consumer.Op == GraphOpIdentity && alias.typ.Kind != GraphValueScalar {
				plan.aliases[consumer.ID] = graphSROAAlias{
					path: cloneIntPath(alias.path),
					typ:  cloneGraphValueType(alias.typ),
				}
				work = append(work, consumer.ID)
				continue
			}
			if use.input == 0 && consumer.Op == GraphOpProject {
				projected, _ := projectGraphValueType(alias.typ, consumer.Path)
				child := graphSROAAlias{
					path: appendGraphPath(alias.path, consumer.Path),
					typ:  projected,
				}
				plan.aliases[consumer.ID] = child
				if projected.Kind != GraphValueScalar {
					work = append(work, consumer.ID)
				}
				continue
			}
			alias.residual = true
		}
		plan.aliases[id] = alias
	}
	return plan
}

func (p *graphSROAPlan) slotForPath(path []int) (graphSROASlot, bool) {
	for _, slot := range p.slots {
		if graphPathsEqual(slot.path, path) {
			return slot, true
		}
	}
	return graphSROASlot{}, false
}

func (p *graphSROAPlan) slotIDsUnder(prefix []int) []NodeID {
	ids := make([]NodeID, 0)
	for _, slot := range p.slots {
		if graphPathHasPrefix(slot.path, prefix) {
			ids = append(ids, slot.id)
		}
	}
	return ids
}

type graphValueLeaf struct {
	path []int
}

func graphValueLeaves(typ GraphValueType, prefix []int) []graphValueLeaf {
	if typ.Kind == GraphValueScalar {
		return []graphValueLeaf{{path: cloneIntPath(prefix)}}
	}
	leaves := make([]graphValueLeaf, 0)
	if typ.Kind == GraphValueStruct {
		for i, child := range typ.Fields {
			leaves = append(leaves, graphValueLeaves(child, appendGraphPath(prefix, []int{i}))...)
		}
		return leaves
	}
	for i := 0; i < typ.Length; i++ {
		leaves = append(leaves, graphValueLeaves(*typ.Element, appendGraphPath(prefix, []int{i}))...)
	}
	return leaves
}

func graphTypeHasOversizedArray(typ GraphValueType, max int) bool {
	if typ.Kind == GraphValueArray {
		if typ.Length > max {
			return true
		}
		return graphTypeHasOversizedArray(*typ.Element, max)
	}
	for _, field := range typ.Fields {
		if graphTypeHasOversizedArray(field, max) {
			return true
		}
	}
	return false
}

func validateGraphValueType(typ GraphValueType, active map[*GraphValueType]bool) error {
	switch typ.Kind {
	case GraphValueScalar:
		if len(typ.Fields) != 0 || typ.Element != nil || typ.Length != 0 {
			return fmt.Errorf("scalar type carries aggregate shape")
		}
	case GraphValueStruct:
		if len(typ.Fields) == 0 || typ.Element != nil || typ.Length != 0 {
			return fmt.Errorf("struct type must have fields and no array shape")
		}
		for i, field := range typ.Fields {
			if err := validateGraphValueType(field, active); err != nil {
				return fmt.Errorf("field %d: %w", i, err)
			}
		}
	case GraphValueArray:
		if typ.Length <= 0 || typ.Element == nil || len(typ.Fields) != 0 {
			return fmt.Errorf("array type must have positive length, one element type, and no struct fields")
		}
		if active[typ.Element] {
			return fmt.Errorf("recursive array element type")
		}
		active[typ.Element] = true
		err := validateGraphValueType(*typ.Element, active)
		delete(active, typ.Element)
		if err != nil {
			return fmt.Errorf("array element: %w", err)
		}
	default:
		return fmt.Errorf("unknown value kind %q", typ.Kind)
	}
	return nil
}

func validateAggregateGraphNode(node AggregateGraphNode, nodes map[NodeID]AggregateGraphNode) error {
	if node.Op != GraphOpProject && len(node.Path) != 0 {
		return fmt.Errorf("aggregate graph: %s node %q carries a projection path", node.Op, node.ID)
	}
	scalar := node.Type.Kind == GraphValueScalar
	switch node.Op {
	case GraphOpInput, GraphOpConstant:
		if !scalar || len(node.Inputs) != 0 {
			return fmt.Errorf("aggregate graph: %s node %q must be a zero-input scalar", node.Op, node.ID)
		}
	case GraphOpIdentity, GraphOpSROASlot:
		if len(node.Inputs) != 1 || !graphValueTypesEqual(node.Type, nodes[node.Inputs[0]].Type) {
			return fmt.Errorf("aggregate graph: %s node %q must preserve one input type", node.Op, node.ID)
		}
		if node.Op == GraphOpSROASlot && !scalar {
			return fmt.Errorf("aggregate graph: sroa slot %q must be scalar", node.ID)
		}
	case GraphOpAdd, GraphOpMultiply:
		if !scalar || len(node.Inputs) != 2 || nodes[node.Inputs[0]].Type.Kind != GraphValueScalar || nodes[node.Inputs[1]].Type.Kind != GraphValueScalar {
			return fmt.Errorf("aggregate graph: %s node %q must have two scalar inputs", node.Op, node.ID)
		}
	case GraphOpAggregate, GraphOpAggregateRebuild:
		if scalar {
			return fmt.Errorf("aggregate graph: %s node %q must have aggregate type", node.Op, node.ID)
		}
		want, overflow := graphValueLeafCount(node.Type, len(node.Inputs)+1)
		if overflow || want != len(node.Inputs) {
			return fmt.Errorf("aggregate graph: %s node %q has %d scalar inputs, want %d", node.Op, node.ID, len(node.Inputs), want)
		}
		for _, input := range node.Inputs {
			if nodes[input].Type.Kind != GraphValueScalar {
				return fmt.Errorf("aggregate graph: %s node %q input %q is not scalar", node.Op, node.ID, input)
			}
		}
	case GraphOpProject:
		if len(node.Inputs) != 1 || len(node.Path) == 0 {
			return fmt.Errorf("aggregate graph: project node %q must have one input and a non-empty static path", node.ID)
		}
		projected, err := projectGraphValueType(nodes[node.Inputs[0]].Type, node.Path)
		if err != nil {
			return fmt.Errorf("aggregate graph: project node %q: %w", node.ID, err)
		}
		if !graphValueTypesEqual(node.Type, projected) {
			return fmt.Errorf("aggregate graph: project node %q result type does not match path", node.ID)
		}
	case GraphOpDynamicProject:
		if len(node.Inputs) != 2 {
			return fmt.Errorf("aggregate graph: dynamic project node %q must have aggregate and index inputs", node.ID)
		}
		source := nodes[node.Inputs[0]].Type
		if source.Kind != GraphValueArray || nodes[node.Inputs[1]].Type.Kind != GraphValueScalar || !graphValueTypesEqual(node.Type, *source.Element) {
			return fmt.Errorf("aggregate graph: dynamic project node %q requires an array, scalar index, and matching element result", node.ID)
		}
	default:
		return fmt.Errorf("aggregate graph: node %q has unknown op %q", node.ID, node.Op)
	}
	return nil
}

func graphValueLeafCount(typ GraphValueType, limit int) (int, bool) {
	if typ.Kind == GraphValueScalar {
		return 1, false
	}
	count := 0
	if typ.Kind == GraphValueStruct {
		for _, child := range typ.Fields {
			childCount, overflow := graphValueLeafCount(child, limit-count)
			if overflow || childCount > limit-count {
				return count, true
			}
			count += childCount
		}
		return count, false
	}
	elementCount, overflow := graphValueLeafCount(*typ.Element, limit)
	if overflow || elementCount > limit {
		return 0, true
	}
	if typ.Length > limit/elementCount {
		return 0, true
	}
	return typ.Length * elementCount, false
}

func projectGraphValueType(typ GraphValueType, path []int) (GraphValueType, error) {
	current := typ
	for depth, index := range path {
		if index < 0 {
			return GraphValueType{}, fmt.Errorf("path index %d at depth %d is out of range", index, depth)
		}
		switch current.Kind {
		case GraphValueStruct:
			if index >= len(current.Fields) {
				return GraphValueType{}, fmt.Errorf("path index %d at depth %d is out of range", index, depth)
			}
			current = current.Fields[index]
		case GraphValueArray:
			if index >= current.Length {
				return GraphValueType{}, fmt.Errorf("path index %d at depth %d is out of range", index, depth)
			}
			current = *current.Element
		default:
			return GraphValueType{}, fmt.Errorf("path continues through scalar at depth %d", depth)
		}
	}
	return cloneGraphValueType(current), nil
}

func aggregateGraphIsAcyclic(nodes []AggregateGraphNode) bool {
	indegree := make(map[NodeID]int, len(nodes))
	consumers := make(map[NodeID][]NodeID, len(nodes))
	for _, node := range nodes {
		indegree[node.ID] = len(node.Inputs)
		for _, input := range node.Inputs {
			consumers[input] = append(consumers[input], node.ID)
		}
	}
	ready := make([]NodeID, 0, len(nodes))
	for id, degree := range indegree {
		if degree == 0 {
			ready = append(ready, id)
		}
	}
	sort.Slice(ready, func(i, j int) bool { return ready[i] < ready[j] })
	seen := 0
	for len(ready) > 0 {
		id := ready[0]
		ready = ready[1:]
		seen++
		for _, consumer := range consumers[id] {
			indegree[consumer]--
			if indegree[consumer] == 0 {
				ready = append(ready, consumer)
				sort.Slice(ready, func(i, j int) bool { return ready[i] < ready[j] })
			}
		}
	}
	return seen == len(nodes)
}

func freshSROASlotID(root NodeID, path []int, used map[NodeID]bool) NodeID {
	parts := make([]string, len(path))
	for i, index := range path {
		parts[i] = fmt.Sprint(index)
	}
	base := NodeID(string(root) + "$sroa$" + strings.Join(parts, "_"))
	id := base
	for suffix := 1; used[id]; suffix++ {
		id = NodeID(fmt.Sprintf("%s$%d", base, suffix))
	}
	used[id] = true
	return id
}

func scalarGraphValueType() GraphValueType {
	return GraphValueType{Kind: GraphValueScalar}
}

func cloneAggregateGraph(graph AggregateGraph) AggregateGraph {
	result := AggregateGraph{
		Nodes:   make([]AggregateGraphNode, len(graph.Nodes)),
		Outputs: cloneNodeIDs(graph.Outputs),
	}
	for i, node := range graph.Nodes {
		result.Nodes[i] = cloneAggregateGraphNode(node)
	}
	if graph.Provenance != nil {
		result.Provenance = make([]GraphProvenanceExpression, len(graph.Provenance))
		for i, expression := range graph.Provenance {
			result.Provenance[i] = cloneGraphProvenanceExpression(expression)
		}
	}
	return result
}

func cloneAggregateGraphNode(node AggregateGraphNode) AggregateGraphNode {
	node.Inputs = cloneNodeIDs(node.Inputs)
	node.Type = cloneGraphValueType(node.Type)
	node.Path = cloneIntPath(node.Path)
	return node
}

func cloneGraphValueType(typ GraphValueType) GraphValueType {
	cloned := GraphValueType{Kind: typ.Kind, Length: typ.Length}
	if typ.Fields != nil {
		cloned.Fields = make([]GraphValueType, len(typ.Fields))
		for i, field := range typ.Fields {
			cloned.Fields[i] = cloneGraphValueType(field)
		}
	}
	if typ.Element != nil {
		element := cloneGraphValueType(*typ.Element)
		cloned.Element = &element
	}
	return cloned
}

func cloneGraphProvenanceExpression(expression GraphProvenanceExpression) GraphProvenanceExpression {
	expression.Path = cloneIntPath(expression.Path)
	return expression
}

func cloneIntPath(path []int) []int {
	if path == nil {
		return nil
	}
	cloned := make([]int, len(path))
	copy(cloned, path)
	return cloned
}

func appendGraphPath(prefix, suffix []int) []int {
	path := make([]int, 0, len(prefix)+len(suffix))
	path = append(path, prefix...)
	path = append(path, suffix...)
	return path
}

func graphPathsEqual(left, right []int) bool {
	return slices.Equal(left, right)
}

func graphPathHasPrefix(path, prefix []int) bool {
	if len(prefix) > len(path) {
		return false
	}
	return graphPathsEqual(path[:len(prefix)], prefix)
}

func graphValueTypesEqual(left, right GraphValueType) bool {
	if left.Kind != right.Kind || left.Length != right.Length || len(left.Fields) != len(right.Fields) || (left.Element == nil) != (right.Element == nil) {
		return false
	}
	for i := range left.Fields {
		if !graphValueTypesEqual(left.Fields[i], right.Fields[i]) {
			return false
		}
	}
	if left.Element != nil && !graphValueTypesEqual(*left.Element, *right.Element) {
		return false
	}
	return true
}
