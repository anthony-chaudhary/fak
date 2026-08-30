package compute

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode/utf8"
)

// NodeID is a stable, backend-neutral identity for one graph value. Passes retain IDs for
// surviving values so receipts can count the exact nodes each pass changed.
type NodeID string

// GraphOp identifies an operation without naming a device or backend lowering.
type GraphOp string

const (
	GraphOpInput    GraphOp = "input"
	GraphOpConstant GraphOp = "constant"
	GraphOpIdentity GraphOp = "identity"
	GraphOpAdd      GraphOp = "add"
	GraphOpMultiply GraphOp = "multiply"
)

// GraphNode is the first typed compute IR node. Nodes are pure dataflow operations and Outputs
// are their only observable roots. Value is meaningful for constants.
type GraphNode struct {
	ID     NodeID   `json:"id"`
	Op     GraphOp  `json:"op"`
	Inputs []NodeID `json:"inputs"`
	Value  float64  `json:"value"`
}

// Graph is a typed acyclic dataflow graph. Node order is representation order, while Outputs
// order is observable and therefore never sorted by a cleanup pass.
type Graph struct {
	Nodes   []GraphNode `json:"nodes"`
	Outputs []NodeID    `json:"outputs"`
}

// Validate rejects malformed graphs before they reach any pass or backend. Every accepted op is
// explicitly defined above and gets its arity checked here.
func (g Graph) Validate() error {
	if len(g.Nodes) == 0 {
		return fmt.Errorf("compute graph: must contain at least one node")
	}
	if len(g.Outputs) == 0 {
		return fmt.Errorf("compute graph: must contain at least one output")
	}

	nodes := make(map[NodeID]GraphNode, len(g.Nodes))
	for i, node := range g.Nodes {
		if node.ID == "" || !utf8.ValidString(string(node.ID)) {
			return fmt.Errorf("compute graph: node %d has invalid empty or non-UTF-8 ID", i)
		}
		if node.Op == "" || !utf8.ValidString(string(node.Op)) {
			return fmt.Errorf("compute graph: node %q has invalid empty or non-UTF-8 op", node.ID)
		}
		if _, exists := nodes[node.ID]; exists {
			return fmt.Errorf("compute graph: duplicate node ID %q", node.ID)
		}
		if math.IsNaN(node.Value) || math.IsInf(node.Value, 0) {
			return fmt.Errorf("compute graph: node %q has non-finite value", node.ID)
		}
		if err := validateGraphOp(node); err != nil {
			return err
		}
		nodes[node.ID] = node
	}

	for _, node := range g.Nodes {
		for _, input := range node.Inputs {
			if _, ok := nodes[input]; !ok {
				return fmt.Errorf("compute graph: node %q references unknown input %q", node.ID, input)
			}
		}
	}
	for i, output := range g.Outputs {
		if _, ok := nodes[output]; !ok {
			return fmt.Errorf("compute graph: output %d references unknown node %q", i, output)
		}
	}
	if !graphIsAcyclic(g) {
		return fmt.Errorf("compute graph: dependency cycle detected")
	}
	return nil
}

func validateGraphOp(node GraphNode) error {
	var want int
	switch node.Op {
	case GraphOpInput, GraphOpConstant:
		want = 0
	case GraphOpIdentity:
		want = 1
	case GraphOpAdd, GraphOpMultiply:
		want = 2
	default:
		return fmt.Errorf("compute graph: node %q has unknown op %q", node.ID, node.Op)
	}
	if len(node.Inputs) != want {
		return fmt.Errorf("compute graph: %s node %q has %d inputs, want %d", node.Op, node.ID, len(node.Inputs), want)
	}
	return nil
}

func graphIsAcyclic(g Graph) bool {
	indegree := make(map[NodeID]int, len(g.Nodes))
	consumers := make(map[NodeID][]NodeID, len(g.Nodes))
	for _, node := range g.Nodes {
		indegree[node.ID] = len(node.Inputs)
		for _, input := range node.Inputs {
			consumers[input] = append(consumers[input], node.ID)
		}
	}
	ready := make([]NodeID, 0, len(g.Nodes))
	for id, degree := range indegree {
		if degree == 0 {
			ready = append(ready, id)
		}
	}
	seen := 0
	for len(ready) > 0 {
		id := ready[len(ready)-1]
		ready = ready[:len(ready)-1]
		seen++
		for _, consumer := range consumers[id] {
			indegree[consumer]--
			if indegree[consumer] == 0 {
				ready = append(ready, consumer)
			}
		}
	}
	return seen == len(g.Nodes)
}

// StableIR returns the exact serialized graph representation used by pipeline convergence and
// receipts. It validates rather than silently serializing an unusable graph.
func (g Graph) StableIR() ([]byte, error) {
	if err := g.Validate(); err != nil {
		return nil, err
	}
	b, err := json.Marshal(g)
	if err != nil {
		return nil, fmt.Errorf("compute graph: serialize stable IR: %w", err)
	}
	return b, nil
}

// Digest returns the lowercase SHA-256 digest of StableIR.
func (g Graph) Digest() (string, error) {
	ir, err := g.StableIR()
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(ir)
	return hex.EncodeToString(digest[:]), nil
}

// GraphPassName is typed so receipts cannot mix pass names with node or backend identifiers.
type GraphPassName string

const (
	CanonicalizePassName GraphPassName = "canonicalize"
	DCEPassName          GraphPassName = "dce"
)

// GraphPass is the backend-neutral rewrite seam. The pipeline gives Apply a private graph copy,
// validates its result, and computes ChangedNodes itself instead of trusting a pass claim.
type GraphPass interface {
	Name() GraphPassName
	Apply(Graph) (Graph, error)
}

// GraphPassReceipt records one pass invocation in execution order.
type GraphPassReceipt struct {
	Round int           `json:"round"`
	Name  GraphPassName `json:"name"`
	// ChangedNodes counts stable node IDs added, removed, or changed in place. Pure
	// representation-order movement is not a node rewrite and is intentionally excluded.
	ChangedNodes int `json:"changed_nodes"`
}

// GraphPipelineReceipt is the deterministic proof emitted only after a whole round reaches a
// fixpoint. Passes is ordered exactly as executed, including repeated convergence rounds.
type GraphPipelineReceipt struct {
	Passes           []GraphPassReceipt `json:"passes"`
	Rounds           int                `json:"rounds"`
	Stable           bool               `json:"stable"`
	FinalGraphDigest string             `json:"final_graph_digest"`
}

// GraphPipeline repeatedly runs its ordered pass list until a whole round leaves StableIR
// byte-identical. MaxRounds is a fail-closed bound against oscillating or defective passes.
type GraphPipeline struct {
	Passes    []GraphPass
	MaxRounds int
}

const defaultGraphPipelineMaxRounds = 64

// CanonicalGraphPipeline constructs the first native compiler spine. Repeating canonicalization
// around DCE is intentional: each cleanup may expose work for the next round.
func CanonicalGraphPipeline() GraphPipeline {
	return GraphPipeline{
		Passes: []GraphPass{
			CanonicalizeGraphPass{},
			DeadCodeEliminationPass{},
			CanonicalizeGraphPass{},
		},
		MaxRounds: defaultGraphPipelineMaxRounds,
	}
}

// Run executes the configured passes to convergence without mutating input. On invalid input,
// invalid pass output, pass failure, or non-convergence it returns a zero graph and no success
// digest, preventing callers from accidentally executing an unverified intermediate graph.
func (p GraphPipeline) Run(input Graph) (Graph, GraphPipelineReceipt, error) {
	var receipt GraphPipelineReceipt
	if len(p.Passes) == 0 {
		return Graph{}, receipt, fmt.Errorf("compute graph pipeline: must contain at least one pass")
	}
	if p.MaxRounds <= 0 {
		return Graph{}, receipt, fmt.Errorf("compute graph pipeline: MaxRounds must be positive")
	}
	if err := input.Validate(); err != nil {
		return Graph{}, receipt, fmt.Errorf("compute graph pipeline: invalid input: %w", err)
	}

	current := cloneGraph(input)
	for round := 1; round <= p.MaxRounds; round++ {
		startIR, err := current.StableIR()
		if err != nil {
			return Graph{}, GraphPipelineReceipt{}, fmt.Errorf("compute graph pipeline: round %d start: %w", round, err)
		}
		for passIndex, pass := range p.Passes {
			name, err := graphPassName(pass)
			if err != nil {
				return Graph{}, GraphPipelineReceipt{}, fmt.Errorf("compute graph pipeline: round %d pass %d: %w", round, passIndex+1, err)
			}
			before := cloneGraph(current)
			candidate, err := applyGraphPass(pass, cloneGraph(current))
			if err != nil {
				return Graph{}, GraphPipelineReceipt{}, fmt.Errorf("compute graph pipeline: round %d pass %q: %w", round, name, err)
			}
			candidate = cloneGraph(candidate)
			if err := candidate.Validate(); err != nil {
				return Graph{}, GraphPipelineReceipt{}, fmt.Errorf("compute graph pipeline: round %d pass %q produced invalid graph: %w", round, name, err)
			}
			receipt.Passes = append(receipt.Passes, GraphPassReceipt{
				Round:        round,
				Name:         name,
				ChangedNodes: changedGraphNodeCount(before, candidate),
			})
			current = candidate
		}

		receipt.Rounds = round
		endIR, err := current.StableIR()
		if err != nil {
			return Graph{}, GraphPipelineReceipt{}, fmt.Errorf("compute graph pipeline: round %d result: %w", round, err)
		}
		if bytes.Equal(startIR, endIR) {
			digest := sha256.Sum256(endIR)
			receipt.Stable = true
			receipt.FinalGraphDigest = hex.EncodeToString(digest[:])
			return cloneGraph(current), receipt, nil
		}
	}
	return Graph{}, receipt, fmt.Errorf("compute graph pipeline: did not converge within %d rounds", p.MaxRounds)
}

func graphPassName(pass GraphPass) (name GraphPassName, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("pass Name panicked: %v", recovered)
		}
	}()
	if pass == nil {
		return "", fmt.Errorf("nil pass")
	}
	name = pass.Name()
	if strings.TrimSpace(string(name)) == "" || !utf8.ValidString(string(name)) {
		return "", fmt.Errorf("pass has invalid empty, whitespace, or non-UTF-8 name")
	}
	return name, nil
}

func applyGraphPass(pass GraphPass, graph Graph) (result Graph, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result = Graph{}
			err = fmt.Errorf("Apply panicked: %v", recovered)
		}
	}()
	return pass.Apply(graph)
}

// CanonicalizeGraphPass folds identities and emits a stable lexicographic topological order. It
// preserves operand order because the first spine has no dtype contract proving a rewrite safe.
type CanonicalizeGraphPass struct{}

func (CanonicalizeGraphPass) Name() GraphPassName { return CanonicalizePassName }

func (CanonicalizeGraphPass) Apply(input Graph) (Graph, error) {
	if err := input.Validate(); err != nil {
		return Graph{}, err
	}
	g := cloneGraph(input)
	aliases := make(map[NodeID]NodeID)
	for _, node := range g.Nodes {
		if node.Op == GraphOpIdentity {
			aliases[node.ID] = node.Inputs[0]
		}
	}
	resolve := func(id NodeID) NodeID {
		for {
			next, ok := aliases[id]
			if !ok {
				return id
			}
			id = next
		}
	}

	nodes := make([]GraphNode, 0, len(g.Nodes)-len(aliases))
	for _, node := range g.Nodes {
		if node.Op == GraphOpIdentity {
			continue
		}
		node.Inputs = cloneNodeIDs(node.Inputs)
		if node.Inputs == nil {
			node.Inputs = []NodeID{}
		}
		for i, input := range node.Inputs {
			node.Inputs[i] = resolve(input)
		}
		nodes = append(nodes, node)
	}
	outputs := cloneNodeIDs(g.Outputs)
	for i, output := range outputs {
		outputs[i] = resolve(output)
	}
	result := Graph{Nodes: nodes, Outputs: outputs}
	ordered, err := stableTopologicalOrder(result)
	if err != nil {
		return Graph{}, err
	}
	result.Nodes = ordered
	if err := result.Validate(); err != nil {
		return Graph{}, err
	}
	return result, nil
}

// DeadCodeEliminationPass removes every node not reachable backwards from an observable output.
type DeadCodeEliminationPass struct{}

func (DeadCodeEliminationPass) Name() GraphPassName { return DCEPassName }

func (DeadCodeEliminationPass) Apply(input Graph) (Graph, error) {
	if err := input.Validate(); err != nil {
		return Graph{}, err
	}
	nodes := make(map[NodeID]GraphNode, len(input.Nodes))
	for _, node := range input.Nodes {
		nodes[node.ID] = node
	}
	live := make(map[NodeID]bool, len(input.Nodes))
	work := cloneNodeIDs(input.Outputs)
	for len(work) > 0 {
		id := work[len(work)-1]
		work = work[:len(work)-1]
		if live[id] {
			continue
		}
		live[id] = true
		work = append(work, nodes[id].Inputs...)
	}
	kept := make([]GraphNode, 0, len(live))
	for _, node := range input.Nodes {
		if live[node.ID] {
			node.Inputs = cloneNodeIDs(node.Inputs)
			kept = append(kept, node)
		}
	}
	result := Graph{Nodes: kept, Outputs: cloneNodeIDs(input.Outputs)}
	if err := result.Validate(); err != nil {
		return Graph{}, err
	}
	return result, nil
}

func stableTopologicalOrder(g Graph) ([]GraphNode, error) {
	byID := make(map[NodeID]GraphNode, len(g.Nodes))
	indegree := make(map[NodeID]int, len(g.Nodes))
	consumers := make(map[NodeID][]NodeID, len(g.Nodes))
	for _, node := range g.Nodes {
		byID[node.ID] = node
		indegree[node.ID] = len(node.Inputs)
		for _, input := range node.Inputs {
			consumers[input] = append(consumers[input], node.ID)
		}
	}
	ready := make([]NodeID, 0, len(g.Nodes))
	for id, degree := range indegree {
		if degree == 0 {
			ready = append(ready, id)
		}
	}
	sort.Slice(ready, func(i, j int) bool { return ready[i] < ready[j] })
	ordered := make([]GraphNode, 0, len(g.Nodes))
	for len(ready) > 0 {
		id := ready[0]
		ready = ready[1:]
		ordered = append(ordered, byID[id])
		for _, consumer := range consumers[id] {
			indegree[consumer]--
			if indegree[consumer] == 0 {
				ready = append(ready, consumer)
				sort.Slice(ready, func(i, j int) bool { return ready[i] < ready[j] })
			}
		}
	}
	if len(ordered) != len(g.Nodes) {
		return nil, fmt.Errorf("compute graph: dependency cycle detected during canonicalization")
	}
	return ordered, nil
}

func changedGraphNodeCount(before, after Graph) int {
	afterNodes := make(map[NodeID]GraphNode, len(after.Nodes))
	for _, node := range after.Nodes {
		afterNodes[node.ID] = node
	}
	changed := 0
	seen := make(map[NodeID]bool, len(before.Nodes))
	for _, node := range before.Nodes {
		seen[node.ID] = true
		other, ok := afterNodes[node.ID]
		if !ok || !graphNodesEqual(node, other) {
			changed++
		}
	}
	for _, node := range after.Nodes {
		if !seen[node.ID] {
			changed++
		}
	}
	return changed
}

func graphNodesEqual(a, b GraphNode) bool {
	if a.ID != b.ID || a.Op != b.Op || math.Float64bits(a.Value) != math.Float64bits(b.Value) || len(a.Inputs) != len(b.Inputs) {
		return false
	}
	for i := range a.Inputs {
		if a.Inputs[i] != b.Inputs[i] {
			return false
		}
	}
	return true
}

func cloneGraph(g Graph) Graph {
	nodes := make([]GraphNode, len(g.Nodes))
	for i, node := range g.Nodes {
		node.Inputs = cloneNodeIDs(node.Inputs)
		nodes[i] = node
	}
	return Graph{Nodes: nodes, Outputs: cloneNodeIDs(g.Outputs)}
}

func cloneNodeIDs(ids []NodeID) []NodeID {
	if ids == nil {
		return nil
	}
	cloned := make([]NodeID, len(ids))
	copy(cloned, ids)
	return cloned
}
