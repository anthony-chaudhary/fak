package compute

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestGraphPipelineConvergesAndMatchesEagerOracle(t *testing.T) {
	input := tinyGraphFixture()
	originalIR, err := input.StableIR()
	if err != nil {
		t.Fatalf("fixture StableIR: %v", err)
	}

	optimized, first, err := CanonicalGraphPipeline().Run(input)
	if err != nil {
		t.Fatalf("first pipeline run: %v", err)
	}
	stableIR, err := optimized.StableIR()
	if err != nil {
		t.Fatalf("optimized StableIR: %v", err)
	}
	wantStableIR := []byte(`{"nodes":[{"id":"two","op":"constant","inputs":[],"value":2},{"id":"x","op":"input","inputs":[],"value":0},{"id":"sum","op":"add","inputs":["x","two"],"value":0}],"outputs":["sum"]}`)
	if !bytes.Equal(stableIR, wantStableIR) {
		t.Fatalf("stable IR = %s, want %s", stableIR, wantStableIR)
	}
	if bytes.Equal(originalIR, stableIR) {
		t.Fatal("first pipeline run left deliberately non-canonical/dead fixture unchanged")
	}

	got, err := evaluateTinyGraph(optimized, map[NodeID]float64{"x": 3})
	if err != nil {
		t.Fatalf("evaluate optimized graph: %v", err)
	}
	if want := eagerTinyGraph(3); got != want {
		t.Fatalf("optimized graph output = %v, eager oracle = %v", got, want)
	}

	wantFirstOrder := []GraphPassName{
		CanonicalizePassName, DCEPassName, CanonicalizePassName,
		CanonicalizePassName, DCEPassName, CanonicalizePassName,
	}
	if gotOrder := receiptOrder(first); !reflect.DeepEqual(gotOrder, wantFirstOrder) {
		t.Fatalf("first pass order = %v, want %v", gotOrder, wantFirstOrder)
	}
	if !first.Stable || first.Rounds != 2 {
		t.Fatalf("first receipt stable/rounds = %t/%d, want true/2", first.Stable, first.Rounds)
	}
	wantFirstRounds := []int{1, 1, 1, 2, 2, 2}
	if gotRounds := receiptRounds(first); !reflect.DeepEqual(gotRounds, wantFirstRounds) {
		t.Fatalf("first receipt rounds = %v, want %v", gotRounds, wantFirstRounds)
	}
	wantFirstChanges := []int{1, 1, 0, 0, 0, 0}
	if gotChanges := receiptChanges(first); !reflect.DeepEqual(gotChanges, wantFirstChanges) {
		t.Fatalf("first changed-node counts = %v, want %v", gotChanges, wantFirstChanges)
	}

	wantDigestBytes := sha256.Sum256(stableIR)
	wantDigest := hex.EncodeToString(wantDigestBytes[:])
	if first.FinalGraphDigest != wantDigest {
		t.Fatalf("first final digest = %q, want sha256(stable IR) %q", first.FinalGraphDigest, wantDigest)
	}

	secondGraph, second, err := CanonicalGraphPipeline().Run(optimized)
	if err != nil {
		t.Fatalf("second pipeline run: %v", err)
	}
	secondIR, err := secondGraph.StableIR()
	if err != nil {
		t.Fatalf("second StableIR: %v", err)
	}
	if !bytes.Equal(secondIR, stableIR) {
		t.Fatalf("second run changed stable IR:\nfirst:  %s\nsecond: %s", stableIR, secondIR)
	}
	if second.FinalGraphDigest != first.FinalGraphDigest {
		t.Fatalf("second digest = %q, first = %q", second.FinalGraphDigest, first.FinalGraphDigest)
	}
	wantSecondOrder := []GraphPassName{CanonicalizePassName, DCEPassName, CanonicalizePassName}
	if gotOrder := receiptOrder(second); !reflect.DeepEqual(gotOrder, wantSecondOrder) {
		t.Fatalf("second pass order = %v, want %v", gotOrder, wantSecondOrder)
	}
	if !second.Stable || second.Rounds != 1 {
		t.Fatalf("second receipt stable/rounds = %t/%d, want true/1", second.Stable, second.Rounds)
	}
	wantSecondRounds := []int{1, 1, 1}
	if gotRounds := receiptRounds(second); !reflect.DeepEqual(gotRounds, wantSecondRounds) {
		t.Fatalf("second receipt rounds = %v, want %v", gotRounds, wantSecondRounds)
	}
	for _, pass := range second.Passes {
		if pass.ChangedNodes != 0 {
			t.Fatalf("stable second run pass %q changed %d nodes", pass.Name, pass.ChangedNodes)
		}
	}

	inputIR, err := input.StableIR()
	if err != nil {
		t.Fatalf("input StableIR after pipeline: %v", err)
	}
	if !bytes.Equal(inputIR, originalIR) {
		t.Fatal("pipeline mutated its input graph")
	}
}

func TestGraphPipelinePreservesIEEESignedZero(t *testing.T) {
	input := Graph{
		Nodes: []GraphNode{
			{ID: "out", Op: GraphOpIdentity, Inputs: []NodeID{"negative-zero"}},
			{ID: "negative-zero", Op: GraphOpConstant, Value: math.Copysign(0, -1)},
		},
		Outputs: []NodeID{"out"},
	}

	optimized, _, err := CanonicalGraphPipeline().Run(input)
	if err != nil {
		t.Fatalf("pipeline run: %v", err)
	}
	got, err := evaluateTinyGraph(optimized, nil)
	if err != nil {
		t.Fatalf("evaluate optimized graph: %v", err)
	}
	want := eagerSignedZero()
	if got != want || math.Signbit(got) != math.Signbit(want) {
		t.Fatalf("optimized graph output = %v (signbit %t), eager oracle = %v (signbit %t)",
			got, math.Signbit(got), want, math.Signbit(want))
	}
}

func TestGraphPipelineFailsClosed(t *testing.T) {
	tests := []struct {
		name  string
		graph Graph
		want  string
	}{
		{
			name:  "missing output",
			graph: Graph{Nodes: []GraphNode{{ID: "x", Op: GraphOpInput}}},
			want:  "at least one output",
		},
		{
			name: "dangling input",
			graph: Graph{
				Nodes:   []GraphNode{{ID: "sum", Op: GraphOpAdd, Inputs: []NodeID{"missing", "x"}}, {ID: "x", Op: GraphOpInput}},
				Outputs: []NodeID{"sum"},
			},
			want: "unknown input",
		},
		{
			name: "cycle",
			graph: Graph{
				Nodes: []GraphNode{
					{ID: "a", Op: GraphOpIdentity, Inputs: []NodeID{"b"}},
					{ID: "b", Op: GraphOpIdentity, Inputs: []NodeID{"a"}},
				},
				Outputs: []NodeID{"a"},
			},
			want: "cycle",
		},
		{
			name: "unknown op",
			graph: Graph{
				Nodes:   []GraphNode{{ID: "x", Op: GraphOp("custom")}},
				Outputs: []NodeID{"x"},
			},
			want: "unknown op",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, receipt, err := CanonicalGraphPipeline().Run(tc.graph)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Run error = %v, want containing %q", err, tc.want)
			}
			if len(got.Nodes) != 0 || len(got.Outputs) != 0 {
				t.Fatalf("invalid graph escaped on error: %+v", got)
			}
			if receipt.Stable || receipt.FinalGraphDigest != "" {
				t.Fatalf("invalid graph received success receipt: %+v", receipt)
			}
		})
	}

	invalidPasses := []struct {
		name string
		pass GraphPass
		want string
	}{
		{
			name: "dangling input",
			pass: graphPassFunc{
				name: "bad-output",
				apply: func(g Graph) (Graph, error) {
					for i := range g.Nodes {
						if g.Nodes[i].ID == "sum" {
							g.Nodes[i].Inputs[0] = "not-a-node"
							return g, nil
						}
					}
					return Graph{}, errors.New("fixture has no sum node")
				},
			},
			want: "unknown input",
		},
		{
			name: "unknown op",
			pass: graphPassFunc{
				name: "unknown-op-output",
				apply: func(g Graph) (Graph, error) {
					for i := range g.Nodes {
						if g.Nodes[i].ID == "sum" {
							g.Nodes[i].Op = GraphOp("custom")
							return g, nil
						}
					}
					return Graph{}, errors.New("fixture has no sum node")
				},
			},
			want: "unknown op",
		},
	}
	for _, tc := range invalidPasses {
		t.Run("invalid pass output/"+tc.name, func(t *testing.T) {
			pipeline := GraphPipeline{
				Passes:    []GraphPass{CanonicalizeGraphPass{}, tc.pass},
				MaxRounds: 1,
			}
			got, receipt, err := pipeline.Run(tinyGraphFixture())
			if err == nil || !strings.Contains(err.Error(), string(tc.pass.Name())) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("invalid pass output error = %v, want pass %q and %q", err, tc.pass.Name(), tc.want)
			}
			if !reflect.DeepEqual(got, Graph{}) {
				t.Fatalf("invalid pass output returned non-zero graph: %+v", got)
			}
			if !reflect.DeepEqual(receipt, GraphPipelineReceipt{}) {
				t.Fatalf("invalid pass output returned non-zero receipt: %+v", receipt)
			}
		})
	}
}

func eagerSignedZero() float64 {
	return math.Copysign(0, -1)
}

func tinyGraphFixture() Graph {
	return Graph{
		// Deliberately non-topological, with an identity and a dead constant.
		Nodes: []GraphNode{
			{ID: "dead", Op: GraphOpConstant, Value: 99},
			{ID: "sum", Op: GraphOpAdd, Inputs: []NodeID{"x", "two"}},
			{ID: "out", Op: GraphOpIdentity, Inputs: []NodeID{"sum"}},
			{ID: "x", Op: GraphOpInput},
			{ID: "two", Op: GraphOpConstant, Value: 2},
		},
		Outputs: []NodeID{"out"},
	}
}

// eagerTinyGraph is intentionally independent of the graph interpreter below: it is the
// reference program whose observable output the compiler pipeline must preserve.
func eagerTinyGraph(x float64) float64 {
	dead := 99.0
	_ = dead
	return x + 2
}

func evaluateTinyGraph(g Graph, inputs map[NodeID]float64) (float64, error) {
	values := make(map[NodeID]float64, len(g.Nodes))
	for _, node := range g.Nodes {
		switch node.Op {
		case GraphOpInput:
			value, ok := inputs[node.ID]
			if !ok {
				return 0, fmt.Errorf("missing input %q", node.ID)
			}
			values[node.ID] = value
		case GraphOpConstant:
			values[node.ID] = node.Value
		case GraphOpIdentity:
			values[node.ID] = values[node.Inputs[0]]
		case GraphOpAdd:
			values[node.ID] = values[node.Inputs[0]] + values[node.Inputs[1]]
		case GraphOpMultiply:
			values[node.ID] = values[node.Inputs[0]] * values[node.Inputs[1]]
		default:
			return 0, fmt.Errorf("unsupported test op %q", node.Op)
		}
	}
	if len(g.Outputs) != 1 {
		return 0, errors.New("test graph must have one output")
	}
	return values[g.Outputs[0]], nil
}

func receiptOrder(receipt GraphPipelineReceipt) []GraphPassName {
	order := make([]GraphPassName, len(receipt.Passes))
	for i, pass := range receipt.Passes {
		order[i] = pass.Name
	}
	return order
}

func receiptChanges(receipt GraphPipelineReceipt) []int {
	changes := make([]int, len(receipt.Passes))
	for i, pass := range receipt.Passes {
		changes[i] = pass.ChangedNodes
	}
	return changes
}

func receiptRounds(receipt GraphPipelineReceipt) []int {
	rounds := make([]int, len(receipt.Passes))
	for i, pass := range receipt.Passes {
		rounds[i] = pass.Round
	}
	return rounds
}

type graphPassFunc struct {
	name  GraphPassName
	apply func(Graph) (Graph, error)
}

func (p graphPassFunc) Name() GraphPassName          { return p.name }
func (p graphPassFunc) Apply(g Graph) (Graph, error) { return p.apply(g) }
