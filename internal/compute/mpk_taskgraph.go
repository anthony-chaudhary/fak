// Package compute — MPK SM-level task-graph schema, partition invariants, and
// task-descriptor contract, ported from the mirage-project/mirage MPK
// (OSDI 2026) task-graph path: scripts/parse_task_graph.py and
//
// In mirage, tensor programs lower to SM-level task graphs: each node is a
// block of work that one or more SMs execute, and edges are tensor
// dependencies. The partitioner assigns every task to a set of SMs under a
// per-partition SM budget, and the decentralized in-kernel runtime executes
// the partition with cross-operator pipelines and compute-communication
// overlap. This file ports the graph schema, the partition invariants, and
// the task-descriptor contract into a closed in-memory host oracle: no GPU,
// no kernel compilation, no network. The oracle simulates the partitioned
// graph deterministically and reports makespan plus a
// compute-communication overlap count, so a partition can be checked for
// schedulability and pipeline structure on the host.
package compute

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// ErrMPKTaskgraphInvalid is the sentinel returned when a task graph,
// partition, or task descriptor violates the MPK contract.
var ErrMPKTaskgraphInvalid = errors.New("mpk taskgraph: invalid")

// MPKTaskKind classifies a node in the SM-level task graph.
type MPKTaskKind int

const (
	// MPKTaskCompute is a block of tensor work executed on its assigned SMs.
	MPKTaskCompute MPKTaskKind = iota
	// MPKTaskComm is a communication block (e.g. an SM-to-SM or
	// rank-to-rank tensor transfer) that carries a tensor between two
	// compute tasks.
	MPKTaskComm
	// MPKTaskBarrier is a synchronization block with no tensor payload.
	MPKTaskBarrier
)

// String renders the kind for receipts and diagnostics.
func (k MPKTaskKind) String() string {
	switch k {
	case MPKTaskCompute:
		return "compute"
	case MPKTaskComm:
		return "comm"
	case MPKTaskBarrier:
		return "barrier"
	default:
		return fmt.Sprintf("kind(%d)", int(k))
	}
}

// MPKTaskNode is one SM-level task in the graph, mirroring a node emitted by
// parse_task_graph.py: an operator block that occupies a positive number of
// SMs and a bounded shared-memory footprint.
type MPKTaskNode struct {
	ID     string      `json:"id"`
	Kind   MPKTaskKind `json:"kind"`
	Op     string      `json:"op"`
	SMs    int         `json:"sms"`
	Shared int64       `json:"shared_bytes"`
}

// MPKTaskEdge is a tensor dependency between two tasks. Compute-to-compute
// edges carry zero bytes (the tensor is produced and consumed in place);
// communication edges carry a positive byte count for the transferred tensor.
type MPKTaskEdge struct {
	From   string `json:"from"`
	To     string `json:"to"`
	Tensor string `json:"tensor"`
	Bytes  int64  `json:"bytes"`
}

// MPKTaskGraph is the SM-level task graph: nodes plus directed tensor
// dependency edges.
type MPKTaskGraph struct {
	Nodes []MPKTaskNode `json:"nodes"`
	Edges []MPKTaskEdge `json:"edges"`
}

// MPKPartition assigns every task to a set of SMs under a per-partition SM
// budget, mirroring partition_graph.py.
type MPKPartition struct {
	// Assign maps task ID to the SM index the task executes on.
	Assign map[string]int `json:"assign"`
	// SMs is the partition's SM budget.
	SMs int `json:"sms"`
}

// MPKTaskDescriptor is the contract the in-kernel runtime consumes to launch
// one task: its identity, kind, operator, assigned SM, and the tensor
// references it reads and writes.
type MPKTaskDescriptor struct {
	TaskID     string      `json:"task_id"`
	Kind       MPKTaskKind `json:"kind"`
	Op         string      `json:"op"`
	SM         int         `json:"sm"`
	InTensors  []string    `json:"in_tensors"`
	OutTensors []string    `json:"out_tensors"`
}

// MPKOracleReceipt is the closed host-oracle verdict for a partitioned graph.
type MPKOracleReceipt struct {
	// Makespan is the simulated completion time of the last task, in task
	// units (each task occupies its SMs for one unit).
	Makespan int `json:"makespan"`
	// Tasks is the number of tasks scheduled.
	Tasks int `json:"tasks"`
	// Overlap counts communication tasks whose execution window overlaps a
	// compute task on a different SM — the cross-operator
	// compute-communication overlap the MPK runtime is built to exploit.
	Overlap int `json:"overlap"`
}

// MPKValidationResult collects every contract violation found in a graph,
// partition, or descriptor.
type MPKValidationResult struct {
	OK         bool
	Violations []string
}

func (r MPKValidationResult) Error() string {
	return strings.Join(r.Violations, "; ")
}

// MPKValidateGraph checks the task-graph schema: unique task IDs, valid
// kinds, positive SM counts, non-negative shared-memory footprints, edge
// endpoints that exist, no self-edges, no duplicate edges, communication
// edges carrying a positive byte count, compute edges carrying zero bytes,
// and acyclicity.
func MPKValidateGraph(g *MPKTaskGraph) MPKValidationResult {
	var v MPKValidationResult
	if g == nil {
		v.Violations = []string{"nil graph"}
		return v
	}
	seen := map[string]bool{}
	for _, n := range g.Nodes {
		if n.ID == "" {
			v.Violations = append(v.Violations, "task with empty ID")
			continue
		}
		if seen[n.ID] {
			v.Violations = append(v.Violations, fmt.Sprintf("duplicate task ID %q", n.ID))
		}
		seen[n.ID] = true
		if n.Kind < MPKTaskCompute || n.Kind > MPKTaskBarrier {
			v.Violations = append(v.Violations, fmt.Sprintf("task %q has unknown kind %v", n.ID, n.Kind))
		}
		if n.SMs <= 0 {
			v.Violations = append(v.Violations, fmt.Sprintf("task %q must occupy a positive number of SMs (got %d)", n.ID, n.SMs))
		}
		if n.Shared < 0 {
			v.Violations = append(v.Violations, fmt.Sprintf("task %q has negative shared-memory footprint %d", n.ID, n.Shared))
		}
	}
	edgeSeen := map[[2]string]bool{}
	for _, e := range g.Edges {
		if !seen[e.From] {
			v.Violations = append(v.Violations, fmt.Sprintf("edge %q->%q references unknown task %q", e.From, e.To, e.From))
		}
		if !seen[e.To] {
			v.Violations = append(v.Violations, fmt.Sprintf("edge %q->%q references unknown task %q", e.From, e.To, e.To))
		}
		if e.From == e.To {
			v.Violations = append(v.Violations, fmt.Sprintf("self-edge on task %q", e.From))
		}
		key := [2]string{e.From, e.To}
		if edgeSeen[key] {
			v.Violations = append(v.Violations, fmt.Sprintf("duplicate edge %q->%q", e.From, e.To))
		}
		edgeSeen[key] = true
	}
	// Acyclicity via Kahn's algorithm.
	indeg := map[string]int{}
	for _, n := range g.Nodes {
		indeg[n.ID] = 0
	}
	adj := map[string][]string{}
	for _, e := range g.Edges {
		if seen[e.From] && seen[e.To] && e.From != e.To {
			adj[e.From] = append(adj[e.From], e.To)
			indeg[e.To]++
		}
	}
	queue := make([]string, 0, len(g.Nodes))
	for _, n := range g.Nodes {
		if indeg[n.ID] == 0 {
			queue = append(queue, n.ID)
		}
	}
	sorted := 0
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		sorted++
		for _, to := range adj[id] {
			indeg[to]--
			if indeg[to] == 0 {
				queue = append(queue, to)
			}
		}
	}
	if sorted != len(g.Nodes) {
		v.Violations = append(v.Violations, "graph contains a cycle")
	}
	v.OK = len(v.Violations) == 0
	return v
}

// MPKValidatePartition checks the partition invariants against a valid graph:
// every task is assigned exactly once, no task is assigned more SMs than the
// partition budget allows, and every cross-SM tensor dependency is carried by
// a communication task (a compute task on one SM may not hand a tensor
// directly to a task on another SM without an explicit comm task between
// them).
func MPKValidatePartition(g *MPKTaskGraph, p *MPKPartition) MPKValidationResult {
	var v MPKValidationResult
	if g == nil {
		v.Violations = []string{"nil graph"}
		return v
	}
	if p == nil {
		v.Violations = []string{"nil partition"}
		return v
	}
	if p.SMs <= 0 {
		v.Violations = append(v.Violations, fmt.Sprintf("partition SM budget must be positive (got %d)", p.SMs))
	}
	kindByID := map[string]MPKTaskKind{}
	smsByID := map[string]int{}
	for _, n := range g.Nodes {
		kindByID[n.ID] = n.Kind
		smsByID[n.ID] = n.SMs
	}
	assigned := map[string]bool{}
	for id, sm := range p.Assign {
		if _, ok := kindByID[id]; !ok {
			v.Violations = append(v.Violations, fmt.Sprintf("partition assigns unknown task %q", id))
			continue
		}
		if assigned[id] {
			v.Violations = append(v.Violations, fmt.Sprintf("task %q assigned more than once", id))
		}
		assigned[id] = true
		if sm < 0 || sm >= p.SMs {
			v.Violations = append(v.Violations, fmt.Sprintf("task %q assigned to SM %d outside budget [0,%d)", id, sm, p.SMs))
		}
		if smsByID[id] > p.SMs {
			v.Violations = append(v.Violations, fmt.Sprintf("task %q needs %d SMs but the partition budget is %d", id, smsByID[id], p.SMs))
		}
	}
	for _, n := range g.Nodes {
		if !assigned[n.ID] {
			v.Violations = append(v.Violations, fmt.Sprintf("task %q is not assigned to any SM", n.ID))
		}
	}
	// Cross-SM tensor dependencies must be carried by a communication task:
	// a compute task may not hand a tensor directly to a task on a different
	// SM. For any positive-byte edge, if the source is a compute task on SM X
	// and the destination is on SM Y != X, then the destination must be a
	// comm task (handoff to the carrier) or the source must be a comm task
	// (delivery from the carrier).
	producedOn := map[string]int{}
	for _, n := range g.Nodes {
		if n.Kind == MPKTaskCompute {
			if sm, ok := p.Assign[n.ID]; ok {
				producedOn[n.ID] = sm
			}
		}
	}
	for _, e := range g.Edges {
		if e.Bytes <= 0 {
			continue
		}
		fromSM, okF := producedOn[e.From]
		toSM, okT := p.Assign[e.To]
		if !okF || !okT {
			continue
		}
		if fromSM == toSM {
			continue
		}
		if kindByID[e.From] != MPKTaskComm && kindByID[e.To] != MPKTaskComm {
			v.Violations = append(v.Violations, fmt.Sprintf("cross-SM tensor %q flows from compute task %q (SM %d) to task %q (SM %d) without a communication task", e.Tensor, e.From, fromSM, e.To, toSM))
		}
	}
	v.OK = len(v.Violations) == 0
	return v
}

// MPKValidateDescriptor checks the task-descriptor contract against a valid
// graph and partition: the descriptor's SM matches the partition assignment,
// its kind and operator match the graph node, and its input/output tensor
// references match the graph's edges (a communication task must reference
// exactly one input and one output tensor).
func MPKValidateDescriptor(g *MPKTaskGraph, p *MPKPartition, d *MPKTaskDescriptor) MPKValidationResult {
	var v MPKValidationResult
	if g == nil || p == nil || d == nil {
		v.Violations = []string{"nil graph, partition, or descriptor"}
		return v
	}
	var node *MPKTaskNode
	for i := range g.Nodes {
		if g.Nodes[i].ID == d.TaskID {
			node = &g.Nodes[i]
			break
		}
	}
	if node == nil {
		v.Violations = append(v.Violations, fmt.Sprintf("descriptor references unknown task %q", d.TaskID))
		return v
	}
	sm, ok := p.Assign[d.TaskID]
	if !ok {
		v.Violations = append(v.Violations, fmt.Sprintf("task %q is not assigned in the partition", d.TaskID))
		return v
	}
	if d.SM != sm {
		v.Violations = append(v.Violations, fmt.Sprintf("descriptor SM %d does not match partition assignment %d for task %q", d.SM, sm, d.TaskID))
	}
	if d.Kind != node.Kind {
		v.Violations = append(v.Violations, fmt.Sprintf("descriptor kind %v does not match graph kind %v for task %q", d.Kind, node.Kind, d.TaskID))
	}
	if d.Op != node.Op {
		v.Violations = append(v.Violations, fmt.Sprintf("descriptor op %q does not match graph op %q for task %q", d.Op, node.Op, d.TaskID))
	}
	inTensors := map[string]bool{}
	outTensors := map[string]bool{}
	for _, e := range g.Edges {
		if e.To == d.TaskID {
			inTensors[e.Tensor] = true
		}
		if e.From == d.TaskID {
			outTensors[e.Tensor] = true
		}
	}
	for _, t := range d.InTensors {
		if !inTensors[t] {
			v.Violations = append(v.Violations, fmt.Sprintf("descriptor input tensor %q is not produced by an edge into task %q", t, d.TaskID))
		}
	}
	for _, t := range d.OutTensors {
		if !outTensors[t] {
			v.Violations = append(v.Violations, fmt.Sprintf("descriptor output tensor %q is not consumed by an edge out of task %q", t, d.TaskID))
		}
	}
	if node.Kind == MPKTaskComm {
		if len(d.InTensors) != 1 {
			v.Violations = append(v.Violations, fmt.Sprintf("communication task %q must reference exactly one input tensor (got %d)", d.TaskID, len(d.InTensors)))
		}
		if len(d.OutTensors) != 1 {
			v.Violations = append(v.Violations, fmt.Sprintf("communication task %q must reference exactly one output tensor (got %d)", d.TaskID, len(d.OutTensors)))
		}
	}
	v.OK = len(v.Violations) == 0
	return v
}

// MPKTaskgraphOracle is the closed in-memory host oracle: it simulates the
// partitioned task graph deterministically and reports the makespan plus the
// compute-communication overlap count. It refuses to run on a graph or
// partition that fails validation.
func MPKTaskgraphOracle(g *MPKTaskGraph, p *MPKPartition) (MPKOracleReceipt, error) {
	gv := MPKValidateGraph(g)
	if !gv.OK {
		return MPKOracleReceipt{}, fmt.Errorf("%w: %s", ErrMPKTaskgraphInvalid, gv.Error())
	}
	pv := MPKValidatePartition(g, p)
	if !pv.OK {
		return MPKOracleReceipt{}, fmt.Errorf("%w: %s", ErrMPKTaskgraphInvalid, pv.Error())
	}
	kindByID := map[string]MPKTaskKind{}
	for _, n := range g.Nodes {
		kindByID[n.ID] = n.Kind
	}
	deps := map[string][]string{}
	dependents := map[string][]string{}
	indeg := map[string]int{}
	for _, n := range g.Nodes {
		indeg[n.ID] = 0
	}
	for _, e := range g.Edges {
		deps[e.To] = append(deps[e.To], e.From)
		dependents[e.From] = append(dependents[e.From], e.To)
		indeg[e.To]++
	}
	// Deterministic ready-queue simulation: each task occupies its assigned
	// SM for one time unit, starting no earlier than every dependency has
	// finished and no earlier than the SM is free.
	smNextFree := map[int]int{}
	start := map[string]int{}
	finish := map[string]int{}
	ready := make([]string, 0, len(g.Nodes))
	for _, n := range g.Nodes {
		if indeg[n.ID] == 0 {
			ready = append(ready, n.ID)
		}
	}
	scheduled := 0
	for len(ready) > 0 {
		sort.Strings(ready)
		id := ready[0]
		ready = ready[1:]
		sm := p.Assign[id]
		earliest := 0
		for _, d := range deps[id] {
			if finish[d] > earliest {
				earliest = finish[d]
			}
		}
		if smNextFree[sm] > earliest {
			earliest = smNextFree[sm]
		}
		start[id] = earliest
		finish[id] = earliest + 1
		smNextFree[sm] = earliest + 1
		scheduled++
		for _, to := range dependents[id] {
			indeg[to]--
			if indeg[to] == 0 {
				ready = append(ready, to)
			}
		}
	}
	makespan := 0
	for _, f := range finish {
		if f > makespan {
			makespan = f
		}
	}
	// Count compute-communication overlap: a comm task whose window
	// overlaps a compute task on a different SM.
	overlap := 0
	for _, n := range g.Nodes {
		if kindByID[n.ID] != MPKTaskComm {
			continue
		}
		commSM := p.Assign[n.ID]
		commStart, commFinish := start[n.ID], finish[n.ID]
		for _, m := range g.Nodes {
			if kindByID[m.ID] != MPKTaskCompute {
				continue
			}
			if p.Assign[m.ID] == commSM {
				continue
			}
			if start[m.ID] < commFinish && finish[m.ID] > commStart {
				overlap++
				break
			}
		}
	}
	return MPKOracleReceipt{Makespan: makespan, Tasks: scheduled, Overlap: overlap}, nil
}

// MPKBuildDemoGraph builds a small cross-SM pipeline: a producer compute
// task on SM 0, a communication task on SM 1, and a consumer compute task on
// SM 1. It is the canonical fixture for the oracle and descriptor tests.
func MPKBuildDemoGraph() (*MPKTaskGraph, *MPKPartition) {
	g := &MPKTaskGraph{
		Nodes: []MPKTaskNode{
			{ID: "producer", Kind: MPKTaskCompute, Op: "gemm", SMs: 1, Shared: 16384},
			{ID: "comm", Kind: MPKTaskComm, Op: "sm2sm", SMs: 1, Shared: 0},
			{ID: "consumer", Kind: MPKTaskCompute, Op: "gemm", SMs: 1, Shared: 16384},
		},
		Edges: []MPKTaskEdge{
			{From: "producer", To: "comm", Tensor: "act", Bytes: 4096},
			{From: "comm", To: "consumer", Tensor: "act", Bytes: 4096},
		},
	}
	p := &MPKPartition{
		Assign: map[string]int{"producer": 0, "comm": 1, "consumer": 1},
		SMs:    2,
	}
	return g, p
}
