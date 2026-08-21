package agenttopo

import (
	"errors"
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/comm"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Ready reports that the agenttopo leaf is linked into the build.
func Ready() bool { return true }

var (
	// ErrNilGroup is returned when a topology is declared without a comm.Group.
	ErrNilGroup = errors.New("agenttopo: nil group")
	// ErrNilTopology is returned when an expansion has no valid base topology.
	ErrNilTopology = errors.New("agenttopo: nil topology")
	// ErrNoMember is returned when a node or edge endpoint is not in the group.
	ErrNoMember = errors.New("agenttopo: no such member")
	// ErrInvalidNodeID is returned when a proposed node has no stable identity.
	ErrInvalidNodeID = errors.New("agenttopo: empty node identity")
	// ErrIdentityRewrite is returned when a proposal re-declares an existing node.
	ErrIdentityRewrite = errors.New("agenttopo: existing node identity rewrite")
	// ErrDuplicateNode is returned when a proposal repeats a new node identity.
	ErrDuplicateNode = errors.New("agenttopo: duplicate proposed node")
	// ErrDuplicateEdge is returned when an explicit declaration repeats an edge.
	ErrDuplicateEdge = errors.New("agenttopo: duplicate edge")
	// ErrCycle is returned when a declared edge set is not a DAG.
	ErrCycle = errors.New("agenttopo: topology contains a cycle")
	// ErrInvalidLimits is returned unless every expansion bound is positive.
	ErrInvalidLimits = errors.New("agenttopo: expansion limits must be positive")
	// ErrExpansionLimit is matched by every typed expansion limit refusal.
	ErrExpansionLimit = errors.New("agenttopo: expansion limit exceeded")
	// ErrDepthLimit identifies a maximum-depth refusal.
	ErrDepthLimit = errors.New("agenttopo: expansion depth limit exceeded")
	// ErrWidthLimit identifies a maximum-width refusal.
	ErrWidthLimit = errors.New("agenttopo: expansion width limit exceeded")
	// ErrTotalNodesLimit identifies a maximum-total-nodes refusal.
	ErrTotalNodesLimit = errors.New("agenttopo: expansion total-nodes limit exceeded")
	// ErrNoNeighborOutput is returned when CombineIn lacks an in-neighbor output.
	ErrNoNeighborOutput = errors.New("agenttopo: missing neighbor output")
)

// Edge is a directed handoff from one member to another.
type Edge struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// ExpansionProposal is an additive graph delta. Nodes must be new identities;
// edges may connect any base or proposed node.
type ExpansionProposal struct {
	Nodes []comm.Member `json:"nodes"`
	Edges []Edge        `json:"edges"`
}

// ExpansionLimits bounds the admitted topology. Depth counts nodes on the
// longest path; width counts nodes in the largest longest-path depth layer.
type ExpansionLimits struct {
	MaxDepth      int `json:"max_depth"`
	MaxWidth      int `json:"max_width"`
	MaxTotalNodes int `json:"max_total_nodes"`
}

// ExpansionLimit names a closed expansion limit class.
type ExpansionLimit string

const (
	LimitDepth      ExpansionLimit = "depth"
	LimitWidth      ExpansionLimit = "width"
	LimitTotalNodes ExpansionLimit = "total_nodes"
)

// ExpansionLimitError reports the exact bound that refused an expansion.
type ExpansionLimitError struct {
	Limit  ExpansionLimit `json:"limit"`
	Actual int            `json:"actual"`
	Max    int            `json:"max"`
}

func (e *ExpansionLimitError) Error() string {
	return fmt.Sprintf("%v: %s=%d max=%d", ErrExpansionLimit, e.Limit, e.Actual, e.Max)
}

// Unwrap makes both the generic and limit-specific refusal match errors.Is.
func (e *ExpansionLimitError) Unwrap() []error {
	return []error{ErrExpansionLimit, expansionLimitSentinel(e.Limit)}
}

// Exchange is the declared neighborhood of one node. In and Out preserve the
// edge declaration order, not rank order, so reducers receive a stable caller
// chosen ordering.
type Exchange struct {
	Node comm.Member   `json:"node"`
	In   []comm.Member `json:"in"`
	Out  []comm.Member `json:"out"`
}

// Topology is a named DAG over a deterministic comm.Group.
type Topology struct {
	name    string
	group   *comm.Group
	edges   []Edge
	members map[string]comm.Member
	in      map[string][]string
	out     map[string][]string
}

// Declare validates and stores an explicit directed acyclic topology over group.
func Declare(name string, group *comm.Group, edges []Edge) (*Topology, error) {
	if group == nil {
		return nil, ErrNilGroup
	}
	if group.Size() == 0 {
		return nil, comm.ErrEmptyGroup
	}
	members := make(map[string]comm.Member, group.Size())
	for _, m := range membersOf(group) {
		members[m.ID] = m
	}

	t := &Topology{
		name:    name,
		group:   group,
		edges:   make([]Edge, 0, len(edges)),
		members: members,
		in:      make(map[string][]string, group.Size()),
		out:     make(map[string][]string, group.Size()),
	}
	seen := make(map[Edge]struct{}, len(edges))
	for _, e := range edges {
		if _, ok := members[e.From]; !ok {
			return nil, fmt.Errorf("%w: edge from %q", ErrNoMember, e.From)
		}
		if _, ok := members[e.To]; !ok {
			return nil, fmt.Errorf("%w: edge to %q", ErrNoMember, e.To)
		}
		if e.From == e.To {
			return nil, fmt.Errorf("%w: self edge %q", ErrCycle, e.From)
		}
		if _, ok := seen[e]; ok {
			return nil, fmt.Errorf("%w: %s->%s", ErrDuplicateEdge, e.From, e.To)
		}
		seen[e] = struct{}{}
		t.edges = append(t.edges, e)
		t.out[e.From] = append(t.out[e.From], e.To)
		t.in[e.To] = append(t.in[e.To], e.From)
	}
	if err := t.validateAcyclic(); err != nil {
		return nil, err
	}
	return t, nil
}

// AdmitExpansion validates an additive proposal and returns a new topology.
// The base and proposal are never mutated, including on refusal.
func AdmitExpansion(base *Topology, proposal ExpansionProposal, limits ExpansionLimits) (*Topology, error) {
	if base == nil || base.group == nil || base.group.Size() == 0 {
		return nil, ErrNilTopology
	}
	if limits.MaxDepth <= 0 || limits.MaxWidth <= 0 || limits.MaxTotalNodes <= 0 {
		return nil, fmt.Errorf("%w: depth=%d width=%d total_nodes=%d", ErrInvalidLimits, limits.MaxDepth, limits.MaxWidth, limits.MaxTotalNodes)
	}

	proposedNodes := append([]comm.Member(nil), proposal.Nodes...)
	sort.Slice(proposedNodes, func(i, j int) bool { return proposedNodes[i].ID < proposedNodes[j].ID })
	seenNodes := make(map[string]struct{}, len(proposedNodes))
	allNodes := append([]comm.Member(nil), base.Nodes()...)
	for _, node := range proposedNodes {
		if node.ID == "" {
			return nil, ErrInvalidNodeID
		}
		if _, exists := base.members[node.ID]; exists {
			return nil, fmt.Errorf("%w: %q", ErrIdentityRewrite, node.ID)
		}
		if _, duplicate := seenNodes[node.ID]; duplicate {
			return nil, fmt.Errorf("%w: %q", ErrDuplicateNode, node.ID)
		}
		seenNodes[node.ID] = struct{}{}
		allNodes = append(allNodes, node)
	}

	proposedEdges := append([]Edge(nil), proposal.Edges...)
	sort.Slice(proposedEdges, func(i, j int) bool {
		if proposedEdges[i].From != proposedEdges[j].From {
			return proposedEdges[i].From < proposedEdges[j].From
		}
		return proposedEdges[i].To < proposedEdges[j].To
	})
	allEdges := append(base.Edges(), proposedEdges...)

	membership, err := base.group.Membership(0)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNilTopology, err)
	}
	group, err := comm.New(base.group.WaveID(), membership.ParentTraceID, allNodes)
	if err != nil {
		return nil, err
	}
	expanded, err := Declare(base.name, group, allEdges)
	if err != nil {
		return nil, err
	}
	if expanded.Size() > limits.MaxTotalNodes {
		return nil, newExpansionLimitError(LimitTotalNodes, expanded.Size(), limits.MaxTotalNodes)
	}
	depth, width := expanded.dimensions()
	if depth > limits.MaxDepth {
		return nil, newExpansionLimitError(LimitDepth, depth, limits.MaxDepth)
	}
	if width > limits.MaxWidth {
		return nil, newExpansionLimitError(LimitWidth, width, limits.MaxWidth)
	}
	return expanded, nil
}

// Linear declares rank(i) -> rank(i+1) edges over the group.
func Linear(name string, group *comm.Group) (*Topology, error) {
	ms := membersOf(group)
	edges := make([]Edge, 0, max(0, len(ms)-1))
	for i := 0; i+1 < len(ms); i++ {
		edges = append(edges, Edge{From: ms[i].ID, To: ms[i+1].ID})
	}
	return Declare(name, group, edges)
}

// Star declares root -> every other member in rank order.
func Star(name string, group *comm.Group, root string) (*Topology, error) {
	if group == nil {
		return nil, ErrNilGroup
	}
	if _, err := group.Rank(root); err != nil {
		return nil, fmt.Errorf("%w: root %q", ErrNoMember, root)
	}
	edges := make([]Edge, 0, max(0, group.Size()-1))
	for _, m := range membersOf(group) {
		if m.ID == root {
			continue
		}
		edges = append(edges, Edge{From: root, To: m.ID})
	}
	return Declare(name, group, edges)
}

// Name is the topology's declaration name.
func (t *Topology) Name() string {
	if t == nil {
		return ""
	}
	return t.name
}

// Size is the number of nodes.
func (t *Topology) Size() int {
	if t == nil || t.group == nil {
		return 0
	}
	return t.group.Size()
}

// Edges returns the declared edge list in declaration order.
func (t *Topology) Edges() []Edge {
	if t == nil {
		return nil
	}
	return append([]Edge(nil), t.edges...)
}

// Nodes returns members in comm.Group rank order.
func (t *Topology) Nodes() []comm.Member {
	if t == nil || t.group == nil {
		return nil
	}
	return membersOf(t.group)
}

// LaneCount returns the number of distinct non-empty lanes represented by nodes.
func (t *Topology) LaneCount() int {
	seen := map[string]struct{}{}
	for _, m := range t.Nodes() {
		if m.Lane == "" {
			continue
		}
		seen[m.Lane] = struct{}{}
	}
	if len(seen) == 0 && t.Size() > 0 {
		return 1
	}
	return len(seen)
}

// NeighborExchange returns node's declared in/out-neighbors in edge declaration
// order.
func (t *Topology) NeighborExchange(node string) (Exchange, error) {
	if t == nil || t.group == nil {
		return Exchange{}, ErrNilGroup
	}
	m, ok := t.members[node]
	if !ok {
		return Exchange{}, fmt.Errorf("%w: %q", ErrNoMember, node)
	}
	ex := Exchange{Node: m}
	for _, id := range t.in[node] {
		ex.In = append(ex.In, t.members[id])
	}
	for _, id := range t.out[node] {
		ex.Out = append(ex.Out, t.members[id])
	}
	return ex, nil
}

// CombineIn folds a node's declared in-neighbor outputs through modelroute.Combine.
// outputs is keyed by neighbor member ID; votes are passed to Combine in the
// declaration order returned by NeighborExchange.
func (t *Topology) CombineIn(node string, outputs map[string]string, reduce modelroute.Reduction) (modelroute.Result, error) {
	ex, err := t.NeighborExchange(node)
	if err != nil {
		return modelroute.Result{}, err
	}
	if len(ex.In) == 0 {
		return modelroute.Result{}, fmt.Errorf("%w: %q has no in-neighbors", ErrNoMember, node)
	}
	votes := make([]modelroute.Vote, 0, len(ex.In))
	for _, m := range ex.In {
		out, ok := outputs[m.ID]
		if !ok {
			return modelroute.Result{}, fmt.Errorf("%w: %s->%s", ErrNoNeighborOutput, m.ID, node)
		}
		votes = append(votes, modelroute.Vote{
			Member: modelroute.Member{Model: m.ID, Weight: m.Weight, Role: m.Lane},
			Output: out,
		})
	}
	return modelroute.Combine(reduce, votes)
}

func (t *Topology) validateAcyclic() error {
	color := make(map[string]uint8, len(t.members))
	var visit func(string) error
	visit = func(id string) error {
		switch color[id] {
		case 1:
			return fmt.Errorf("%w: back-edge at %q", ErrCycle, id)
		case 2:
			return nil
		}
		color[id] = 1
		for _, next := range t.out[id] {
			if err := visit(next); err != nil {
				return err
			}
		}
		color[id] = 2
		return nil
	}

	ids := make([]string, 0, len(t.members))
	for id := range t.members {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		if color[id] == 0 {
			if err := visit(id); err != nil {
				return err
			}
		}
	}
	return nil
}

func (t *Topology) dimensions() (int, int) {
	depths := make(map[string]int, len(t.members))
	var depthOf func(string) int
	depthOf = func(id string) int {
		if depth := depths[id]; depth != 0 {
			return depth
		}
		depth := 1
		for _, parent := range t.in[id] {
			depth = max(depth, depthOf(parent)+1)
		}
		depths[id] = depth
		return depth
	}

	layers := make(map[int]int)
	maxDepth, maxWidth := 0, 0
	for _, member := range t.Nodes() {
		depth := depthOf(member.ID)
		layers[depth]++
		maxDepth = max(maxDepth, depth)
		maxWidth = max(maxWidth, layers[depth])
	}
	return maxDepth, maxWidth
}

func newExpansionLimitError(limit ExpansionLimit, actual, max int) error {
	return &ExpansionLimitError{Limit: limit, Actual: actual, Max: max}
}

func expansionLimitSentinel(limit ExpansionLimit) error {
	switch limit {
	case LimitDepth:
		return ErrDepthLimit
	case LimitWidth:
		return ErrWidthLimit
	case LimitTotalNodes:
		return ErrTotalNodesLimit
	default:
		return ErrExpansionLimit
	}
}

func membersOf(g *comm.Group) []comm.Member {
	if g == nil {
		return nil
	}
	out := make([]comm.Member, 0, g.Size())
	for r := 0; r < g.Size(); r++ {
		m, err := g.Member(r)
		if err == nil {
			out = append(out, m)
		}
	}
	return out
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
