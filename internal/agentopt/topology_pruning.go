package agentopt

import (
	"sort"
)

// Family 11: Multi-agent orchestration patterns.
//
// Topology pruning optimizes agent communication graphs by identifying and
// eliminating redundant intermediate hops (passthrough nodes that neither
// transform state nor filter actions) and pruning low-gain communication
// edges while preserving overall output semantics.

// AgentNode models an agent participating in a multi-agent orchestration workflow.
type AgentNode struct {
	ID              string         `json:"id"`
	Role            string         `json:"role,omitempty"`
	TransformsState bool           `json:"transforms_state"`
	FiltersActions  bool           `json:"filters_actions"`
	IsPassthrough   bool           `json:"is_passthrough,omitempty"`
	Metadata        map[string]any `json:"metadata,omitempty"`
}

// CanBypass indicates whether the node functions as an uninformative intermediary
// that neither alters state nor filters downstream actions.
func (n AgentNode) CanBypass() bool {
	if n.IsPassthrough {
		return true
	}
	return !n.TransformsState && !n.FiltersActions
}

// CommunicationEdge represents a directed message or state transfer between two agents.
type CommunicationEdge struct {
	Source           string         `json:"source"`
	Target           string         `json:"target"`
	MessageCount     int            `json:"message_count"`
	TransformedState bool           `json:"transformed_state"`
	InformationGain  float64        `json:"information_gain"`
	Metadata         map[string]any `json:"metadata,omitempty"`
}

// Utility measures the overall utility score for this communication link.
// State-transforming edges receive a utility premium over information gain.
func (e CommunicationEdge) Utility() float64 {
	score := e.InformationGain
	if e.TransformedState {
		score += 1.0
	}
	return score
}

// MeasureEdgeUtility returns the evaluated communication edge utility.
func MeasureEdgeUtility(e CommunicationEdge) float64 {
	return e.Utility()
}

// MeasureGraphUtility computes the cumulative utility of a slice of communication edges.
func MeasureGraphUtility(edges []CommunicationEdge) float64 {
	var total float64
	for _, e := range edges {
		total += MeasureEdgeUtility(e)
	}
	return total
}

// MeasureAverageInformationGain computes the mean information gain across edges.
func MeasureAverageInformationGain(edges []CommunicationEdge) float64 {
	if len(edges) == 0 {
		return 0.0
	}
	var total float64
	for _, e := range edges {
		total += e.InformationGain
	}
	return total / float64(len(edges))
}

// CommunicationGraph models the full interaction graph of a multi-agent system.
type CommunicationGraph struct {
	Nodes map[string]AgentNode `json:"nodes"`
	Edges []CommunicationEdge  `json:"edges"`
}

// NewCommunicationGraph constructs an empty communication graph.
func NewCommunicationGraph() *CommunicationGraph {
	return &CommunicationGraph{
		Nodes: make(map[string]AgentNode),
		Edges: make([]CommunicationEdge, 0),
	}
}

// AddNode registers an agent node in the communication graph.
func (g *CommunicationGraph) AddNode(node AgentNode) {
	if g.Nodes == nil {
		g.Nodes = make(map[string]AgentNode)
	}
	g.Nodes[node.ID] = node
}

// AddEdge registers a directed communication link in the graph.
func (g *CommunicationGraph) AddEdge(edge CommunicationEdge) {
	g.Edges = append(g.Edges, edge)
}

// TotalUtility calculates the sum of utilities for all edges in the graph.
func (g *CommunicationGraph) TotalUtility() float64 {
	return MeasureGraphUtility(g.Edges)
}

// AverageInformationGain returns the average information gain across all edges in the graph.
func (g *CommunicationGraph) AverageInformationGain() float64 {
	return MeasureAverageInformationGain(g.Edges)
}

// EdgeBypass records an eliminated intermediate hop where upstream and downstream nodes
// are connected directly across a bypassed passthrough agent.
type EdgeBypass struct {
	BypassedNodeID          string  `json:"bypassed_node_id"`
	Source                  string  `json:"source"`
	Target                  string  `json:"target"`
	CombinedInformationGain float64 `json:"combined_information_gain"`
	CombinedMessageCount    int     `json:"combined_message_count"`
	TransformedState        bool    `json:"transformed_state"`
	Reason                  string  `json:"reason"`
}

// CompressedTopology represents the pruned and optimized multi-agent topology.
type CompressedTopology struct {
	Nodes    map[string]AgentNode `json:"nodes"`
	Edges    []CommunicationEdge  `json:"edges"`
	Bypasses []EdgeBypass         `json:"bypasses,omitempty"`
}

// PrunedGraph is the resulting graph topology following topology pruning.
type PrunedGraph = CompressedTopology

// HasNode checks whether a given node exists in the topology.
func (ct *CompressedTopology) HasNode(id string) bool {
	if ct.Nodes == nil {
		return false
	}
	_, ok := ct.Nodes[id]
	return ok
}

// HasEdge checks whether a directed edge from source to target exists.
func (ct *CompressedTopology) HasEdge(source, target string) bool {
	for _, e := range ct.Edges {
		if e.Source == source && e.Target == target {
			return true
		}
	}
	return false
}

// FindEdge locates an edge between source and target, if present.
func (ct *CompressedTopology) FindEdge(source, target string) (CommunicationEdge, bool) {
	for _, e := range ct.Edges {
		if e.Source == source && e.Target == target {
			return e, true
		}
	}
	return CommunicationEdge{}, false
}

// NodeCount returns the number of active nodes in the compressed topology.
func (ct *CompressedTopology) NodeCount() int {
	return len(ct.Nodes)
}

// EdgeCount returns the number of active edges in the compressed topology.
func (ct *CompressedTopology) EdgeCount() int {
	return len(ct.Edges)
}

// TotalUtility calculates the combined utility for all edges in the compressed topology.
func (ct *CompressedTopology) TotalUtility() float64 {
	return MeasureGraphUtility(ct.Edges)
}

// AverageInformationGain returns the average information gain across all edges in the compressed topology.
func (ct *CompressedTopology) AverageInformationGain() float64 {
	return MeasureAverageInformationGain(ct.Edges)
}

// PruneReport summarizes the reduction metrics, eliminated nodes, and bypassed hops.
type PruneReport struct {
	OriginalNodesCount           int                 `json:"original_nodes_count"`
	PrunedNodesCount             int                 `json:"pruned_nodes_count"`
	RemainingNodesCount          int                 `json:"remaining_nodes_count"`
	OriginalEdgesCount           int                 `json:"original_edges_count"`
	PrunedEdgesCount             int                 `json:"pruned_edges_count"`
	RemainingEdgesCount          int                 `json:"remaining_edges_count"`
	RemovedNodes                 []string            `json:"removed_nodes"`
	RemovedEdges                 []CommunicationEdge `json:"removed_edges"`
	Bypasses                     []EdgeBypass        `json:"bypasses"`
	EdgesPrunedByInformationGain int                 `json:"edges_pruned_by_information_gain"`
	PassthroughHopsEliminated    int                 `json:"passthrough_hops_eliminated"`
	ExecutionHopsSaved           int                 `json:"execution_hops_saved"`
	InitialUtility               float64             `json:"initial_utility"`
	FinalUtility                 float64             `json:"final_utility"`
}

// GraphPruner defines the interface for topology reduction on multi-agent communication structures.
type GraphPruner interface {
	PruneTopology(graph CommunicationGraph, minInformationGain float64) (PrunedGraph, PruneReport)
}

// TopologyPruner eliminates redundant hops and low-gain edges across agent communication graphs.
type TopologyPruner struct {
	PreserveTerminalNodes bool
}

// NewTopologyPruner creates a default topology pruner.
func NewTopologyPruner() *TopologyPruner {
	return &TopologyPruner{
		PreserveTerminalNodes: true,
	}
}

// NewGraphPruner creates a default GraphPruner instance.
func NewGraphPruner() GraphPruner {
	return NewTopologyPruner()
}

// PruneTopology prunes redundant communication edges and collapses intermediate passthrough hops.
func (p *TopologyPruner) PruneTopology(graph CommunicationGraph, minInformationGain float64) (PrunedGraph, PruneReport) {
	report := PruneReport{
		OriginalNodesCount: len(graph.Nodes),
		OriginalEdgesCount: len(graph.Edges),
		RemovedNodes:       make([]string, 0),
		RemovedEdges:       make([]CommunicationEdge, 0),
		Bypasses:           make([]EdgeBypass, 0),
		InitialUtility:     MeasureGraphUtility(graph.Edges),
	}

	activeNodes := make(map[string]AgentNode, len(graph.Nodes))
	for k, v := range graph.Nodes {
		activeNodes[k] = v
	}

	// Step 1: Filter out low information gain edges that do not transform state.
	// State-transforming edges are always preserved to guarantee output semantics.
	activeEdges := make([]CommunicationEdge, 0, len(graph.Edges))
	for _, e := range graph.Edges {
		if !e.TransformedState && e.InformationGain < minInformationGain {
			report.RemovedEdges = append(report.RemovedEdges, e)
			report.EdgesPrunedByInformationGain++
			continue
		}
		activeEdges = append(activeEdges, e)
	}

	// Step 2: Iteratively collapse and bypass intermediate passthrough nodes.
	bypassedSet := make(map[string]bool)

	for {
		var eligibleNodeIDs []string
		for id, node := range activeNodes {
			if bypassedSet[id] || !node.CanBypass() {
				continue
			}
			var inCount, outCount int
			for _, e := range activeEdges {
				if e.Target == id && e.Source != id {
					inCount++
				}
				if e.Source == id && e.Target != id {
					outCount++
				}
			}
			if inCount > 0 && outCount > 0 {
				eligibleNodeIDs = append(eligibleNodeIDs, id)
			}
		}

		if len(eligibleNodeIDs) == 0 {
			break
		}

		sort.Strings(eligibleNodeIDs)
		targetID := eligibleNodeIDs[0]
		bypassedSet[targetID] = true

		var inEdges []CommunicationEdge
		var outEdges []CommunicationEdge
		var otherEdges []CommunicationEdge

		for _, e := range activeEdges {
			if e.Target == targetID && e.Source != targetID {
				inEdges = append(inEdges, e)
			} else if e.Source == targetID && e.Target != targetID {
				outEdges = append(outEdges, e)
			} else if e.Source != targetID && e.Target != targetID {
				otherEdges = append(otherEdges, e)
			} else {
				// Self-cycle on bypassed node
				report.RemovedEdges = append(report.RemovedEdges, e)
			}
		}

		for _, inE := range inEdges {
			report.RemovedEdges = append(report.RemovedEdges, inE)
		}
		for _, outE := range outEdges {
			report.RemovedEdges = append(report.RemovedEdges, outE)
		}

		bypassEdgesMap := make(map[string]CommunicationEdge)

		for _, inE := range inEdges {
			for _, outE := range outEdges {
				src := inE.Source
				dst := outE.Target
				if src == dst {
					continue // avoid self-cycle
				}

				combinedGain := inE.InformationGain
				if outE.InformationGain > combinedGain {
					combinedGain = outE.InformationGain
				}

				combinedCount := inE.MessageCount
				if outE.MessageCount > combinedCount {
					combinedCount = outE.MessageCount
				}

				combinedState := inE.TransformedState || outE.TransformedState

				key := src + "->" + dst
				if existing, exists := bypassEdgesMap[key]; exists {
					existing.MessageCount += combinedCount
					if combinedGain > existing.InformationGain {
						existing.InformationGain = combinedGain
					}
					existing.TransformedState = existing.TransformedState || combinedState
					bypassEdgesMap[key] = existing
				} else {
					bypassEdgesMap[key] = CommunicationEdge{
						Source:           src,
						Target:           dst,
						MessageCount:     combinedCount,
						TransformedState: combinedState,
						InformationGain:  combinedGain,
					}
				}

				bypass := EdgeBypass{
					BypassedNodeID:          targetID,
					Source:                  src,
					Target:                  dst,
					CombinedInformationGain: combinedGain,
					CombinedMessageCount:    combinedCount,
					TransformedState:        combinedState,
					Reason:                  "eliminated redundant worker hop lacking state transformation and action filtering",
				}
				report.Bypasses = append(report.Bypasses, bypass)
				report.ExecutionHopsSaved++
			}
		}

		// Rebuild activeEdges: merge bypass edges into otherEdges
		activeEdges = otherEdges
		for _, bEdge := range bypassEdgesMap {
			merged := false
			for i := range activeEdges {
				if activeEdges[i].Source == bEdge.Source && activeEdges[i].Target == bEdge.Target {
					activeEdges[i].MessageCount += bEdge.MessageCount
					if bEdge.InformationGain > activeEdges[i].InformationGain {
						activeEdges[i].InformationGain = bEdge.InformationGain
					}
					activeEdges[i].TransformedState = activeEdges[i].TransformedState || bEdge.TransformedState
					merged = true
					break
				}
			}
			if !merged {
				activeEdges = append(activeEdges, bEdge)
			}
		}

		delete(activeNodes, targetID)
		report.RemovedNodes = append(report.RemovedNodes, targetID)
		report.PassthroughHopsEliminated++
	}

	sort.Slice(activeEdges, func(i, j int) bool {
		if activeEdges[i].Source != activeEdges[j].Source {
			return activeEdges[i].Source < activeEdges[j].Source
		}
		return activeEdges[i].Target < activeEdges[j].Target
	})

	report.PrunedNodesCount = len(report.RemovedNodes)
	report.RemainingNodesCount = len(activeNodes)
	report.PrunedEdgesCount = len(report.RemovedEdges)
	report.RemainingEdgesCount = len(activeEdges)
	report.FinalUtility = MeasureGraphUtility(activeEdges)

	prunedGraph := CompressedTopology{
		Nodes:    activeNodes,
		Edges:    activeEdges,
		Bypasses: report.Bypasses,
	}

	return prunedGraph, report
}
