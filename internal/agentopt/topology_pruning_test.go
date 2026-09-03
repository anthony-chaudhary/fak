package agentopt

import (
	"fmt"
	"testing"
)

// simulateGraphExecution simulates sequential workflow evaluation through a communication graph.
// It starts at the entry node, follows outgoing edges whose target agents are active, applies
// state transformations, and returns the final state reaching the terminal sink and the hop count.
func simulateGraphExecution(nodes map[string]AgentNode, edges []CommunicationEdge, startID, endID, initialPayload string) (string, int, error) {
	state := initialPayload
	current := startID
	hops := 0
	visited := make(map[string]bool)

	for current != endID {
		visited[current] = true
		currNode, exists := nodes[current]
		if !exists {
			return "", hops, fmt.Errorf("node %s not found in topology", current)
		}

		if currNode.TransformsState {
			state = state + "->" + currNode.Role + "{" + currNode.ID + "}"
		}

		// Locate next active outgoing link
		var nextHop string
		var foundLink bool
		for _, e := range edges {
			if e.Source == current && !visited[e.Target] {
				if _, targetExists := nodes[e.Target]; targetExists {
					nextHop = e.Target
					foundLink = true
					break
				}
			}
		}

		if !foundLink {
			return "", hops, fmt.Errorf("no forward link found from %s towards %s", current, endID)
		}

		current = nextHop
		hops++
	}

	// Apply final node transform if present
	if endNode, exists := nodes[endID]; exists && endNode.TransformsState {
		state = state + "->" + endNode.Role + "{" + endNode.ID + "}"
	}

	return state, hops, nil
}

func TestTopologyPruning(t *testing.T) {
	t.Run("demonstrates_redundant_hop_removal_preserving_output_semantics", func(t *testing.T) {
		pruner := NewTopologyPruner()

		// Construct multi-agent graph with:
		// - 4 functional agents (orchestrator, parser, compute, sink)
		// - 2 redundant intermediate passthrough hops (router1, router2)
		// - 3 low-gain noise edges (heartbeat, telemetry ping, keepalive)
		graph := NewCommunicationGraph()

		graph.AddNode(AgentNode{
			ID:              "agent_orchestrator",
			Role:            "orchestrator",
			TransformsState: true,
			FiltersActions:  true,
		})
		graph.AddNode(AgentNode{
			ID:              "agent_parser",
			Role:            "parser",
			TransformsState: true,
			FiltersActions:  false,
		})
		graph.AddNode(AgentNode{
			ID:              "agent_router_1",
			Role:            "router",
			TransformsState: false,
			FiltersActions:  false,
			IsPassthrough:   true,
		})
		graph.AddNode(AgentNode{
			ID:              "agent_router_2",
			Role:            "proxy",
			TransformsState: false,
			FiltersActions:  false,
			IsPassthrough:   true,
		})
		graph.AddNode(AgentNode{
			ID:              "agent_compute",
			Role:            "compute_engine",
			TransformsState: true,
			FiltersActions:  false,
		})
		graph.AddNode(AgentNode{
			ID:              "agent_sink",
			Role:            "result_sink",
			TransformsState: true,
			FiltersActions:  true,
		})

		// Essential workflow edges
		graph.AddEdge(CommunicationEdge{
			Source:           "agent_orchestrator",
			Target:           "agent_parser",
			MessageCount:     3,
			TransformedState: true,
			InformationGain:  0.85,
		})
		graph.AddEdge(CommunicationEdge{
			Source:           "agent_parser",
			Target:           "agent_router_1",
			MessageCount:     2,
			TransformedState: true,
			InformationGain:  0.75,
		})
		graph.AddEdge(CommunicationEdge{
			Source:           "agent_router_1",
			Target:           "agent_router_2",
			MessageCount:     2,
			TransformedState: false,
			InformationGain:  0.60,
		})
		graph.AddEdge(CommunicationEdge{
			Source:           "agent_router_2",
			Target:           "agent_compute",
			MessageCount:     2,
			TransformedState: false,
			InformationGain:  0.60,
		})
		graph.AddEdge(CommunicationEdge{
			Source:           "agent_compute",
			Target:           "agent_sink",
			MessageCount:     1,
			TransformedState: true,
			InformationGain:  0.90,
		})

		// Low-gain noise edges that do not transform state
		graph.AddEdge(CommunicationEdge{
			Source:           "agent_orchestrator",
			Target:           "agent_router_1",
			MessageCount:     1,
			TransformedState: false,
			InformationGain:  0.02, // low gain heartbeat
		})
		graph.AddEdge(CommunicationEdge{
			Source:           "agent_parser",
			Target:           "agent_sink",
			MessageCount:     1,
			TransformedState: false,
			InformationGain:  0.01, // uninformative telemetry
		})
		graph.AddEdge(CommunicationEdge{
			Source:           "agent_router_1",
			Target:           "agent_sink",
			MessageCount:     1,
			TransformedState: false,
			InformationGain:  0.03, // uninformative ping
		})

		// 1. Simulate execution on original uncompressed graph
		origOutput, origHops, err := simulateGraphExecution(
			graph.Nodes,
			graph.Edges,
			"agent_orchestrator",
			"agent_sink",
			"query_init",
		)
		if err != nil {
			t.Fatalf("failed to simulate uncompressed graph: %v", err)
		}
		if origHops != 5 {
			t.Fatalf("expected 5 hops in original graph, got %d", origHops)
		}

		// 2. Prune topology with minimum information gain threshold of 0.05
		prunedGraph, report := pruner.PruneTopology(*graph, 0.05)

		// 3. Verify node and edge reductions
		if report.OriginalNodesCount != 6 {
			t.Errorf("expected 6 original nodes, got %d", report.OriginalNodesCount)
		}
		if report.RemainingNodesCount != 4 {
			t.Errorf("expected 4 remaining nodes, got %d", report.RemainingNodesCount)
		}
		if report.PassthroughHopsEliminated != 2 {
			t.Errorf("expected 2 passthrough hops eliminated, got %d", report.PassthroughHopsEliminated)
		}
		if report.EdgesPrunedByInformationGain != 3 {
			t.Errorf("expected 3 edges pruned by information gain, got %d", report.EdgesPrunedByInformationGain)
		}

		// Passthrough routers must be eliminated
		if prunedGraph.HasNode("agent_router_1") {
			t.Errorf("expected agent_router_1 to be pruned")
		}
		if prunedGraph.HasNode("agent_router_2") {
			t.Errorf("expected agent_router_2 to be pruned")
		}

		// Functional agents must remain
		for _, expectedID := range []string{"agent_orchestrator", "agent_parser", "agent_compute", "agent_sink"} {
			if !prunedGraph.HasNode(expectedID) {
				t.Errorf("expected %s to be preserved in pruned topology", expectedID)
			}
		}

		// Direct bypass link must be established between parser and compute
		bypassLink, found := prunedGraph.FindEdge("agent_parser", "agent_compute")
		if !found {
			t.Fatalf("expected direct edge from agent_parser to agent_compute")
		}
		if !bypassLink.TransformedState {
			t.Errorf("expected bypass edge to preserve TransformedState = true")
		}
		if bypassLink.InformationGain < 0.75 {
			t.Errorf("expected bypass edge information gain >= 0.75, got %f", bypassLink.InformationGain)
		}

		// 4. Simulate execution on pruned topology and verify identical output semantics
		prunedOutput, prunedHops, err := simulateGraphExecution(
			prunedGraph.Nodes,
			prunedGraph.Edges,
			"agent_orchestrator",
			"agent_sink",
			"query_init",
		)
		if err != nil {
			t.Fatalf("failed to simulate pruned graph: %v", err)
		}

		// Output semantics must be perfectly preserved
		if origOutput != prunedOutput {
			t.Fatalf("output semantics diverged!\noriginal: %s\npruned:   %s", origOutput, prunedOutput)
		}

		// Hop count must be reduced from 5 to 3
		if prunedHops != 3 {
			t.Fatalf("expected 3 hops in pruned graph, got %d", prunedHops)
		}
		if report.ExecutionHopsSaved < 2 {
			t.Errorf("expected at least 2 execution hops saved, got %d", report.ExecutionHopsSaved)
		}
	})

	t.Run("preserves_action_filtering_nodes", func(t *testing.T) {
		pruner := NewTopologyPruner()
		graph := NewCommunicationGraph()

		graph.AddNode(AgentNode{
			ID:              "agent_producer",
			Role:            "producer",
			TransformsState: true,
			FiltersActions:  false,
		})
		// Security filter does not transform state, but filters actions!
		graph.AddNode(AgentNode{
			ID:              "agent_security_filter",
			Role:            "security_guardrail",
			TransformsState: false,
			FiltersActions:  true,
		})
		graph.AddNode(AgentNode{
			ID:              "agent_consumer",
			Role:            "consumer",
			TransformsState: true,
			FiltersActions:  false,
		})

		graph.AddEdge(CommunicationEdge{
			Source:           "agent_producer",
			Target:           "agent_security_filter",
			MessageCount:     2,
			TransformedState: true,
			InformationGain:  0.8,
		})
		graph.AddEdge(CommunicationEdge{
			Source:           "agent_security_filter",
			Target:           "agent_consumer",
			MessageCount:     2,
			TransformedState: false,
			InformationGain:  0.8,
		})

		prunedGraph, report := pruner.PruneTopology(*graph, 0.1)

		// Security filter must NOT be bypassed because FiltersActions == true
		if !prunedGraph.HasNode("agent_security_filter") {
			t.Errorf("agent_security_filter must be preserved because it filters actions")
		}
		if report.PassthroughHopsEliminated != 0 {
			t.Errorf("expected 0 passthrough hops eliminated, got %d", report.PassthroughHopsEliminated)
		}
	})

	t.Run("preserves_state_transforming_edges_despite_low_gain", func(t *testing.T) {
		pruner := NewTopologyPruner()
		graph := NewCommunicationGraph()

		graph.AddNode(AgentNode{
			ID:              "src",
			Role:            "source",
			TransformsState: true,
		})
		graph.AddNode(AgentNode{
			ID:              "dst",
			Role:            "destination",
			TransformsState: true,
		})

		// Low information gain, but alters state!
		graph.AddEdge(CommunicationEdge{
			Source:           "src",
			Target:           "dst",
			MessageCount:     1,
			TransformedState: true,
			InformationGain:  0.01,
		})

		prunedGraph, report := pruner.PruneTopology(*graph, 0.1)

		if !prunedGraph.HasEdge("src", "dst") {
			t.Errorf("state-transforming edge must be preserved despite low information gain")
		}
		if report.EdgesPrunedByInformationGain != 0 {
			t.Errorf("expected 0 edges pruned by information gain, got %d", report.EdgesPrunedByInformationGain)
		}
	})

	t.Run("multi_in_multi_out_passthrough_bypass", func(t *testing.T) {
		pruner := NewTopologyPruner()
		graph := NewCommunicationGraph()

		graph.AddNode(AgentNode{ID: "in1", Role: "producer1", TransformsState: true})
		graph.AddNode(AgentNode{ID: "in2", Role: "producer2", TransformsState: true})
		graph.AddNode(AgentNode{ID: "relay", Role: "router", IsPassthrough: true})
		graph.AddNode(AgentNode{ID: "out1", Role: "worker1", TransformsState: true})
		graph.AddNode(AgentNode{ID: "out2", Role: "worker2", TransformsState: true})

		graph.AddEdge(CommunicationEdge{Source: "in1", Target: "relay", InformationGain: 0.5, MessageCount: 2})
		graph.AddEdge(CommunicationEdge{Source: "in2", Target: "relay", InformationGain: 0.6, MessageCount: 3})
		graph.AddEdge(CommunicationEdge{Source: "relay", Target: "out1", InformationGain: 0.5, MessageCount: 2})
		graph.AddEdge(CommunicationEdge{Source: "relay", Target: "out2", InformationGain: 0.6, MessageCount: 3})

		prunedGraph, report := pruner.PruneTopology(*graph, 0.1)

		if prunedGraph.HasNode("relay") {
			t.Errorf("relay node must be eliminated")
		}
		if report.RemainingNodesCount != 4 {
			t.Errorf("expected 4 remaining nodes, got %d", report.RemainingNodesCount)
		}

		// Must connect each input to each output: in1->out1, in1->out2, in2->out1, in2->out2
		for _, src := range []string{"in1", "in2"} {
			for _, dst := range []string{"out1", "out2"} {
				if !prunedGraph.HasEdge(src, dst) {
					t.Errorf("expected bypass edge from %s to %s", src, dst)
				}
			}
		}
	})

	t.Run("utility_and_gain_metrics", func(t *testing.T) {
		edge1 := CommunicationEdge{
			Source:           "a",
			Target:           "b",
			InformationGain:  0.4,
			TransformedState: true,
		}
		edge2 := CommunicationEdge{
			Source:           "b",
			Target:           "c",
			InformationGain:  0.6,
			TransformedState: false,
		}

		if util1 := MeasureEdgeUtility(edge1); util1 != 1.4 {
			t.Errorf("expected utility 1.4, got %f", util1)
		}
		if util2 := MeasureEdgeUtility(edge2); util2 != 0.6 {
			t.Errorf("expected utility 0.6, got %f", util2)
		}

		edges := []CommunicationEdge{edge1, edge2}
		if totUtil := MeasureGraphUtility(edges); totUtil != 2.0 {
			t.Errorf("expected total utility 2.0, got %f", totUtil)
		}
		if avgGain := MeasureAverageInformationGain(edges); avgGain != 0.5 {
			t.Errorf("expected average gain 0.5, got %f", avgGain)
		}
	})

	t.Run("graph_pruner_interface_compliance", func(t *testing.T) {
		var pruner GraphPruner = NewGraphPruner()
		graph := NewCommunicationGraph()
		graph.AddNode(AgentNode{ID: "worker1", TransformsState: true})
		graph.AddNode(AgentNode{ID: "worker2", TransformsState: true})
		graph.AddEdge(CommunicationEdge{
			Source:           "worker1",
			Target:           "worker2",
			InformationGain:  0.9,
			TransformedState: true,
		})

		prunedGraph, report := pruner.PruneTopology(*graph, 0.1)
		if prunedGraph.NodeCount() != 2 {
			t.Errorf("expected 2 nodes, got %d", prunedGraph.NodeCount())
		}
		if report.RemainingEdgesCount != 1 {
			t.Errorf("expected 1 edge, got %d", report.RemainingEdgesCount)
		}
	})
}
