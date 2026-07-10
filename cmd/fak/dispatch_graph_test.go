package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// lanedRoute is a routed candidate in lane "L" declaring the given prerequisites.
func lanedRoute(num int, blockedBy ...string) dispatchtick.IssueRoute {
	return dispatchtick.IssueRoute{Number: num, Title: "n", Lane: "L", ExpectedSteps: 1, BlockedBy: blockedBy}
}

func graphPayload(issues ...dispatchtick.IssueRoute) dispatchtick.RouterPayload {
	lanes := map[string]dispatchtick.RouterLaneGroup{}
	nums := make([]int, 0, len(issues))
	for _, iss := range issues {
		nums = append(nums, iss.Number)
	}
	lanes["L"] = dispatchtick.RouterLaneGroup{Count: len(nums), StepBudget: len(nums), Issues: nums}
	return dispatchtick.RouterPayload{
		Issues: issues,
		Lanes:  lanes,
		Counts: dispatchtick.RouterCounts{Open: len(nums), Routed: len(nums), SkippedByReason: map[string]int{}},
	}
}

// TestBuildDispatchGraphChain pins the linear case: A<-B<-C renders one chain rooted at the single
// dispatchable node, with the two dependents nested in order.
func TestBuildDispatchGraphChain(t *testing.T) {
	g := buildDispatchGraph(graphPayload(
		lanedRoute(101),        // root
		lanedRoute(102, "101"), // waits on 101
		lanedRoute(104, "102"), // waits on 102
	))
	if len(g.Roots) != 1 || g.Roots[0] != 101 {
		t.Fatalf("root should be exactly #101, got %v", g.Roots)
	}
	if g.Counts.Blocked != 2 {
		t.Errorf("2 dependents should be blocked, got %d", g.Counts.Blocked)
	}
	if len(g.Chains) != 1 {
		t.Fatalf("want one chain, got %d", len(g.Chains))
	}
	top := g.Chains[0]
	if top.Issue != 101 || len(top.Blocks) != 1 || top.Blocks[0].Issue != 102 {
		t.Fatalf("chain should be 101 -> 102, got %+v", top)
	}
	if len(top.Blocks[0].Blocks) != 1 || top.Blocks[0].Blocks[0].Issue != 104 {
		t.Fatalf("chain should descend 102 -> 104, got %+v", top.Blocks[0])
	}
	if len(g.Cycles) != 0 {
		t.Errorf("a linear chain has no cycles, got %v", g.Cycles)
	}
}

// TestBuildDispatchGraphDiamond pins the converging case: D waits on both B and C, which both wait on
// A. D must appear exactly once (diamonds converge, not duplicate) and record both its prerequisites.
func TestBuildDispatchGraphDiamond(t *testing.T) {
	g := buildDispatchGraph(graphPayload(
		lanedRoute(101),               // A root
		lanedRoute(102, "101"),        // B
		lanedRoute(103, "101"),        // C
		lanedRoute(104, "102", "103"), // D waits on B and C
	))
	if len(g.Roots) != 1 || g.Roots[0] != 101 {
		t.Fatalf("root should be exactly #101, got %v", g.Roots)
	}
	// D (#104) is recorded once in the edges with both open prerequisites.
	var dEdge *dispatchGraphEdge
	for i := range g.Edges {
		if g.Edges[i].Issue == 104 {
			dEdge = &g.Edges[i]
		}
	}
	if dEdge == nil || len(dEdge.BlockedBy) != 2 || dEdge.BlockedBy[0] != 102 || dEdge.BlockedBy[1] != 103 {
		t.Fatalf("#104 edge should record both prerequisites #102,#103, got %+v", dEdge)
	}
	// #104 appears exactly once across the whole chain forest (no duplication under both parents).
	if n := countChainNode(g.Chains, 104); n != 1 {
		t.Fatalf("diamond apex #104 must appear exactly once in the forest, appeared %d times", n)
	}
}

// TestBuildDispatchGraphCycle pins the 3-cycle: the engine breaks only 2-cycles, so a 3-cycle leaves
// all three blocked (no root, no chain) and must surface in the dedicated cycles section.
func TestBuildDispatchGraphCycle(t *testing.T) {
	g := buildDispatchGraph(graphPayload(
		lanedRoute(201, "203"),
		lanedRoute(202, "201"),
		lanedRoute(203, "202"),
	))
	if len(g.Roots) != 0 {
		t.Errorf("a pure 3-cycle has no dispatchable root, got roots %v", g.Roots)
	}
	if len(g.Chains) != 0 {
		t.Errorf("no root can reach a 3-cycle, so there are no chains, got %d", len(g.Chains))
	}
	if len(g.Cycles) != 1 {
		t.Fatalf("want exactly one cycle, got %v", g.Cycles)
	}
	cyc := g.Cycles[0]
	if len(cyc) != 3 || cyc[0] != 201 || cyc[1] != 202 || cyc[2] != 203 {
		t.Fatalf("cycle should be the sorted set {201,202,203}, got %v", cyc)
	}
}

// TestBuildDispatchGraphUnroutedNotRoot pins that an unblocked but UNROUTED issue (no lane) is not a
// root -- "root" means dispatchable this tick, and an unrouted issue is not dispatchable.
func TestBuildDispatchGraphUnroutedNotRoot(t *testing.T) {
	g := buildDispatchGraph(dispatchtick.RouterPayload{
		Issues: []dispatchtick.IssueRoute{
			{Number: 301, Title: "unrouted", Lane: "", ExpectedSteps: 1}, // no lane -> not dispatchable
			lanedRoute(302),
		},
		Lanes:  map[string]dispatchtick.RouterLaneGroup{"L": {Count: 1, StepBudget: 1, Issues: []int{302}}},
		Counts: dispatchtick.RouterCounts{Open: 2, Routed: 1, SkippedByReason: map[string]int{}},
	})
	if len(g.Roots) != 1 || g.Roots[0] != 302 {
		t.Fatalf("only the laned #302 is a root, got %v", g.Roots)
	}
}

// countChainNode counts how many times an issue number appears across the chain forest.
func countChainNode(nodes []dispatchGraphNode, num int) int {
	n := 0
	for _, node := range nodes {
		if node.Issue == num {
			n++
		}
		n += countChainNode(node.Blocks, num)
	}
	return n
}
