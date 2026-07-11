package main

// dispatch_graph.go -- `fak dispatch graph`, the read-only companion to `fak dispatch route`. Route
// answers "what dispatches this tick"; graph answers "what is the dependency SHAPE of the backlog" --
// which leaves are roots (dispatchable now), which are waiting behind an open prerequisite, and which
// sit in a dependency CYCLE that will never dispatch until a human breaks an edge.
//
// It reads the same "depends-on:/blocked-by: #N" edges the router parsed onto IssueRoute.BlockedBy, so
// the graph and the live dependency hold (dispatch_prereq.go) are one source of truth: a node is a
// ROOT here iff the hold would let it dispatch (absent from dispatchorder.BlockedByOpenPrereq). It
// routes BEFORE the prereq hold (dispatchRoutedBeforePrereqHold) so held dependents still carry their
// edges -- the hold moves them into the edge-less skipped set, which the graph must see through.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// dispatchGraph is the rendered dependency model: roots (dispatchable now), the declared edges, the
// chains rooted at each dispatchable prerequisite, and the cycles no root can reach.
type dispatchGraph struct {
	Roots  []int               `json:"roots"`
	Edges  []dispatchGraphEdge `json:"edges"`
	Chains []dispatchGraphNode `json:"chains"`
	Cycles [][]int             `json:"cycles"`
	Counts dispatchGraphCounts `json:"counts"`
}

// dispatchGraphEdge is one dependent and the OPEN prerequisites it still waits on (closed/absent
// prerequisites are dropped -- they fail open and never appear as an edge).
type dispatchGraphEdge struct {
	Issue     int   `json:"issue"`
	BlockedBy []int `json:"blocked_by"`
}

// dispatchGraphNode is a node in a chain tree: an issue and the dependents it blocks (its children).
type dispatchGraphNode struct {
	Issue     int                 `json:"issue"`
	BlockedBy []int               `json:"blocked_by,omitempty"`
	Blocks    []dispatchGraphNode `json:"blocks,omitempty"`
}

type dispatchGraphCounts struct {
	Candidates int `json:"candidates"`
	Roots      int `json:"roots"`
	Blocked    int `json:"blocked"`
	Chains     int `json:"chains"`
	Cycles     int `json:"cycles"`
}

func runDispatchGraph(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch graph", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: current directory)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	root := strings.TrimSpace(*workspace)
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch graph: getwd: %v\n", err)
			return 1
		}
		root = wd
	}
	payload, err := dispatchRoutedBeforePrereqHold(root, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch graph: %v\n", err)
		return 1
	}
	graph := buildDispatchGraph(payload)
	if *asJSON {
		if err := writeIndentedJSON(stdout, graph); err != nil {
			fmt.Fprintf(stderr, "fak dispatch graph: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, renderDispatchGraph(graph))
	return 0
}

// buildDispatchGraph is the pure model builder: it projects the routed payload into the dependency
// graph. It uses TWO edge views on purpose -- dispatchorder.BlockedByOpenPrereq (which breaks 2-cycles
// exactly as the live hold does) drives roots/chains so "root == dispatchable this tick", and the
// declared-edge graph (unbroken) drives cycle detection, so a cycle the engine silently broke is still
// surfaced for a human to see.
func buildDispatchGraph(payload dispatchtick.RouterPayload) dispatchGraph {
	// Candidate universe = routed issues (with edges) ∪ still-open skipped issues (presence only),
	// identical to the live hold's universe so the two agree on what counts as an open prerequisite.
	cands := make([]dispatchorder.Candidate, 0, len(payload.Issues)+len(payload.SkippedHumanBlocked))
	present := map[string]bool{}
	laned := map[string]bool{}
	for _, iss := range payload.Issues {
		id := strconv.Itoa(iss.Number)
		present[id] = true
		if iss.Lane != "" {
			laned[id] = true
		}
		cands = append(cands, dispatchorder.Candidate{ID: id, BlockedBy: iss.BlockedBy})
	}
	for _, sk := range payload.SkippedHumanBlocked {
		id := strconv.Itoa(sk.Number)
		present[id] = true
		cands = append(cands, dispatchorder.Candidate{ID: id})
	}

	// Declared-edge graph (dependent -> present prerequisites), for cycle detection.
	declared := map[string][]string{}
	for _, iss := range payload.Issues {
		id := strconv.Itoa(iss.Number)
		for _, p := range iss.BlockedBy {
			if present[p] {
				declared[id] = append(declared[id], p)
			}
		}
	}

	blocked := dispatchorder.BlockedByOpenPrereq(cands) // dependent -> OPEN prereqs, 2-cycles broken

	// Roots = dispatchable candidates (laned) with no open prerequisite this tick.
	var roots []int
	for _, iss := range payload.Issues {
		id := strconv.Itoa(iss.Number)
		if laned[id] && len(blocked[id]) == 0 {
			roots = append(roots, iss.Number)
		}
	}
	sort.Ints(roots)

	// Edges (for JSON): every dependent with at least one open prerequisite.
	var edges []dispatchGraphEdge
	for _, iss := range payload.Issues {
		id := strconv.Itoa(iss.Number)
		open := blocked[id]
		if len(open) == 0 {
			continue
		}
		edges = append(edges, dispatchGraphEdge{Issue: iss.Number, BlockedBy: atois(open)})
	}
	sort.Slice(edges, func(i, j int) bool { return edges[i].Issue < edges[j].Issue })

	// Chains: invert `blocked` into waiters[prereq] = dependents, then grow an indented forest from
	// each root. A globally-visited set keeps each node printed once (diamonds converge) and stops a
	// chain that grazes a cycle from looping.
	waiters := map[string][]string{}
	for dep, prereqs := range blocked {
		for _, p := range prereqs {
			waiters[p] = append(waiters[p], dep)
		}
	}
	visited := map[string]bool{}
	var chains []dispatchGraphNode
	for _, r := range roots {
		id := strconv.Itoa(r)
		if len(waiters[id]) == 0 {
			continue // a free root with no dependents is not a chain
		}
		chains = append(chains, growChain(id, waiters, blocked, visited))
	}

	cycles := detectCycles(declared)

	blockedCount := 0
	for _, iss := range payload.Issues {
		if len(blocked[strconv.Itoa(iss.Number)]) > 0 {
			blockedCount++
		}
	}
	return dispatchGraph{
		Roots:  roots,
		Edges:  edges,
		Chains: chains,
		Cycles: cycles,
		Counts: dispatchGraphCounts{
			Candidates: len(payload.Issues),
			Roots:      len(roots),
			Blocked:    blockedCount,
			Chains:     len(chains),
			Cycles:     len(cycles),
		},
	}
}

// growChain builds the dependent subtree rooted at id, descending through waiters (things blocked by
// id). visited is global across the whole forest so a node prints once.
func growChain(id string, waiters map[string][]string, blocked map[string][]string, visited map[string]bool) dispatchGraphNode {
	visited[id] = true
	node := dispatchGraphNode{Issue: atoi(id), BlockedBy: atois(blocked[id])}
	kids := append([]string(nil), waiters[id]...)
	sort.Slice(kids, func(i, j int) bool { return atoi(kids[i]) < atoi(kids[j]) })
	for _, k := range kids {
		if visited[k] {
			continue
		}
		child := growChain(k, waiters, blocked, visited)
		node.Blocks = append(node.Blocks, child)
	}
	return node
}

// detectCycles returns every dependency cycle in the declared-edge graph as a sorted issue-number
// list, via Tarjan's SCC: any component of size >= 2 is a cycle, and a size-1 component is a cycle
// only if the node depends on itself. Deterministic (inputs sorted), so a test can pin the output.
func detectCycles(declared map[string][]string) [][]int {
	nodes := make([]string, 0, len(declared))
	for n := range declared {
		nodes = append(nodes, n)
	}
	sort.Slice(nodes, func(i, j int) bool { return atoi(nodes[i]) < atoi(nodes[j]) })

	index := map[string]int{}
	low := map[string]int{}
	onStack := map[string]bool{}
	var stack []string
	next := 0
	var out [][]int

	var strong func(v string)
	strong = func(v string) {
		index[v] = next
		low[v] = next
		next++
		stack = append(stack, v)
		onStack[v] = true
		succ := append([]string(nil), declared[v]...)
		sort.Slice(succ, func(i, j int) bool { return atoi(succ[i]) < atoi(succ[j]) })
		for _, w := range succ {
			if _, seen := index[w]; !seen {
				strong(w)
				if low[w] < low[v] {
					low[v] = low[w]
				}
			} else if onStack[w] {
				if index[w] < low[v] {
					low[v] = index[w]
				}
			}
		}
		if low[v] == index[v] {
			var comp []string
			for {
				w := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				onStack[w] = false
				comp = append(comp, w)
				if w == v {
					break
				}
			}
			if len(comp) >= 2 || (len(comp) == 1 && dependsOnSelf(declared, comp[0])) {
				nums := atois(comp)
				sort.Ints(nums)
				out = append(out, nums)
			}
		}
	}
	for _, n := range nodes {
		if _, seen := index[n]; !seen {
			strong(n)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i][0] < out[j][0] })
	return out
}

func dependsOnSelf(declared map[string][]string, id string) bool {
	for _, p := range declared[id] {
		if p == id {
			return true
		}
	}
	return false
}

func renderDispatchGraph(g dispatchGraph) string {
	var b strings.Builder
	fmt.Fprintf(&b, "dispatch graph: %d candidate(s), %d root(s), %d blocked, %d chain(s), %d cycle(s)\n",
		g.Counts.Candidates, g.Counts.Roots, g.Counts.Blocked, g.Counts.Chains, g.Counts.Cycles)
	fmt.Fprintf(&b, "  roots (dispatchable now): %s\n", intList(g.Roots))
	if len(g.Chains) > 0 {
		fmt.Fprintln(&b, "  chains (waiting behind an open prerequisite):")
		for _, c := range g.Chains {
			renderChainNode(&b, c, 2)
		}
	}
	if len(g.Cycles) > 0 {
		fmt.Fprintln(&b, "  cycles (SOFT-held — no root can reach these; break an edge to dispatch):")
		for _, cyc := range g.Cycles {
			parts := make([]string, len(cyc))
			for i, n := range cyc {
				parts[i] = fmt.Sprintf("#%d", n)
			}
			fmt.Fprintf(&b, "    %s\n", strings.Join(parts, " → "))
		}
	}
	return b.String()
}

func renderChainNode(b *strings.Builder, node dispatchGraphNode, depth int) {
	indent := strings.Repeat("  ", depth)
	if len(node.BlockedBy) > 0 {
		fmt.Fprintf(b, "%s#%d (blocked by %s)\n", indent, node.Issue, hashList(node.BlockedBy))
	} else {
		fmt.Fprintf(b, "%s#%d\n", indent, node.Issue)
	}
	for _, child := range node.Blocks {
		renderChainNode(b, child, depth+1)
	}
}

func hashList(nums []int) string {
	if len(nums) == 0 {
		return "-"
	}
	parts := make([]string, len(nums))
	for i, n := range nums {
		parts[i] = fmt.Sprintf("#%d", n)
	}
	return strings.Join(parts, ", ")
}

// atoi / atois convert the string IDs the dependency engine speaks back to issue numbers. A
// non-numeric id (never produced by this path, which stringifies issue numbers) folds to 0.
func atoi(s string) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return n
}

func atois(ss []string) []int {
	out := make([]int, len(ss))
	for i, s := range ss {
		out[i] = atoi(s)
	}
	return out
}
