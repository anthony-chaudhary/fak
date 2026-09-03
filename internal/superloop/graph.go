package superloop

// graph.go — the MODULARITY-OBSERVABILITY rung over the registry. WALK and DRIVE read a
// SINGLE intent's member STATUS (debt, dark, satisfied); GRAPH reads the whole
// registry's SHAPE — the composition DAG the super loops form by DESCENDING one another
// over KindSuperloop edges. Three structural invariants keep that nesting safe, and
// until now they lived only in the package tests:
//
//	resolves      — every descend ref points at a registered intent (no dangling edge)
//	acyclic       — the descend edges form a DAG, so a walk's descent terminates
//	root-rooted   — every intent is REACHABLE from the root ("tend"), so none escapes it
//	once-only     — no scorecard key is walked by two intents, so no debt folds twice
//
// [Graph] FOLDS those into one report an operator or agent can read at RUNTIME — the
// modular structure made OBSERVABLE, not merely test-enforced. It is PURE: it reads the
// data-only registry and no files/clock, the same fold [Registry]/[Names] are.
//
// The payoff view is FAN-IN. A super loop DESCENDED by more than one parent (drain-issues,
// under both improve-loops and run-the-night) is a REUSED module — the whole point of
// nesting instead of copying members — and the report names it as SHARED. A scorecard
// reused that same way is not reuse but the once-only VIOLATION (its debt would fold
// twice), so the report reds it. The two look alike in a flat registry; the graph fold is
// what tells them apart.

import (
	"fmt"
	"sort"
)

// GraphSchema is the versioned payload tag the `--json` graph emits.
const GraphSchema = "fak.superloop-graph.v1"

// RootIntent is the name of the ROOT super loop — the intent every other intent must be
// reachable from over KindSuperloop edges (the "super loop of super loops"). Depth and
// reachability in [Graph] are measured from here; the no-escape test pins that the
// registry actually reaches every intent from it.
const RootIntent = "tend"

// IntentNode is one super loop as a node in the composition DAG: its own facts plus its
// place in the graph — the sub-super-loops it DESCENDS (its out-edges / module
// dependencies), the parents that descend IT (fan-in / in-edges), its minimum descend
// depth below the root, and whether the root reaches it at all.
type IntentNode struct {
	Name       string             `json:"name"`
	Title      string             `json:"title"`
	Members    int                `json:"members"`
	KindCounts map[MemberKind]int `json:"kind_counts,omitempty"`
	// Descends are the KindSuperloop member refs, in declaration order — this node's
	// out-edges in the composition DAG.
	Descends []string `json:"descends,omitempty"`
	// Parents are the intents that descend THIS node (its in-edges), sorted.
	Parents []string `json:"parents,omitempty"`
	// FanIn is len(Parents): how many parents reuse this node.
	FanIn int `json:"fan_in"`
	// Depth is the minimum descend depth from the root (root = 0); -1 if unreachable.
	Depth     int  `json:"depth"`
	Reachable bool `json:"reachable"`
	Root      bool `json:"root"`
	// LeafIntent is true when the node descends NO sub-super-loop: it drives only
	// concrete surfaces (scorecards/loops/gardens/utilization), an interior node whose
	// children are all leaves.
	LeafIntent bool `json:"leaf_intent"`
	// Shared is FanIn > 1: a module REUSED by multiple parents — the modularity payoff a
	// flat registry hides.
	Shared bool `json:"shared"`
	// LeafDenominator is the count of distinct non-superloop leaf members reachable
	// by descending from this intent in the registry DAG — the structural leaf denominator.
	LeafDenominator int `json:"leaf_denominator"`
}

// ScorecardClash is one once-only violation: a scorecard key walked by two or more
// intents reachable from the root, whose debt would therefore fold into the root
// aggregate more than once.
type ScorecardClash struct {
	Ref     string   `json:"ref"`
	Intents []string `json:"intents"`
}

// GraphReport folds the whole registry into its composition DAG plus the structural
// invariants that keep the nesting safe — the modular structure made observable. Verdict
// is OK only when the graph resolves, is acyclic, roots every intent, and counts each
// scorecard once; otherwise ACTION names the first failing invariant in Finding.
type GraphReport struct {
	Schema   string       `json:"schema"`
	Root     string       `json:"root"`
	Intents  int          `json:"intents"`
	Edges    int          `json:"edges"` // KindSuperloop descend edges
	MaxDepth int          `json:"max_depth"`
	Nodes    []IntentNode `json:"nodes"` // in declaration order

	// Dangling are descend edges ("parent -> ref") whose ref resolves to no registered
	// intent — a broken module reference that would permanently red its parent's walk.
	Dangling []string `json:"dangling,omitempty"`

	// Acyclic is true when the descend edges form a DAG; Cycle carries the back-edge path
	// when they do not.
	Acyclic bool     `json:"acyclic"`
	Cycle   []string `json:"cycle,omitempty"`

	// RootReaches counts intents reachable from the root; Orphans lists those not.
	RootReaches int      `json:"root_reaches"`
	Orphans     []string `json:"orphans,omitempty"`

	// OnceOnly is true when no scorecard is double-counted; DoubleCounted lists any that
	// are.
	OnceOnly      bool             `json:"once_only"`
	DoubleCounted []ScorecardClash `json:"double_counted,omitempty"`

	// Shared lists the super loops with fan-in > 1 (reused modules), sorted.
	Shared []string `json:"shared,omitempty"`

	// TotalLeafDenominator is the total count of distinct leaf surfaces reachable from the root.
	TotalLeafDenominator int `json:"total_leaf_denominator"`

	Verdict string `json:"verdict"`
	Finding string `json:"finding"`
	Reason  string `json:"reason"`
}

// Graph folds the package registry into its composition DAG and the structural
// invariants, measuring depth and reachability from [RootIntent]. Pure: it reads the
// data-only registry and nothing else.
func Graph() GraphReport { return graphOf(registry, RootIntent) }

// graphOf is Graph over an EXPLICIT registry and root, so the failure modes (a cycle, an
// orphan, a double-counted scorecard, a dangling ref) are testable on synthetic
// registries without mutating the package one. All resolution is scoped to reg — the
// helpers never consult the global registry — so a synthetic graph folds in isolation.
func graphOf(reg []Super, root string) GraphReport {
	rep := GraphReport{Schema: GraphSchema, Root: root, Intents: len(reg)}

	byName := make(map[string]Super, len(reg))
	for _, s := range reg {
		byName[s.Name] = s
	}
	resolves := func(name string) bool { _, ok := byName[name]; return ok }

	// Build the descend (out) and parent (in) edge maps over KindSuperloop, and record any
	// edge whose ref does not resolve.
	descends := map[string][]string{}
	parents := map[string][]string{}
	for _, s := range reg {
		for _, m := range s.Members {
			if m.Kind != KindSuperloop {
				continue
			}
			descends[s.Name] = append(descends[s.Name], m.Ref)
			parents[m.Ref] = append(parents[m.Ref], s.Name)
			rep.Edges++
			if !resolves(m.Ref) {
				rep.Dangling = append(rep.Dangling, s.Name+" -> "+m.Ref)
			}
		}
	}
	sort.Strings(rep.Dangling)

	// Acyclic over EVERY intent (a cycle unreachable from the root is still a cycle), and
	// reachability + minimum depth over a BFS from the root.
	rep.Acyclic, rep.Cycle = detectCycle(reg, descends, resolves)
	reach, depth := reachDepth(root, descends, resolves)
	rep.RootReaches = len(reach)

	// Once-only over the reachable set (the invariant's domain): a scorecard walked by
	// two reachable intents folds its debt twice.
	clashes := doubleCountedScorecards(reg, reach)
	rep.OnceOnly = len(clashes) == 0
	rep.DoubleCounted = clashes

	for _, s := range reg {
		node := IntentNode{
			Name:       s.Name,
			Title:      s.Title,
			Members:    len(s.Members),
			KindCounts: kindCounts(s),
			Descends:   descends[s.Name],
			Parents:    sortedCopy(parents[s.Name]),
			FanIn:      len(parents[s.Name]),
			Depth:      -1,
			Root:       s.Name == root,
			LeafIntent: len(descends[s.Name]) == 0,
		}
		if d, ok := depth[s.Name]; ok {
			node.Depth = d
			node.Reachable = true
			if d > rep.MaxDepth {
				rep.MaxDepth = d
			}
		} else {
			rep.Orphans = append(rep.Orphans, s.Name)
		}
		node.Shared = node.FanIn > 1
		if node.Shared {
			rep.Shared = append(rep.Shared, s.Name)
		}
		node.LeafDenominator = reachableLeafCount(s.Name, byName, rep.Acyclic)
		rep.Nodes = append(rep.Nodes, node)
	}
	sort.Strings(rep.Orphans)
	sort.Strings(rep.Shared)
	rep.TotalLeafDenominator = reachableLeafCount(root, byName, rep.Acyclic)

	rep.Verdict, rep.Finding, rep.Reason = graphVerdict(rep)
	return rep
}

// detectCycle runs a coloring DFS over the descend edges of every intent and returns
// whether the graph is acyclic plus, when it is not, the back-edge path witnessing the
// first cycle found. Unresolvable refs are skipped (they are surfaced as dangling, not a
// cycle).
func detectCycle(reg []Super, descends map[string][]string, resolves func(string) bool) (bool, []string) {
	const (
		visiting = 1
		done     = 2
	)
	state := map[string]int{}
	var cycle []string
	var visit func(name string, path []string) bool
	visit = func(name string, path []string) bool {
		switch state[name] {
		case visiting:
			// Back edge: the cycle is path from the first occurrence of name, closed.
			start := 0
			for i, p := range path {
				if p == name {
					start = i
					break
				}
			}
			cycle = append(append([]string(nil), path[start:]...), name)
			return true
		case done:
			return false
		}
		state[name] = visiting
		for _, next := range descends[name] {
			if !resolves(next) {
				continue
			}
			if visit(next, append(path, name)) {
				return true
			}
		}
		state[name] = done
		return false
	}
	for _, s := range reg {
		if visit(s.Name, nil) {
			return false, cycle
		}
	}
	return true, nil
}

// reachDepth BFS-walks the descend edges from root and returns the reachable set and each
// reachable intent's MINIMUM depth (root = 0). Unresolvable refs are not followed.
func reachDepth(root string, descends map[string][]string, resolves func(string) bool) (map[string]bool, map[string]int) {
	reach := map[string]bool{}
	depth := map[string]int{}
	if !resolves(root) {
		return reach, depth
	}
	reach[root] = true
	depth[root] = 0
	queue := []string{root}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, next := range descends[cur] {
			if !resolves(next) || reach[next] {
				continue
			}
			reach[next] = true
			depth[next] = depth[cur] + 1
			queue = append(queue, next)
		}
	}
	return reach, depth
}

// doubleCountedScorecards folds the scorecard membership over the reachable set and
// returns any scorecard key held by two or more intents — the once-only violation. Each
// intent is counted once (a scorecard listed twice by ONE intent is not a cross-intent
// double-count). The clash list and each clash's intents are sorted for a stable report.
func doubleCountedScorecards(reg []Super, reach map[string]bool) []ScorecardClash {
	holders := map[string]map[string]bool{}
	for _, s := range reg {
		if !reach[s.Name] {
			continue
		}
		for _, m := range s.Members {
			if m.Kind != KindScorecard {
				continue
			}
			if holders[m.Ref] == nil {
				holders[m.Ref] = map[string]bool{}
			}
			holders[m.Ref][s.Name] = true
		}
	}
	var clashes []ScorecardClash
	for ref, in := range holders {
		if len(in) < 2 {
			continue
		}
		intents := make([]string, 0, len(in))
		for name := range in {
			intents = append(intents, name)
		}
		sort.Strings(intents)
		clashes = append(clashes, ScorecardClash{Ref: ref, Intents: intents})
	}
	sort.Slice(clashes, func(i, j int) bool { return clashes[i].Ref < clashes[j].Ref })
	return clashes
}

// graphVerdict grades the folded graph, naming the FIRST failing invariant in priority
// order (dangling ref, cycle, orphan, double-count) or reporting the sound structure.
func graphVerdict(rep GraphReport) (verdict, finding, reason string) {
	switch {
	case len(rep.Dangling) > 0:
		return "ACTION", "structure_dangling",
			fmt.Sprintf("%d descend edge(s) point at no registered intent: %v — resolve or drop them", len(rep.Dangling), rep.Dangling)
	case !rep.Acyclic:
		return "ACTION", "structure_cycle",
			fmt.Sprintf("the descend edges are not acyclic: %v — a walk's descent would not terminate", rep.Cycle)
	case len(rep.Orphans) > 0:
		return "ACTION", "structure_orphan",
			fmt.Sprintf("%d intent(s) are unreachable from root %q: %v — nest them under the root or a member so the root walk sees them", len(rep.Orphans), rep.Root, rep.Orphans)
	case len(rep.DoubleCounted) > 0:
		return "ACTION", "structure_double_count",
			fmt.Sprintf("%d scorecard(s) are walked by two intents, folding their debt twice: %v — nest the shared surface once", len(rep.DoubleCounted), clashRefs(rep.DoubleCounted))
	default:
		return "OK", "structure_sound",
			fmt.Sprintf("%d intents, %d descend edges, max depth %d: resolves, acyclic, every intent rooted at %q, every scorecard counted once", rep.Intents, rep.Edges, rep.MaxDepth, rep.Root)
	}
}

func clashRefs(cs []ScorecardClash) []string {
	out := make([]string, 0, len(cs))
	for _, c := range cs {
		out = append(out, c.Ref)
	}
	return out
}

func kindCounts(s Super) map[MemberKind]int {
	counts := map[MemberKind]int{}
	for _, m := range s.Members {
		counts[m.Kind]++
	}
	return counts
}

func sortedCopy(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	out := append([]string(nil), in...)
	sort.Strings(out)
	return out
}

// reachableLeafCount returns the count of distinct non-superloop leaf surfaces
// reachable from start by descending the registry DAG.
func reachableLeafCount(start string, byName map[string]Super, acyclic bool) int {
	visited := map[string]bool{}
	leaves := map[string]bool{}
	var visit func(name string)
	visit = func(name string) {
		if visited[name] {
			return
		}
		visited[name] = true
		s, ok := byName[name]
		if !ok {
			return
		}
		for _, m := range s.Members {
			if m.Kind == KindSuperloop {
				visit(m.Ref)
			} else {
				leaves[memberKey(m)] = true
			}
		}
	}
	visit(start)
	return len(leaves)
}
