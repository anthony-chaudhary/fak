package superloop

import (
	"reflect"
	"testing"
)

// TestGraphShippedRegistrySound is the LIVE witness of the four structural invariants
// the static tests enforce (resolves, acyclic, root-rooted, once-only): folded over the
// shipped registry, [Graph] must report a sound graph. This makes the invariants
// observable at runtime — a future drift becomes a red verdict here, not merely a red
// test elsewhere.
func TestGraphShippedRegistrySound(t *testing.T) {
	rep := Graph()

	if rep.Schema != GraphSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, GraphSchema)
	}
	if rep.Verdict != "OK" || rep.Finding != "structure_sound" {
		t.Fatalf("shipped registry graph = %q/%q (%s); want OK/structure_sound", rep.Verdict, rep.Finding, rep.Reason)
	}
	if len(rep.Dangling) != 0 {
		t.Errorf("no descend edge may dangle, got %v", rep.Dangling)
	}
	if !rep.Acyclic || len(rep.Cycle) != 0 {
		t.Errorf("shipped graph must be acyclic, got acyclic=%v cycle=%v", rep.Acyclic, rep.Cycle)
	}
	if len(rep.Orphans) != 0 {
		t.Errorf("every intent must be reachable from the root, orphans=%v", rep.Orphans)
	}
	if rep.RootReaches != rep.Intents {
		t.Errorf("root reaches %d of %d intents; want all", rep.RootReaches, rep.Intents)
	}
	if !rep.OnceOnly || len(rep.DoubleCounted) != 0 {
		t.Errorf("no scorecard may be double-counted, got %v", rep.DoubleCounted)
	}
	if rep.Root != RootIntent {
		t.Errorf("root = %q, want %q", rep.Root, RootIntent)
	}
	if rep.Intents != len(Registry()) {
		t.Errorf("intents = %d, want %d", rep.Intents, len(Registry()))
	}
}

// TestGraphRootNodeAndDepth pins the root node (depth 0, no parent) and that every other
// node carries a finite descend depth — the reachability invariant per-node, plus the
// nesting is real (max depth > 1).
func TestGraphRootNodeAndDepth(t *testing.T) {
	rep := Graph()
	byName := map[string]IntentNode{}
	for _, n := range rep.Nodes {
		byName[n.Name] = n
	}
	root, ok := byName[RootIntent]
	if !ok {
		t.Fatalf("root intent %q missing from node table", RootIntent)
	}
	if !root.Root || root.Depth != 0 || root.FanIn != 0 {
		t.Errorf("root node = %+v; want Root=true Depth=0 FanIn=0", root)
	}
	for _, n := range rep.Nodes {
		if !n.Reachable {
			t.Errorf("intent %q is not reachable from the root", n.Name)
			continue
		}
		if n.Name != RootIntent && n.Depth < 1 {
			t.Errorf("non-root intent %q has depth %d; want >= 1", n.Name, n.Depth)
		}
	}
	if rep.MaxDepth < 2 {
		t.Errorf("MaxDepth = %d; the shipped registry nests several levels", rep.MaxDepth)
	}
}

// TestGraphSharedModuleFanIn pins that a genuinely reused module is surfaced: drain-issues
// is descended by both improve-loops and run-the-night today, so it must be Shared with
// FanIn>=2 and both parents named — the modularity payoff a flat registry hides.
func TestGraphSharedModuleFanIn(t *testing.T) {
	rep := Graph()
	found := false
	for _, s := range rep.Shared {
		if s == "drain-issues" {
			found = true
		}
	}
	if !found {
		t.Fatalf("drain-issues should be a shared module, got Shared=%v", rep.Shared)
	}
	for _, n := range rep.Nodes {
		if n.Name != "drain-issues" {
			continue
		}
		if !n.Shared || n.FanIn < 2 {
			t.Errorf("drain-issues node = %+v; want Shared=true FanIn>=2", n)
		}
		if len(n.Parents) != n.FanIn {
			t.Errorf("drain-issues has %d parents but FanIn %d", len(n.Parents), n.FanIn)
		}
	}
}

// TestGraphEdgeCountMatchesDescends cross-checks the summed per-node Descends against the
// reported Edges total (all descend refs resolve in the shipped registry).
func TestGraphEdgeCountMatchesDescends(t *testing.T) {
	rep := Graph()
	sum := 0
	for _, n := range rep.Nodes {
		sum += len(n.Descends)
	}
	if sum != rep.Edges {
		t.Errorf("summed Descends %d != Edges %d", sum, rep.Edges)
	}
}

// --- synthetic-registry drift witnesses (graphOf against hand-built graphs) ---

func gNode(name string, members ...Member) Super {
	return Super{Name: name, Title: name, Members: members}
}

func gSub(ref string) Member  { return Member{Kind: KindSuperloop, Ref: ref} }
func gCard(ref string) Member { return Member{Kind: KindScorecard, Ref: ref} }

// TestGraphDetectsCycle: a synthetic back edge reds ACTION and names the closed cycle.
func TestGraphDetectsCycle(t *testing.T) {
	reg := []Super{
		gNode("tend", gSub("a")),
		gNode("a", gSub("b")),
		gNode("b", gSub("a")), // cycle a -> b -> a
	}
	rep := graphOf(reg, "tend")
	if rep.Acyclic {
		t.Fatal("cyclic registry must not report acyclic")
	}
	if rep.Verdict != "ACTION" || rep.Finding != "structure_cycle" {
		t.Errorf("verdict/finding = %q/%q; want ACTION/structure_cycle", rep.Verdict, rep.Finding)
	}
	if len(rep.Cycle) < 2 || rep.Cycle[0] != rep.Cycle[len(rep.Cycle)-1] {
		t.Errorf("cycle path %v should close on itself", rep.Cycle)
	}
}

// TestGraphDetectsOrphan: an intent never wired under the root is flagged unreachable and
// reds ACTION (dangling clean, so orphan is the surfaced fault).
func TestGraphDetectsOrphan(t *testing.T) {
	reg := []Super{
		gNode("tend", gSub("a")),
		gNode("a"),
		gNode("stray", gCard("x")), // registered but unreachable from the root
	}
	rep := graphOf(reg, "tend")
	if !rep.Acyclic {
		t.Fatalf("acyclic tree misreported cyclic: %v", rep.Cycle)
	}
	if !reflect.DeepEqual(rep.Orphans, []string{"stray"}) {
		t.Errorf("orphans = %v; want [stray]", rep.Orphans)
	}
	for _, n := range rep.Nodes {
		if n.Name == "stray" && (n.Reachable || n.Depth != -1) {
			t.Errorf("orphan node = %+v; want Reachable=false Depth=-1", n)
		}
	}
	if rep.Verdict != "ACTION" || rep.Finding != "structure_orphan" {
		t.Errorf("verdict/finding = %q/%q; want ACTION/structure_orphan", rep.Verdict, rep.Finding)
	}
}

// TestGraphDetectsDoubleCountedScorecard: one scorecard under two reachable intents is the
// once-only violation and reds ACTION with both intents named.
func TestGraphDetectsDoubleCountedScorecard(t *testing.T) {
	reg := []Super{
		gNode("tend", gSub("a"), gSub("b")),
		gNode("a", gCard("dup")),
		gNode("b", gCard("dup")), // same scorecard walked by a and b
	}
	rep := graphOf(reg, "tend")
	if rep.OnceOnly {
		t.Fatal("a scorecard under two intents must break once-only")
	}
	if len(rep.DoubleCounted) != 1 || rep.DoubleCounted[0].Ref != "dup" {
		t.Fatalf("double-counted = %v; want one clash on dup", rep.DoubleCounted)
	}
	if !reflect.DeepEqual(rep.DoubleCounted[0].Intents, []string{"a", "b"}) {
		t.Errorf("clash intents = %v; want [a b]", rep.DoubleCounted[0].Intents)
	}
	if rep.Finding != "structure_double_count" {
		t.Errorf("finding = %q; want structure_double_count", rep.Finding)
	}
}

// TestGraphDoubleCountIgnoresUnreachable: a scorecard shared by two intents that are NOT
// reachable from the root is not a root-fold double-count — the invariant's domain is the
// reachable set (and the unreachable intents surface as orphans instead).
func TestGraphDoubleCountIgnoresUnreachable(t *testing.T) {
	reg := []Super{
		gNode("tend"),
		gNode("x", gCard("dup")), // both unreachable from tend
		gNode("y", gCard("dup")),
	}
	rep := graphOf(reg, "tend")
	if len(rep.DoubleCounted) != 0 {
		t.Errorf("unreachable scorecard reuse is not a root double-count, got %v", rep.DoubleCounted)
	}
	// The first surfaced fault is the orphaned pair, not a double-count.
	if rep.Finding != "structure_orphan" {
		t.Errorf("finding = %q; want structure_orphan", rep.Finding)
	}
}

// TestGraphDetectsDanglingEdge: a descend edge to an unregistered intent is drift, reds
// ACTION, and is NOT mistaken for a cycle.
func TestGraphDetectsDanglingEdge(t *testing.T) {
	reg := []Super{
		gNode("tend", gSub("ghost")), // "ghost" is not registered
	}
	rep := graphOf(reg, "tend")
	if !rep.Acyclic {
		t.Errorf("a dangling edge is not a cycle")
	}
	if len(rep.Dangling) != 1 || rep.Dangling[0] != "tend -> ghost" {
		t.Fatalf("dangling = %v; want [\"tend -> ghost\"]", rep.Dangling)
	}
	if rep.Verdict != "ACTION" || rep.Finding != "structure_dangling" {
		t.Errorf("verdict/finding = %q/%q; want ACTION/structure_dangling", rep.Verdict, rep.Finding)
	}
}

// TestGraphDeterministic: the fold has no clock/map-order leak — two folds of the same
// registry are byte-identical in every ordered field.
func TestGraphDeterministic(t *testing.T) {
	if !reflect.DeepEqual(Graph(), Graph()) {
		t.Error("Graph() is not deterministic across calls")
	}
}
