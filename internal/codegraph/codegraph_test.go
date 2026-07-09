package codegraph

import (
	"reflect"
	"testing"
)

func hitIDs(hits []Hit) []NodeID {
	out := make([]NodeID, len(hits))
	for i, h := range hits {
		out[i] = h.ID
	}
	return out
}

func findHit(hits []Hit, id NodeID) (Hit, bool) {
	for _, h := range hits {
		if h.ID == id {
			return h, true
		}
	}
	return Hit{}, false
}

func diamond() *Graph {
	g := NewGraph()
	g.AddEdge("d", "a", "calls")
	g.AddEdge("a", "b", "calls")
	g.AddEdge("a", "c", "calls")
	g.AddEdge("b", "c", "calls")
	return g
}

func TestForwardReachAndShortestPath(t *testing.T) {
	g := diamond()
	hits := g.Reaches("d")
	if got := hitIDs(hits); !reflect.DeepEqual(got, []NodeID{"a", "b", "c"}) {
		t.Fatalf("Reaches(d) = %v, want [a b c] nearest-first", got)
	}
	c, _ := findHit(hits, "c")
	if c.Dist != 2 {
		t.Errorf("dist to c = %d, want 2", c.Dist)
	}
	// Shortest path d->c goes through a, not the longer d->a->b->c.
	if !reflect.DeepEqual(c.Path, []NodeID{"d", "a", "c"}) {
		t.Errorf("shortest path to c = %v, want [d a c]", c.Path)
	}
}

func TestReverseDependents(t *testing.T) {
	g := diamond()
	deps := g.Dependents("c")
	if got := hitIDs(deps); !reflect.DeepEqual(got, []NodeID{"a", "b", "d"}) {
		t.Fatalf("Dependents(c) = %v, want [a b d]", got)
	}
	d, _ := findHit(deps, "d")
	if d.Dist != 2 || !reflect.DeepEqual(d.Path, []NodeID{"c", "a", "d"}) {
		t.Errorf("impact path to d = dist %d %v, want dist 2 [c a d]", d.Dist, d.Path)
	}
}

func TestEdgeKindFilterAndDepthBound(t *testing.T) {
	g := diamond()
	g.AddEdge("d", "z", "references") // a different relation

	// Filtered to "calls": z must not appear.
	for _, h := range g.Reaches("d", "calls") {
		if h.ID == "z" {
			t.Fatal("calls-filtered traversal followed a references edge to z")
		}
	}
	// Unfiltered: z is a direct neighbor.
	if _, ok := findHit(g.Reaches("d"), "z"); !ok {
		t.Error("unfiltered traversal missed z")
	}

	// Depth bound of 1 keeps only the immediate neighbors.
	near := g.BFS("d", Traversal{MaxDepth: 1, EdgeKinds: []string{"calls"}})
	if got := hitIDs(near); !reflect.DeepEqual(got, []NodeID{"a"}) {
		t.Errorf("depth-1 Reaches(d) = %v, want [a]", got)
	}
}

func TestBuildCallGraph(t *testing.T) {
	src := `package p

func a() { b(); c() }
func b() { c() }
func c() {}
func d() { a() }
`
	g, err := BuildCallGraph(src)
	if err != nil {
		t.Fatalf("BuildCallGraph: %v", err)
	}
	if g.NodeCount() != 4 {
		t.Fatalf("node count = %d, want 4", g.NodeCount())
	}
	// d reaches a (1), then b and c (2).
	reach := g.Reaches("d")
	if got := hitIDs(reach); !reflect.DeepEqual(got, []NodeID{"a", "b", "c"}) {
		t.Fatalf("Reaches(d) = %v, want [a b c]", got)
	}
	c, _ := findHit(reach, "c")
	if !reflect.DeepEqual(c.Path, []NodeID{"d", "a", "c"}) {
		t.Errorf("call path d->c = %v, want [d a c]", c.Path)
	}
	// c is the most-depended-on leaf: a, b directly, d transitively.
	if got := hitIDs(g.Dependents("c")); !reflect.DeepEqual(got, []NodeID{"a", "b", "d"}) {
		t.Errorf("Dependents(c) = %v, want [a b d]", got)
	}
}
