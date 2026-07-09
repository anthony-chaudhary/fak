// Package codegraph is a directed code knowledge-graph with breadth-first
// traversal — the "what reaches / what depends on this" seam (#3439, epic #3434,
// the capstone). Nodes are code entities (functions, files, packages); edges are
// typed relations between them (calls, imports, references). BFS answers the two
// questions the flat retrieval layers (#3435-#3438) cannot: forward reachability
// ("what does f end up calling") and reverse reachability ("what would break if I
// change c"), each as a shortest, deterministic path.
//
// The graph is in-memory and holds both a forward and a reverse adjacency list, so
// dependents are a plain BFS on the reverse edges — no re-derivation. Traversal is
// unweighted, so BFS yields shortest paths by construction; ties break by node id
// for a stable, reproducible walk.
package codegraph

import (
	"go/ast"
	"go/parser"
	"go/token"
	"sort"
)

// NodeID is a code entity's stable identity (a qualified name, a path).
type NodeID string

// Node is one entity in the graph.
type Node struct {
	ID   NodeID
	Kind string // "func", "file", "package", ...
}

// Edge is a typed directed relation from one entity to another.
type Edge struct {
	From, To NodeID
	Kind     string // "calls", "imports", "references", ...
}

// Graph is a directed multigraph with forward and reverse adjacency. The zero value
// is not ready; use NewGraph.
type Graph struct {
	nodes map[NodeID]Node
	fwd   map[NodeID][]Edge
	rev   map[NodeID][]Edge
}

// NewGraph returns an empty graph.
func NewGraph() *Graph {
	return &Graph{
		nodes: map[NodeID]Node{},
		fwd:   map[NodeID][]Edge{},
		rev:   map[NodeID][]Edge{},
	}
}

// AddNode records an entity. Re-adding updates its kind.
func (g *Graph) AddNode(id NodeID, kind string) { g.nodes[id] = Node{ID: id, Kind: kind} }

// AddEdge records a typed relation, auto-creating either endpoint that is not yet a
// node (with an empty kind) so a graph can be built edges-first.
func (g *Graph) AddEdge(from, to NodeID, kind string) {
	if _, ok := g.nodes[from]; !ok {
		g.AddNode(from, "")
	}
	if _, ok := g.nodes[to]; !ok {
		g.AddNode(to, "")
	}
	e := Edge{From: from, To: to, Kind: kind}
	g.fwd[from] = append(g.fwd[from], e)
	g.rev[to] = append(g.rev[to], e)
}

// NodeCount is the number of entities.
func (g *Graph) NodeCount() int { return len(g.nodes) }

// Traversal parameterizes a BFS. MaxDepth <= 0 means unbounded. EdgeKinds nil/empty
// means follow every edge kind; otherwise only the listed kinds. Reverse walks the
// reverse edges (dependents) instead of the forward edges (reachable).
type Traversal struct {
	MaxDepth  int
	EdgeKinds []string
	Reverse   bool
}

// Hit is one reached entity: its id, its BFS distance from the start (>=1), and the
// shortest path from start to it (inclusive of both ends).
type Hit struct {
	ID   NodeID
	Dist int
	Path []NodeID
}

func kindSet(kinds []string) map[string]bool {
	if len(kinds) == 0 {
		return nil
	}
	m := make(map[string]bool, len(kinds))
	for _, k := range kinds {
		m[k] = true
	}
	return m
}

// BFS returns every entity reachable from start under t, nearest-first. The start
// node itself is not a hit (distance 0 is the query). Results are ordered by
// (distance, id) so the walk is fully deterministic; each hit carries the shortest
// path that discovered it.
func (g *Graph) BFS(start NodeID, t Traversal) []Hit {
	adj := g.fwd
	if t.Reverse {
		adj = g.rev
	}
	allow := kindSet(t.EdgeKinds)

	dist := map[NodeID]int{start: 0}
	pred := map[NodeID]NodeID{}
	queue := []NodeID{start}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		d := dist[cur]
		if t.MaxDepth > 0 && d >= t.MaxDepth {
			continue
		}
		// Deterministic neighbor order: the far endpoint's id, ascending.
		edges := append([]Edge(nil), adj[cur]...)
		sort.Slice(edges, func(i, j int) bool { return neighbor(edges[i], t.Reverse) < neighbor(edges[j], t.Reverse) })
		for _, e := range edges {
			if allow != nil && !allow[e.Kind] {
				continue
			}
			nb := neighbor(e, t.Reverse)
			if _, seen := dist[nb]; seen {
				continue // first discovery = shortest path in unweighted BFS
			}
			dist[nb] = d + 1
			pred[nb] = cur
			queue = append(queue, nb)
		}
	}

	hits := make([]Hit, 0, len(dist))
	for id, d := range dist {
		if id == start {
			continue
		}
		hits = append(hits, Hit{ID: id, Dist: d, Path: buildPath(pred, start, id)})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Dist != hits[j].Dist {
			return hits[i].Dist < hits[j].Dist
		}
		return hits[i].ID < hits[j].ID
	})
	return hits
}

// neighbor returns the "other end" of an edge for the direction being walked.
func neighbor(e Edge, reverse bool) NodeID {
	if reverse {
		return e.From
	}
	return e.To
}

// buildPath reconstructs start..id from the predecessor map (id -> pred).
func buildPath(pred map[NodeID]NodeID, start, id NodeID) []NodeID {
	var rev []NodeID
	for cur := id; ; cur = pred[cur] {
		rev = append(rev, cur)
		if cur == start {
			break
		}
	}
	for i, j := 0, len(rev)-1; i < j; i, j = i+1, j-1 {
		rev[i], rev[j] = rev[j], rev[i]
	}
	return rev
}

// Reaches is forward BFS: everything start ends up depending on, over the given edge
// kinds (all kinds if none given).
func (g *Graph) Reaches(start NodeID, kinds ...string) []Hit {
	return g.BFS(start, Traversal{EdgeKinds: kinds})
}

// Dependents is reverse BFS: everything that (transitively) depends on start — what
// would be impacted by changing it — over the given edge kinds.
func (g *Graph) Dependents(start NodeID, kinds ...string) []Hit {
	return g.BFS(start, Traversal{EdgeKinds: kinds, Reverse: true})
}

// BuildCallGraph parses a single Go source file and returns its intra-file call
// graph: a "func" node per top-level function, and a "calls" edge from a caller to
// each top-level function it invokes by bare name. It is intentionally scoped to
// same-file, same-name resolution — enough to demonstrate the graph end-to-end
// without a type checker.
func BuildCallGraph(src string) (*Graph, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "src.go", src, 0)
	if err != nil {
		return nil, err
	}
	g := NewGraph()
	// First pass: register every top-level function as a node.
	funcs := map[string]bool{}
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Recv == nil {
			funcs[fn.Name.Name] = true
			g.AddNode(NodeID(fn.Name.Name), "func")
		}
	}
	// Second pass: an edge per call to a known top-level function.
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Recv != nil || fn.Body == nil {
			continue
		}
		caller := NodeID(fn.Name.Name)
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			if id, ok := call.Fun.(*ast.Ident); ok && funcs[id.Name] {
				g.AddEdge(caller, NodeID(id.Name), "calls")
			}
			return true
		})
	}
	return g, nil
}
