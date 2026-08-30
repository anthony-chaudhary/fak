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
	"fmt"
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
	Kind string // "func", "method", "file", "package", ...
	Name string // the simple entity name (call resolution key); defaults to ID
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

// AddNode records an entity. Re-adding updates its kind; the name defaults to the
// id for hand-built graphs (BuildCallGraph sets a distinct simple name).
func (g *Graph) AddNode(id NodeID, kind string) {
	n := g.nodes[id]
	n.ID, n.Kind = id, kind
	if n.Name == "" {
		n.Name = string(id)
	}
	g.nodes[id] = n
}

// addNamed records an entity with an explicit simple name (the call-resolution key).
func (g *Graph) addNamed(id NodeID, kind, name string) {
	g.nodes[id] = Node{ID: id, Kind: kind, Name: name}
}

// Nodes returns every entity, ordered by id — the enumeration a caller needs to
// resolve a simple name (e.g. "Search") to its qualified node id ("(*Index).Search").
func (g *Graph) Nodes() []Node {
	out := make([]Node, 0, len(g.nodes))
	for _, n := range g.nodes {
		out = append(out, n)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// NodesByName returns the ids of every entity whose simple name matches — a call to
// "Search" may resolve to several methods named Search on different types.
func (g *Graph) NodesByName(name string) []NodeID {
	var out []NodeID
	for _, n := range g.Nodes() {
		if n.Name == name {
			out = append(out, n.ID)
		}
	}
	return out
}

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

// BuildCallGraph parses a single Go source file and returns its call graph. See
// BuildCallGraphFiles for the details; this is the one-file convenience.
func BuildCallGraph(src string) (*Graph, error) { return BuildCallGraphFiles(src) }

// BuildCallGraphFiles parses one or more Go source files (a package) and returns
// their call graph: a node per function AND per method, and a "calls" edge from a
// caller to each function/method it invokes by name. Resolution is syntactic and
// name-based — enough to be useful across a real, method-heavy package without a
// type checker:
//
//   - Free functions are keyed by name ("Foo"); methods by "(Recv).Method"
//     ("(*Index).Search"), so the two never collide in the node set.
//   - A bare call `foo()` resolves to every node named foo; a selector call
//     `x.Bar()` resolves to every method named Bar (the receiver type is not
//     checked — a documented syntactic approximation, not a type resolver).
//   - Self-recursion is not emitted as an edge.
//
// The earlier free-functions-only version produced an EMPTY graph on real Go
// (which is mostly methods); this is the dogfood-driven fix (#3439).
func BuildCallGraphFiles(srcs ...string) (*Graph, error) {
	fset := token.NewFileSet()
	g := NewGraph()

	type fnDecl struct {
		id   NodeID
		body *ast.BlockStmt
	}
	var decls []fnDecl
	nameIndex := map[string][]NodeID{} // simple name -> node ids (call resolution)

	// Pass 1: register every function and method as a node.
	for i, src := range srcs {
		file, err := parser.ParseFile(fset, fmt.Sprintf("src%d.go", i), src, 0)
		if err != nil {
			return nil, err
		}
		for _, decl := range file.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			name := fd.Name.Name
			id := NodeID(name)
			kind := "func"
			if fd.Recv != nil {
				id = NodeID("(" + receiverTypeName(fd.Recv) + ")." + name)
				kind = "method"
			}
			g.addNamed(id, kind, name)
			nameIndex[name] = append(nameIndex[name], id)
			decls = append(decls, fnDecl{id: id, body: fd.Body})
		}
	}

	// Pass 2: an edge per call to a known function/method.
	seen := map[string]bool{}
	for _, d := range decls {
		if d.body == nil {
			continue
		}
		ast.Inspect(d.body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			var callee string
			switch fn := call.Fun.(type) {
			case *ast.Ident:
				callee = fn.Name
			case *ast.SelectorExpr:
				callee = fn.Sel.Name
			}
			if callee == "" {
				return true
			}
			for _, tgt := range nameIndex[callee] {
				if tgt == d.id {
					continue // ignore self-recursion
				}
				key := string(d.id) + "\x00" + string(tgt)
				if seen[key] {
					continue
				}
				seen[key] = true
				g.AddEdge(d.id, tgt, "calls")
			}
			return true
		})
	}
	return g, nil
}

// receiverTypeName renders a method receiver's type for the node id: "T", "*T", or
// the base of a generic receiver "T[P]".
func receiverTypeName(recv *ast.FieldList) string {
	if recv == nil || len(recv.List) == 0 {
		return "?"
	}
	switch t := recv.List[0].Type.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			return "*" + id.Name
		}
	case *ast.IndexExpr: // generic receiver T[P]
		if id, ok := t.X.(*ast.Ident); ok {
			return id.Name + "[]"
		}
	case *ast.IndexListExpr: // generic receiver T[P, Q]
		if id, ok := t.X.(*ast.Ident); ok {
			return id.Name + "[]"
		}
	}
	return "?"
}

// StronglyConnectedComponents returns the graph's strongly connected components
// in deterministic order. Nodes inside a component are sorted by ID; components
// are ordered by their first ID. Edge kinds restrict the traversal when provided.
func (g *Graph) StronglyConnectedComponents(kinds ...string) [][]NodeID {
	allow := kindSet(kinds)
	index := 0
	indices := make(map[NodeID]int, len(g.nodes))
	lowlink := make(map[NodeID]int, len(g.nodes))
	onStack := make(map[NodeID]bool, len(g.nodes))
	stack := make([]NodeID, 0, len(g.nodes))
	components := make([][]NodeID, 0)

	var visit func(NodeID)
	visit = func(id NodeID) {
		indices[id] = index
		lowlink[id] = index
		index++
		stack = append(stack, id)
		onStack[id] = true

		edges := append([]Edge(nil), g.fwd[id]...)
		sort.Slice(edges, func(i, j int) bool { return edges[i].To < edges[j].To })
		for _, edge := range edges {
			if allow != nil && !allow[edge.Kind] {
				continue
			}
			to := edge.To
			if _, seen := indices[to]; !seen {
				visit(to)
				if lowlink[to] < lowlink[id] {
					lowlink[id] = lowlink[to]
				}
			} else if onStack[to] && indices[to] < lowlink[id] {
				lowlink[id] = indices[to]
			}
		}

		if lowlink[id] != indices[id] {
			return
		}
		component := make([]NodeID, 0, 1)
		for {
			last := len(stack) - 1
			member := stack[last]
			stack = stack[:last]
			onStack[member] = false
			component = append(component, member)
			if member == id {
				break
			}
		}
		sort.Slice(component, func(i, j int) bool { return component[i] < component[j] })
		components = append(components, component)
	}

	for _, node := range g.Nodes() {
		if _, seen := indices[node.ID]; !seen {
			visit(node.ID)
		}
	}
	sort.Slice(components, func(i, j int) bool { return components[i][0] < components[j][0] })
	return components
}
