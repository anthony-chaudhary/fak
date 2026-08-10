package codegraph

import (
	"go/ast"
	"go/parser"
	"go/token"
	"time"
)

var comparisonSources = []string{
	`package p

type T struct{}
func Start() { Step() }
func Step() { (&T{}).Method() }
`,
	`package p
func (t *T) Method() { Leaf() }
func Leaf() {}
`,
}

// ComparisonArm records a same-source call-graph implementation. Unavailable
// rows intentionally carry zero measurements until the real tool executes;
// wrappers around this package are not independent product witnesses.
type ComparisonArm struct {
	Name            string
	Kind            string
	Available       bool
	Correct         bool
	Latency         time.Duration
	Checks          int
	PassedChecks    int
	MissedChecks    int
	FalseNodes      int
	FalseEdges      int
	Nodes           int
	DirectEdges     int
	ForwardHits     int
	ReverseHits     int
	PathErrors      int
	CPUSeconds      float64
	PeakRSSBytes    int64
	InputBytes      int64
	NetworkBytes    int64
	OperatorSeconds float64
	CostUSD         float64
	Note            string
}

type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func runNativeComparison() ComparisonArm {
	a := ComparisonArm{Name: "fak native syntactic Go call graph", Kind: "native", Available: true, Checks: 4, Note: "multi-file function/method graph with forward and reverse transitive paths"}
	for _, src := range comparisonSources {
		a.InputBytes += int64(len(src))
	}
	start := time.Now()
	g, err := BuildCallGraphFiles(comparisonSources...)
	if err != nil {
		a.Latency = time.Since(start)
		a.FalseNodes++
		return a
	}
	forward := g.Reaches("Start", "calls")
	reverse := g.Dependents("Leaf", "calls")
	direct := g.BFS("Start", Traversal{EdgeKinds: []string{"calls"}, MaxDepth: 1})
	a.Latency = time.Since(start)
	a.Nodes = g.NodeCount()
	a.DirectEdges = len(direct)
	a.ForwardHits = len(forward)
	a.ReverseHits = len(reverse)
	if a.Nodes == 4 {
		a.PassedChecks++
	} else {
		a.FalseNodes++
	}
	if len(direct) == 1 && direct[0].ID == "Step" {
		a.PassedChecks++
	} else {
		a.FalseEdges++
	}
	if pathMatches(forward, "Leaf", 3, []NodeID{"Start", "Step", "(*T).Method", "Leaf"}) {
		a.PassedChecks++
	} else {
		a.PathErrors++
	}
	if pathMatches(reverse, "Start", 3, []NodeID{"Leaf", "(*T).Method", "Step", "Start"}) {
		a.PassedChecks++
	} else {
		a.PathErrors++
	}
	a.Correct = a.PassedChecks == a.Checks && a.FalseNodes == 0 && a.FalseEdges == 0 && a.PathErrors == 0
	return a
}

func pathMatches(hits []Hit, id NodeID, dist int, path []NodeID) bool {
	for _, h := range hits {
		if h.ID != id || h.Dist != dist || len(h.Path) != len(path) {
			continue
		}
		for i := range path {
			if h.Path[i] != path[i] {
				return false
			}
		}
		return true
	}
	return false
}

func runDirectASTBaseline() ComparisonArm {
	a := ComparisonArm{Name: "go/ast direct-call scan", Kind: "baseline", Available: true, Checks: 4, Note: "tuned no-graph baseline extracts direct call names but does not resolve nodes or compute forward/reverse paths"}
	start := time.Now()
	nodes := map[string]bool{}
	edges := map[string]bool{}
	for i, src := range comparisonSources {
		a.InputBytes += int64(len(src))
		f, err := parser.ParseFile(token.NewFileSet(), "src.go", src, 0)
		if err != nil {
			a.FalseNodes++
			continue
		}
		_ = i
		for _, decl := range f.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			nodes[fd.Name.Name] = true
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				switch fn := call.Fun.(type) {
				case *ast.Ident:
					edges[fd.Name.Name+"->"+fn.Name] = true
				case *ast.SelectorExpr:
					edges[fd.Name.Name+"->"+fn.Sel.Name] = true
				}
				return true
			})
		}
	}
	a.Latency = time.Since(start)
	a.Nodes = len(nodes)
	a.DirectEdges = len(edges)
	if len(nodes) == 4 {
		a.PassedChecks++
	}
	if edges["Start->Step"] && edges["Step->Method"] && edges["Method->Leaf"] {
		a.PassedChecks++
	}
	a.MissedChecks = 2
	a.Correct = false
	return a
}

func unavailableComparisonArm(name, note string) ComparisonArm {
	return ComparisonArm{Name: name, Kind: "external", Note: note}
}

func CompareLocal() ComparisonResult {
	arms := []ComparisonArm{
		runNativeComparison(),
		runDirectASTBaseline(),
		unavailableComparisonArm("golang.org/x/tools/go/callgraph", "requires pinned x/tools with equivalent loading and traversal"),
		unavailableComparisonArm("gopls call hierarchy", "requires a real pinned gopls/LSP session"),
		unavailableComparisonArm("Go guru callers and callees", "requires pinned guru execution"),
		unavailableComparisonArm("CodeQL Go call graph", "requires a real CodeQL database and equivalent query"),
		unavailableComparisonArm("SCIP Go code intelligence graph", "requires pinned scip-go indexing and equivalent traversal"),
	}
	return ComparisonResult{Workload: "build the same two-file Go function/method call graph and return exact direct edges, forward reach, reverse dependents, distances, and shortest paths", Arms: arms}
}
