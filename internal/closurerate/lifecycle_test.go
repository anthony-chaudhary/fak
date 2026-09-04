package closurerate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"testing"
)

// TestClosureRateLifecycle verifies core throughput and honesty counters on a known ledger fixture.
func TestClosureRateLifecycle(t *testing.T) {
	t.Parallel()

	m := Fold(fixtureLedger, 4.0)
	if m.Total != 10 {
		t.Fatalf("expected total 10, got %d", m.Total)
	}
	if m.Closed != 8 {
		t.Fatalf("expected closed 8, got %d", m.Closed)
	}
	if m.Witnessed != 6 {
		t.Fatalf("expected witnessed 6, got %d", m.Witnessed)
	}
	if m.ClaimedWithoutWitness != 2 {
		t.Fatalf("expected 2 unwitnessed closes, got %d", m.ClaimedWithoutWitness)
	}
}

// TestClosureRateBenchmarkMaturity verifies that bench_test.go defines a substantive BenchmarkClosureRate.
func TestClosureRateBenchmarkMaturity(t *testing.T) {
	benchPath := filepath.Join(".", "bench_test.go")
	content, err := os.ReadFile(benchPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", benchPath, err)
	}

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, benchPath, content, 0)
	if err != nil {
		t.Fatalf("failed to parse %s: %v", benchPath, err)
	}

	hasBenchmarkClosureRate := false
	for _, decl := range node.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "BenchmarkClosureRate" {
			continue
		}
		if fn.Body == nil || len(fn.Body.List) == 0 {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			if forStmt, ok := n.(*ast.ForStmt); ok {
				if forStmt.Cond != nil {
					ast.Inspect(forStmt.Cond, func(cn ast.Node) bool {
						if id, ok := cn.(*ast.Ident); ok && id.Name == "N" {
							hasBenchmarkClosureRate = true
						}
						return true
					})
				}
			}
			return true
		})
	}

	if !hasBenchmarkClosureRate {
		t.Errorf("benchmark_test.go must define BenchmarkClosureRate containing a b.N loop")
	}
}
