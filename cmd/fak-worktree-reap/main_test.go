package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// fak-test:runtime unit est=100ms lane=default
func TestNativeReapRoutesShareBoundedImplementation(t *testing.T) {
	for _, tc := range []struct{ file, fn, call string }{{"main.go", "main", "Run"}, {"../fak/worktree_worker_reap.go", "worktreeWorkerReap", "RunSingle"}} {
		f, err := parser.ParseFile(token.NewFileSet(), tc.file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || fn.Name.Name != tc.fn {
				continue
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				c, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				s, ok := c.Fun.(*ast.SelectorExpr)
				if !ok {
					return true
				}
				x, ok := s.X.(*ast.Ident)
				if ok && x.Name == "worktreereapcli" && s.Sel.Name == tc.call {
					found = true
				}
				return true
			})
		}
		if !found {
			t.Fatalf("%s does not invoke shared %s", tc.file, tc.call)
		}
	}
}

// fak-test:runtime unit est=100ms lane=default
func TestNativeReapDependencyBoundary(t *testing.T) {
	root := filepath.Clean("../..")
	const prefix = "github.com/anthony-chaudhary/fak/"
	seen := map[string]bool{}
	queue := []string{"cmd/fak-worktree-reap"}
	for len(queue) > 0 {
		pkg := queue[0]
		queue = queue[1:]
		if seen[pkg] {
			continue
		}
		seen[pkg] = true
		if pkg == "cmd/fak" || pkg == "internal/model" || strings.HasPrefix(pkg, "internal/model/") || pkg == "internal/agent" || strings.HasPrefix(pkg, "internal/agent/") || pkg == "internal/compute" || strings.HasPrefix(pkg, "internal/compute/") {
			t.Fatalf("inference dependency introduced: %s", pkg)
		}
		entries, err := os.ReadDir(filepath.Join(root, pkg))
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
				continue
			}
			f, err := parser.ParseFile(token.NewFileSet(), filepath.Join(root, pkg, e.Name()), nil, parser.ImportsOnly)
			if err != nil {
				t.Fatal(err)
			}
			for _, imp := range f.Imports {
				p, err := strconv.Unquote(imp.Path.Value)
				if err != nil {
					t.Fatal(err)
				}
				if strings.HasPrefix(p, prefix) {
					queue = append(queue, strings.TrimPrefix(p, prefix))
				}
			}
		}
	}
}
