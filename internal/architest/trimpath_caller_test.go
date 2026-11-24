package architest

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// Tests in this module must not locate files through runtime.Caller. Under
// -trimpath (the lane and land-verify default, and the cure for path-keyed
// GOCACHE growth) the recorded source path is module-qualified, not a directory,
// so such a test goes red only for the builds that keep the cache bounded.
// internal/testenv/repotest resolves the package dir and repo root from the working
// directory instead.

// trimpathScanDotDirs are the dot-directories that hold module tests; every
// other dot-directory is tool state (.git, .worktrees, .gocache, ...).
var trimpathScanDotDirs = map[string]bool{".github": true, ".claude": true, ".agents": true, ".githooks": true}

func TestTestsDoNotLocateFilesViaRuntimeCaller(t *testing.T) {
	root := repoRoot(t)
	var offenders []string
	scanned := 0
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if p == root {
				return nil
			}
			name := d.Name()
			if name == "testdata" || name == "vendor" || name == "node_modules" ||
				(strings.HasPrefix(name, ".") && !trimpathScanDotDirs[name]) {
				return filepath.SkipDir
			}
			if _, err := os.Stat(filepath.Join(p, "go.mod")); err == nil {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, "_test.go") {
			return nil
		}
		src, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		scanned++
		if !bytes.Contains(src, []byte("runtime.Caller(")) {
			return nil
		}
		lines, err := runtimeCallerLines(p, src)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		for _, line := range lines {
			offenders = append(offenders, filepath.ToSlash(rel)+":"+strconv.Itoa(line))
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", root, err)
	}
	if scanned == 0 {
		t.Fatalf("scanned no _test.go files under %s; the walk no longer covers the module", root)
	}
	if len(offenders) > 0 {
		t.Fatalf("tests call runtime.Caller, which names no directory under -trimpath; "+
			"use internal/testenv/repotest (PackageDir, Root, Path) instead:\n  %s", strings.Join(offenders, "\n  "))
	}
}

func TestRuntimeCallerLinesFlagsOnlyFileResults(t *testing.T) {
	src := []byte(`package p

import rt "runtime"

// rt.Caller(0) in a comment
var fixture = "_, f, _, _ := rt.Caller(0)"

func f() {
	_, file, _, _ := rt.Caller(0)
	pc, _, _, _ := rt.Caller(2)
	use(rt.Caller(1))
	_, _ = file, pc
}
`)
	lines, err := runtimeCallerLines("p_test.go", src)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{9, 11}; len(lines) != len(want) || lines[0] != want[0] || lines[1] != want[1] {
		t.Fatalf("runtimeCallerLines = %v, want %v", lines, want)
	}
}

// runtimeCallerLines returns the lines of runtime.Caller calls whose file
// result is kept. A call assigned with a blank file result (frame or function
// lookups via pc) is fine; any other use is flagged because its file result may
// flow anywhere. The import name is resolved so a renamed import is caught.
func runtimeCallerLines(path string, src []byte) ([]int, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, src, parser.SkipObjectResolution)
	if err != nil {
		return nil, err
	}
	pkgName := ""
	for _, imp := range file.Imports {
		if imp.Path.Value != `"runtime"` {
			continue
		}
		pkgName = "runtime"
		if imp.Name != nil {
			pkgName = imp.Name.Name
		}
	}
	if pkgName == "" || pkgName == "_" {
		return nil, nil
	}
	isCaller := func(e ast.Expr) bool {
		call, ok := e.(*ast.CallExpr)
		if !ok {
			return false
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "Caller" {
			return false
		}
		id, ok := sel.X.(*ast.Ident)
		return ok && id.Name == pkgName
	}
	fileDiscarded := map[ast.Expr]bool{}
	var lines []int
	ast.Inspect(file, func(n ast.Node) bool {
		if as, ok := n.(*ast.AssignStmt); ok && len(as.Rhs) == 1 && len(as.Lhs) == 4 && isCaller(as.Rhs[0]) {
			if id, ok := as.Lhs[1].(*ast.Ident); ok && id.Name == "_" {
				fileDiscarded[as.Rhs[0]] = true
			}
		}
		if e, ok := n.(ast.Expr); ok && isCaller(e) && !fileDiscarded[e] {
			lines = append(lines, fset.Position(e.Pos()).Line)
		}
		return true
	})
	return lines, nil
}
