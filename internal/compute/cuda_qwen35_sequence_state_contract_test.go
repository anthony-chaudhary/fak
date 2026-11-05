package compute

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// This is a source contract for the CUDA-only entry point, not device execution.
// An ordinary allocation failure after GDN execution must invalidate recurrent
// state just as a retained-quant kernel failure does.
// fak-test:runtime fast est=1ms lane=default
func TestCUDASequenceFailedUnwindPoisonsAdvancedState(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "cuda_qwen35_sequence.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var entry *ast.FuncDecl
	for _, decl := range f.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "Qwen35SequencePrefill" {
			entry = fn
		}
	}
	if entry == nil {
		t.Fatal("missing CUDA sequence entry point")
	}
	isIdent := func(expr ast.Expr, name string) bool {
		id, ok := expr.(*ast.Ident)
		return ok && id.Name == name
	}
	calls := func(node ast.Node, method string) bool {
		found := false
		ast.Inspect(node, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if ok {
				selector, ok := call.Fun.(*ast.SelectorExpr)
				found = found || ok && isIdent(selector.X, "c") && selector.Sel.Name == method
			}
			return true
		})
		return found
	}
	var cleanup *ast.DeferStmt
	for _, stmt := range entry.Body.List {
		if d, ok := stmt.(*ast.DeferStmt); ok && calls(d, "qwen35SequenceReleaseSinceLocked") {
			cleanup = d
		}
	}
	if cleanup == nil {
		t.Fatal("missing failure cleanup defer")
	}
	literal, ok := cleanup.Call.Fun.(*ast.FuncLit)
	if !ok || len(literal.Body.List) != 1 {
		t.Fatal("cleanup must be a single failure-guarded closure")
	}
	failure, ok := literal.Body.List[0].(*ast.IfStmt)
	if !ok || !isIdent(failure.Cond, "failed") {
		t.Fatal("successful completion must not poison state")
	}
	guarded := false
	for _, stmt := range failure.Body.List {
		if guard, ok := stmt.(*ast.IfStmt); ok && calls(guard.Body, "qwen35SequencePoisonStates") {
			condition, ok := guard.Cond.(*ast.BinaryExpr)
			guarded = ok && condition.Op == token.LOR && isIdent(condition.X, "statesMayHaveAdvanced")
		}
	}
	if !guarded {
		t.Fatal("failed stateful execution must poison independently of the returned error type")
	}
	assignments, launches := 0, 0
	ast.Inspect(entry.Body, func(node ast.Node) bool {
		if assignment, ok := node.(*ast.AssignStmt); ok && len(assignment.Lhs) == 1 && isIdent(assignment.Lhs[0], "statesMayHaveAdvanced") {
			assignments++
			if assignment.Tok == token.DEFINE && (len(assignment.Rhs) != 1 || !isIdent(assignment.Rhs[0], "false") || assignment.Pos() >= cleanup.Pos()) {
				t.Error("stateful marker must start false before failure cleanup is registered")
			}
		}
		block, ok := node.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			assignment, ok := stmt.(*ast.AssignStmt)
			if !ok || !calls(assignment, "qwen35SequenceGDNLocked") {
				continue
			}
			launches++
			if i == 0 {
				t.Fatal("GDN launch has no preceding stateful marker")
			}
			marker, ok := block.List[i-1].(*ast.AssignStmt)
			if !ok || marker.Tok != token.ASSIGN || len(marker.Lhs) != 1 || !isIdent(marker.Lhs[0], "statesMayHaveAdvanced") || len(marker.Rhs) != 1 || !isIdent(marker.Rhs[0], "true") || marker.Pos() <= cleanup.End() {
				t.Error("mark stateful execution immediately before calling GDN, including failed calls")
			}
		}
		return true
	})
	if assignments != 2 || launches != 1 {
		t.Fatalf("stateful marker must not be reset: assignments=%d, GDN sites=%d", assignments, launches)
	}
}
