package model

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestQwen35BatchDiagnosticIncludesSplit(t *testing.T) {
	// Check the actual dispatcher, without sleep-based timing thresholds or a
	// replacement split implementation. This is an ordering guard, not a timer
	// accuracy, backend execution, or end-to-end latency witness.
	file, err := parser.ParseFile(token.NewFileSet(), "qwen35_batch_decode.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var body *ast.BlockStmt
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "stepBatchQwen35HybridQ4KIndependent" {
			body = fn.Body
		}
	}
	if body == nil {
		t.Fatal("missing independent batch dispatcher")
	}
	var split, closeSplit, diagnostic token.Pos
	var rows string
	var calls int
	ast.Inspect(body, func(node ast.Node) bool {
		if call, ok := node.(*ast.CallExpr); ok {
			if name, ok := call.Fun.(*ast.Ident); ok && name.Name == "splitScaledLogits" {
				calls++
			}
		}
		return true
	})
	for _, stmt := range body.List {
		switch stmt := stmt.(type) {
		case *ast.AssignStmt:
			if len(stmt.Lhs) != 1 || len(stmt.Rhs) != 1 {
				continue
			}
			name, ok := stmt.Lhs[0].(*ast.Ident)
			if !ok {
				continue
			}
			if name.Name == "diagSplit" {
				closeSplit = stmt.Pos()
			}
			if call, ok := stmt.Rhs[0].(*ast.CallExpr); ok {
				if fn, ok := call.Fun.(*ast.Ident); ok && fn.Name == "splitScaledLogits" {
					split, rows = stmt.Pos(), name.Name
				}
			}
		case *ast.IfStmt:
			if name, ok := stmt.Cond.(*ast.Ident); ok && name.Name == "qwen35BatchDiagEnabled" {
				diagnostic = stmt.Pos()
			}
		}
	}
	if calls != 1 || split == token.NoPos || closeSplit <= split || diagnostic <= closeSplit {
		t.Fatalf("split must execute once before its timing and diagnostic: calls=%d split=%d close=%d diagnostic=%d", calls, split, closeSplit, diagnostic)
	}
	last, ok := body.List[len(body.List)-1].(*ast.ReturnStmt)
	if !ok || last.Pos() <= diagnostic || len(last.Results) != 2 {
		t.Fatal("dispatcher must return the split rows after the diagnostic")
	}
	gotRows, rowsOK := last.Results[0].(*ast.Ident)
	accepted, acceptedOK := last.Results[1].(*ast.Ident)
	if !rowsOK || gotRows.Name != rows || !acceptedOK || accepted.Name != "true" {
		t.Fatal("dispatcher must return the already measured rows as accepted")
	}
}
