package compute

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// Estimate only; not timed. Parses source without linking or executing CUDA.
// fak-test:runtime fast est=20ms lane=default
func TestCUDACloneTensorSerializesAllocationAndCopy(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "cuda.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var clone *ast.FuncDecl
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "CloneTensor" || fn.Recv == nil || len(fn.Recv.List) != 1 {
			continue
		}
		ptr, ok := fn.Recv.List[0].Type.(*ast.StarExpr)
		if !ok {
			continue
		}
		name, ok := ptr.X.(*ast.Ident)
		if ok && name.Name == "cudaBackend" {
			clone = fn
			break
		}
	}
	if clone == nil || clone.Body == nil {
		t.Fatal("cudaBackend.CloneTensor declaration missing")
	}
	isMutexCall := func(call *ast.CallExpr, method string) bool {
		if call == nil || len(call.Args) != 0 {
			return false
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || selector.Sel.Name != method {
			return false
		}
		owner, ok := selector.X.(*ast.Ident)
		return ok && owner.Name == "cudaMu"
	}
	hasEntryGuard := func(body []ast.Stmt) bool {
		if len(body) < 2 {
			return false
		}
		lock, ok := body[0].(*ast.ExprStmt)
		if !ok {
			return false
		}
		call, ok := lock.X.(*ast.CallExpr)
		if !ok || !isMutexCall(call, "Lock") {
			return false
		}
		unlock, ok := body[1].(*ast.DeferStmt)
		return ok && isMutexCall(unlock.Call, "Unlock")
	}
	if !hasEntryGuard(clone.Body.List) {
		t.Fatal("CloneTensor must lock cudaMu and defer its unlock before validation, allocation, or copy")
	}
	// The guard must reject the pre-fix body and an unlock that would run before
	// the device operation. These controls do not execute any backend code.
	if hasEntryGuard(clone.Body.List[2:]) {
		t.Fatal("guard accepted a body with entry serialization removed")
	}
	unlock := clone.Body.List[1].(*ast.DeferStmt)
	earlyUnlock := append([]ast.Stmt(nil), clone.Body.List...)
	earlyUnlock[1] = &ast.ExprStmt{X: unlock.Call}
	if hasEntryGuard(earlyUnlock) {
		t.Fatal("guard accepted an immediate unlock")
	}
}
