package compute

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=1s
func TestCUDAInkling2ResidentAllocationIsDeviceOnly(t *testing.T) {
	file := parseGoFile(t, "cuda_inkling2.go")
	fn := namedFunction(t, file, "Ink2Alloc")
	calls := selectorCalls(fn.Body)
	if calls["dallocDeviceClass"] != 1 {
		t.Fatalf("Ink2Alloc dallocDeviceClass calls = %d, want 1", calls["dallocDeviceClass"])
	}
	for _, forbidden := range []string{"dallocClass", "dallocManagedClass", "fcuda_malloc_managed_tagged"} {
		if calls[forbidden] != 0 {
			t.Fatalf("Ink2Alloc calls fallback allocator %s", forbidden)
		}
	}

	allocator := namedFunction(t, parseGoFile(t, "cuda.go"), "dallocDeviceClass")
	allocatorCalls := selectorCalls(allocator.Body)
	if allocatorCalls["fcuda_malloc_tagged"] != 1 {
		t.Fatalf("dallocDeviceClass fcuda_malloc_tagged calls = %d, want 1", allocatorCalls["fcuda_malloc_tagged"])
	}
	if allocatorCalls["fcuda_malloc_managed_tagged"] != 0 {
		t.Fatal("dallocDeviceClass silently falls back to managed memory")
	}
}

func parseGoFile(t *testing.T, name string) *ast.File {
	t.Helper()
	path := filepath.Join(".", name)
	f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return f
}

func namedFunction(t *testing.T, file *ast.File, name string) *ast.FuncDecl {
	t.Helper()
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == name {
			return fn
		}
	}
	t.Fatalf("function %s not found", name)
	return nil
}

func selectorCalls(node ast.Node) map[string]int {
	calls := make(map[string]int)
	ast.Inspect(node, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
			calls[sel.Sel.Name]++
		}
		return true
	})
	return calls
}
