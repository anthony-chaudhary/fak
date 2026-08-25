package compute

// #6412's applied fault-boundary path lives in `-tags cuda` sources that the default build
// neither compiles nor links (linking needs libfakcuda + the CUDA runtime), so a plain
// `go test ./internal/compute/` cannot execute the gate it adds. This witness parses those
// sources and pins the WIRING SHAPE that the tagged tests (cuda_devicefault_test.go,
// cuda_qwen35_gdn_test.go) execute on a GPU host:
//
//  1. the registered cuda backend constructs a session fault latch (cuda.go init),
//  2. the backend publishes it through the DeviceFaultReporter capability,
//  3. the GDN decode entry admits through the latch as its FIRST statement — before
//     geometry, locks, or any allocation, and
//  4. the kernel-failure branch observes the fault into the latch.
//
// It follows the package's cuda_header_preflight_test.go precedent: a deterministic,
// GPU-free structural check that makes a refactor which silently drops the fail-closed gate
// fail the default CI run instead of the next paid GPU acceptance run. Behavior stays
// witnessed by the tagged tests; this guards presence, not semantics.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func parseCUDAWiringSource(t *testing.T, filename string) *ast.File {
	t.Helper()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filename, nil, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse %s: %v", filename, err)
	}
	return file
}

// cudaBackendFuncDecl finds the named method on *cudaBackend.
func cudaBackendFuncDecl(t *testing.T, file *ast.File, name string) *ast.FuncDecl {
	t.Helper()
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != name || fn.Recv == nil || len(fn.Recv.List) != 1 {
			continue
		}
		star, ok := fn.Recv.List[0].Type.(*ast.StarExpr)
		if !ok {
			continue
		}
		if ident, ok := star.X.(*ast.Ident); ok && ident.Name == "cudaBackend" {
			return fn
		}
	}
	t.Fatalf("%s: no func (c *cudaBackend) %s found", file.Name.Name, name)
	return nil
}

// countLatchCalls counts calls of the form <recv>.faultLatch.<method>(...) under root.
func countLatchCalls(root ast.Node, method string) int {
	count := 0
	ast.Inspect(root, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != method {
			return true
		}
		inner, ok := sel.X.(*ast.SelectorExpr)
		if ok && inner.Sel.Name == "faultLatch" {
			count++
		}
		return true
	})
	return count
}

func countSelectorCalls(root ast.Node, method string) int {
	count := 0
	ast.Inspect(root, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == method {
			count++
		}
		return true
	})
	return count
}

func TestCUDAInitConstructsSessionFaultLatch(t *testing.T) {
	file := parseCUDAWiringSource(t, "cuda.go")
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "init" || fn.Recv != nil {
			continue
		}
		found := false
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "NewDeviceFaultLatch" {
				found = true
			}
			return true
		})
		if !found {
			t.Fatal("cuda.go init() registers the backend without constructing its session fault latch")
		}
		return
	}
	t.Fatal("cuda.go: no init() found")
}

func TestCUDABackendPublishesDeviceFaultReporterCapability(t *testing.T) {
	file := parseCUDAWiringSource(t, "cuda_backend_state.go")
	fn := cudaBackendFuncDecl(t, file, "DeviceFaultLatch")
	if fn.Type.Results == nil || len(fn.Type.Results.List) != 1 {
		t.Fatal("DeviceFaultLatch method does not return the single latch the reporter capability requires")
	}
}

func TestCUDAQwen35GDNDecodeAdmitsThroughLatchFirst(t *testing.T) {
	file := parseCUDAWiringSource(t, "cuda_specialized.go")
	fn := cudaBackendFuncDecl(t, file, "Qwen35GDNDecode")
	if len(fn.Body.List) == 0 {
		t.Fatal("Qwen35GDNDecode has an empty body")
	}
	// The gate must be the FIRST statement: admitting after geometry validation or after
	// allocations would let a poisoned session spend device work (or panic on a poisoned
	// operand) before the typed refusal.
	if got := countLatchCalls(fn.Body.List[0], "Admit"); got != 1 {
		t.Fatalf("Qwen35GDNDecode first statement contains %d faultLatch.Admit calls, want exactly 1", got)
	}
	observations := 0
	for _, helperName := range []string{"allocateQwen35GDN", "failQwen35GDN"} {
		if got := countSelectorCalls(fn.Body, helperName); got != 1 {
			t.Fatalf("Qwen35GDNDecode calls %s %d times, want exactly 1", helperName, got)
		}
		helper := cudaBackendFuncDecl(t, file, helperName)
		observations += countLatchCalls(helper.Body, "ObserveError")
	}
	if observations < 2 {
		t.Fatalf("Qwen35GDNDecode helpers observe %d fault sites into the latch, want >= 2 (strict allocation + kernel status)", observations)
	}
}
