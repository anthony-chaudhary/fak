package compute

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

// This executes the production status gate with CPU-owned stand-ins. It does
// not initialize Vulkan or establish driver, device, or native-free behavior.
// fak-test:runtime fast est=1ms lane=default
func TestVulkanTeardownAfterFlushRetainsOwners(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		status int
		class  VulkanErrorClass
		cause  error
	}{
		{-4, VulkanClassDeviceLost, ErrVulkanDeviceLost},
		{-13, VulkanClassSubmissionFailed, ErrVulkanSubmissionFailed},
		{2, VulkanClassSubmissionFailed, ErrVulkanSubmissionFailed},
	} {
		t.Run(fmt.Sprint(tc.status), func(t *testing.T) {
			owners := struct{ homes, staging, accounted int }{2, 8192, 65536}
			before := owners
			releases := 0
			for attempt := 0; attempt < 2; attempt++ {
				err := vulkanTeardownAfterFlush(tc.status, func() {
					releases++
					owners.homes, owners.staging, owners.accounted = 0, 0, 0
				})
				if releases != 0 || owners != before {
					t.Fatalf("attempt %d: failed fence released ownership: releases=%d owners=%+v", attempt, releases, owners)
				}
				var typed *BackendError
				if !errors.As(err, &typed) || !errors.Is(err, tc.cause) {
					t.Fatalf("attempt %d: missing typed failure: %v", attempt, err)
				}
				if typed.Backend != "vulkan" || typed.Site != "TeardownResources" || typed.Class != tc.class ||
					!strings.Contains(typed.Message, fmt.Sprintf("fault %d;", tc.status)) {
					t.Fatalf("attempt %d: failure identity changed: %+v", attempt, typed)
				}
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestVulkanTeardownAfterFlushSuccess(t *testing.T) {
	t.Parallel()
	owned, releases := true, 0
	for attempt := 1; attempt <= 2; attempt++ {
		if err := vulkanTeardownAfterFlush(0, func() {
			releases++
			owned = false
		}); err != nil {
			t.Fatal(err)
		}
		if owned || releases != attempt {
			t.Fatalf("attempt %d: release calls=%d owned=%t", attempt, releases, owned)
		}
	}
}

// Bind the CPU-tested gate to the actual tagged method without compiling or
// initializing Vulkan. The unchanged native release body stays inside its
// success-only closure; this source check is not a device execution witness.
// fak-test:runtime fast est=3ms lane=default
func TestVulkanTeardownAfterFlushWiring(t *testing.T) {
	t.Parallel()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "vulkan.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var method *ast.FuncDecl
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if ok && fn.Name.Name == "TeardownResources" && fn.Recv != nil {
			method = fn
			break
		}
	}
	if method == nil || len(method.Recv.List) != 1 || method.Body == nil || len(method.Body.List) != 4 {
		t.Fatal("teardown must lock, defer unlock, flush, then return the status gate")
	}
	render := func(node ast.Node) string {
		var out bytes.Buffer
		if err := format.Node(&out, fset, node); err != nil {
			t.Fatal(err)
		}
		return out.String()
	}
	if render(method.Recv.List[0].Type) != "*vulkanBackend" {
		t.Fatal("teardown gate must be wired to the actual Vulkan backend")
	}
	for i, want := range []string{"vulkanMu.Lock()", "defer vulkanMu.Unlock()", "C.fvk_batch_flush()"} {
		if got := render(method.Body.List[i]); got != want {
			t.Fatalf("teardown step %d = %q, want %q", i, got, want)
		}
	}
	ret, ok := method.Body.List[3].(*ast.ReturnStmt)
	if !ok || len(ret.Results) != 1 {
		t.Fatal("teardown must return the gate error")
	}
	call, ok := ret.Results[0].(*ast.CallExpr)
	if !ok || render(call.Fun) != "vulkanTeardownAfterFlush" || len(call.Args) != 2 ||
		render(call.Args[0]) != "int(C.fvk_submission_status())" {
		t.Fatal("teardown must pass the sticky status immediately after the plain flush")
	}
	release, ok := call.Args[1].(*ast.FuncLit)
	if !ok || render(release.Body) != `{
	v.freeHomesLocked()
	if v.q4kStagePtr != nil {
		C.fvk_free(v.q4kStagePtr)
		v.dlUsed -= v.q4kStageBytes
		v.q4kStagePtr = nil
		v.q4kStageBytes = 0
	}
	C.fvk_trim_pool()
}` {
		t.Fatal("existing teardown releases must remain inside the success-only gate, in order")
	}
}
